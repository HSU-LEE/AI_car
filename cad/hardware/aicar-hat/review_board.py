"""Board review checks DRC does not cover. Run with KiCad's Python.

1. Through-hole tails against tall Raspberry Pi 4 parts (positions from the official mechanical drawing).
2. Bottleneck copper width between key points of each power net, across both layers.
"""
import heapq

import pcbnew

OX, OY = 100.0, 100.0
STEP = 0.2
to = pcbnew.ToMM

# name: (x0, y0, x1, y1, height above the Pi board, mm). None means the height is not on the drawing.
PI_PARTS = {
    "PoE header": (59.1, 7.2, 64.1, 12.2, None),
    "SoC (heatsink)": (21.7, 16.0, 36.8, 31.0, None),
    "RAM": (39.0, 16.0, 50.5, 31.0, 2.4),
    "CSI connector": (45.0, 33.2, 48.0, 55.8, 5.5),
    "DSI connector": (1.5, 16.8, 5.5, 38.5, 5.5),
    "Audio jack": (50.5, 43.5, 57.5, 58.0, 6.0),
    "USB-C": (7.0, 49.0, 15.5, 57.0, 3.2),
    "HDMI0": (23.0, 49.5, 29.5, 57.0, 3.0),
    "HDMI1": (36.5, 49.5, 43.0, 57.0, 3.0),
}
TAIL = 2.5  # Typical through-hole tail below the HAT after soldering, mm.
GAP = 11.0  # HAT underside above the Pi board with an 8.5 mm socket and 11 mm standoffs.


def tails(board):
    print("--- through-hole tails over Pi parts")
    print(f"tail tip about {GAP - TAIL:.1f} mm above the Pi board")
    for f in board.GetFootprints():
        ref = f.GetReference()
        if ref == "J1" or ref.startswith("H"):
            continue
        for p in f.Pads():
            if not p.HasHole():
                continue
            x, y = to(p.GetPosition().x) - OX, to(p.GetPosition().y) - OY
            for name, (x0, y0, x1, y1, z) in PI_PARTS.items():
                if x0 - 0.5 <= x <= x1 + 0.5 and y0 - 0.5 <= y <= y1 + 0.5:
                    note = "height unknown" if z is None else f"{z} mm, clear by {GAP - TAIL - z:.1f}"
                    print(f"  {ref}.{p.GetNumber()} ({x:.1f}, {y:.1f}) net {p.GetNetname() or '-'} over {name}: {note}")


def copper_map(board, net):
    """Grid cells carrying this net, per layer, plus layer links at vias and through-hole pads."""
    cols, rows = int(66 / STEP), int(58 / STEP)
    cells = {pcbnew.F_Cu: set(), pcbnew.B_Cu: set()}
    links = set()

    def cell(x, y):
        return int(round((to(x) - OX) / STEP)), int(round((to(y) - OY) / STEP))

    def pos(i, j):
        return pcbnew.VECTOR2I(pcbnew.FromMM(OX + i * STEP), pcbnew.FromMM(OY + j * STEP))

    for z in board.Zones():
        if z.GetNetname() != net:
            continue
        layer = z.GetFirstLayer()
        polys = z.GetFilledPolysList(layer)
        for n in range(polys.OutlineCount()):
            one = pcbnew.SHAPE_POLY_SET()
            one.AddOutline(polys.COutline(n))
            b = polys.COutline(n).BBox()
            i0, j0 = cell(b.GetLeft(), b.GetTop())
            i1, j1 = cell(b.GetRight(), b.GetBottom())
            for i in range(max(0, i0), min(cols, i1 + 1)):
                for j in range(max(0, j0), min(rows, j1 + 1)):
                    if one.Contains(pos(i, j)):
                        cells[layer].add((i, j))
    for t in board.GetTracks():
        if t.GetNetname() != net:
            continue
        if t.Type() == pcbnew.PCB_VIA_T:
            c = cell(t.GetPosition().x, t.GetPosition().y)
            links.add(c)
            for layer in cells:
                cells[layer].add(c)
            continue
        s, e, hw = t.GetStart(), t.GetEnd(), to(t.GetWidth()) / 2
        length = max(to(t.GetLength()), 0.01)
        steps = int(length / (STEP / 2)) + 1
        r = int(round(hw / STEP))
        for k in range(steps + 1):
            x = s.x + (e.x - s.x) * k // steps
            y = s.y + (e.y - s.y) * k // steps
            ci, cj = cell(x, y)
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if (di * STEP) ** 2 + (dj * STEP) ** 2 <= hw ** 2 + STEP ** 2 / 2:
                        cells[t.GetLayer()].add((ci + di, cj + dj))
    terminals = {}
    for f in board.GetFootprints():
        for p in f.Pads():
            if p.GetNetname() != net:
                continue
            bb = p.GetBoundingBox()
            i0, j0 = cell(bb.GetLeft(), bb.GetTop())
            i1, j1 = cell(bb.GetRight(), bb.GetBottom())
            pad_cells = [(i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)]
            for layer in cells:
                if p.IsOnLayer(layer):
                    cells[layer].update(pad_cells)
            c = cell(p.GetPosition().x, p.GetPosition().y)
            terminals[f"{f.GetReference()}.{p.GetNumber()}"] = (c, p.GetLayerSet().Contains(pcbnew.F_Cu))
            if p.HasHole():
                links.add(c)
    return cells, links, terminals


def clearance_map(cells):
    """Half-width of copper at each cell: distance to the nearest non-copper cell, in mm."""
    dist = {}
    frontier = []
    for c in cells:
        i, j = c
        if any((i + a, j + b) not in cells for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1))):
            dist[c] = STEP
            frontier.append(c)
    d = STEP
    while frontier:
        d += STEP
        nxt = []
        for i, j in frontier:
            for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (i + a, j + b)
                if n in cells and n not in dist:
                    dist[n] = d
                    nxt.append(n)
        frontier = nxt
    return dist


def bottleneck(cells, links, start, goal):
    """Widest path from start to goal; returns its narrowest copper width, mm."""
    dist = {layer: clearance_map(cells[layer]) for layer in cells}
    layers = list(cells)
    best = {}
    heap = []
    for layer in layers:
        if start in dist[layer]:
            heap.append((-dist[layer][start], start, layer))
            best[(start, layer)] = dist[layer][start]
    heapq.heapify(heap)
    while heap:
        w, c, layer = heapq.heappop(heap)
        w = -w
        if c == goal:
            return 2 * w
        i, j = c
        moves = [((i + a, j + b), layer) for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        if c in links:
            moves += [(c, other) for other in layers if other != layer]
        for n, lay in moves:
            if n not in dist[lay]:
                continue
            nw = min(w, dist[lay][n]) if c != n else w
            if nw > best.get((n, lay), 0):
                best[(n, lay)] = nw
                heapq.heappush(heap, (-nw, n, lay))
    return 0.0


def widths(board):
    print("--- narrowest copper on the best path (mm; about 1 mm carries 2.5-3 A on 1 oz outer copper)")
    for net, pairs in (
            ("GND", [("J2.2", "U4.9"), ("J2.2", "U5.9"), ("J2.2", "U6.3"), ("J3.2", "U4.9"), ("U4.9", "J1.6"),
                     ("U5.9", "J5.1"), ("U6.3", "J3.2")]),
            ("BAT1", [("J2.1", "Q1.2")]), ("BAT2", [("J3.1", "Q2.2")]),
            ("VOR", [("Q1.6", "Q3.2"), ("Q2.6", "Q3.2")]), ("VRAW", [("Q3.6", "F1.1")]),
            ("VSW", [("F1.2", "U4.7"), ("F1.2", "U5.7"), ("F1.2", "U6.13"), ("F1.2", "U6.24")]),
            ("5V_PI", [("L1.2", "J1.2"), ("L1.2", "J1.4")]), ("5V_SRV", [("L2.2", "J5.2")]),
            ("MOT1", [("U6.1", "J4.1")]), ("MOT2", [("U6.5", "J4.2")])):
        cells, links, terms = copper_map(board, net)
        for a, b in pairs:
            w = bottleneck(cells, links, terms[a][0], terms[b][0])
            print(f"  {net:7} {a:6} -> {b:6} {w:.1f}")


def main():
    board = pcbnew.LoadBoard("aicar-hat.kicad_pcb")
    tails(board)
    widths(board)


if __name__ == "__main__":
    main()
