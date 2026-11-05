"""Check that nothing sits in the space ports need for plugs, wires and a soldering iron. Run with KiCad's Python.

Each keep-clear box is x0, y0, x1, y1 in mm from the board's top-left corner. Parts are compared by pads and body outline.
"""
import pcbnew

OX, OY = 100.0, 100.0
to = pcbnew.ToMM
MARGIN = 2.0    # Around plug housings, wire pads.
IRON = 1.0      # Beside J1's top-side solder joints.
SCREW = 2.5     # M2.5 screw head radius, rounded up.


def pads_box(f):
    bs = [p.GetBoundingBox() for p in f.Pads()]
    return (min(to(b.GetLeft()) for b in bs) - OX, min(to(b.GetTop()) for b in bs) - OY,
            max(to(b.GetRight()) for b in bs) - OX, max(to(b.GetBottom()) for b in bs) - OY)


def body_box(f):
    boxes = [p.GetBoundingBox() for p in f.Pads()]
    for g in f.GraphicalItems():
        if g.GetLayer() in (pcbnew.F_SilkS, pcbnew.F_Fab) and not isinstance(g, pcbnew.PCB_TEXT):
            boxes.append(g.GetBoundingBox())
    return (min(to(b.GetLeft()) for b in boxes) - OX, min(to(b.GetTop()) for b in boxes) - OY,
            max(to(b.GetRight()) for b in boxes) - OX, max(to(b.GetBottom()) for b in boxes) - OY)


def grow(b, m):
    return b[0] - m, b[1] - m, b[2] + m, b[3] + m


def overlap(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def main():
    board = pcbnew.LoadBoard("aicar-hat.kicad_pcb")
    parts = {f.GetReference(): f for f in board.GetFootprints()}
    zones = {}
    for ref in ("J5", "J6", "J7"):      # Plug housing is about 2.54 mm wide, centred on the pins.
        zones[f"{ref} plug"] = (ref, grow(pads_box(parts[ref]), MARGIN))
    zones["J4 solder"] = ("J4", grow(pads_box(parts["J4"]), MARGIN))
    for ref in ("J2", "J3"):            # Housing plus room for fingers on the plug's open side.
        zones[f"{ref} plug"] = (ref, grow(body_box(parts[ref]), 0.3))
    x0, y0, x1, y1 = pads_box(parts["J1"])
    zones["J1 solder"] = ("J1", (x0, y0, x1, y1 + IRON))
    # Common 3-pin slide switch: about 11.6 x 4 mm, centred on the middle pin.
    p = {q.GetNumber(): q.GetPosition() for q in parts["SW1"].Pads()}
    sx, sy = to(p["2"].x) - OX, to(p["2"].y) - OY
    zones["SW1 body"] = ("SW1", (sx - 2.0, sy - 5.8, sx + 2.0, sy + 5.8))
    for name, (hx, hy) in {"H1": (3.5, 3.5), "H2": (61.5, 3.5), "H3": (3.5, 52.5), "H4": (61.5, 52.5)}.items():
        zones[f"{name} screw"] = (name, (hx - SCREW, hy - SCREW, hx + SCREW, hy + SCREW))

    bad = 0
    for label, (owner, zone) in zones.items():
        hits = [r for r, f in parts.items()
                if r != owner and r != "J1" and not r.startswith("H") and overlap(zone, body_box(f))]
        print(f"  {label:10} {'OK' if not hits else 'HIT ' + ', '.join(sorted(hits))}")
        bad += len(hits)
    print("port conflicts", bad)


if __name__ == "__main__":
    main()
