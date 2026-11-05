"""KiCad helper: DSN export without GND (the pours carry it), SES import, GND stitching, zone fill."""
import json
import math
import re
import sys

import pcbnew
import wx

from layout import H, NOTCH, OX, OY, W

VIA_D, VIA_DRILL = 0.7, 0.35
GAP = 0.2       # Copper gap kept around a stitching via.
MM = pcbnew.FromMM


def export(path, dsn):
    board = pcbnew.LoadBoard(path)
    if not pcbnew.ExportSpecctraDSN(board, dsn):
        sys.exit("DSN export failed")
    text = open(dsn).read()
    text, n = re.subn(r'\(net "?GND"?\s*\(pins[^)]*\)\s*\)', '', text)
    print("GND removed from router:", n)
    open(dsn, "w").write(text)


def fill(board):
    pcbnew.ZONE_FILLER(board).Fill(board.Zones())


def seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    t = 0 if dx == dy == 0 else max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)


def stitch(board, targets=None):
    """Add GND vias on a grid and beside each GND pad wherever they fit, tying F.Cu islands to the B.Cu plane."""
    gnd = board.FindNet("GND")
    tracks, vias, pads = [], [], []
    to = pcbnew.ToMM
    for t in board.GetTracks():
        if t.Type() == pcbnew.PCB_VIA_T:
            p = t.GetPosition()
            vias.append((to(p.x), to(p.y), to(t.GetWidth(pcbnew.F_Cu)) / 2, t.GetNetname()))
        else:
            s, e = t.GetStart(), t.GetEnd()
            tracks.append((to(s.x), to(s.y), to(e.x), to(e.y), to(t.GetWidth()) / 2, t.GetNetname()))
    for f in board.GetFootprints():
        for p in f.Pads():
            b = p.GetBoundingBox()
            pads.append((to(b.GetLeft()), to(b.GetTop()), to(b.GetRight()), to(b.GetBottom()),
                         p.GetNetname(), p.HasHole()))

    r = VIA_D / 2

    def fits(x, y):
        lx, ly = x - OX, y - OY
        if not (r + 0.5 < lx < W - r - 0.5 and r + 0.5 < ly < H - r - 0.5):
            return False
        for cx, cy in ((3, 3), (W - 3, 3), (3, H - 3), (W - 3, H - 3)):
            if (lx < 3 or lx > W - 3) and (ly < 3 or ly > H - 3) and math.hypot(lx - cx, ly - cy) > 3 - r - 0.4:
                return False
        if NOTCH[0] - r - 0.5 < lx < NOTCH[1] + r + 0.5 and ly > NOTCH[2] - r - 0.5:
            return False
        for hx, hy in ((3.5, 3.5), (61.5, 3.5), (3.5, 52.5), (61.5, 52.5)):
            if math.hypot(lx - hx, ly - hy) < 1.35 + r + 0.5:
                return False
        for ax, ay, bx, by, hw, net in tracks:
            if net != "GND" and seg_dist(x, y, ax, ay, bx, by) < r + hw + GAP:
                return False
        for vx, vy, vr, net in vias:
            if math.hypot(x - vx, y - vy) < r + vr + 0.3:
                return False
        for l, t, rt, b, net, hole in pads:
            cx, cy = max(l, min(x, rt)), max(t, min(y, b))
            if math.hypot(x - cx, y - cy) < r + (0.3 if hole or net != "GND" else 0.05):
                return False
        return True

    def place(x, y):
        v = pcbnew.PCB_VIA(board)
        v.SetPosition(pcbnew.VECTOR2I(MM(x), MM(y)))
        v.SetWidth(MM(VIA_D))
        v.SetDrill(MM(VIA_DRILL))
        v.SetNet(gnd)
        board.Add(v)
        vias.append((x, y, r, "GND"))

    if targets is not None:
        return fits, place, tracks, pads

    added = 0
    # Beside each GND SMD pad first, so no pad depends on a thin F.Cu island.
    for l, t, rt, b, net, hole in pads:
        if net != "GND" or hole:
            continue
        cx, cy = (l + rt) / 2, (t + b) / 2
        if any(math.hypot(cx - vx, cy - vy) < 1.8 for vx, vy, _, n in vias if n == "GND"):
            continue
        half = max(rt - l, b - t) / 2
        for dist in (half + 0.45, half + 0.7, half + 1.0):
            done = False
            for a in range(0, 360, 30):
                x, y = cx + dist * math.cos(math.radians(a)), cy + dist * math.sin(math.radians(a))
                if fits(x, y):
                    place(x, y)
                    added += 1
                    done = True
                    break
            if done:
                break
    # Then a grid across the board.
    for gx in range(0, int(W), 5):
        for gy in range(0, int(H), 5):
            x, y = OX + gx + 2.5, OY + gy + 2.5
            if fits(x, y):
                place(x, y)
                added += 1
    print("stitching vias", added)


def tie_islands(board):
    """Give every F.Cu and B.Cu GND fragment that has no via or through-hole pad its own via."""
    fill(board)
    fits, place, _, _ = stitch(board, targets=True)
    to = pcbnew.ToMM
    ties = [(p.GetPosition()) for f in board.GetFootprints() for p in f.Pads() if p.HasHole() and p.GetNetname() == "GND"]
    ties += [t.GetPosition() for t in board.GetTracks() if t.Type() == pcbnew.PCB_VIA_T and t.GetNetname() == "GND"]
    added = 0
    for z in board.Zones():
        if z.GetNetname() != "GND":
            continue
        layer = z.GetFirstLayer()
        polys = z.GetFilledPolysList(layer)
        for i in range(polys.OutlineCount()):
            sub = pcbnew.SHAPE_POLY_SET()
            sub.AddOutline(polys.COutline(i))
            if any(sub.Contains(p) for p in ties):
                continue
            b = polys.COutline(i).BBox()
            found = None
            step = MM(0.2)
            for px in range(b.GetLeft(), b.GetRight(), step):
                for py in range(b.GetTop(), b.GetBottom(), step):
                    if sub.Contains(pcbnew.VECTOR2I(px, py)) and fits(to(px), to(py)):
                        found = (to(px), to(py))
                        break
                if found:
                    break
            if found:
                place(*found)
                ties.append(pcbnew.VECTOR2I(MM(found[0]), MM(found[1])))
                added += 1
    return added


def gnd_components(board):
    """Split the filled GND copper into fragments and group the ones joined by vias, pins or tracks."""
    frags = []
    for z in board.Zones():
        if z.GetNetname() != "GND":
            continue
        layer = z.GetFirstLayer()
        polys = z.GetFilledPolysList(layer)
        for i in range(polys.OutlineCount()):
            sub = pcbnew.SHAPE_POLY_SET()
            sub.AddOutline(polys.COutline(i))
            frags.append((layer, sub, polys.COutline(i).BBox()))
    parent = list(range(len(frags)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def at(pos, layer=None):
        return [i for i, (lay, sub, _) in enumerate(frags) if (layer is None or lay == layer) and sub.Contains(pos)]

    links = [at(t.GetPosition()) for t in board.GetTracks()
             if t.Type() == pcbnew.PCB_VIA_T and t.GetNetname() == "GND"]
    # A pin's own centre is a hole in the fill, so look just outside its pad, all the way round.
    for f in board.GetFootprints():
        for p in f.Pads():
            if p.HasHole() and p.GetNetname() == "GND":
                c, r = p.GetPosition(), max(p.GetSize().x, p.GetSize().y) // 2 + MM(0.15)
                ring = []
                for k in range(16):
                    a = math.radians(k * 22.5)
                    q = pcbnew.VECTOR2I(c.x + int(r * math.cos(a)), c.y + int(r * math.sin(a)))
                    ring += at(q)
                links.append(sorted(set(ring)))
    links += [at(t.GetStart(), t.GetLayer()) + at(t.GetEnd(), t.GetLayer()) for t in board.GetTracks()
              if t.Type() != pcbnew.PCB_VIA_T and t.GetNetname() == "GND"]
    for group in links:
        for other in group[1:]:
            parent[find(other)] = find(group[0])
    groups = {}
    for i in range(len(frags)):
        groups.setdefault(find(i), []).append(i)
    return frags, sorted(groups.values(), key=len, reverse=True)


def link(path):
    """Add a via wherever a GND fragment is cut off from the main group and overlaps it on the other layer."""
    board = pcbnew.LoadBoard(path)
    to = pcbnew.ToMM
    added = 0
    for _ in range(8):
        fill(board)
        frags, groups = gnd_components(board)
        if len(groups) == 1:
            break
        area = lambda g: sum(frags[i][2].GetWidth() * frags[i][2].GetHeight() for i in g)
        main = set(max(groups, key=area))
        fits, place, _, _ = stitch(board, targets=True)
        progress = False
        for group in groups:
            if set(group) == main:
                continue
            found = None
            for i in group:
                layer, sub, box = frags[i]
                others = [frags[m][1] for m in main if frags[m][0] != layer]
                for px in range(box.GetLeft(), box.GetRight(), MM(0.15)):
                    for py in range(box.GetTop(), box.GetBottom(), MM(0.15)):
                        q = pcbnew.VECTOR2I(px, py)
                        if sub.Contains(q) and any(o.Contains(q) for o in others) and fits(to(px), to(py)):
                            found = (to(px), to(py))
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                place(*found)
                added += 1
                progress = True
                continue
            # Too small for a via: run a short stub from a GND pad in this fragment out to the main plane.
            _, _, tracks, _ = stitch(board, targets=True)
            for i in group:
                layer, sub, box = frags[i]
                others = [frags[m][1] for m in main if frags[m][0] != layer]
                pads = [p for f in board.GetFootprints() for p in f.Pads()
                        if p.GetNetname() == "GND" and not p.HasHole() and p.IsOnLayer(layer)
                        and box.Intersects(p.GetBoundingBox())]
                for p in pads:
                    px, py = to(p.GetPosition().x), to(p.GetPosition().y)
                    done = False
                    for dist in (0.9, 1.1, 1.3, 1.6, 2.0, 2.5, 3.0, 3.5):
                        for ang in range(0, 360, 15):
                            x, y = px + dist * math.cos(math.radians(ang)), py + dist * math.sin(math.radians(ang))
                            if not fits(x, y) or not any(o.Contains(pcbnew.VECTOR2I(MM(x), MM(y))) for o in others):
                                continue
                            clear = True
                            for ax, ay, bx, by, hw, n in tracks:
                                if n == "GND":
                                    continue
                                if any(seg_dist(ax + (bx - ax) * k / 20, ay + (by - ay) * k / 20, px, py, x, y) < hw + 0.3
                                       for k in range(21)):
                                    clear = False
                                    break
                            for f2 in board.GetFootprints():
                                for q in f2.Pads():
                                    if not clear or q.GetNetname() == "GND":
                                        continue
                                    b = q.GetBoundingBox()
                                    cx = (to(b.GetLeft()) + to(b.GetRight())) / 2
                                    cy = (to(b.GetTop()) + to(b.GetBottom())) / 2
                                    half = max(to(b.GetWidth()), to(b.GetHeight())) / 2
                                    if seg_dist(cx, cy, px, py, x, y) < half + 0.2:
                                        clear = False
                            if not clear:
                                continue
                            place(x, y)
                            s = pcbnew.PCB_TRACK(board)
                            s.SetStart(pcbnew.VECTOR2I(MM(px), MM(py)))
                            s.SetEnd(pcbnew.VECTOR2I(MM(x), MM(y)))
                            s.SetWidth(MM(0.3))
                            s.SetLayer(layer)
                            s.SetNet(board.FindNet("GND"))
                            board.Add(s)
                            added += 1
                            progress = done = True
                            break
                        if done:
                            break
                    if done:
                        break
                if progress:
                    break
        if not progress:
            break
    fill(board)
    frags, groups = gnd_components(board)
    pcbnew.SaveBoard(path, board)
    print("link vias", added, "gnd groups", len(groups))
    for g in groups[1:]:
        for i in g:
            b = frags[i][2]
            print("  cut off", board.GetLayerName(frags[i][0]), round(to(b.GetLeft()) - OX, 1), round(to(b.GetTop()) - OY, 1),
                  round(to(b.GetRight()) - OX, 1), round(to(b.GetBottom()) - OY, 1))


def repair(path, drc=None):
    """Tie GND fragments together, then connect any pad DRC still lists as unconnected with a short stub and via."""
    board = pcbnew.LoadBoard(path)
    total = 0
    for _ in range(3):
        n = tie_islands(board)
        total += n
        if not n:
            break
    stubs = 0
    if drc:
        data = json.load(open(drc))
        fits, place, tracks, pads = stitch(board, targets=True)
        for u in data.get("unconnected_items", []):
            for item in u["items"]:
                if "[GND]" not in item["description"] or item["pos"]["x"] < 0:
                    continue
                px, py = item["pos"]["x"], item["pos"]["y"]
                done = False
                for dist in (0.9, 1.2, 1.5, 1.9, 2.4, 3.0):
                    for a in range(0, 360, 20):
                        x, y = px + dist * math.cos(math.radians(a)), py + dist * math.sin(math.radians(a))
                        if not fits(x, y):
                            continue
                        clear = True
                        for ax, ay, bx, by, hw, n in tracks:
                            if n == "GND":
                                continue
                            for k in range(21):     # Sample the track so a crossing cannot slip through.
                                qx, qy = ax + (bx - ax) * k / 20, ay + (by - ay) * k / 20
                                if seg_dist(qx, qy, px, py, x, y) < hw + 0.3:
                                    clear = False
                                    break
                            if not clear:
                                break
                        if not clear:
                            continue
                        place(x, y)
                        t = pcbnew.PCB_TRACK(board)
                        t.SetStart(pcbnew.VECTOR2I(MM(px), MM(py)))
                        t.SetEnd(pcbnew.VECTOR2I(MM(x), MM(y)))
                        t.SetWidth(MM(0.3))
                        t.SetLayer(pcbnew.F_Cu)
                        t.SetNet(board.FindNet("GND"))
                        board.Add(t)
                        stubs += 1
                        done = True
                        break
                    if done:
                        break
    fill(board)
    pcbnew.SaveBoard(path, board)
    print("island vias", total, "pad stubs", stubs)


def tidy_silk(board):
    """Put every reference upright beside its part, clear of pads, vias, bodies and other text."""
    to = pcbnew.ToMM
    size, thick = 0.8, 0.13
    pads, bodies = [], {}
    for f in board.GetFootprints():
        for p in f.Pads():
            b = p.GetBoundingBox()
            pads.append((to(b.GetLeft()), to(b.GetTop()), to(b.GetRight()), to(b.GetBottom())))
        b = f.GetBoundingBox(False)     # Body and courtyard, without text.
        bodies[f.GetReference()] = (to(b.GetLeft()), to(b.GetTop()), to(b.GetRight()), to(b.GetBottom()))
    for t in board.GetTracks():
        if t.Type() == pcbnew.PCB_VIA_T:
            p, rr = t.GetPosition(), to(t.GetWidth(pcbnew.F_Cu)) / 2
            pads.append((to(p.x) - rr, to(p.y) - rr, to(p.x) + rr, to(p.y) + rr))
    taken = []
    for d in board.GetDrawings():
        if isinstance(d, pcbnew.PCB_TEXT) and d.GetLayer() == pcbnew.F_SilkS:
            b = d.GetBoundingBox()
            taken.append((to(b.GetLeft()), to(b.GetTop()), to(b.GetRight()), to(b.GetBottom())))

    def hit(a, b, m=0.06):
        return a[0] < b[2] + m and a[2] > b[0] - m and a[1] < b[3] + m and a[3] > b[1] - m

    placed = hidden = 0
    bodies.pop("J1", None)      # J1 sits on the bottom side.
    for f in sorted(board.GetFootprints(), key=lambda f: -len(f.GetReference())):
        ref = f.GetReference()
        text = f.Reference()
        if ref.startswith("H") or ref == "J1":
            text.SetVisible(False)
            continue
        l, t_, r_, b_ = bodies[ref]
        cx, cy = (l + r_) / 2, (t_ + b_) / 2
        tw, th = 0.72 * len(ref) * size + 0.1, size + 0.1
        best = None
        for gap in (0.05, 0.3, 0.6, 1.0, 1.5):
            spots = []
            for angle, w, h in ((0, tw, th), (90, th, tw)):
                spots += [(angle, w, h, cx, t_ - h / 2 - gap), (angle, w, h, cx, b_ + h / 2 + gap),
                          (angle, w, h, l - w / 2 - gap, cy), (angle, w, h, r_ + w / 2 + gap, cy)]
            for angle, w, h, x, y in spots:
                box = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
                lx, ly = x - OX, y - OY
                if not (w / 2 + 0.2 < lx < W - w / 2 - 0.2 and h / 2 + 0.2 < ly < H - h / 2 - 0.2):
                    continue
                if NOTCH[0] - w / 2 - 0.2 < lx < NOTCH[1] + w / 2 + 0.2 and ly > NOTCH[2] - h / 2 - 0.2:
                    continue
                if any(hit(box, p) for p in pads) or any(hit(box, q) for q in taken):
                    continue
                if any(hit(box, v, 0) for k, v in bodies.items() if k != ref):
                    continue
                best = (x, y, box, angle)
                break
            if best:
                break
        text.SetTextSize(pcbnew.VECTOR2I(MM(size), MM(size)))
        text.SetTextThickness(MM(thick))
        if not best:
            text.SetVisible(False)
            hidden += 1
            print("  no room for", ref)
            continue
        text.SetVisible(True)
        text.SetKeepUpright(True)
        text.SetHorizJustify(pcbnew.GR_TEXT_H_ALIGN_CENTER)
        text.SetVertJustify(pcbnew.GR_TEXT_V_ALIGN_CENTER)
        text.SetTextAngleDegrees(best[3])
        text.SetPosition(pcbnew.VECTOR2I(MM(best[0]), MM(best[1])))
        taken.append(best[2])
        placed += 1
    print("silk refs placed", placed, "hidden", hidden)


def import_ses(path, ses):
    board = pcbnew.LoadBoard(path)
    if not pcbnew.ImportSpecctraSES(board, ses):
        sys.exit("SES import failed")
    for tr in board.GetTracks():    # Freerouting necks down at tight pads; hold them to the 0.15 mm rule.
        if tr.Type() != pcbnew.PCB_VIA_T and tr.GetWidth() < MM(0.15):
            tr.SetWidth(MM(0.15))
    stitch(board)
    fill(board)
    pcbnew.SaveBoard(path, board)
    print("tracks", len(board.GetTracks()))


if __name__ == "__main__":
    {"export": export, "import": import_ses, "repair": repair, "link": link}[sys.argv[1]](*sys.argv[2:])
