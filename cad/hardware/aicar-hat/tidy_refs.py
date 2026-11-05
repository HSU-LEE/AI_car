"""Place every reference beside its part, clear of pads, part bodies and other silk text.

Run with KiCad closed, using KiCad's Python: safe to repeat after moving parts.
Most constrained parts choose first, so tight spots do not lose out to parts with room to spare.
"""
import pcbnew
import wx

from scale_refs import ref_size

PATH ="aicar-hat.kicad_pcb"
OX, OY, W, H = 100.0, 100.0, 65.0, 56.5
NOTCH = (44.5, 48.5, 35.5)
HOLES = ((3.5, 3.5), (61.5, 3.5), (3.5, 52.5), (61.5, 52.5))
GAPS = (0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 1.8)
SIDE = {"R23": "left", "R24": "left", "D6": "right", "C13": "below", "C14": "below",
        "C22": "below", "C23": "below", "C24": "below"}     # Parts that read better with their number on one side.
MANUAL = {"C13": (8.9, 20.0, 0)}     # D4 blocks the centred spot under C13.
to = pcbnew.ToMM
MM = pcbnew.FromMM


def box_of(item):
    b = item.GetBoundingBox()
    return to(b.GetLeft()), to(b.GetTop()), to(b.GetRight()), to(b.GetBottom())


def union(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def hit(a, b, m=0.05):
    return a[0] < b[2] + m and a[2] > b[0] - m and a[1] < b[3] + m and a[3] > b[1] - m


def body_box(f):
    """Pads plus silk and fab outline: what you see of the part, without its wider courtyard."""
    boxes = [box_of(p) for p in f.Pads()]
    for g in f.GraphicalItems():
        if g.GetLayer() in (pcbnew.F_SilkS, pcbnew.F_Fab) and not isinstance(g, pcbnew.PCB_TEXT):
            boxes.append(box_of(g))
    return union(boxes)


def main():
    app = wx.App(False)
    board = pcbnew.LoadBoard(PATH)
    for f in board.GetFootprints():
        if f.GetReference().startswith("H"):
            continue
        text = f.Reference()
        size, stroke = ref_size(f)
        text.SetTextSize(pcbnew.VECTOR2I(MM(size), MM(size)))
        text.SetTextThickness(MM(stroke))
    for d in board.GetDrawings():   # Connector labels share one size.
        if isinstance(d, pcbnew.PCB_TEXT) and d.GetLayer() == pcbnew.F_SilkS:
            d.SetTextSize(pcbnew.VECTOR2I(MM(0.9), MM(0.9)))
            d.SetTextThickness(MM(0.14))
    parts = [f for f in board.GetFootprints() if f.GetLayer() == pcbnew.F_Cu and f.GetReference()[0] != "H"]
    bodies = {f.GetReference(): body_box(f) for f in parts}
    pads = [box_of(p) for f in board.GetFootprints() for p in f.Pads()]
    taken = [box_of(d) for d in board.GetDrawings()
             if isinstance(d, pcbnew.PCB_TEXT) and d.GetLayer() == pcbnew.F_SilkS]
    by_ref = {f.GetReference(): f for f in parts}
    for ref, (x, y, angle) in MANUAL.items():
        t = by_ref[ref].Reference()
        t.SetVisible(True)
        t.SetKeepUpright(True)
        t.SetTextAngleDegrees(angle)
        t.SetPosition(pcbnew.VECTOR2I(MM(OX + x), MM(OY + y)))
        taken.append(box_of(t))

    def free(box, ref):
        cx, cy = (box[0] + box[2]) / 2 - OX, (box[1] + box[3]) / 2 - OY
        hw, hh = (box[2] - box[0]) / 2, (box[3] - box[1]) / 2
        if not (hw + 0.3 < cx < W - hw - 0.3 and hh + 0.3 < cy < H - hh - 0.3):
            return False
        if NOTCH[0] - hw - 0.3 < cx < NOTCH[1] + hw + 0.3 and cy > NOTCH[2] - hh - 0.3:
            return False
        for hx, hy in HOLES:
            if abs(cx - hx) < 1.5 + hw and abs(cy - hy) < 1.5 + hh:
                return False
        if any(hit(box, p) for p in pads) or any(hit(box, q) for q in taken):
            return False
        return not any(hit(box, v) for k, v in bodies.items() if k != ref)

    def options(f):
        ref = f.GetReference()
        text = f.Reference()
        size = to(text.GetTextHeight())
        l, t, r, b = bodies[ref]
        cx, cy = (l + r) / 2, (t + b) / 2
        tw, th = 0.8 * len(ref) * size + 0.1, size + 0.1
        tall = False    # Above first everywhere, so numbers read the same across the board.
        out = []
        for gap in GAPS:
            above = (cx, t - th / 2 - gap, 0, tw, th)
            below = (cx, b + th / 2 + gap, 0, tw, th)
            right = (r + tw / 2 + gap, cy, 0, tw, th)
            left = (l - tw / 2 - gap, cy, 0, tw, th)
            named = {"left": left, "right": right, "above": above, "below": below}
            if ref in SIDE:
                out.append(named[SIDE[ref]])
            out += [right, left, above, below] if tall else [above, below, left, right]
        # Match the number to a footprint rotated 90 degrees.
        sideways = []
        for gap in GAPS:
            sideways += [(r + th / 2 + gap, cy, 90, th, tw), (l - th / 2 - gap, cy, 90, th, tw),
                         (cx, t - tw / 2 - gap, 90, th, tw), (cx, b + tw / 2 + gap, 90, th, tw)]
        return out + sideways

    def feasible(f):
        ref = f.GetReference()
        return [o for o in options(f) if free((o[0] - o[3] / 2, o[1] - o[4] / 2, o[0] + o[3] / 2, o[1] + o[4] / 2), ref)]

    left = [f for f in parts if f.GetReference() not in MANUAL]
    placed, failed = 0, []
    while left:
        choices = [(len(feasible(f)), i, f) for i, f in enumerate(left)]
        _, _, f = min(choices, key=lambda c: c[:2])
        left.remove(f)
        opts = feasible(f)
        if not opts:
            failed.append(f.GetReference())
            continue
        x, y, angle, w, h = opts[0]
        text = f.Reference()
        text.SetVisible(True)
        text.SetKeepUpright(True)
        text.SetHorizJustify(pcbnew.GR_TEXT_H_ALIGN_CENTER)
        text.SetVertJustify(pcbnew.GR_TEXT_V_ALIGN_CENTER)
        text.SetTextAngleDegrees(angle)
        text.SetPosition(pcbnew.VECTOR2I(MM(x), MM(y)))
        taken.append((x - w / 2, y - h / 2, x + w / 2, y + h / 2))
        placed += 1
    pcbnew.SaveBoard(PATH, board)
    print("placed", placed + len(failed), "manual", sorted(MANUAL), "no room", sorted(set(failed) - set(MANUAL)))


if __name__ == "__main__":
    main()
