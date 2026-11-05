"""Size each reference to its part, moving only the ones that then collide. Hand-placed numbers otherwise stay put.

Usage, with KiCad's Python: scale_refs.py [board file]. Close KiCad before running it on the real board.
"""
import sys

import pcbnew
import wx

OX, OY, W, H = 100.0, 100.0, 65.0, 56.5
NOTCH = (44.5, 48.5, 35.5)
HOLES = ((3.5, 3.5), (61.5, 3.5), (3.5, 52.5), (61.5, 52.5))
GAPS = (0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 1.8)
# First match on the footprint name wins: (name part, text height mm).
SIZES = (("0603", 0.65), ("TestPoint", 0.65),
         ("1206", 0.8), ("TSOT", 0.8), ("SOT-23", 0.8), ("MSOP", 0.8), ("SolderWire", 0.8),
         ("SOIC", 0.8), ("D_SMA", 0.8), ("D_SMB", 0.8), ("PinHeader", 0.8), ("JST", 0.8), ("PinSocket", 0.8),
         ("L_APV", 1.0), ("CP_Elec", 1.0), ("SSOP", 1.0))
to = pcbnew.ToMM
MM = pcbnew.FromMM


def ref_size(f):
    """Text height and stroke for a footprint, in mm."""
    name = f.GetFPIDAsString()
    size = next(s for part, s in SIZES if part in name)
    return size, max(0.1, round(size * 0.15, 2))


def box_of(item):
    b = item.GetBoundingBox()
    return to(b.GetLeft()), to(b.GetTop()), to(b.GetRight()), to(b.GetBottom())


def hit(a, b, m=0.05):
    return a[0] < b[2] + m and a[2] > b[0] - m and a[1] < b[3] + m and a[3] > b[1] - m


def body_box(f):
    boxes = [box_of(p) for p in f.Pads()]
    for g in f.GraphicalItems():
        if g.GetLayer() in (pcbnew.F_SilkS, pcbnew.F_Fab) and not isinstance(g, pcbnew.PCB_TEXT):
            boxes.append(box_of(g))
    return min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)


def text_box(ref, size, x, y, angle):
    long_side, short_side = 0.8 * len(ref) * size + 0.1, size + 0.1
    w, h = (short_side, long_side) if round(angle) % 180 == 90 else (long_side, short_side)
    return x - w / 2, y - h / 2, x + w / 2, y + h / 2


def main(path):
    app = wx.App(False)
    board = pcbnew.LoadBoard(path)
    parts = [f for f in board.GetFootprints() if f.GetReference()[0] != "H"]
    front = [f for f in parts if f.GetLayer() == pcbnew.F_Cu]
    bodies = {f.GetReference(): body_box(f) for f in front}
    pads = [box_of(p) for f in board.GetFootprints() for p in f.Pads()]
    fixed = [box_of(d) for d in board.GetDrawings()
             if isinstance(d, pcbnew.PCB_TEXT) and d.GetLayer() == pcbnew.F_SilkS]
    placed = {}

    grew = set()
    for f in parts:
        size, stroke = ref_size(f)
        t = f.Reference()
        if MM(size) > t.GetTextHeight():
            grew.add(f.GetReference())
        t.SetTextSize(pcbnew.VECTOR2I(MM(size), MM(size)))
        t.SetTextThickness(MM(stroke))

    def free(box, ref):
        cx, cy = (box[0] + box[2]) / 2 - OX, (box[1] + box[3]) / 2 - OY
        hw, hh = (box[2] - box[0]) / 2, (box[3] - box[1]) / 2
        if not (hw + 0.2 < cx < W - hw - 0.2 and hh + 0.2 < cy < H - hh - 0.2):
            return False
        if NOTCH[0] - hw - 0.2 < cx < NOTCH[1] + hw + 0.2 and cy > NOTCH[2] - hh - 0.2:
            return False
        if any(abs(cx - hx) < 1.5 + hw and abs(cy - hy) < 1.5 + hh for hx, hy in HOLES):
            return False
        if any(hit(box, p) for p in pads) or any(hit(box, q) for q in fixed):
            return False
        if any(hit(box, q) for r, q in placed.items() if r != ref):
            return False
        return not any(hit(box, v, 0) for k, v in bodies.items() if k != ref)

    # Numbers that still fit stay where they are; the rest move, largest text first.
    order = sorted(front, key=lambda f: -ref_size(f)[0])
    stay, move = [], []
    current = {}
    for f in order:
        t = f.Reference()
        p = t.GetPosition()
        current[f.GetReference()] = text_box(f.GetReference(), ref_size(f)[0], to(p.x), to(p.y), t.GetTextAngleDegrees())
    placed.update(current)
    for f in order:
        ref = f.GetReference()
        if free(current[ref], ref) and not hit(current[ref], bodies[ref], 0.12 if ref in grew else 0):
            stay.append(f)
        else:
            move.append(f)
    for f in move:
        placed.pop(f.GetReference())

    moved, failed = [], []
    for f in move:
        ref = f.GetReference()
        size = ref_size(f)[0]
        l, t, r, b = bodies[ref]
        cx, cy = (l + r) / 2, (t + b) / 2
        text = f.Reference()
        vertical = round(text.GetTextAngleDegrees()) % 180 == 90
        for angle in ((90, 0) if vertical else (0, 90)):
            long_side, short_side = 0.8 * len(ref) * size + 0.1, size + 0.1
            w, h = (short_side, long_side) if angle == 90 else (long_side, short_side)
            done = False
            for gap in GAPS:
                for x, y in ((cx, t - h / 2 - gap), (cx, b + h / 2 + gap), (r + w / 2 + gap, cy), (l - w / 2 - gap, cy)):
                    box = text_box(ref, size, x, y, angle)
                    if free(box, ref):
                        text.SetVisible(True)
                        text.SetKeepUpright(True)
                        text.SetHorizJustify(pcbnew.GR_TEXT_H_ALIGN_CENTER)
                        text.SetVertJustify(pcbnew.GR_TEXT_V_ALIGN_CENTER)
                        text.SetTextAngleDegrees(angle)
                        text.SetPosition(pcbnew.VECTOR2I(MM(x), MM(y)))
                        placed[ref] = box
                        moved.append(ref)
                        done = True
                        break
                if done:
                    break
            if done:
                break
        else:
            failed.append(ref)
    pcbnew.SaveBoard(path, board)
    print("kept", len(stay), "moved", sorted(moved), "no room", sorted(failed))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "aicar-hat.kicad_pcb")
