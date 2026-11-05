"""Make every routed track 0.4 mm, except the short escapes from U7's 0.5 mm pitch pins.

Run with KiCad's Python after routing: finish_widths.py <board>. Tracks and vias stay where routed, apart from
the 5V_PI run between R24's pads, which is centred so 0.4 mm fits there.
"""
import sys

import pcbnew
import wx

OX, OY = 100.0, 100.0
MM = pcbnew.FromMM
WIDTH = MM(0.4)
NECK = MM(0.25)

# (net, from, to): where to split a track at U7. The part touching the pin stays narrow.
SPLITS = [
    ("ADC1", (36.28, 29.00), (38.90, 29.00), (37.30, 29.00)),
    ("ADC2", (37.65, 30.75), (38.90, 29.50), (38.25, 30.15)),
    ("GND", (38.90, 27.50), (37.60, 27.50), (38.00, 27.50)),
    ("GND", (38.90, 28.50), (40.35, 28.50), (39.80, 28.50)),
    ("3V3", (43.10, 28.50), (44.80, 28.50), (44.00, 28.50)),
    ("SDA", (43.10, 28.00), (41.99, 28.00), (42.20, 28.00)),
]
# Whole tracks that stay narrow: too short to split, and right at a pin.
NARROW = [
    ("ADC2", (38.95, 29.55), (38.90, 29.50)),
    ("GND", (38.90, 28.50), (37.90, 28.50)),
    ("SCL", (42.34, 26.74), (43.10, 27.50)),
    ("SCL", (43.10, 27.42), (43.10, 27.50)),
]
# The 5V_PI run between R24's pads: old end points to new end points.
MOVES = [
    ((34.68, 35.64), (35.44, 36.41), (34.68, 35.64), (35.50, 36.41)),
    ((35.44, 36.41), (35.44, 38.05), (35.50, 36.41), (35.50, 38.00)),
    ((35.44, 38.05), (35.68, 38.29), (35.50, 38.00), (36.00, 38.50)),
    ((35.68, 38.29), (35.86, 38.29), (36.00, 38.50), (36.30, 38.78)),
    ((35.86, 38.29), (35.93, 38.35), (36.30, 38.78), (36.60, 39.05)),
    ((35.93, 38.35), (37.13, 39.56), (36.60, 39.05), (37.13, 39.56)),
]


def near(point, xy):
    return abs(pcbnew.ToMM(point.x) - OX - xy[0]) < 0.015 and abs(pcbnew.ToMM(point.y) - OY - xy[1]) < 0.015


def pt(xy):
    return pcbnew.VECTOR2I(MM(OX + xy[0]), MM(OY + xy[1]))


def same(t, a, b):
    return (near(t.GetStart(), a) and near(t.GetEnd(), b)) or (near(t.GetStart(), b) and near(t.GetEnd(), a))


def main(path):
    board = pcbnew.LoadBoard(path)
    tracks = [t for t in board.GetTracks() if t.Type() == pcbnew.PCB_TRACE_T]

    for t in tracks:
        t.SetWidth(WIDTH)

    moved = 0
    for old_a, old_b, new_a, new_b in MOVES:
        match = [t for t in tracks if t.GetNetname() == "5V_PI" and same(t, old_a, old_b)]
        if len(match) != 1:
            sys.exit(f"5V_PI track not found: {old_a} {old_b}")
        match[0].SetStart(pt(new_a))
        match[0].SetEnd(pt(new_b))
        moved += 1

    for net, a, b in NARROW:
        match = [t for t in tracks if t.GetNetname() == net and same(t, a, b)]
        if len(match) != 1:
            sys.exit(f"narrow track not found: {net} {a} {b}")
        match[0].SetWidth(NECK)

    for net, a, b, cut in SPLITS:
        match = [t for t in tracks if t.GetNetname() == net and same(t, a, b)]
        if len(match) != 1:
            sys.exit(f"split track not found: {net} {a} {b}")
        t = match[0]
        pin_end = b if net in ("ADC1", "ADC2") else a       # The end that sits on a U7 pin.
        far_end = a if pin_end == b else b
        t.SetStart(pt(far_end))
        t.SetEnd(pt(cut))
        t.SetWidth(WIDTH)
        neck = pcbnew.PCB_TRACK(board)
        neck.SetStart(pt(cut))
        neck.SetEnd(pt(pin_end))
        neck.SetWidth(NECK)
        neck.SetLayer(t.GetLayer())
        neck.SetNet(t.GetNet())
        neck.SetLocked(t.IsLocked())
        board.Add(neck)

    pcbnew.ZONE_FILLER(board).Fill(board.Zones())
    pcbnew.SaveBoard(path, board)
    print("tracks", len(tracks), "moved", moved, "necks", len(NARROW) + len(SPLITS))


if __name__ == "__main__":
    app = wx.App(False)
    main(sys.argv[1])
