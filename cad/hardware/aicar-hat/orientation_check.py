"""List where pin 1 of each polarised SMT part must sit, to compare with the JLCPCB placement preview.

Run with KiCad's Python: orientation_check.py <board> <out.csv>. Directions are as seen from the top, with the
board in KiCad's orientation. Pin 1 is the cathode on diodes and LEDs and the plus side on electrolytics.
"""
import csv
import sys

import pcbnew
import wx

POLAR = ("D", "Q", "U")
ELECTROLYTIC = ("C5", "C6", "C7")


def direction(dx, dy):
    span = max(abs(dx), abs(dy))
    vertical = "top" if dy < -span / 4 else "bottom" if dy > span / 4 else ""
    horizontal = "left" if dx < -span / 4 else "right" if dx > span / 4 else ""
    return "-".join(p for p in (vertical, horizontal) if p)


def main(path, out):
    board = pcbnew.LoadBoard(path)
    rows = []
    for fp in board.GetFootprints():
        ref = fp.GetReference()
        if ref[0] not in POLAR and ref not in ELECTROLYTIC:
            continue
        pad = next(p for p in fp.Pads() if p.GetNumber() == "1")
        centre, pin = fp.GetPosition(), pad.GetPosition()
        rows.append((ref, fp.GetValue(), direction(pin.x - centre.x, pin.y - centre.y)))
    rows.sort(key=lambda r: (r[0][0], int(r[0][1:])))
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Designator", "Part", "Pin 1 position"])
        w.writerows(rows)
    print("polarised parts", len(rows))


if __name__ == "__main__":
    app = wx.App(False)
    main(sys.argv[1], sys.argv[2])
