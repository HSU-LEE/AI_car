"""Shrink silk text taller than 0.65 mm down to 0.65 mm. Smaller text and all positions stay as they are.

Usage, with KiCad's Python: clamp_text.py [board file]. Close KiCad before running it on the real board.
"""
import sys

import pcbnew
import wx

LIMIT = pcbnew.FromMM(0.65)
STROKE = pcbnew.FromMM(0.1)


def clamp(text):
    if max(text.GetTextHeight(), text.GetTextWidth()) <= LIMIT:
        return 0
    text.SetTextSize(pcbnew.VECTOR2I(LIMIT, LIMIT))
    text.SetTextThickness(min(text.GetTextThickness(), STROKE))
    return 1


def main(path):
    app = wx.App(False)
    board = pcbnew.LoadBoard(path)
    changed = sum(clamp(f.Reference()) for f in board.GetFootprints())
    labels = sum(clamp(d) for d in board.GetDrawings()
                 if isinstance(d, pcbnew.PCB_TEXT) and d.GetLayer() in (pcbnew.F_SilkS, pcbnew.B_SilkS))
    pcbnew.SaveBoard(path, board)
    print("numbers shrunk", changed, "labels shrunk", labels)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "aicar-hat.kicad_pcb")
