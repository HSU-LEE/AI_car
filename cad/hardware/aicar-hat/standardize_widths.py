"""Cap all track widths at 0.4 mm after routing, then refill copper zones.

Use KiCad's Python: standardize_widths.py <board.kicad_pcb>
"""
import sys

import pcbnew

MM = pcbnew.FromMM
MAX_WIDTH = MM(0.4)


def main(path):
    board = pcbnew.LoadBoard(path)
    changed = 0
    for track in board.GetTracks():
        if track.Type() != pcbnew.PCB_VIA_T and track.GetWidth() > MAX_WIDTH:
            track.SetWidth(MAX_WIDTH)
            changed += 1

    # C3.2 formerly met the VOR bus only through the 1.5 mm trace's width.
    # Add a short explicit connection so the 0.4 mm bus still reaches its pad.
    a, b = (MM(144.5), MM(110.775)), (MM(144.5), MM(111.64))
    exists = any(t.Type() != pcbnew.PCB_VIA_T and t.GetNetname() == "VOR"
                 and {(t.GetStart().x, t.GetStart().y), (t.GetEnd().x, t.GetEnd().y)} == {a, b}
                 for t in board.GetTracks())
    if not exists:
        stub = pcbnew.PCB_TRACK(board)
        stub.SetStart(pcbnew.VECTOR2I(*a))
        stub.SetEnd(pcbnew.VECTOR2I(*b))
        stub.SetWidth(MAX_WIDTH)
        stub.SetLayer(pcbnew.F_Cu)
        stub.SetNet(board.FindNet("VOR"))
        board.Add(stub)

    pcbnew.ZONE_FILLER(board).Fill(board.Zones())
    pcbnew.SaveBoard(path, board)
    print("tracks narrowed", changed, "VOR stub added", not exists)


if __name__ == "__main__":
    main(sys.argv[1])
