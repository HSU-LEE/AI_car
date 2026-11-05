"""One-off: turn the autorouted board into a clean start for hand layout. Run with KiCad's Python."""
import sys
import uuid

import pcbnew
import wx

PATH = "aicar-hat.kicad_pcb"
OX, OY = 100.0, 100.0
MM = pcbnew.FromMM


def pt(x, y):
    return pcbnew.VECTOR2I(MM(OX + x), MM(OY + y))


def guide(board, label, x0, y0, x1, y1):
    """Draw a Pi 4 part outline on Cmts.User, which never reaches the Gerbers."""
    s = pcbnew.PCB_SHAPE(board)
    s.SetShape(pcbnew.SHAPE_T_RECT)
    s.SetStart(pt(x0, y0))
    s.SetEnd(pt(x1, y1))
    s.SetLayer(pcbnew.Cmts_User)
    s.SetWidth(MM(0.15))
    board.Add(s)
    t = pcbnew.PCB_TEXT(board)
    t.SetText(label)
    t.SetPosition(pt((x0 + x1) / 2, (y0 + y1) / 2))
    t.SetLayer(pcbnew.Cmts_User)
    t.SetTextSize(pcbnew.VECTOR2I(MM(1.0), MM(1.0)))
    t.SetTextThickness(MM(0.12))
    board.Add(t)


def strip_copper_add_rule_area(path, x0, y0, x1, y1):
    """Edit the saved file: removing items through pcbnew's Python breaks its type table in this build."""
    text = open(path).read()
    # Top-level tracks and vias sit at one tab; zone fills sit deeper. Remove each balanced block.
    for start in ("\n\t(segment", "\n\t(arc", "\n\t(via", "(filled_polygon"):
        out, i = [], 0
        while True:
            k = text.find(start, i)
            if k < 0:
                out.append(text[i:])
                break
            out.append(text[i:k])
            depth, n = 0, k
            while True:
                if text[n] == "(":
                    depth += 1
                elif text[n] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                n += 1
            i = n + 1
        text = "".join(out)
    pts = " ".join(f"(xy {OX + x} {OY + y})" for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)))
    # Flags any through-hole pin placed above the Pi's PoE header.
    zone = f"""
\t(zone
\t\t(net 0)
\t\t(net_name "")
\t\t(layer "B.Cu")
\t\t(uuid "{uuid.uuid5(uuid.NAMESPACE_DNS, 'aicar-hat/poe-keepout')}")
\t\t(name "PoE header: no through-hole pins")
\t\t(hatch edge 0.5)
\t\t(connect_pads (clearance 0))
\t\t(min_thickness 0.25)
\t\t(filled_areas_thickness no)
\t\t(keepout (tracks allowed) (vias allowed) (pads not_allowed) (copperpour allowed) (footprints allowed))
\t\t(fill (thermal_gap 0.5) (thermal_bridge_width 0.5))
\t\t(polygon (pts {pts}))
\t)
"""
    end = text.rstrip().rfind(")")
    open(path, "w").write(text[:end] + zone + text[end:])


def main():
    app = wx.App(False)
    board = pcbnew.LoadBoard(PATH)
    for f in board.GetFootprints():
        ref = f.GetReference()
        fixed = ref == "J1" or ref.startswith("H")
        f.SetLocked(fixed)      # The Pi header and mounting holes must stay on the Pi's grid.
        if not ref.startswith("H"):
            f.Reference().SetVisible(True)
    guide(board, "Pi PoE", 59.1, 7.2, 64.1, 12.2)
    guide(board, "Pi SoC", 21.7, 16.0, 36.8, 31.0)
    guide(board, "Pi RAM", 39.0, 16.0, 50.5, 31.0)
    guide(board, "Pi audio", 50.5, 43.5, 57.5, 56.0)
    pcbnew.SaveBoard(PATH, board)
    strip_copper_add_rule_area(PATH, 59.1, 7.2, 64.1, 12.0)


if __name__ == "__main__":
    sys.exit(main())
