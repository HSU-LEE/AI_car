"""Route the hand-placed board: fixed power copper first, then Freerouting for the signals.

Usage, with KiCad's Python: route_board.py power|pretie <board> | dsn <board> <dsn> | ses <board> <ses>
Coordinates are mm from the board's top-left corner. Part positions and silk text are never changed.
"""
import re
import sys

import pcbnew
import wx

OX, OY = 100.0, 100.0
MM = pcbnew.FromMM

# (net, width, layer, [pad names or (x, y)])
POWER = [
    # Pack 1: connector, TVS, MOSFET sources.
    ("BAT1", 2.0, "F", ["J2.1", (59.5, 13.5), (59.5, 9.0), "D1.1"]),
    ("BAT1", 1.0, "F", ["D1.1", (54.4, 9.0), (54.4, 11.64), (53.3, 11.64)]),
    ("BAT1", 1.2, "F", [(53.3, 10.36), (53.3, 12.91)]),
    ("BAT1", 0.6, "F", ["Q1.1", (53.3, 12.91)]), ("BAT1", 0.6, "F", ["Q1.2", (53.3, 11.64)]),
    ("BAT1", 0.6, "F", ["Q1.3", (53.3, 10.36)]),
    # Pack 2.
    ("BAT2", 2.0, "F", ["J3.1", (59.5, 23.5), (59.5, 22.0), "D2.1"]),
    ("BAT2", 1.0, "F", ["D2.1", (54.4, 22.0), (54.4, 24.64), (53.3, 24.64)]),
    ("BAT2", 1.2, "F", [(53.3, 23.36), (53.3, 25.91)]),
    ("BAT2", 0.6, "F", ["Q2.1", (53.3, 25.91)]), ("BAT2", 0.6, "F", ["Q2.2", (53.3, 24.64)]),
    ("BAT2", 0.6, "F", ["Q2.3", (53.3, 23.36)]),
    # Ideal diode outputs to the power switch.
    ("VOR", 1.2, "F", [(47.8, 9.09), (47.8, 26.9)]),
    ("VOR", 0.6, "F", ["Q1.5", (47.8, 9.09)]), ("VOR", 0.6, "F", ["Q1.6", (47.8, 10.36)]),
    ("VOR", 0.6, "F", ["Q1.7", (47.8, 11.64)]), ("VOR", 0.6, "F", ["Q1.8", (47.8, 12.91)]),
    ("VOR", 0.6, "F", ["Q2.5", (47.8, 22.09)]), ("VOR", 0.6, "F", ["Q2.6", (47.8, 23.36)]),
    ("VOR", 0.6, "F", ["Q2.7", (47.8, 24.64)]), ("VOR", 0.6, "F", ["Q2.8", (47.8, 25.91)]),
    ("VOR", 1.5, "F", [(47.03, 11.64), (43.1, 11.64)]),
    ("VOR", 1.0, "F", [(43.1, 10.36), (43.1, 12.91)]),
    ("VOR", 0.6, "F", ["Q3.1", (43.1, 12.91)]), ("VOR", 0.6, "F", ["Q3.2", (43.1, 11.64)]),
    ("VOR", 0.6, "F", ["Q3.3", (43.1, 10.36)]),
    ("VOR", 0.4, "F", ["R5.1", (47.8, 19.33)]),
    ("VOR", 0.4, "F", ["R7.1", (46.05, 24.64)]),
    ("VOR", 0.4, "F", [(47.03, 25.91), "TP1.1"]),
    ("VOR", 0.4, "F", ["U1.6", (51.64, 13.9), (47.8, 13.9)]),
    ("VOR", 0.4, "F", ["U2.6", (51.64, 26.9), (47.8, 26.9)]),
    # Ideal diode controller supply and sense.
    ("BAT1", 0.4, "F", [(54.4, 11.64), (54.4, 17.17), "R1.1"]),
    ("BAT1", 0.4, "F", [(54.4, 17.17), (53.0, 17.45), "U1.4"]),
    ("BAT1", 0.3, "F", ["U1.1", (49.36, 14.75)]),
    ("BAT1", 0.4, "B", [(49.36, 14.75), (54.4, 14.75)]),
    ("BAT2", 0.4, "F", [(54.4, 24.64), (54.4, 29.5), "R2.1"]),
    ("BAT2", 0.4, "F", [(54.4, 29.5), (53.0, 30.45), "U2.4"]),
    ("BAT2", 0.3, "F", ["U2.1", (49.36, 27.75)]),
    ("BAT2", 0.4, "B", [(49.36, 27.75), (54.4, 27.75)]),
    # Switch output to the fuse.
    ("VRAW", 1.6, "F", [(36.0, 9.09), (36.0, 12.91)]),
    ("VRAW", 0.6, "F", ["Q3.5", (36.0, 9.09)]), ("VRAW", 0.6, "F", ["Q3.6", (36.0, 10.36)]),
    ("VRAW", 0.6, "F", ["Q3.7", (36.0, 11.64)]), ("VRAW", 0.6, "F", ["Q3.8", (36.0, 12.91)]),
    ("VRAW", 1.6, "F", [(36.0, 11.0), "F1.1"]),
    # Main bus: trunk down x=31.3, branches to both bucks, bulk caps, TVS and the motor driver.
    ("VSW", 2.0, "F", ["F1.2", (31.3, 11.0)]),
    ("VSW", 1.6, "F", [(31.3, 10.47), (31.3, 26.0)]),
    ("VSW", 1.5, "F", [(31.3, 10.47), "C11.1", "C10.1", (23.6, 10.4)]),
    ("VSW", 0.8, "F", [(23.6, 10.4), (22.9, 10.4)]), ("VSW", 0.8, "F", [(22.9, 9.09), (22.9, 10.4)]),
    ("VSW", 0.6, "F", ["U4.7", (22.9, 10.36)]), ("VSW", 0.6, "F", ["U4.8", (22.9, 9.09)]),
    ("VSW", 0.6, "F", ["R10.1", (31.3, 14.0)]),
    ("VSW", 1.2, "F", [(31.3, 21.0), "D3.1"]),
    ("VSW", 1.2, "F", ["D3.1", (25.5, 20.6), "C5.1"]),
    ("VSW", 0.8, "F", [(31.3, 26.0), (29.65, 27.6), (29.65, 31.95), "C21.1"]),
    ("VSW", 1.5, "F", ["C21.1", "C20.1", (23.6, 31.9)]),
    ("VSW", 0.8, "F", [(23.6, 31.9), (22.9, 31.87)]), ("VSW", 0.8, "F", [(22.9, 30.59), (22.9, 31.87)]),
    ("VSW", 0.6, "F", ["U5.7", (22.9, 31.87)]), ("VSW", 0.6, "F", ["U5.8", (22.9, 30.59)]),
    ("VSW", 0.6, "F", [(29.65, 31.95), "R14.1"]),
    ("VSW", 1.2, "F", [(31.3, 26.0), (33.5, 26.0)]),
    ("VSW", 1.0, "F", [(33.5, 26.0), (33.5, 42.5), "C6.1"]),
    ("VSW", 0.8, "F", [(33.5, 33.5), (53.0, 33.5)]),
    ("VSW", 1.0, "F", [(53.0, 33.5), (53.2, 35.0), "C30.1"]),
    ("VSW", 0.8, "F", [(53.2, 35.0), (53.2, 39.75)]),
    ("VSW", 0.4, "F", ["U6.13", (53.2, 39.75)]), ("VSW", 0.4, "F", ["U6.14", (53.2, 39.75)]),
    ("VSW", 0.6, "F", ["C31.1", (53.2, 39.72)]),
    # VM1 (pin 24) feeds channel A, on the back so the logic pins can escape left.
    ("VSW", 0.8, "B", [(53.2, 38.0), (53.2, 47.4)]),
    ("VSW", 0.4, "F", ["U6.24", (53.2, 47.4)]),
    # Pi 5V.
    ("LXA", 0.6, "F", ["U4.2", (15.7, 10.6)]), ("LXA", 1.0, "F", [(15.7, 10.6), "L1.1"]),
    ("LXA", 0.4, "F", [(13.4, 10.5), (13.4, 7.72), "C15.2"]),
    ("5V_PI", 1.5, "F", ["L1.2", (6.2, 9.0), (6.2, 2.23), "J1.2"]),
    ("5V_PI", 1.0, "F", ["J1.2", "J1.4"]),
    ("5V_PI", 1.5, "F", ["L1.2", (6.0, 13.0), (6.0, 15.53)]),
    ("5V_PI", 1.0, "F", ["C12.1", "C13.1", "C14.1"]),
    ("5V_PI", 0.8, "F", [(6.0, 15.53), (6.0, 21.0), "D4.1"]),
    ("5V_PI", 0.4, "F", ["D4.1", (7.5, 22.5)]), ("5V_PI", 0.4, "F", ["TP3.1", (7.5, 22.5)]),
    ("5V_PI", 0.6, "B", [(7.5, 22.5), (12.0, 27.0), (12.0, 45.0), (17.0, 48.5), "J6.1"]),
    ("5V_PI", 0.4, "F", ["C14.1", (13.3, 15.53), (14.0, 18.6), (19.67, 18.6), "R12.1"]),
    ("5V_PI", 0.4, "F", [(19.67, 18.6), (24.72, 18.6), "C18.1"]),
    # Servo 5V.
    ("LXB", 0.6, "F", ["U5.2", (15.7, 32.1)]), ("LXB", 1.0, "F", [(15.7, 32.1), "L2.1"]),
    ("LXB", 0.4, "F", [(13.5, 31.0), "C25.2"]),
    ("5V_SRV", 1.5, "F", ["L2.2", (6.0, 34.5), (6.0, 37.53)]),
    ("5V_SRV", 1.2, "F", ["C22.1", "C23.1", "C24.1"]),
    ("5V_SRV", 1.2, "F", ["C24.1", (14.2, 37.53), (14.2, 44.0), "C7.1"]),
    ("5V_SRV", 1.0, "F", [(14.2, 44.0), (14.2, 48.4), (11.04, 48.4), "J5.2"]),
    ("5V_SRV", 0.4, "F", [(14.2, 39.6), (20.67, 39.6), "R16.1"]),
    ("5V_SRV", 0.4, "F", [(20.67, 39.6), (24.22, 39.6), "C28.1"]),
    # Motor outputs: MOT1 on the front, MOT2 around the outside on the back.
    ("MOT1", 0.35, "F", ["U6.1", (62.7, 46.25)]), ("MOT1", 0.35, "F", ["U6.2", (62.7, 46.25)]),
    ("MOT1", 0.8, "F", [(62.7, 46.25), (62.7, 48.0), (56.5, 48.5), "J4.1"]),
    ("MOT2", 0.35, "F", ["U6.5", (62.6, 43.65)]), ("MOT2", 0.35, "F", ["U6.6", (62.6, 43.65)]),
    ("MOT2", 0.8, "F", [(62.6, 43.65), (63.9, 43.6)]),
    ("MOT2", 0.8, "B", [(63.9, 43.6), (63.9, 49.5), (53.0, 49.5), "J4.2"]),
    # GND spine on the back: routed signals may not cut the return path between packs, bucks and motor driver.
    ("GND", 1.0, "B", ["J2.2", (59.7, 16.0), (59.7, 26.0), "J3.2"]),
    ("GND", 1.0, "B", [(59.7, 20.0), (5.0, 20.0)]),
    ("GND", 1.0, "B", [(21.0, 7.6), (21.0, 33.5)]),
    ("GND", 1.0, "B", [(21.0, 33.5), "J6.2"]),
    ("GND", 1.0, "B", [(21.0, 7.6), (18.53, 6.4), "J1.9"]),
    ("GND", 1.0, "B", [(59.7, 26.0), (59.7, 39.0), (60.4, 41.05)]),
    # GND pads the pour cannot reach.
    ("GND", 0.3, "F", ["D5.1", "D6.1", (38.71, 38.7)]),
    ("GND", 0.25, "F", ["U7.1", (37.6, 27.5)]), ("GND", 0.25, "F", ["U7.3", (37.9, 28.5), (37.6, 27.5)]),
    ("GND", 0.3, "F", ["U1.2", "U1.3", (49.36, 18.8)]),
    # Test point on the servo rail.
    ("5V_SRV", 0.6, "F", ["TP4.1", (5.5, 33.0)]),
    # Motor driver GND.
    ("GND", 0.4, "F", ["U6.9", (62.7, 41.05)]), ("GND", 0.4, "F", ["U6.10", (62.7, 41.05)]),
    ("GND", 0.4, "F", ["U6.3", (62.7, 44.95)]), ("GND", 0.4, "F", ["U6.4", (62.7, 44.95)]),
    ("GND", 0.4, "F", [(54.0, 40.72), (54.0, 42.68)]),
    ("GND", 0.3, "F", ["U6.15", (54.0, 40.72)]), ("GND", 0.3, "F", ["U6.16", (54.0, 41.38)]),
    ("GND", 0.3, "F", ["U6.17", (54.0, 42.03)]), ("GND", 0.3, "F", ["U6.18", (54.0, 42.68)]),
    ("GND", 0.6, "F", [(54.0, 41.7), (52.4, 41.7), "C31.2"]),
    ("GND", 0.6, "F", [(52.4, 41.7), "C32.2"]),
    ("GND", 0.6, "F", ["C31.2", (50.5, 42.2)]), ("GND", 0.6, "F", ["C32.2", (50.5, 42.2)]),
]

# (net, x, y)
VIAS = [
    ("BAT1", 49.36, 14.75), ("BAT1", 54.4, 14.75), ("BAT2", 49.36, 27.75), ("BAT2", 54.4, 27.75),
    ("5V_PI", 7.5, 22.5), ("VSW", 53.2, 38.0), ("VSW", 53.2, 47.4), ("MOT2", 63.9, 43.6),
    ("GND", 62.7, 41.05), ("GND", 62.7, 44.95), ("GND", 60.4, 41.05), ("GND", 50.5, 42.2), ("GND", 38.71, 38.7), ("GND", 37.6, 27.5), ("GND", 49.36, 18.8),
    # Under the buck exposed pads.
    ("GND", 18.8, 10.2), ("GND", 20.2, 10.2), ("GND", 18.8, 11.8), ("GND", 20.2, 11.8), ("GND", 19.5, 11.0),
    ("GND", 18.8, 31.7), ("GND", 20.2, 31.7), ("GND", 18.8, 33.3), ("GND", 20.2, 33.3), ("GND", 19.5, 32.5),
]


def pt(x, y):
    return pcbnew.VECTOR2I(MM(OX + x), MM(OY + y))


def power(path):
    board = pcbnew.LoadBoard(path)
    pads = {}
    for f in board.GetFootprints():
        for p in f.Pads():
            pos = p.GetPosition()
            pads[f"{f.GetReference()}.{p.GetNumber()}"] = (pcbnew.ToMM(pos.x) - OX, pcbnew.ToMM(pos.y) - OY)
    for name, width, layer, points in POWER:
        coords = [pads[p] if isinstance(p, str) else p for p in points]
        for a, b in zip(coords, coords[1:]):
            t = pcbnew.PCB_TRACK(board)
            t.SetStart(pt(*a))
            t.SetEnd(pt(*b))
            t.SetWidth(MM(width))
            t.SetLayer(pcbnew.F_Cu if layer == "F" else pcbnew.B_Cu)
            t.SetNet(board.FindNet(name))
            t.SetLocked(True)
            board.Add(t)
    for name, x, y in VIAS:
        v = pcbnew.PCB_VIA(board)
        v.SetPosition(pt(x, y))
        v.SetWidth(MM(0.7))
        v.SetDrill(MM(0.35))
        v.SetNet(board.FindNet(name))
        v.SetLocked(True)
        board.Add(v)
    pcbnew.SaveBoard(path, board)
    print("power tracks", len(POWER), "vias", len(VIAS))


def pretie(path):
    """Give each GND SMD pad its own short stub and via before routing, so signals cannot wall it off."""
    import math

    import pcb_tool
    board = pcbnew.LoadBoard(path)
    to = pcbnew.ToMM
    fits, place, tracks, _ = pcb_tool.stitch(board, targets=True)
    gnd_vias = [(to(t.GetPosition().x), to(t.GetPosition().y)) for t in board.GetTracks()
                if t.Type() == pcbnew.PCB_VIA_T and t.GetNetname() == "GND"]
    others = []
    for f in board.GetFootprints():
        for q in f.Pads():
            if q.GetNetname() != "GND":
                b = q.GetBoundingBox()
                others.append((to(b.GetLeft()), to(b.GetTop()), to(b.GetRight()), to(b.GetBottom())))
    added, failed = 0, []

    def clear(px, py, x, y):
        for ax, ay, bx, by, hw, n in tracks:
            if n != "GND" and any(pcb_tool.seg_dist(ax + (bx - ax) * k / 20, ay + (by - ay) * k / 20, px, py, x, y) < hw + 0.25
                                  for k in range(21)):
                return False
        for l, t, r, b in others:
            for k in range(11):
                sx, sy = px + (x - px) * k / 10, py + (y - py) * k / 10
                if l - 0.2 < sx < r + 0.2 and t - 0.2 < sy < b + 0.2:
                    return False
        return True

    for f in board.GetFootprints():
        for p in f.Pads():
            if p.GetNetname() != "GND" or p.HasHole() or f.GetReference().startswith("TP"):
                continue
            b = p.GetBoundingBox()
            px, py = to(p.GetPosition().x), to(p.GetPosition().y)
            half = max(to(b.GetWidth()), to(b.GetHeight())) / 2
            if any(math.hypot(px - vx, py - vy) < half + 0.6 for vx, vy in gnd_vias):
                continue
            done = False
            for dist in (half + 0.5, half + 0.7, half + 0.9, half + 1.2, half + 1.6, half + 2.0):
                for ang in range(0, 360, 15):
                    x, y = px + dist * math.cos(math.radians(ang)), py + dist * math.sin(math.radians(ang))
                    if fits(x, y) and clear(px, py, x, y):
                        place(x, y)
                        gnd_vias.append((x, y))
                        t = pcbnew.PCB_TRACK(board)
                        t.SetStart(pcbnew.VECTOR2I(MM(px), MM(py)))
                        t.SetEnd(pcbnew.VECTOR2I(MM(x), MM(y)))
                        t.SetWidth(MM(0.3))
                        t.SetLayer(pcbnew.F_Cu)
                        t.SetNet(board.FindNet("GND"))
                        t.SetLocked(True)
                        board.Add(t)
                        added += 1
                        done = True
                        break
                if done:
                    break
            if not done:
                failed.append(f"{f.GetReference()}.{p.GetNumber()}")
    for t in board.GetTracks():
        if t.Type() == pcbnew.PCB_VIA_T and t.GetNetname() == "GND":
            t.SetLocked(True)
    pcbnew.SaveBoard(path, board)
    print("pad ties", added, "no room", failed)


def dsn(path, out):
    board = pcbnew.LoadBoard(path)
    if not pcbnew.ExportSpecctraDSN(board, out):
        sys.exit("DSN export failed")
    text = open(out).read()
    text, n = re.subn(r'\(net "?GND"?\s*\(pins[^)]*\)\s*\)', "", text)
    # The stragglers on power nets are small bias and sense pins, so route them at signal width.
    i = text.index("(network")
    head, net = text[:i], text[i:]
    net, classes = re.subn(r"\(width \d+(\.\d+)?\)", "(width 400)", net)
    open(out, "w").write(head + net)
    print("GND nets removed", n, "class widths set", classes)


def ses(path, routed):
    board = pcbnew.LoadBoard(path)
    if not pcbnew.ImportSpecctraSES(board, routed):
        sys.exit("SES import failed")
    for t in board.GetTracks():    # Freerouting necks down at tight pads; hold them to the 0.15 mm rule.
        if t.Type() != pcbnew.PCB_VIA_T and t.GetWidth() < MM(0.15):
            t.SetWidth(MM(0.15))
    pcbnew.SaveBoard(path, board)
    print("tracks", len(board.GetTracks()))


if __name__ == "__main__":
    {"power": power, "pretie": pretie, "dsn": dsn, "ses": ses}[sys.argv[1]](*sys.argv[2:])
