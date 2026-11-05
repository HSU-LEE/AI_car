"""Build aicar-hat.kicad_pcb from design.py using KiCad's pcbnew.

Run with KiCad's own Python. Coordinates are mm from the HAT's top-left corner,
with the Pi USB ports to the right, matching the Pi's top view.
"""
import json
import sys
from pathlib import Path

import pcbnew
import wx

from design import PARTS
from layout import OX, OY, PLACE, POWER, W, H, NOTCH, LABELS, VIAS

HERE = Path(__file__).parent
LIB = "/Applications/KiCad/KiCad.app/Contents/SharedSupport/footprints/"


def mm(x):
    return pcbnew.FromMM(x)


def pt(x, y):
    return pcbnew.VECTOR2I(mm(OX + x), mm(OY + y))


def segment(board, a, b, layer=pcbnew.Edge_Cuts, width=0.1):
    s = pcbnew.PCB_SHAPE(board)
    s.SetShape(pcbnew.SHAPE_T_SEGMENT)
    s.SetStart(pt(*a))
    s.SetEnd(pt(*b))
    s.SetLayer(layer)
    s.SetWidth(mm(width))
    board.Add(s)


def arc(board, c, start, end):
    s = pcbnew.PCB_SHAPE(board)
    s.SetShape(pcbnew.SHAPE_T_ARC)
    s.SetCenter(pt(*c))
    s.SetStart(pt(*start))
    s.SetEnd(pt(*end))
    s.SetLayer(pcbnew.Edge_Cuts)
    s.SetWidth(mm(0.1))
    board.Add(s)


def outline(board):
    r = 3.0
    nx0, nx1, ny = NOTCH
    segment(board, (r, 0), (W - r, 0))
    arc(board, (W - r, r), (W - r, 0), (W, r))
    segment(board, (W, r), (W, H - r))
    arc(board, (W - r, H - r), (W, H - r), (W - r, H))
    segment(board, (W - r, H), (nx1, H))
    segment(board, (nx1, H), (nx1, ny))
    segment(board, (nx1, ny), (nx0, ny))
    segment(board, (nx0, ny), (nx0, H))
    segment(board, (nx0, H), (r, H))
    arc(board, (r, H - r), (r, H), (0, H - r))
    segment(board, (0, H - r), (0, r))
    arc(board, (r, r), (0, r), (r, 0))


def load(fp):
    lib, name = fp.split(":")
    return pcbnew.FootprintLoad(LIB + lib + ".pretty", name)


def add_zone(board, net, layer, x0, y0, x1, y1, priority=0, clearance=0.25):
    z = pcbnew.ZONE(board)
    z.SetLayer(layer)
    z.SetNet(net)
    z.SetAssignedPriority(priority)
    z.SetLocalClearance(mm(clearance))
    z.SetMinThickness(mm(0.2))
    z.SetThermalReliefGap(mm(0.3))
    z.SetThermalReliefSpokeWidth(mm(0.5))
    z.SetPadConnection(pcbnew.ZONE_CONNECTION_THERMAL)
    z.SetIslandRemovalMode(pcbnew.ISLAND_REMOVAL_MODE_ALWAYS)
    o = z.Outline()
    o.NewOutline()
    for p in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        o.Append(pt(*p))
    board.Add(z)
    return z


def main(path):
    app = wx.App(False)     # pcbnew's library loader needs an app object.
    board = pcbnew.BOARD()
    ds = board.GetDesignSettings()
    ds.m_TrackMinWidth = mm(0.15)
    ds.m_ViasMinSize = mm(0.6)
    ds.m_ViasMinDrill = mm(0.3)
    ds.m_MinClearance = mm(0.15)
    ds.SetAuxOrigin(pt(0, H))    # Bottom-left corner, so Gerbers and placement share positive coordinates.
    ds.m_CopperEdgeClearance = mm(0.2)
    nc = ds.m_NetSettings.GetDefaultNetclass()
    nc.SetClearance(mm(0.15))
    nc.SetTrackWidth(mm(0.25))
    nc.SetViaDiameter(mm(0.7))
    nc.SetViaDrill(mm(0.35))

    nets = {}

    def net(name):
        if name not in nets:
            n = pcbnew.NETINFO_ITEM(board, name)
            board.Add(n)
            nets[name] = n
        return nets[name]

    outline(board)
    for x, y in ((3.5, 3.5), (61.5, 3.5), (3.5, 52.5), (61.5, 52.5)):
        h = load("MountingHole:MountingHole_2.7mm_M2.5")
        h.SetPosition(pt(x, y))
        h.SetReference(f"H{len([f for f in board.GetFootprints() if f.GetReference().startswith('H')]) + 1}")
        h.SetAttributes(pcbnew.FP_EXCLUDE_FROM_POS_FILES | pcbnew.FP_EXCLUDE_FROM_BOM)
        h.Reference().SetVisible(False)
        h.Value().SetVisible(False)
        board.Add(h)

    missing = [r for r in PARTS if r not in PLACE]
    if missing:
        sys.exit(f"unplaced: {missing}")
    for ref, (value, fp, lcsc, pins) in PARTS.items():
        x, y, rot, side = PLACE[ref]
        f = load(fp)
        f.SetReference(ref)
        f.SetValue(value)
        board.Add(f)    # Flip needs the footprint to belong to a board.
        f.SetPosition(pt(x, y))
        if side == "B":
            f.Flip(f.GetPosition(), pcbnew.FLIP_DIRECTION_LEFT_RIGHT)
        f.SetOrientationDegrees(rot)
        if lcsc:
            f.SetField("LCSC", lcsc)
            for field in f.GetFields():
                if field.GetName() == "LCSC":
                    field.SetVisible(False)
        for pad in f.Pads():
            n = pad.GetNumber()
            if n in pins and pins[n]:
                pad.SetNet(net(pins[n]))
        ref_text = f.Reference()
        ref_text.SetTextSize(pcbnew.VECTOR2I(mm(0.7), mm(0.7)))
        ref_text.SetTextThickness(mm(0.12))
        f.Value().SetVisible(False)

    pads = {}
    for f in board.GetFootprints():
        for p in f.Pads():
            pos = p.GetPosition()
            pads.setdefault(f"{f.GetReference()}.{p.GetNumber()}", []).append(
                (round(pcbnew.ToMM(pos.x) - OX, 3), round(pcbnew.ToMM(pos.y) - OY, 3)))

    # J1 must land exactly on the Pi header, whatever the footprint's orientation.
    for pin, want in (("1", (8.37, 4.77)), ("2", (8.37, 2.23)), ("3", (10.91, 4.77)), ("40", (56.63, 2.23))):
        got = pads[f"J1.{pin}"][0]
        if abs(got[0] - want[0]) > 0.02 or abs(got[1] - want[1]) > 0.02:
            print(f"J1 pin {pin} at {got}, want {want}")

    def resolve(p):
        return tuple(pads[p][0]) if isinstance(p, str) else p

    for name, width, layer, points in POWER:
        lay = pcbnew.F_Cu if layer == "F" else pcbnew.B_Cu
        pts = [resolve(p) for p in points]
        for a, b in zip(pts, pts[1:]):
            t = pcbnew.PCB_TRACK(board)
            t.SetStart(pt(*a))
            t.SetEnd(pt(*b))
            t.SetWidth(mm(width))
            t.SetLayer(lay)
            t.SetNet(net(name))
            t.SetLocked(True)
            board.Add(t)
    for name, x, y in VIAS:
        v = pcbnew.PCB_VIA(board)
        v.SetPosition(pt(*resolve(x)) if isinstance(x, str) else pt(x, y))
        v.SetWidth(mm(0.7))
        v.SetDrill(mm(0.35))
        v.SetNet(net(name))
        v.SetLocked(True)
        board.Add(v)

    gnd = net("GND")
    add_zone(board, gnd, pcbnew.B_Cu, -1, -1, W + 1, H + 1, priority=0)
    add_zone(board, gnd, pcbnew.F_Cu, -1, -1, W + 1, H + 1, priority=0)

    for text, x, y, layer in LABELS:
        t = pcbnew.PCB_TEXT(board)
        t.SetText(text)
        t.SetPosition(pt(x, y))
        t.SetLayer(pcbnew.F_SilkS if layer == "F" else pcbnew.B_SilkS)
        t.SetTextSize(pcbnew.VECTOR2I(mm(1.0), mm(1.0)))
        t.SetTextThickness(mm(0.15))
        if layer == "B":
            t.SetMirrored(True)
        board.Add(t)

    board.SetCopperLayerCount(2)
    pcbnew.SaveBoard(str(path), board)
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "pads.json").write_text(json.dumps(pads, indent=0))
    print("saved", path)


if __name__ == "__main__":
    main(HERE / (sys.argv[1] if len(sys.argv) > 1 else "aicar-hat.kicad_pcb"))
