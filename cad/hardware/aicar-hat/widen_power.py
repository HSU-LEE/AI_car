"""Widen power tracks to 0.5 mm wherever design rules allow, leaving the rest at their current width.

Run with KiCad's Python: widen_power.py <board>. Each pass runs DRC and puts any widened track it flags back.
"""
import json
import subprocess
import sys
from pathlib import Path

import pcbnew
import wx

CLI = "/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli"
NETS = {"BAT1", "BAT2", "VOR", "VRAW", "VSW", "5V_PI", "5V_SRV", "LXA", "LXB"}
WIDE = pcbnew.FromMM(0.5)
MAX_PASSES = 12


def main(path):
    report = Path(path).with_suffix(".drc.json")
    board = pcbnew.LoadBoard(path)
    original = {}
    for t in board.GetTracks():
        if t.Type() == pcbnew.PCB_TRACE_T and t.GetNetname() in NETS and pcbnew.ToMM(t.GetWidth()) > 0.39:
            original[t.m_Uuid.AsString()] = t.GetWidth()
            t.SetWidth(WIDE)
    print("widened", len(original))
    restored = 0
    for n in range(MAX_PASSES):
        pcbnew.ZONE_FILLER(board).Fill(board.Zones())
        pcbnew.SaveBoard(path, board)
        subprocess.run([CLI, "pcb", "drc", "--severity-error", "--format", "json", "-o", str(report), path],
                       capture_output=True, check=False)
        data = json.loads(report.read_text())
        flagged = {i.get("uuid") for v in data["violations"] for i in v["items"]} & set(original)
        print(f"pass {n + 1}: violations {len(data['violations'])}, restoring {len(flagged)}", flush=True)
        if not flagged:
            break
        for t in board.GetTracks():
            if t.Type() == pcbnew.PCB_TRACE_T and t.m_Uuid.AsString() in flagged:
                t.SetWidth(original[t.m_Uuid.AsString()])
                restored += 1
        for u in flagged:
            del original[u]
    print("kept at 0.5 mm", len(original), "put back", restored)


if __name__ == "__main__":
    app = wx.App(False)
    main(sys.argv[1])
