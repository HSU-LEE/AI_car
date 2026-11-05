"""Snap parts to rows and columns on a 0.5 mm grid. Run with KiCad closed, using KiCad's Python.

Coordinates are mm from the board's top-left corner. J1 and the mounting holes stay on the Pi's grid.
"""
import pcbnew
import wx

PATH = "aicar-hat.kicad_pcb"
OX, OY = 100.0, 100.0
MM = pcbnew.FromMM

# ref: (x, y, rotation)
PLACE = {
    # Pi buck. The servo buck below repeats it 22 mm lower.
    "L1": (9.5, 11.0, 180), "U4": (19.5, 11.0, 0), "C15": (15.0, 8.0, 90),
    "C10": (24.5, 9.0, 90), "C11": (27.0, 9.0, 90),
    "C16": (25.0, 12.5, 0), "R10": (28.0, 12.5, 180),
    "R11": (15.5, 15.5, 0), "C17": (18.5, 15.5, 0), "R12": (21.5, 15.5, 0), "R13": (27.5, 15.5, 0),
    "C18": (24.5, 15.5, 0),
    "C12": (4.0, 17.0, 270), "C13": (8.0, 17.0, 270), "C14": (12.0, 17.0, 270),
    "D4": (5.0, 23.35, 270), "TP3": (10.0, 23.0, 0), "TP2": (32.5, 26.0, 0),

    # Top row of the power path.
    "F1": (33.0, 11.0, 180), "Q3": (39.5, 11.0, 180), "C3": (39.5, 7.5, 0),

    # Undervoltage cutoff: three columns on three rows.
    "R9": (37.5, 16.0, 0), "R6": (44.0, 16.0, 90),
    "R8": (37.5, 19.5, 0), "U3": (41.0, 19.5, 0), "R5": (44.0, 19.5, 90),
    "C33": (37.5, 23.0, 0), "C4": (41.0, 23.0, 0), "R7": (44.0, 23.0, 90),
    "SW1": (33.5, 16.0, 0),

    # Pack inputs: pack 2 repeats pack 1 13 mm lower.
    "Q1": (50.5, 11.0, 180), "D1": (56.0, 11.0, 270), "U1": (50.5, 16.0, 0), "R1": (55.5, 16.0, 0),
    "Q2": (50.5, 24.0, 180), "D2": (56.0, 24.0, 270), "U2": (50.5, 29.0, 0), "R2": (55.5, 29.0, 0),
    "J2": (62.1, 13.5, 270), "J3": (62.1, 23.5, 270),

    # ADC.
    "D3": (28.0, 23.0, 270), "U7": (41.0, 29.0, 0), "TP1": (45.5, 29.0, 0),
    "R3": (32.5, 29.0, 0), "C1": (36.0, 29.0, 180), "R4": (32.5, 32.0, 0), "C2": (36.0, 32.0, 0),

    # Bus capacitors and LEDs.
    "C5": (20.0, 23.0, 180), "C6": (37.5, 44.0, 0),
    "R23": (36.0, 36.0, 0), "D5": (39.0, 36.0, 0), "R24": (36.0, 38.0, 0), "D6": (39.0, 38.0, 0),

    # Motor driver.
    "U6": (56.0, 43.0, 180), "C30": (55.5, 35.0, 0), "C31": (50.0, 39.5, 270), "C32": (50.0, 43.0, 270),
    "J4": (52.0, 52.2, 0),

    # Servo output and sensors, one row above the bottom connectors.
    "C7": (20.0, 44.0, 0), "TP4": (3.0, 33.0, 0),
    "R20": (13.5, 47.0, 0), "C34": (8.0, 47.0, 0), "R21": (27.5, 47.0, 0), "R22": (30.5, 47.0, 0),
    "C35": (29.5, 44.0, 0), "TP5": (3.0, 47.0, 0),
    "J5": (8.5, 52.2, 90), "J6": (18.5, 52.2, 90), "J7": (29.5, 52.2, 90),
}
for a, b in (("L1", "L2"), ("U4", "U5"), ("C15", "C25"), ("C10", "C20"), ("C11", "C21"), ("C16", "C26"),
             ("R10", "R14"), ("R11", "R15"), ("C17", "C27"), ("R12", "R16"), ("R13", "R17"), ("C18", "C28"),
             ("C12", "C22"), ("C13", "C23"), ("C14", "C24")):
    x, y, r = PLACE[a]
    PLACE[b] = (x, y + 22.0, r)

# Hand-tuned lower buck clearances; keep them when this table is run again.
PLACE.update({"C20": (25.0, 31.0, 90), "C21": (28.0, 31.0, 90), "C25": (14.5, 29.0, 90)})

# Front-only repack: spread the signal parts while keeping power loops and ports in place.
PLACE.update({
    "C1": (35.5, 29.0, 180), "C3": (37.0, 7.5, 0), "C33": (37.5, 24.5, 0),
    "C34": (7.0, 47.0, 0), "C35": (29.5, 43.5, 0), "C4": (41.5, 23.0, 0),
    "D5": (39.5, 35.0, 0), "D6": (39.5, 37.5, 0),
    "R1": (55.5, 16.5, 0), "R2": (55.5, 29.5, 0),
    "R10": (29.0, 14.0, 180), "R11": (15.0, 16.5, 0),
    "R12": (20.5, 17.5, 0), "R13": (28.5, 16.5, 0),
    "R14": (29.0, 35.5, 180), "R15": (15.0, 38.0, 0),
    "R16": (21.5, 38.5, 0), "R17": (28.5, 38.0, 0),
    "R20": (12.0, 47.0, 0), "R21": (28.0, 46.0, 0), "R22": (29.5, 48.5, 0),
    "R23": (35.5, 35.0, 0), "R24": (35.5, 37.5, 0),
    "R3": (31.5, 29.0, 0), "R4": (32.0, 32.0, 0),
    "R5": (46.5, 18.5, 90), "R6": (45.0, 14.5, 90),
    "R7": (45.0, 23.5, 90), "R8": (38.0, 19.5, 0), "R9": (38.0, 16.0, 0),
    "TP1": (46.5, 29.0, 0), "TP2": (32.5, 25.5, 0),
    "TP4": (2.5, 33.0, 0), "TP5": (2.0, 47.0, 0),
    "U3": (42.5, 19.0, 0), "U7": (41.0, 28.5, 0),
})

LABELS = {"BAT1 +": (61.5, 9.5), "BAT2 +": (61.5, 19.7), "PWR": (42.8, 36.0), "ST": (42.0, 38.0)}

def main():
    app = wx.App(False)
    board = pcbnew.LoadBoard(PATH)
    fixed = {f.GetReference() for f in board.GetFootprints() if f.IsLocked()}
    refs = {f.GetReference() for f in board.GetFootprints()} - fixed
    if refs != set(PLACE):
        raise SystemExit(f"table mismatch: missing {sorted(refs - set(PLACE))}, extra {sorted(set(PLACE) - refs)}")
    for f in board.GetFootprints():
        ref = f.GetReference()
        if ref in PLACE:
            x, y, rot = PLACE[ref]
            f.SetOrientationDegrees(rot)
            f.SetPosition(pcbnew.VECTOR2I(MM(OX + x), MM(OY + y)))
    for d in board.GetDrawings():
        if isinstance(d, pcbnew.PCB_TEXT) and d.GetText() in LABELS:
            x, y = LABELS[d.GetText()]
            d.SetPosition(pcbnew.VECTOR2I(MM(OX + x), MM(OY + y)))
    pcbnew.SaveBoard(PATH, board)
    print("placed", len(PLACE))


if __name__ == "__main__":
    main()
