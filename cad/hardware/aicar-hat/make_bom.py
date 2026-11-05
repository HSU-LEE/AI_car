"""JLCPCB BOM and placement files from design.py and KiCad's position export."""
import csv
import re
from collections import defaultdict

from design import PARTS

# Added to KiCad's rotation for parts whose JLCPCB model faces another way (JLCKicadTools table).
# Starting values only: the placement preview on the order page has the final say.
ROTATION = [(r"^SOIC-", 270), (r"^SOT-23", -90), (r"^TSOT-23", 180), (r"^SSOP-", 270), (r"^CP_Elec_8x10", 180)]

# Parts soldered by hand: LCSC number and note, each checked against its LCSC page.
HAND = {
    "J1": ("C2977589", "2x20 socket, 8.5 mm tall, straight. Fit on the bottom side, solder on top."),
    "J2": ("C158012", "JST B2B-XH-A"),
    "J3": ("C158012", "JST B2B-XH-A"),
    "J4": ("", "Solder the motor wires straight to the pads."),
    "J5": ("C2895846", "Break 3 pins off the 40-pin strip."),
    "J6": ("C2895846", "Break 3 pins off the 40-pin strip."),
    "J7": ("C2895846", "Break 4 pins off the 40-pin strip."),
    "SW1": ("C2895846", "Break 3 pins off the strip. Shorting pins 1 and 2 turns the board on."),
}


def rotated(footprint, rotation):
    extra = next((deg for pattern, deg in ROTATION if re.match(pattern, footprint)), 0)
    return f"{(float(rotation) + extra) % 360:g}"


groups = defaultdict(list)
hand = []
for ref, (value, fp, lcsc, pins) in PARTS.items():
    if lcsc:
        comment = "Red LED" if lcsc == "C2286" else value    # D5 and D6 are one part with two labels.
        groups[(comment, fp.split(":")[1], lcsc)].append(ref)
    elif not ref.startswith("TP"):
        hand.append((ref, value, fp.split(":")[1], *HAND[ref]))

with open("fab/bom-jlcpcb.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Comment", "Designator", "Footprint", "LCSC Part #"])
    for (value, fp, lcsc), refs in sorted(groups.items(), key=lambda kv: kv[0][2]):
        w.writerow([value, ",".join(sorted(refs)), fp, lcsc])

with open("fab/hand-solder.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Designator", "Part", "Footprint", "LCSC Part #", "Note"])
    w.writerows(hand)

smt = {r for refs in groups.values() for r in refs}
with open("fab/pos.csv") as src, open("fab/cpl-jlcpcb.csv", "w", newline="") as dst:
    w = csv.writer(dst)
    w.writerow(["Designator", "Mid X", "Mid Y", "Layer", "Rotation"])
    for row in csv.DictReader(src):
        if row["Ref"] in smt:
            layer = "Top" if row["Side"] == "top" else "Bottom"
            footprint = PARTS[row["Ref"]][1].split(":")[1]
            w.writerow([row["Ref"], row["PosX"], row["PosY"], layer, rotated(footprint, row["Rot"])])

print("SMT lines", len(groups), "SMT parts", len(smt), "hand", len(hand))
