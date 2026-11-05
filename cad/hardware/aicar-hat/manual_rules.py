"""Add hand-routing net classes and track widths to aicar-hat.kicad_pro. Run with KiCad closed; safe to repeat."""
import json
from pathlib import Path

PRO = Path(__file__).with_name("aicar-hat.kicad_pro")

# name: (track width, via diameter, via drill, clearance, nets)
CLASSES = {
    "Battery": (0.4, 0.8, 0.4, 0.2, ["BAT1", "BAT2", "VOR", "VRAW", "VSW"]),
    "Power5V": (0.4, 0.8, 0.4, 0.2, ["5V_PI", "5V_SRV", "LXA", "LXB"]),
    "Ground": (0.4, 0.7, 0.35, 0.15, ["GND"]),
    "Motor": (0.4, 0.8, 0.4, 0.2, ["MOT1", "MOT2"]),
}
WIDTHS = [0.3, 0.35, 0.4]
VIAS = [(0.7, 0.35), (0.8, 0.4)]

p = json.loads(PRO.read_text())
settings = p["net_settings"]
default = next(c for c in settings["classes"] if c["name"] == "Default")
default["track_width"] = 0.4
settings["classes"] = [default]
settings["netclass_patterns"] = []
for priority, (name, (width, via_d, via_drill, clearance, nets)) in enumerate(CLASSES.items()):
    settings["classes"].append({**default, "name": name, "track_width": width, "via_diameter": via_d,
                                "via_drill": via_drill, "clearance": clearance, "priority": priority})
    settings["netclass_patterns"] += [{"netclass": name, "pattern": net} for net in nets]
design = p["board"]["design_settings"]
design["track_widths"] = [0.0] + WIDTHS     # KiCad keeps slot 0 for "use net class width".
design["via_dimensions"] = [{"diameter": 0.0, "drill": 0.0}] + [{"diameter": d, "drill": h} for d, h in VIAS]
PRO.write_text(json.dumps(p, indent=2))
print("net classes", list(CLASSES), "widths", WIDTHS)
