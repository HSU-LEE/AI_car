"""Summarise out/drc.json."""
import json
from collections import Counter

d = json.load(open("out/drc.json"))
v, u, p = d.get("violations", []), d.get("unconnected_items", []), d.get("schematic_parity", [])
print("violations", len(v), dict(Counter(x["type"] for x in v)))
print("unconnected", len(u))
print("parity", len(p), dict(Counter(x["type"] for x in p)))
for x in v[:14]:
    print(" ", x["type"], "|", " / ".join(i["description"][:46] for i in x["items"]))
nets = Counter()
for x in u:
    for i in x["items"]:
        m = i["description"]
        nets[m[m.find("[") + 1:m.find("]")] if "[" in m else m[:24]] += 1
print("unconnected nets", nets.most_common(14))
for x in p[:6]:
    print(" ", x["type"], x["description"][:70], "|", " / ".join(i["description"][:40] for i in x.get("items", [])))
