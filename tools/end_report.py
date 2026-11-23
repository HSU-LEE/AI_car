"""Where each cm= action really ended, from drive logs and recordings on this Mac. A table per step, to set against a tape.

  .venv/bin/python tools/end_report.py logs/20261123-101500 runs/20261123-102000 ...
  .venv/bin/python tools/end_report.py                       every drive log and recording under logs/, runs/ and .aicar/ends/

Reads ends.csv (saved by every drive log and recording that had one), or the [measure] lines in out.log when there is none.
Columns: target cm, cm at the end (lane end: from the car front, as on screen), error (+ = ended early, still that far
short; - = late), speed and motor command then, how far it slid before stopping, and what it measured to (ref).
"""
import builtins
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "pi" / "lane_follow.py"


def constant(name, default):
    """A number set at the top of lane_follow.py, read from its text (no camera or OpenCV needed)."""
    try:
        m = re.search(rf"^{name}\s*=\s*(-?[0-9.]+)", CODE.read_text(), re.M)
        return float(m.group(1)) if m else default
    except OSError:
        return default


END_BASE_CM = constant("END_BASE_CM", 38.0)
LANEEND_MIN_CM = constant("LANEEND_MIN_CM", 4.5)
LANEEND_MAX_CM = constant("LANEEND_MAX_CM", 28.0)


def num(v):
    try:
        return float(v) if v not in (None, "") else None
    except ValueError:
        return None


def rows_from_csv(path):
    with open(path, newline="") as f:
        return [{k: (v if k in ("how", "ref") else num(v)) for k, v in r.items()} for r in csv.DictReader(f)]


MEASURE = re.compile(r"\[measure (\d+)\] end: line at (\S+)cm \((\w+).*?target (\S+)\) speed (\S+)cm/s command (\S+)")
SLID = re.compile(r"\[measure (\d+)\] stopped: line at (\S+)cm, slid (\S+)cm")


def rows_from_log(path):
    """The [measure] lines a drive printed: fewer columns than ends.csv (no ref), for logs made before it was saved."""
    rows, by_step = [], {}
    text = Path(path).read_text(errors="replace").replace("\r", "\n")
    for line in text.splitlines():
        m = MEASURE.search(line)
        if m:
            row = {"step": num(m.group(1)), "end_z": num(m.group(2)), "how": m.group(3), "target_cm": num(m.group(4)),
                   "speed": num(m.group(5)), "cmd": num(m.group(6)), "ref": None, "ref_z": None, "slid_cm": None}
            rows.append(row)
            by_step[m.group(1)] = row
            continue
        m = SLID.search(line)
        if m and m.group(1) in by_step:
            by_step[m.group(1)]["slid_cm"] = num(m.group(3))
    return rows


def labels(folder):
    """Step number -> a short description, from the course.py kept with that drive. Empty if it cannot be read."""
    path = Path(folder) / "course.py"
    if not path.exists():
        return {}
    scope = {"__name__": "course_report", "__file__": str(path), "__builtins__": builtins, "print": lambda *a, **k: None}
    try:
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), scope)
        describe = scope.get("describe") or (lambda st: str(st.get("do")))
        return {i + 1: describe(st) for i, st in enumerate(scope.get("COURSE") or [])}
    except Exception:
        return {}


def at_end(row):
    """(target, cm at the end, error) in the same units. Lane end: distance from the car front. Line: cm past END_BASE_CM."""
    target = row.get("target_cm")
    if target is None:
        return None, None, None
    if row.get("ref") == "laneend" and row.get("ref_z") is not None:
        goal = min(max(target, LANEEND_MIN_CM), LANEEND_MAX_CM)
        return goal, row["ref_z"], row["ref_z"] - goal
    if row.get("end_z") is None:
        return target, None, None
    got = END_BASE_CM - row["end_z"]
    return target, got, target - got


def report(folders, out=print):
    fmt = lambda v, f="5.1f": "    -" if v is None else format(v, f)
    out(f"{'drive':16s} {'step':>4s} {'target':>6s} {'at end':>6s} {'error':>6s} {'speed':>6s} {'cmd':>5s} {'slid':>5s} {'ref':8s} {'how':6s} action")
    errors_by_step = {}
    shown = 0
    for folder in folders:
        folder = Path(folder)
        if (folder / "ends.csv").exists():
            rows = rows_from_csv(folder / "ends.csv")
        elif (folder / "out.log").exists():
            rows = rows_from_log(folder / "out.log")
        else:
            continue
        names = labels(folder)
        for row in rows:
            target, got, err = at_end(row)
            if target is None:
                continue
            step = int(row["step"]) if row.get("step") is not None else 0
            out(f"{folder.name[:16]:16s} {step:4d} {fmt(target, '6.1f')} {fmt(got, '6.1f')} {fmt(err, '+6.1f')} {fmt(row.get('speed'), '6.1f')}"
                f" {fmt(row.get('cmd'), '5.2f')} {fmt(row.get('slid_cm'), '5.1f')} {(row.get('ref') or '-'):8s} {(row.get('how') or '-'):6s} {names.get(step, '')}")
            shown += 1
            if err is not None:
                errors_by_step.setdefault(step, []).append(err)
    if not shown:
        out("No cm= ends found (ends.csv or [measure] lines). Drive a course with go(cm=...) / to_junction(cm=...) first.")
        return {}
    out("")
    out("By step (error: + ended early, - ended late; cm)")
    for step in sorted(errors_by_step):
        e = errors_by_step[step]
        mean = sum(e) / len(e)
        spread = (sum((x - mean) ** 2 for x in e) / len(e)) ** 0.5
        out(f"  step {step:3d}: {len(e)} ends, mean {mean:+.1f}, spread {spread:.1f}, from {min(e):+.1f} to {max(e):+.1f}")
    return errors_by_step


def main(argv):
    cache = ROOT / ".aicar" / "ends"     # What aicar end-report downloads.
    folders = [Path(a) for a in argv] or sorted([*(ROOT / "logs").glob("2*"), *(ROOT / "runs").glob("2*"),
                                                 *(cache / "logs").glob("2*"), *(cache / "runs").glob("2*")], key=lambda p: p.name)
    report([f for f in folders if f.is_dir()])


if __name__ == "__main__":
    main(sys.argv[1:])
