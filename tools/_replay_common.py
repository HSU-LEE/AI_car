"""Shared by the recording tools: load one recording the way pi/replay.py does and step a Driver through its frames.

Every recording changes module-level settings (vision.json, the course, run.json --set), so each one must run in a fresh
process: the tools use a multiprocessing pool with the "spawn" start method and one task per process.
"""
import ast
import contextlib
import csv
import importlib
import importlib.util
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = Path(os.environ.get("AICAR_RUNS", ROOT / "runs"))     # AICAR_RUNS=/some/folder: recordings kept elsewhere.


def all_runs():
    return sorted(p for p in RUNS.iterdir() if p.is_dir() and (p / "log.csv").exists())


def apply_sets(sets):
    """MODULE.NAME=VALUE overrides (for experiments), for example lane_track.TRACK_FAR_WEIGHT=0.2."""
    for kv in sets or ():
        key, val = kv.split("=", 1)
        mod, name = key.rsplit(".", 1)
        setattr(importlib.import_module(mod), name, ast.literal_eval(val))


def setup(run, code, lane_model, sets=None):
    """Import lane_follow from the code folder and set it up for this recording, as replay.py does. Returns (lf, rows, driver)."""
    sys.path.insert(0, str(Path(code).resolve()))
    with contextlib.redirect_stdout(io.StringIO()):
        lf = importlib.import_module("lane_follow")
        apply_sets(sets)
        run = Path(run)
        lf.load_landmarks(run)
        if (run / lf.VISION_FILE.name).exists():
            lf.load_vision(run / lf.VISION_FILE.name)
        course_list, module = lf.COURSE, None
        if (run / "course.py").exists():
            spec = importlib.util.spec_from_file_location("run_course", run / "course.py")
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
                course_list = module.COURSE
                lf._course = module
            except Exception:
                module = None
        if module is not None:
            for name in dir(module):
                if name.startswith("PERSON_") or name in ("RED_BACK_ON", "JUNCTION_TRACK_ASSIST", "LANE_POINTS"):
                    setattr(lf, name, getattr(module, name))
            lf.JUNCTION_TRACK_ASSIST = getattr(module, "JUNCTION_TRACK_ASSIST", False)    # As replay.py: off before it existed.
            lf.LANE_POINTS = getattr(module, "LANE_POINTS", "bands")     # Likewise: the band points before "pairs" existed.
            apply_sets(sets)        # ... but an override given for an experiment still wins.
        saved = json.loads((run / "run.json").read_text()) if (run / "run.json").exists() else {}
        for kv in saved.get("set", []):
            key, val = kv.split("=", 1)
            try:
                new = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                new = val
            setattr(lf, key, new)
        lf.LANE_MODEL = lane_model
        rows = list(csv.DictReader(open(run / "log.csv")))
        driver = lf.Driver(course_list, base_speed=saved.get("base_speed") or lf.BASE_SPEED,
                           person=lf.load_person(run / lf.PERSON_FILE.name))
    return lf, rows, driver


def frames(run, rows, driver):
    """(row, frame, now) for each recorded frame, with the course stepped as replay.py does before each tick."""
    import cv2
    for row in rows:
        frame = cv2.imread(str(Path(run) / f"{int(row['frame']):05d}.jpg"))
        if frame is None:
            continue
        if row.get("status") != "WAIT ENTER":
            driver.course.press_enter()
        yield row, frame, float(row["t"])


def far_bend(lf, tr):
    """How far the tracker's center line bends away from a straight line at the top of its drawn curve, in lane widths there.

    A straight lane on a flat floor is a straight line on screen. The drawn curve (curve_points) runs up to the farthest
    row with line pixels; this is its gap there from the straight line through its own points at the bottom row and
    halfway up. None without a tracked lane."""
    if tr is None or tr.p is None:
        return None
    pts = tr.curve_points(lf, 0)
    if pts is None or len(pts) < 3:
        return None
    H = lf.HORIZON_Y
    y_top, y_bot = float(pts[:, 1].min()), float(lf.FRAME_H - 1)
    y_mid = (y_top + y_bot) / 2
    _, _, (x_top, x_mid, x_bot) = tr.lines_at(tr.p, np.array([y_top, y_mid, y_bot]) - H)
    x_line = x_bot + (x_mid - x_bot) * 2        # The straight line through bottom and middle, at the top.
    return float(abs(x_top - x_line) / max(tr.p[2] * (y_top - H), 4.0))
