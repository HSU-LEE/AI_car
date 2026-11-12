"""Re-run a recorded drive (runs/<time>/) and write an annotated video. Works on a Mac.

  python replay.py runs/20261029-190000
  python replay.py runs/20261029-190000 --set KP=0.7 KH=1.0
  python replay.py runs/20261029-190000 --frames 120-300

Writes <run>/replay.mp4 and prints a short summary.
"""
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

import lane_follow as lf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--set", nargs="*", default=[], help="override lane_follow settings with KEY=VALUE")
    ap.add_argument("--frames", help="for example 100-250")
    ap.add_argument("--out", help="output video path (default: <run>/replay.mp4)")
    args = ap.parse_args()

    run = Path(args.run)
    # The calibration and course of that run, saved with it. --set below still overrides. A Mac has no vision.json of its own.
    lf.load_landmarks(run)      # The traffic light and crosswalk registered that day (none if the run has none).
    if (run / lf.VISION_FILE.name).exists():
        lf.load_vision(run / lf.VISION_FILE.name)
    else:
        print(f"[warning] This recording has no {lf.VISION_FILE.name}: analysed with the current settings (line color {lf.LINE_COLOR}). If wrong, set --set LINE_COLOR=black etc.")
    course_list, module = lf.COURSE, None
    if (run / "course.py").exists():
        spec = importlib.util.spec_from_file_location("run_course", run / "course.py")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            course_list = module.COURSE
            lf._course = module     # Switches read while driving (SLOWDOWN_ON ...) come from that day's course too.
        except Exception as e:      # A course from an older version of the code may not load any more.
            print(f"[warning] The recording's course.py does not load with this code ({type(e).__name__}: {e}): the current course is used")
            module = None
    else:
        print("[warning] This recording has no course.py: the current course is used")
    if module is not None:
        for name in dir(module):        # The figure's settings are read from the course file too.
            if name.startswith("PERSON_") or name == "RED_BACK_ON":
                setattr(lf, name, getattr(module, name))
    saved = json.loads((run / "run.json").read_text()) if (run / "run.json").exists() else {}
    if saved.get("lane_follow_md5") not in (None, hashlib.md5(Path(lf.__file__).read_bytes()).hexdigest()):
        print("[warning] lane_follow.py differs from the one recorded with, so decisions may differ")
    args.set = list(saved.get("set", [])) + args.set     # Settings of that run first. Yours win.

    for kv in args.set:
        key, val = kv.split("=", 1)
        old = getattr(lf, key)
        try:
            new = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            new = val       # Leave values such as "light" as strings.
        setattr(lf, key, new)
        print(f"{key}: {old} -> {new}")

    if (run / "log.csv").exists():
        rows = list(csv.DictReader(open(run / "log.csv")))
    else:   # Frames only: assume 20 fps.
        print("No log.csv -> replaying frames only (20 fps assumed)")
        rows = [{"frame": p.stem, "t": int(p.stem) / 20, "steer": "nan"}
                for p in sorted(run.glob("[0-9]*.jpg"))]
    if args.frames:
        a, b = map(int, args.frames.split("-"))
        rows = rows[a:b + 1]

    # Same decisions as a real run, including course options and the LEGO figure saved with the recording.
    driver = lf.Driver(course_list, base_speed=saved.get("base_speed") or lf.BASE_SPEED, person=lf.load_person(run / lf.PERSON_FILE.name))
    detect, steering, traffic, course = driver.detect, driver.steering, driver.traffic, driver.course
    out = Path(args.out) if args.out else run / "replay.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 20,
                             (lf.FRAME_W + lf.Driver.PANEL_W, lf.FRAME_H))     # Same right-hand info panel as a live run.

    lost = junctions = gaps = 0
    lights = {"red": 0, "green": 0}
    steers, diffs = [], []
    for row in rows:
        frame = cv2.imread(str(run / f"{int(row['frame']):05d}.jpg"))
        if frame is None:
            continue
        now = float(row["t"])
        before = course.idx
        if row.get("status") != "WAIT ENTER":       # next(): Enter goes on where the recording stopped waiting.
            course.press_enter()
        _, _, dbg = driver.tick(frame, now, show=True)
        light, det = driver.light, driver.det
        junctions += course.idx != before
        gaps += detect.gap_y is not None
        if light:
            lights[light] += 1
        lost += det is None
        steers.append(steering.angle)
        diffs.append(steering.angle - float(row["steer"]))

        view = driver.view(dbg, now)
        cv2.putText(view, f"#{row['frame']}  recorded steer {float(row['steer']):+.1f}",
                    (10, lf.FRAME_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        writer.write(view)
    writer.release()

    s = np.array(steers)
    print(f"\nframes {len(rows)}  LOST {lost} ({100 * lost / max(len(rows), 1):.0f}%)  junctions {junctions}")
    print(f"frames with a lane gap {gaps}  traffic light red {lights['red']} / green {lights['green']} frames")
    if driver.stop_log:     # go_to() matches, the same columns as the stops.csv a drive saves.
        driver.save_stop_log(run / "replay_stops.csv")
        print(f"stop point matches: {run / 'replay_stops.csv'}")
    print(f"steering mean {s.mean():+.1f}  std {s.std():.1f}  "
          f"mean change per frame {np.abs(np.diff(s)).mean() if len(s) > 1 else 0:.2f} deg (larger is more zigzag)")
    if not np.isnan(diffs).all():
        print(f"mean difference from the recorded steering {np.nanmean(np.abs(diffs)):.1f} deg")
    print(f"Video: {out}")


if __name__ == "__main__":
    main()
