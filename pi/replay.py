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
    ap.add_argument("--set", nargs="*", default=[], help="KEY=VALUE 로 lane_follow 설정 덮어쓰기")
    ap.add_argument("--frames", help="예: 100-250")
    ap.add_argument("--out", help="출력 영상 경로 (기본: <run>/replay.mp4)")
    args = ap.parse_args()

    run = Path(args.run)
    # The calibration and course of that run, saved with it. --set below still overrides. A Mac has no vision.json of its own.
    if (run / lf.VISION_FILE.name).exists():
        lf.load_vision(run / lf.VISION_FILE.name)
    else:
        print(f"[주의] 이 녹화에 {lf.VISION_FILE.name}이 없어 현재 설정으로 분석합니다 (선 색 {lf.LINE_COLOR}). 다르면 --set LINE_COLOR=black 등으로 지정")
    course_list = lf.COURSE
    if (run / "course.py").exists():
        spec = importlib.util.spec_from_file_location("run_course", run / "course.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        course_list = module.COURSE
    else:
        print("[주의] 이 녹화에 course.py가 없어 현재 코스로 판단합니다")
    if (run / "course.py").exists():
        for name in dir(module):        # The figure's settings are read from the course file too.
            if name.startswith("PERSON_") or name == "RED_BACK_ON":
                setattr(lf, name, getattr(module, name))
    saved = json.loads((run / "run.json").read_text()) if (run / "run.json").exists() else {}
    if saved.get("lane_follow_md5") not in (None, hashlib.md5(Path(lf.__file__).read_bytes()).hexdigest()):
        print("[주의] 녹화 때와 lane_follow.py가 달라 판단이 다를 수 있습니다")
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
        print("log.csv 없음 -> 프레임만으로 재생 (20fps 가정)")
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
    print(f"\n프레임 {len(rows)}  LOST {lost} ({100 * lost / max(len(rows), 1):.0f}%)  교차로 {junctions}회")
    print(f"차선 끊김(GAP) 보인 프레임 {gaps}  신호등 빨강 {lights['red']} / 초록 {lights['green']} 프레임")
    print(f"조향 평균 {s.mean():+.1f}  표준편차 {s.std():.1f}  "
          f"프레임간 변화량 평균 {np.abs(np.diff(s)).mean() if len(s) > 1 else 0:.2f}도 (클수록 지그재그)")
    if not np.isnan(diffs).all():
        print(f"녹화 당시 조향과 차이 평균 {np.nanmean(np.abs(diffs)):.1f}도")
    print(f"영상: {out}")


if __name__ == "__main__":
    main()
