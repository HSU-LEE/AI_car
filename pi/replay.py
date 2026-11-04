"""Re-run a recorded drive (runs/<time>/) and write an annotated video. Works on a Mac.

  python replay.py runs/20261029-190000
  python replay.py runs/20261029-190000 --set KP=0.7 KH=1.0
  python replay.py runs/20261029-190000 --frames 120-300

Writes <run>/replay.mp4 and prints a short summary.
"""
import argparse
import ast
import csv
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

    for kv in args.set:
        key, val = kv.split("=", 1)
        old = getattr(lf, key)
        try:
            new = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            new = val       # Leave values such as "light" as strings.
        setattr(lf, key, new)
        print(f"{key}: {old} -> {new}")

    run = Path(args.run)
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
    driver = lf.Driver(lf.COURSE, person=lf.PersonDetector.load(lf.HORIZON_Y, run / lf.PERSON_FILE.name) if lf.PERSON_ON else None)
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
