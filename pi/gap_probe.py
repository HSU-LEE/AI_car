"""Log why a lane gap is or is not detected. Motors stay off. Push the car slowly toward a gap.

  python gap_probe.py            Record for 20 seconds.
  python gap_probe.py 30         Record for 30 seconds.

Prints a summary and writes gap_probe.csv. `aicar gap-probe` copies it to the Mac.
"""
import csv
import sys
import time
from collections import Counter

import lane_follow as lf


def main():
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    cam = lf.open_camera()
    det = lf.LaneDetector()
    rows, t0 = [], time.monotonic()
    print(f"{secs:.0f}초 동안 기록합니다. 차를 끊긴 구간 쪽으로 천천히 밀어 주세요.", flush=True)
    try:
        while (now := time.monotonic()) - t0 < secs:
            d, _, _ = det(cam.capture_array(), draw=False)
            pos = None if det.gap_y is None else round(det.gap_y / lf.FRAME_H, 2)
            rows.append((round(now - t0, 2), pos, det.gap_why, d is not None))
    finally:
        cam.stop()
        cam.close()
    with open(lf.VISION_FILE.with_name("gap_probe.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "gap_at", "why_not", "lane_seen"])
        w.writerows(rows)
    gaps = Counter(r[1] for r in rows if r[1] is not None)
    whys = Counter(r[2] for r in rows if r[1] is None)
    print(f"프레임 {len(rows)}개 중 끊김 감지 {sum(gaps.values())}개")
    print("  감지된 위치(화면 비율 → at= 값과 비교):", dict(sorted(gaps.items())))
    print("  감지 안 된 이유:", dict(whys.most_common(6)))
    # How the detected position moved, sampled across the recording.
    print("  시간별:", "  ".join(
        f"{t:.1f}s:{p if p is not None else '-'}" for t, p, _, _ in rows[::max(1, len(rows) // 40)]))


if __name__ == "__main__":
    main()
