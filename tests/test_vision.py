"""Check detection and course logic with synthetic images. Runs on a Mac or a Pi, with no recording and no camera.

  aicar test-vision                       on a Mac (uses .venv)
  python tests/test_vision.py             run it directly (locates the pi/ folder and imports it)

Images are drawn from the current settings (LANE_WIDTH_PX, HORIZON_Y, and the rest),
so the test still applies after the field of view or the track changes. Any failure exits with code 1.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pi"))
import lane_follow as lf  # noqa: E402
from course import COURSE, finish, follow, stop, straight, wait_green  # noqa: E402
from person import AutoViews, PersonDetector  # noqa: E402
import tune_person  # noqa: E402

W, H = lf.FRAME_W, lf.FRAME_H
RED, GREEN = (0, 0, 255), (0, 255, 0)
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  {'OK  ' if ok else 'FAIL'} {name}" + (f"  ({detail})" if detail else ""))


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


# --- Synthetic Images ---
def lane_x(y, shift=0.0):
    w = lf.LANE_WIDTH_PX * (y - lf.HORIZON_Y) / (342 - lf.HORIZON_Y)
    cx = W * lf.CAMERA_X + shift
    return cx - w / 2, cx + w / 2


def lane_frame(y_end=0, left=True, right=True, shift=0.0, bg=30, line=235, crosswalk=False):
    """Lane (line) on a floor (bg). Above y_end, the far side, there is no line, which is a gap."""
    rng = np.random.default_rng(0)
    img = np.clip(rng.normal(bg, 4, (H, W, 3)), 0, 255).astype(np.uint8)
    for y in range(max(y_end, int(H * lf.ROI_TOP) - 10), H):
        xl, xr = lane_x(y, shift)
        if left:
            cv2.line(img, (int(xl) - 5, y), (int(xl) + 5, y), (line,) * 3, 1)
        if right:
            cv2.line(img, (int(xr) - 5, y), (int(xr) + 5, y), (line,) * 3, 1)
    if crosswalk:
        y0 = int(H * 0.68)
        for k in range(8):
            x = int(W * 0.25) + k * int(W * 0.065)
            cv2.rectangle(img, (x, y0), (x + int(W * 0.033), y0 + int(H * 0.1)), (line,) * 3, -1)
    return img


def run_detector(img, n=5, color="white"):
    old = lf.LINE_COLOR
    lf.LINE_COLOR = color
    try:
        d = lf.LaneDetector()
        for _ in range(n):
            det, junction, _ = d(img)
        return d, det
    finally:
        lf.LINE_COLOR = old


def lamp_img(*lamps):
    img = np.full((H, W, 3), 25, np.uint8)
    for (x, y), bgr in lamps:
        cv2.circle(img, (x, y), 8, bgr, -1)
    return img


def feed(course, frames, t, det=(0.0, 0.0), light=None, gap=None, lamps=None):
    for _ in range(frames):
        course.update(False, t, det, False, {"gap": gap}, light, lamps)
        t += 0.05
    return t


def feed_img(course, img, frames, t):
    tl = lf.TrafficLight()
    for _ in range(frames):
        light = tl(img)
        course.update(False, t, (0.0, 0.0), False, {}, light, tl.lamps)
        t += 0.05
    return t


# --- Tests ---
def test_fast_math():
    print("[빠른 계산 = 원래 계산]")
    rng = np.random.default_rng(1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (lf.TOPHAT_K, lf.TOPHAT_K))
    same_t = same_p = 0
    for _ in range(20):
        img = rng.integers(0, 256, (H // 2, W // 2), dtype=np.uint8)
        img = cv2.GaussianBlur(img, (7, 7), 0)
        same_t += np.array_equal(cv2.morphologyEx(img, cv2.MORPH_TOPHAT, k), lf.tophat_ellipse(img, lf.TOPHAT_K))
        same_p += float(np.percentile(img, 99)) == lf.percentile_u8(img, 99)
    check("탑햇(직사각형 분해) = morphologyEx", same_t == 20, f"{same_t}/20")
    check("percentile_u8 = np.percentile", same_p == 20, f"{same_p}/20")


def test_lane():
    print("[차선]")
    _, det = run_detector(lane_frame())
    check("가운데 직선 → 위치 오차 거의 0", det is not None and abs(det[0]) < 0.05, f"{det}")
    _, det_r = run_detector(lane_frame(shift=40))
    check("차로가 오른쪽에 보이면 위치 오차 +", det_r is not None and det_r[0] > 0.05, f"{det_r}")
    _, det_1 = run_detector(lane_frame(right=False))
    check("왼쪽 선만 보여도 중앙 추정", det_1 is not None and abs(det_1[0]) < 0.15, f"{det_1}")


def test_gap():
    print("[차선 끊김 gap]")
    d, _ = run_detector(lane_frame(y_end=0))
    check("이어진 차선 → gap 없음", d.gap_y is None, f"gap_y={d.gap_y}")
    d1, _ = run_detector(lane_frame(y_end=250))
    check("멀리서 끊긴 차선 → gap", d1.gap_y is not None, f"gap_y={d1.gap_y}")
    d2, _ = run_detector(lane_frame(y_end=300))
    check("가까이서 끊기면 gap_y가 더 아래", d2.gap_y is not None and d2.gap_y > (d1.gap_y or 0),
          f"{d1.gap_y} → {d2.gap_y}")
    d3, _ = run_detector(lane_frame(y_end=250, right=False))
    check("한쪽 선만(급커브) → gap 아님", d3.gap_y is None, f"gap_y={d3.gap_y}")
    band_h = (H - int(H * lf.ROI_TOP)) // lf.SLICES
    def blank(img, y0, y1, x0=0, x1=W):
        img[y0:y1, x0:x1] = np.clip(np.random.default_rng(3).normal(30, 4, (y1 - y0, x1 - x0, 3)), 0, 255)
        return img
    y4 = H - 5 * band_h                          # Clear only band 4, the fifth from the bottom, so the line resumes in band 5 above it.
    d6, _ = run_detector(blank(lane_frame(), y4, y4 + band_h))
    check("짧은 끊김(빈 띠 1개, 위에서 선이 다시 보임) → gap", d6.gap_y is not None, f"gap_y={d6.gap_y}")
    img = lane_frame()
    blank(img, H - 4 * band_h, H - 2 * band_h, W // 2, W)     # The right line ends first.
    blank(img, H - 4 * band_h, H - 3 * band_h, 0, W // 2)     # The left line ends one band later.
    d7, _ = run_detector(img)
    check("양쪽 선이 끊기는 위치가 어긋나도 gap", d7.gap_y is not None, f"gap_y={d7.gap_y}")
    d8, _ = run_detector(blank(lane_frame(), int(H * lf.ROI_TOP), int(H * lf.ROI_TOP) + band_h))
    check("맨 위 띠 하나만 빔(먼 선이 흐림) → gap 아님", d8.gap_y is None, f"why={d8.gap_why}")
    img = lane_frame(y_end=250)
    img[int(H * 0.82):, W // 2:] = 30           # The near right line is cut off at the bottom of the frame and is not visible.
    d4, _ = run_detector(img)
    check("가까운 선 하나가 화면 밖으로 잘려도 끊기기 직전에 양쪽이 보이면 gap", d4.gap_y is not None,
          f"gap_y={d4.gap_y}")
    img = lane_frame(y_end=0, bg=200, line=20)  # Bright wood floor plus black tape. Only the bottom corners are dark from vignetting.
    yy, xx = np.mgrid[0:H, 0:W]
    shade = (1 - 0.35 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H) / H) ** 2)).clip(0.6, 1)
    img = (img * shade[..., None]).astype(np.uint8)
    d5, det5 = run_detector(img, color="black")
    check("바닥 하나 + 가장자리 어두움: 가까운 양쪽 선을 지우지 않음",
          d5.left[0] is not None and d5.right[0] is not None, f"left {d5.left[0]} right {d5.right[0]}")


def test_mat_mask():
    print("[매트 마스크]")
    img = lane_frame(y_end=0, bg=25, line=235)
    img[:, :int(W * 0.12)] = 170                # The left side is a wooden border, outside the mat.
    cv2.line(img, (int(W * 0.06), int(H * 0.45)), (int(W * 0.06), H), (250, 250, 250), 8)   # Something white on that border.
    d, det = run_detector(img)
    top = int(H * lf.ROI_TOP)
    mask = d.line_mask(img[top:])
    check("검은 매트 + 나무 테두리: 테두리 위 흰 것은 무시", int((mask[:, :int(W * 0.1)] > 0).sum()) == 0)
    check("검은 매트 + 나무 테두리: 차선은 그대로", det is not None and abs(det[0]) < 0.1, f"{det}")


def test_crosswalk():
    print("[횡단보도]")
    d, _ = run_detector(lane_frame(crosswalk=True))
    check("횡단보도 줄무늬 감지", d.crosswalk_y is not None, f"crosswalk_y={d.crosswalk_y}")
    d0, _ = run_detector(lane_frame())
    check("차선만 있으면 횡단보도 아님", d0.crosswalk_y is None)


def test_track_color():
    print("[트랙 색: 흰 바닥 위 검은 선]")
    for name, kw in (("직선", {}), ("치우침", {"shift": 40}), ("끊김", {"y_end": 250}), ("횡단보도", {"crosswalk": True})):
        da, a = run_detector(lane_frame(bg=30, line=235, **kw), color="white")
        db, b = run_detector(lane_frame(bg=225, line=20, **kw), color="black")
        same = (a is not None and b is not None and np.allclose(a, b, atol=0.02)
                and (da.gap_y, da.crosswalk_y) == (db.gap_y, db.crosswalk_y))
        check(f"{name}: 검은 바닥/흰 선과 같은 결과", same, f"{a} / {b}")


def test_traffic_light():
    print("[신호등]")
    tl = lf.TrafficLight()
    check("왼쪽 빨간 램프", tl(lamp_img(((50, 60), RED))) == "red", tl.state)
    check("오른쪽 초록 램프", tl(lamp_img(((W - 50, 60), GREEN))) == "green", tl.state)
    img = np.full((H, W, 3), 25, np.uint8)
    cv2.circle(img, (50, 60), 9, (150, 230, 150), -1)
    cv2.circle(img, (50, 60), 5, (255, 255, 255), -1)
    check("가운데가 하얀 LED + 옅은 초록 테두리", tl(img) == "green", tl.state)
    img = np.full((H, W, 3), 25, np.uint8)
    cv2.rectangle(img, (10, 40), (220, 55), RED, -1)
    check("길쭉한 빨간 물체 → 무시", tl(img) is None, tl.state)
    check("화면 가운데 램프 → 범위 밖", tl(lamp_img(((W // 2, 60), RED))) is None, tl.state)
    check("차선 영역(아래쪽) 초록 → 무시", tl(lamp_img(((50, int(H * 0.85)), GREEN))) is None, tl.state)


def test_course_logic():
    print("[코스 로직]")
    with quiet():
        c = lf.CourseRunner([straight(until="lane", max_sec=30), finish()])
        t = feed(c, 40, 0.0)
    check("straight(until=lane): 처음부터 보이던 차선으론 안 끝남", c.idx == 0)
    with quiet():
        t = feed(c, 5, t, det=None)
        t = feed(c, lf.TURN_CLEAR_FRAMES + 1, t)
    check("straight(until=lane): 사라졌다 다시 잡히면 끝", c.idx == 1)

    with quiet():
        c = lf.CourseRunner([straight(until="lane", max_sec=30), finish()])
        for near, n in ((True, 5), (False, 20)):     # Over a gap the lane beyond stays visible higher up, but there is no line just ahead.
            for _ in range(n):
                c.update(False, t, (0.0, 0.0), False, {"near": near}); t += 0.05
    check("끊긴 구간 위(바로 앞 선 없음)에서는 직진 계속", c.idx == 0)
    with quiet():
        for _ in range(lf.TURN_CLEAR_FRAMES + 1):
            c.update(False, t, (0.0, 0.0), False, {"near": True}); t += 0.05
    check("건너서 바로 앞 선이 다시 보이면 직진 끝", c.idx == 1)

    with quiet():
        c = lf.CourseRunner([wait_green(max_sec=30), finish()])
        t = feed(c, 20, 0.0, light="red")
    check("wait_green: 빨간불이면 정지 유지", c.idx == 0 and c.stopped(t))
    with quiet():
        t = feed(c, 3, t, light="green")
    check("wait_green: 빨강 → 초록 3프레임이면 출발", c.idx == 1)
    with quiet():
        c = lf.CourseRunner([wait_green(max_sec=30), finish()])
        t = feed(c, 40, 0.0, light="green")
    check("wait_green: 처음부터 초록이면 출발 안 함", c.idx == 0)
    with quiet():
        c = lf.CourseRunner([wait_green(max_sec=1), finish()])
        feed(c, 30, 0.0)
    check("wait_green: 인식 실패해도 max_sec 뒤 출발", c.idx == 1)

    with quiet():
        c = lf.CourseRunner([wait_green(max_sec=30), finish()])
        t = feed_img(c, lamp_img(((50, 50), RED)), 5, 0.0)
        t = feed_img(c, lamp_img(((W - 50, 50), GREEN)), 10, t)
    check("빨간불 다른 곳의 초록 물체 → 출발 안 함", c.idx == 0)
    with quiet():
        t = feed_img(c, lamp_img(((50, 80), GREEN)), 5, t)
    check("빨간 램프 바로 아래 초록 → 출발", c.idx == 1)

    with quiet():
        c = lf.CourseRunner([follow(until="gap", at=0.85, max_sec=30), finish()])
        t = feed(c, 10, 0.0, gap=int(H * 0.75))
    check("until=gap at=0.85: 0.75 지점에서는 아직", c.idx == 0)
    with quiet():
        feed(c, 3, t, gap=int(H * 0.85) + 1)
    check("until=gap at=0.85: 0.85 지점에 오면 끝", c.idx == 1)
    with quiet():
        c = lf.CourseRunner([follow(until="gap", max_sec=30), finish()])
        t = feed(c, 10, 0.0, gap=int(H * 0.35))
    check("until=gap at 없음: 멀리 보이는 끊김으로는 안 끝남", c.idx == 0)
    with quiet():
        feed(c, 3, t, gap=int(H * lf.JUNCTION_Y) + 1)
    check("until=gap at 없음: JUNCTION_Y 지점에 오면 끝", c.idx == 1)

    with quiet():
        c = lf.CourseRunner([straight(until="lane", angle="hold", max_sec=30), finish()])
        c.update(False, 0.0, None, False, {}, None)
        a1, a2 = c.target(None, 0.0, current=12.0), c.target(None, 0.1, current=3.0)
    check('straight(angle="hold"): 시작 각도 유지', a1 == 12.0 and a2 == 12.0, f"{a1}, {a2}")


def test_calibration():
    print("[자동 캘리브레이션 계산]")
    import calibrate as cal
    true_w, true_h, true_shift = 430.0, 70.0, 12.0      # A real track that differs from the current settings.

    def frame(bg, line):
        rng = np.random.default_rng(2)
        img = np.clip(rng.normal(bg, 4, (H, W, 3)), 0, 255).astype(np.uint8)
        for y in range(int(H * lf.ROI_TOP) - 10, H):
            w = true_w * (y - true_h) / (342 - true_h)
            cx = W / 2 + true_shift
            for x in (cx - w / 2, cx + w / 2):
                cv2.line(img, (int(x) - 5, y), (int(x) + 5, y), (line,) * 3, 1)
        return img

    color, _ = cal.detect_track_color([frame(30, 235)])
    check("어두운 바닥 → 흰 선 트랙", color == "white", color)
    color_b, _ = cal.detect_track_color([frame(225, 20)])
    check("밝은 바닥 → 검은 선 트랙", color_b == "black", color_b)
    geo = cal.fit_geometry(cal.measure_lanes([frame(30, 235)] * 6))
    ok = isinstance(geo, dict)
    check("차로 폭 측정", ok and abs(geo["lane_width_px"] - true_w) < 8, geo["lane_width_px"] if ok else geo)
    check("소실점 측정", ok and abs(geo["horizon_y"] - true_h) < 10, geo["horizon_y"] if ok else geo)
    check("카메라 중앙 측정", ok and abs(geo["camera_x"] * W - (W / 2 + true_shift)) < 3,
          geo["camera_x"] if ok else geo)
    old = lf.LINE_COLOR
    lf.LINE_COLOR = "black"
    try:
        geo_b = cal.fit_geometry(cal.measure_lanes([frame(225, 20)] * 6))
    finally:
        lf.LINE_COLOR = old
    check("흰 바닥/검은 선에서도 같은 측정", isinstance(geo_b, dict) and abs(geo_b["lane_width_px"] - true_w) < 8,
          geo_b["lane_width_px"] if isinstance(geo_b, dict) else geo_b)
    # Sharpness measured on the Pi at each lens position (far, middle, near). The near-weighted choice must be 4.5.
    table = {1.5: [349, 159, 92], 3.0: [506, 356, 216], 3.75: [495, 447, 319], 4.5: [408, 462, 418],
             5.25: [305, 388, 467], 6.0: [217, 292, 445], 7.5: [111, 133, 277]}
    check("초점: 차선 영역(가까운 쪽 비중) 기준으로 고름", cal.best_focus(table) == 4.5, cal.best_focus(table))
    check("선이 안 보이면 측정 안 하고 이유를 알려 줌",
          isinstance(cal.fit_geometry(cal.measure_lanes([np.full((H, W, 3), 30, np.uint8)] * 4)), str))


def test_motor():
    print("[모터 최소값]")
    check("직진 최소값", abs(lf.motor_floor(0) - lf.MOTOR_MIN) < 1e-9, lf.motor_floor(0))
    check("최대로 꺾으면 꺾음 최소값", abs(lf.motor_floor(lf.STEER_LIMIT) - lf.MOTOR_MIN_TURN) < 1e-9,
          lf.motor_floor(lf.STEER_LIMIT))
    check("반쯤 꺾으면 그 사이", lf.MOTOR_MIN <= lf.motor_floor(lf.STEER_LIMIT / 2) <= lf.MOTOR_MIN_TURN)
    with quiet():
        d = lf.Driver([follow(until="junction", speed=0.1, max_sec=99), finish()])
        cmds = [d.tick(lane_frame(), i / 30)[1] for i in range(40)]
    check("아주 느린 명령(0.1)도 최소값 이상으로 보냄", min(c for c in cmds if c > 0) >= lf.MOTOR_MIN - 1e-9,
          f"최소 {min(c for c in cmds if c > 0):.2f}")


def test_turn_align_stop():
    print("[tr(): 회전 → 정렬 → 천천히 주행]")
    import course as cs
    with quiet():
        c = lf.CourseRunner([cs.tr(sec=1.0), finish()])
        t = feed(c, 22, 0.0, det=None)                  # Still no lane after the 1 second blind period, so counting starts after that.
        t = feed(c, lf.TURN_END_FRAMES, t, det=(0.3, 0.3))       # New lane: both position and heading match.
    check("새 차선이 잡히면 3초를 안 채우고 회전 끝", c.idx == 1 and t < 3.0, f"{t:.2f}초, 동작 {c.idx + 1}")
    with quiet():
        t = feed(c, 10, t, det=(0.3, 0.2))              # Still not aligned enough.
    check("덜 정렬되면 계속 차선 따라감", c.idx == 1 and not c.stopped(t))
    with quiet():
        t = feed(c, lf.ALIGN_FRAMES, t, det=(0.05, 0.03))       # Lane center, pointed straight.
    check("정렬되면 멈추지 않고 SETTLE_SPEED로 천천히 주행", c.idx == 2 and not c.stopped(t) and c.speed == cs.SETTLE_SPEED, c.status(t))
    with quiet():
        t = feed(c, 25, t, det=(0.05, 0.03))
    check("sec(1초) 뒤 다음 동작", c.idx == 3 or c.done, f"동작 {c.idx + 1}")
    with quiet():
        c = lf.CourseRunner([cs.tr(), finish()])
        t = feed(c, 190, 0.0, det=None)                 # The lane is never found (9.5 s).
        before = c.idx
        t = feed(c, 20, t, det=None)                    # 10.5 s.
    check("차선을 못 찾으면 10초 안전장치로 회전 끝", before == 0 and c.idx >= 1, f"{t:.1f}초, 동작 {c.idx + 1}")
    check("tl(sec=0): 천천히 주행 없이 회전 → 정렬만", [s["do"] for s in cs._flat(cs.tl(sec=0))] == ["left", "follow"])

    with quiet():   # A leftover lane crossing at an angle during a turn may look like a horizontal line, but a new lane still ends the turn.
        c = lf.CourseRunner([cs.tl(sec=0), finish()])
        t = 0.0
        for _ in range(22):
            c.update(False, t, None, False, {}); t += 0.05          # The lane stays lost until the 1 second blind period ends.
        for _ in range(lf.TURN_END_FRAMES):
            c.update(True, t, (0.1, -0.3), False, {"near": True}); t += 0.05   # It looks like a junction, but it is a new lane.
    check("회전: 가로선(교차로)으로 보여도 새 차선이 잡히면 끝", c.idx == 1, f"{t:.2f}초, 동작 {c.idx + 1}")
    with quiet():   # Pointed along the exit lane but off to one side (offset 0.55): end the turn, alignment fixes the offset.
        c = lf.CourseRunner([cs.tr(sec=0), finish()])
        t = feed(c, 6, 0.0, det=(0.05, 0.05))
        t = feed(c, 30, t, det=None)
        for _ in range(lf.TURN_END_FRAMES):
            c.update(False, t, (0.55, 0.1), False, {"near": True}); t += 0.05
    check("회전: 차로 방향은 맞고 옆으로 치우친 차선(0.55)에서도 끝", c.idx == 1, f"동작 {c.idx + 1}")
    with quiet():
        c = lf.CourseRunner([cs.tr(sec=0), finish()])
        t = feed(c, 6, 0.0, det=(0.05, 0.05))
        t = feed(c, 30, t, det=None)
        for _ in range(lf.TURN_END_FRAMES + 2):
            c.update(False, t, (0.55, 0.45), False, {"near": True}); t += 0.05
    check("회전: 치우쳤는데 방향도 틀어진 선(0.55, 0.45)으로는 안 끝남", c.idx == 0)
    with quiet():   # The lane already visible at the start of the turn, before the car turns, must not end the turn.
        c = lf.CourseRunner([cs.tr(sec=0), finish()])
        t = feed(c, 30, 0.0, det=(0.05, 0.05))
    check("회전: 돌기 전부터 보이던 차선으로는 안 끝남", c.idx == 0)
    with quiet():   # A gap that stayed visible during the previous action must not end the next action the moment it starts.
        c = lf.CourseRunner([stop(1.0), follow(until="gap", max_sec=30), finish()])
        t = feed(c, 5, 0.0)                           # No gap at first.
        t = feed(c, 16, t, gap=int(H * 0.8))          # A fake gap appears during the stop, and the stop ends at 1 second.
        started = c.idx
        t = feed(c, 2, t, gap=int(H * 0.8))           # Second frame of the new action.
    check("이전 동작의 감지로 즉시 끝나지 않음 (새 동작에서 다시 3프레임)", started == 1 and c.idx == 1,
          f"동작 {c.idx + 1}")
    with quiet():
        feed(c, 1, t, gap=int(H * 0.8))
    check("새 동작에서 3프레임 계속 보이면 끝", c.idx == 2)
    old = lf.GAP_MAX_HEADING
    lf.GAP_MAX_HEADING = -1.0       # Treat even a tiny heading error as misalignment, so the guard can be checked.
    try:
        d9, _ = run_detector(lane_frame(y_end=250))
    finally:
        lf.GAP_MAX_HEADING = old
    check("차가 차로와 크게 어긋나 있으면 끊김으로 안 봄", d9.gap_y is None and "aligned" in d9.gap_why, d9.gap_why)


def feed_junc(course, frames, t, on, det=(0.0, 0.0)):
    """A junction line at 0.6 of the frame height (on=True), or no line."""
    for _ in range(frames):
        course.update(on, t, det, False, {"junction": int(H * 0.6) if on else None, "near": True})
        t += 0.05
    return t


def test_blind_and_bundles():
    print("[blind_sec / 묶음 동작]")
    import course as cs
    with quiet():   # Quick turn: the old lane goes away and the new lane comes back, all inside the 1 second blind period.
        c = lf.CourseRunner([cs.tr(sec=0), finish()])
        t = feed(c, 6, 0.0, det=(0.05, 0.05))
        t = feed(c, 10, t, det=None)
        t = feed(c, 30, t, det=(0.05, 0.05))
    check("회전: blind 안에 새 차선이 잡혀도 blind 끝나면 회전 끝 (3초 안 채움)", c.idx >= 1 and c.course[c.idx]["until"] == "aligned"
          or c.idx >= 2, f"동작 {c.idx + 1}")
    with quiet():   # A near junction comes into view at 0.8 s and is still in view when the 1.5 second blind period ends.
        c = lf.CourseRunner([cs.to_junction(blind_sec=1.5), finish()])
        t = feed_junc(c, 16, 0.0, False)
        t = feed_junc(c, 20, t, True)
    check("to_junction: blind 중에 들어온 가까운 교차로에서 끝 (안 지나침)", c.idx == 1, f"{t:.2f}초, 동작 {c.idx + 1}")
    with quiet():   # A line that flickers off for only one or two frames during blind still must not end the action.
        c = lf.CourseRunner([cs.to_junction(blind_sec=1.5), finish()])
        t = feed_junc(c, 10, 0.0, True)
        t = feed_junc(c, 2, t, False)
        t = feed_junc(c, 30, t, True)
    check("to_junction: blind 중 1~2프레임 깜빡임으로는 안 끝남", c.idx == 0)

    with quiet():   # nth=2: the first line (a bend) is passed, the second ends the action.
        c = lf.CourseRunner([cs.to_junction(nth=2, blind_sec=0), finish()])
        t = feed_junc(c, 5, 0.0, False)
        t = feed_junc(c, 8, t, True)
        first = (c.idx, c.count, c.final_approach())
        t = feed_junc(c, 8, t, False)
        before = c.final_approach()
        t = feed_junc(c, 8, t, True)
    check("to_junction(nth=2): 첫 선은 지나치고 두 번째 선에서 끝", first[:2] == (0, 1) and c.idx == 1, f"{first} → 동작 {c.idx + 1}")
    check("nth=2: 첫 선을 지나는 동안은 감속/조향 제한 안 함, 그 뒤 다시 적용", not first[2] and before)

    s = cs.follow(3)
    check("follow(3): 시간 동작에는 until이 안 남음", s["until"] is None)
    try:
        cs.follow(3, slow=0.2)
        ok = False
    except ValueError:
        ok = True
    check("follow(3, slow=): sec과 같이 쓰면 에러", ok)
    steps = cs._flat(cs.corner("left", wide=2))
    check("corner: 교차로 → 직진 → 회전 → 정렬 → 천천히 주행",
          [x["do"] for x in steps] == ["follow", "straight", "left", "follow", "follow"]
          and steps[0]["shift"] == 2 and steps[2]["blind"] == round(cs.TURN_BLIND * cs.TURN_BLIND_SPEED / cs.TURN_SPEED, 2), str([x["do"] for x in steps]))
    steps = cs._flat(cs.uturn("right"))
    check("uturn: 최대 조향, blind 1.5초, 정렬까지",
          [x["do"] for x in steps] == ["follow", "right", "follow", "follow"]
          and steps[1]["angle"] is None and steps[1]["blind"] == 1.5 and steps[1]["max"] == 5)
    check("cross(): 교차로선(junction)으로 건넘", cs._flat(cs.cross())[0]["until"] == "junction")
    check('cross(line="crosswalk"): 횡단보도로 건넘', cs._flat(cs.cross(line="crosswalk"))[0]["until"] == "crosswalk")

    with quiet():
        c = lf.CourseRunner([cs.to_end(), finish()])
        t = feed(c, 5, 0.0)
        t = feed_junc(c, 5, t, True)
    check("to_end: 교차로선(가로선)에서도 끝", c.idx == 1, f"동작 {c.idx + 1}")
    with quiet():
        c = lf.CourseRunner([cs.to_end(), finish()])
        t = feed(c, 5, 0.0)
        t = feed(c, 5, t, gap=int(H * 0.3))
    check("to_end: 멀리 있는 끊김으로는 안 끝남", c.idx == 0)
    with quiet():
        t = feed(c, 5, t, gap=int(H * 0.6))
    check("to_end: at(0.5) 아래로 온 차선 끊김에서 끝", c.idx == 1, f"동작 {c.idx + 1}")

    fast, slow = cs._flat(cs.tr(speed=0.7))[0]["blind"], cs._flat(cs.tr(speed=0.3))[0]["blind"]
    check("회전 blind: 빠를수록 짧게, 기준 속도 이하면 그대로", fast < cs.TURN_BLIND and slow == cs.TURN_BLIND,
          f"0.7 → {fast}초, 0.3 → {slow}초")


def test_lost_and_far():
    print("[차선 놓침 후진 / 회전각 / 먼 교차로선]")
    import course as cs
    blank = np.full((H, W, 3), 30, np.uint8)
    with quiet():
        d = lf.Driver([follow(until="junction", max_sec=99), finish()])
        for i in range(10):
            d.tick(lane_frame(), i / 30)
        cmds = [(i / 30, d.tick(blank, i / 30)[1]) for i in range(10, 160)]
    back = [t for t, c in cmds if c < 0]
    check("차선을 놓치면 LOST_HOLD 뒤 후진", back and abs(back[0] - (10 / 30 + lf.LOST_HOLD)) < 0.1,
          f"후진 시작 {back[0] if back else None}")
    check("후진은 LOST_BACK_SEC까지만, 그 뒤 정지", back and back[-1] - back[0] <= lf.LOST_BACK_SEC + 0.05
          and cmds[-1][1] == 0, f"{back[-1] - back[0] if back else 0:.2f}초")
    with quiet():
        for i in range(160, 170):
            d.tick(lane_frame(), i / 30)
        second = [d.tick(blank, i / 30)[1] for i in range(170, 300)]
        for i in range(300, 310):
            d.tick(lane_frame(), i / 30)
        third = [d.tick(blank, i / 30)[1] for i in range(310, 440)]
    check(f"차선을 다시 놓치면 또 후진 (동작당 {lf.LOST_BACK_TRIES}번까지)", min(second) < 0 and min(third) >= 0,
          f"2번째 {min(second):+.2f}, 3번째 {min(third):+.2f}")

    with quiet():   # Retrace: drive forward steering +15, then lose the lane. Backing up replays +15.
        d = lf.Driver([follow(until="junction", max_sec=99), finish()])
        for i in range(30):
            d.tick(lane_frame(shift=60), i / 30)
        hist = [a for _, a, _ in d.history[-10:]]
        steers = [d.tick(blank, i / 30) for i in range(30, 120)]
    back = [st for st, cmd, _ in steers if cmd < 0 and st is not None]
    check("후진하면서 직전에 쓰던 조향을 거꾸로 재생", back and abs(back[0] - hist[-1]) < 3, f"앞 {hist[-1]:+.1f} / 후진 {back[0] if back else None}")

    ym = lf.YawMeter()
    room = np.random.default_rng(3).integers(0, 255, (H, W + 80, 3), np.uint8)
    room = cv2.GaussianBlur(room, (9, 9), 0)
    for k in range(9):      # The view slides 10 px left per frame, as when turning right.
        total = ym(np.ascontiguousarray(room[:, 10 * k:10 * k + W]))
    check("카메라 회전 측정: 화면이 밀린 만큼 (오른쪽 회전 = 음수)", abs(total + 80) < 6, f"{total:+.1f}px (정답 -80)")
    with quiet():
        c = lf.CourseRunner([cs.tr(sec=0), finish()])
        t = feed(c, 22, 0.0, det=None)
        c.turned = lf.TURN_YAW_PX + 1
        feed(c, 1, t, det=None)
    check("회전: 새 차선이 안 보여도 90도(TURN_YAW_PX) 돌면 끝", c.idx == 1, f"동작 {c.idx + 1}")
    def turn_end_time(min_sec):
        with quiet():
            c = lf.CourseRunner([cs.tr(sec=0, min_sec=min_sec), finish()])
            t = feed(c, 6, 0.0, det=(0.05, 0.05))
            t = feed(c, 20, t, det=None)
            c.turned = lf.TURN_YAW_PX + 1     # Already turned 90 degrees, new lane in view too.
            while c.idx == 0 and t < 5:
                c.update(False, t, (0.05, 0.05), False, {"near": True}); t += 0.05
        return t
    t_short, t_long = turn_end_time(0.5), turn_end_time(2.5)
    check("tr(min_sec=2.5): 90도가 되고 새 차선이 보여도 2.5초까지는 계속 돎", t_short < 2.0 and 2.5 <= t_long < 2.8,
          f"min_sec 0.5 → {t_short:.2f}초, 2.5 → {t_long:.2f}초")
    long_turn = cs._flat(cs.tr(4))[0]
    check("tr(4): 최소 4초 돌기, 최대 시간은 자동으로 늘어남", long_turn["min"] == 4 and long_turn["max"] >= 5,
          f"min {long_turn['min']}, max {long_turn['max']}")

    far = lane_frame()
    cv2.line(far, (W // 2 - 110, 175), (W // 2 + 110, 179), (235, 235, 235), 3)    # Thin, slightly slanted line ahead.
    with quiet():
        d = lf.Driver([finish()])
        d.tick(far, 0.0)
    y = d.detect.far_hline(far)
    check("ROI 안쪽의 가늘고 비스듬한 먼 가로선도 감속용으로 잡음", y is not None and 170 <= y <= 190, y)


def test_stripes_and_crossing():
    print("[횡단보도 줄무늬 / 건너기 / 정지 중 조향]")
    import course as cs
    plain = lane_frame()
    striped = lane_frame()
    for k, y in enumerate((230, 270, 310)):     # Three short stripes, staggered, between the lane lines (as on the real track).
        l, r = lane_x(y)
        w = int(0.12 * (r - l))
        x = int(l + (r - l) * (0.3 + 0.2 * k))
        cv2.rectangle(striped, (x, y - w // 4), (x + w, y + w // 4), (235, 235, 235), -1)
    d0, det0 = run_detector(plain)
    d1, det1 = run_detector(striped)
    check("짧은 줄무늬 3개 = 횡단보도", len(d1.stripes) >= lf.STRIPES_MIN and d1.crosswalk_any, f"{len(d1.stripes)}개")
    check("줄무늬는 차선에서 빠짐 (차선 결과가 줄무늬 없을 때와 같음)",
          det0 is not None and det1 is not None and abs(det0[0] - det1[0]) < 0.05 and abs(det0[1] - det1[1]) < 0.1,
          f"{det0} / {det1}")
    check("줄무늬 없는 차선에는 횡단보도 없음", not d0.stripes and not d0.crosswalk_any)
    def stripes_at(xs_frac, ys=(230, 270, 310)):
        img = lane_frame()
        for xf, y in zip(xs_frac, ys):
            l, r = lane_x(y)
            w = int(0.12 * (r - l))
            x = int(l + (r - l) * xf)
            cv2.rectangle(img, (x, y - w // 4), (x + w, y + w // 4), (235, 235, 235), -1)
        return run_detector(img)[0]
    two = stripes_at((0.3, 0.5))
    check(f"줄무늬 2개는 횡단보도 아님 ({lf.STRIPES_MIN}개 이상 필요)", not two.stripes and not two.crosswalk_any, len(two.stripes))
    out = stripes_at((-0.6, -0.4, 1.3))     # Three stripes, but outside the lane lines.
    check("차선 밖의 줄무늬 3개는 횡단보도 아님", not out.stripes, len(out.stripes))
    far_only = np.full((H, W, 3), 30, np.uint8)
    top = int(H * lf.ROI_TOP)
    cv2.line(far_only, (0, top + 8), (40, top + 14), (235, 235, 235), 6)    # Something at the far top-left edge, no lane.
    _, det_far = run_detector(far_only)
    check("멀리 끝에 점 하나만 보이면 차선 아님 (차선 놓침)", det_far is None, det_far)
    right_only = lane_frame(left=False)
    with_obj = right_only.copy()
    top = int(H * lf.ROI_TOP)
    cv2.rectangle(with_obj, (50, top + 25), (120, top + 60), (235, 235, 235), -1)   # A short object off the track, far left.
    d_o, _ = run_detector(with_obj, n=1)
    check("트랙 밖 짧은 물체(먼 쪽 가장자리)는 차선 후보에서 빠짐", all(x is None or x > W * 0.4 for x in d_o.left), d_o.left)
    _, det_r = run_detector(right_only, n=1)
    old_side = lf.SPECK_SIDE
    lf.SPECK_SIDE = 0.0         # Without that removal: no previous position (just after a turn), so the longer line must win.
    try:
        _, det_o = run_detector(with_obj, n=1)
    finally:
        lf.SPECK_SIDE = old_side
    check("폭이 안 맞고 이전 위치도 없으면 긴 선(진짜 차선)을 남김", det_r is not None and det_o is not None
          and abs(det_r[0] - det_o[0]) < 0.1, f"{det_r} / {det_o}")

    steps = cs._flat(cs.cross(line="crosswalk", approach=False))
    check('cross(line="crosswalk", approach=False): 다가가기 없이 바로 건넘', [x["until"] for x in steps] == ["crossed"])
    with quiet():
        c = lf.CourseRunner(steps + [finish()])
        t = 0.0
        for _ in range(20):     # Crosswalk under the car.
            c.update(False, t, (0.0, 0.0), True, {"crosswalk": 300}); t += 0.05
        mid = c.idx
        for _ in range(12):     # Gone from view.
            c.update(False, t, (0.0, 0.0), False, {}); t += 0.05
    check("건너기: 횡단보도가 보이는 동안 계속, 지나가면 끝", mid == 0 and c.idx == 1, f"{mid} → {c.idx}")

    with quiet():
        d = lf.Driver([stop(5), finish()])
        a = [d.tick(lane_frame(shift=sh), i / 30)[0] for i, sh in enumerate([0] * 5 + [80] * 30)]
    a = [x for x in a if x is not None]
    check("정지 중에는 차선이 옆으로 가도 조향 그대로", max(a) - min(a) < 0.5, f"{min(a):+.1f}..{max(a):+.1f}")

    def align_steer(img):
        with quiet():
            d = lf.Driver([follow(until="aligned", max_sec=99), finish()])
            for i in range(20):
                st, _, _ = d.tick(img, i / 30)
        return st
    tilted = np.full((H, W, 3), 30, np.uint8)
    for y0, y1 in ((H, 150),):
        cv2.line(tilted, (int(W * 0.1), y0), (int(W * 0.9), y1), (235, 235, 235), 8)   # One lane line, steep heading.
    with_line = tilted.copy()
    cv2.line(with_line, (0, 230), (W, 230), (235, 235, 235), 8)      # A horizontal (junction) line in view.
    a_free, a_line = align_steer(tilted), align_steer(with_line)
    check("정렬: 가로선이 보이면 큰 방향 오차를 따라 꺾지 않음 (가로선 없을 때보다 작게)",
          a_line is not None and a_free is not None and abs(a_line) <= lf.ALIGN_STEER_MAX + 0.5 and abs(a_free) > abs(a_line),
          f"가로선 없음 {a_free}, 있음 {a_line}")


def test_driver_smoke():
    print("[Driver 전체 한 바퀴]")
    frames = [lane_frame(), lane_frame(shift=30), lane_frame(y_end=250), lane_frame(crosswalk=True),
              lamp_img(((50, 50), RED))]
    ok = True
    with quiet():
        d = lf.Driver(COURSE)
        try:
            for i in range(60):
                steer, cmd, dbg = d.tick(frames[i % len(frames)], i / 30, show=i % 2 == 0)
                if i % 2 == 0:
                    view = d.view(dbg, i / 30, 30)
                    ok &= view.shape[0] == H
                ok &= cmd >= 0 and (steer is None or abs(steer) <= lf.AUTO_STEER_MAX + 1e-6)
        except Exception as e:  # noqa: BLE001
            ok = False
            print(e)
    check("COURSE로 60프레임 tick + 화면 그리기 (오류 없음, 값 범위 정상)", ok)
    import course as cs
    events = [e for e in cs.EVENTS if e]
    missing = [e for e in events if e not in lf.Driver.EVENT_EN or e not in lf.CourseRunner.EVENT_NAMES
               or e not in lf.CourseRunner([finish()]).hits]
    check("course.py의 모든 until 조건에 화면/로그 이름과 카운터가 있음", not missing, missing)
    try:
        with quiet():
            d = lf.Driver([stop(0.1), cs.cross(line="crosswalk", approach=False), finish()])
            for i in range(12):     # stop (next: crossing), then the crossing itself.
                _, _, dbg = d.tick(lane_frame(), i / 30, show=True)
                d.view(dbg, i / 30, 30)
        ok = True
    except Exception as e:  # noqa: BLE001
        ok = False
        print(repr(e))
    check("화면 그리기: 다음/현재 동작이 횡단보도 건너기여도 오류 없음", ok)
    try:
        with quiet():
            d = lf.Driver([stop(0.1), cs.right(), cs.uturn("left"), finish()])
            for i in range(12):     # Next action is a turn with the default (None) angle.
                _, _, dbg = d.tick(lane_frame(), i / 30, show=True)
                d.view(dbg, i / 30, 30)
        ok = True
    except Exception as e:  # noqa: BLE001
        ok = False
        print(repr(e))
    check("화면 그리기: 다음 동작이 기본 각도 회전(right(), uturn)이어도 오류 없음", ok)
    try:
        with quiet():
            import course as cs
            d = lf.Driver([cs.tr(), finish()], start=1)
            for i in range(5):
                d.tick(lane_frame(), i / 30, show=True)
        ok = True
    except Exception as e:  # noqa: BLE001
        ok = False
        print(e)
    check("회전 동작부터 시작(--start)해도 첫 프레임에서 오류 없음", ok)


def minifig(img, x, feet, torso=(40, 40, 200), legs=(150, 70, 30), head=(60, 210, 240)):
    """Draw a LEGO figure standing on the floor. Height follows the floor perspective, as for a real object.

    Head narrower than the shoulders, hands out to the sides, legs narrower than the arms: it fills about 60% of its box, as the real figure does.
    """
    h = int(0.42 * (feet - lf.HORIZON_Y))
    w, top = int(h * 0.62), feet - h
    out = img.copy()
    cv2.rectangle(out, (x - w // 5, top), (x + w // 5, top + int(h * 0.25)), head, -1)
    cv2.rectangle(out, (x - w // 3, top + int(h * 0.27)), (x + w // 3, top + int(h * 0.62)), torso, -1)
    cv2.rectangle(out, (x - w // 2, top + int(h * 0.42)), (x + w // 2, top + int(h * 0.54)), torso, -1)
    cv2.rectangle(out, (x - w // 3, top + int(h * 0.64)), (x - 2, feet), legs, -1)
    cv2.rectangle(out, (x + 2, top + int(h * 0.64)), (x + w // 3, feet), legs, -1)
    return out


def registered_figure(feet=240, back=None, **colors):
    back = lane_frame() if back is None else back
    det = PersonDetector(lf.HORIZON_Y)
    return det, det.add_view(minifig(back, 320, feet, **colors))


def test_person_detect():
    print("\n[레고 사람 인식]")
    det, (ok, msg) = registered_figure()
    check("등록: 배경과 사람 사진으로 등록됨", ok and len(det.views) == 1, msg)
    empty = lane_frame()
    check("등록: 사람이 없으면 등록 거부", not registered_figure()[0].add_view(empty)[0])
    check("인식: 사람 없는 차로에서 없음", det(empty) is None)
    found = det(minifig(empty, 320, 240))
    check("인식: 등록 위치에서 찾음", found is not None and abs(found["feet"] - 240) <= 4, found)
    check("인식: 여러 거리, 위치에서 찾음", all(det(minifig(empty, x, f)) for x in (220, 320, 420) for f in (190, 240, 300)))
    far = det(minifig(empty, 320, 240), (0.7, 0.9))
    check("인식: 정한 가로 범위 밖은 무시", far is None)
    boxes = empty.copy()
    cv2.rectangle(boxes, (300, 180), (340, 240), (40, 40, 200), -1)
    check("오탐: 같은 색 상자 하나는 사람 아님", det(boxes) is None)
    big = minifig(empty, 320, 240)
    cv2.rectangle(big, (100, 60), (520, 300), (150, 70, 30), -1)
    check("오탐: 크기가 맞지 않는 큰 물체는 사람 아님", det(big) is None)
    lamp = empty.copy()
    cv2.circle(lamp, (80, 60), 8, (0, 0, 255), -1)
    check("오탐: 횡단보도/빨간 신호등 화면에서 없음", det(lane_frame(crosswalk=True)) is None and det(lamp) is None)

    black = {"torso": (20, 20, 20), "legs": (20, 20, 20)}
    bright = lane_frame(line=20, bg=200)
    dark = registered_figure(back=bright, **black)[0]
    found = dark(minifig(bright, 320, 240, **black))
    check("검은 옷, 흰 바닥: 머리로 찾고 등록한 곳과 같은 기준", found is not None and abs(found["feet"] / lf.FRAME_H - dark.stop_at(lf.FRAME_H)) < 0.03, found)
    check("검은 옷, 흰 바닥: 검은 차선을 사람으로 안 봄", dark(bright) is None)
    dark = registered_figure(**black)[0]
    found = dark(minifig(empty, 320, 240, **black))
    check("검은 옷, 검은 바닥: 등록한 곳과 같은 기준으로 찾음", found is not None and abs(found["feet"] / lf.FRAME_H - dark.stop_at(lf.FRAME_H)) < 0.03, found)
    white, (_, msg) = registered_figure(back=lane_frame(line=20, bg=120), torso=(235, 235, 235), legs=(235, 235, 235), head=(235, 235, 235))
    check("거의 흰/검정 사람: 등록 때 경고", "주의" in msg, msg)

    print("  -- 배경 사진 없는 등록")
    tall = int(0.42 * (240 - lf.HORIZON_Y))     # Height minifig() draws at feet=240.
    behind = lane_frame()       # Run on the Pi: a same-colored patch behind the figure was joined to it, and the registered height was 108 px instead of 67.
    cv2.rectangle(behind, (305, 240 - tall - 35), (335, 240 - tall - 3), (150, 70, 30), -1)
    joined = registered_figure(back=behind)[0]
    check("등록: 뒤에 같은 색이 있어도 사람 크기만 잼", joined.views[0]["h"] <= tall + 6, (joined.views[0]["h"], tall))
    arm = empty.copy()      # Run on the Pi: with a background shot, an arm that moved was the largest change, and all 8 views registered the arm.
    cv2.rectangle(arm, (0, 150), (260, 230), (120, 160, 210), -1)
    armed = PersonDetector(lf.HORIZON_Y)
    added, msg = armed.add_view(minifig(arm, 400, 240))
    check("등록: 팔이 보여도 팔이 아니라 사람을 등록", added and armed.last_box[0] > 340, msg)
    check("등록: 등록 뒤 팔은 인식 안 하고 사람은 인식", armed(arm) is None and armed(minifig(empty, 400, 240)) is not None)
    check("등록: 사람 없이 팔만 있으면 거부", not PersonDetector(lf.HORIZON_Y).add_view(arm)[0])
    shifted = np.roll(minifig(empty, 320, 240), 6, axis=1)
    moved = PersonDetector(lf.HORIZON_Y)
    check("등록: 차가 살짝 밀려도 등록됨 (배경과 비교 안 함)", moved.add_view(shifted)[0])
    outside = PersonDetector(lf.HORIZON_Y)
    check("등록: 네모 밖의 사람은 거부", not outside.add_view(minifig(empty, 80, 240))[0])
    cut = PersonDetector(lf.HORIZON_Y)
    check("등록: 네모 테두리에 걸린 사람은 거부", not cut.add_view(minifig(empty, 196, 250))[0])
    flat = empty.copy()     # Run on the Pi: 4 views of a wide dull area in the room were accepted, and nothing could ever be found with them.
    cv2.rectangle(flat, (210, 170), (340, 200), (70, 80, 90), -1)
    wide_det = PersonDetector(lf.HORIZON_Y)
    check("등록: 가로로 넓은 물체는 사람으로 안 받음", not wide_det.add_view(flat)[0] and not wide_det.views)
    added, msg = wide_det.add_view(minifig(flat, 400, 300))
    check("등록: 넓은 물체와 사람이 같이 있으면 사람을 고름", added and wide_det.last_box[0] > 340, (msg, wide_det.last_box))
    painted = empty.copy()
    cv2.rectangle(painted, (0, 0), (639, 215), (40, 40, 200), -1)
    tone = PersonDetector(lf.HORIZON_Y)     # The figure's color fills the room too.
    solid_red = {"torso": (40, 40, 200), "legs": (40, 40, 200), "head": (40, 40, 200)}
    check("등록: 사람 색이 주변과 같으면 거부", not tone.add_view(minifig(painted, 320, 330, **solid_red))[0] and not tone.views)

    print("  -- 자동 촬영")
    side = {"torso": (40, 40, 200), "legs": (40, 40, 200), "head": (60, 210, 240)}
    auto_det = PersonDetector(lf.HORIZON_Y)
    auto = AutoViews(auto_det)

    def still(img, t0, sec=1.0):
        notes = [auto.update(img, t0 + i / 30) for i in range(int(sec * 30))]
        return t0 + sec, [n for n in notes if n]

    tt, notes = still(minifig(empty, 320, 240), 0.0)
    check("자동: 멈춰 있으면 한 장 찍음", len(auto_det.views) == 1 and len(notes) == 1, notes)
    tt, notes = still(minifig(empty, 320, 240), tt, 3.0)
    check("자동: 계속 멈춰 있으면 더 안 찍음", len(auto_det.views) == 1 and not notes, notes)
    hand = minifig(empty, 320, 240, **side)
    cv2.rectangle(hand, (100, 150), (260, 230), (120, 160, 210), -1)
    tt, notes = still(hand, tt)
    check("자동: 손이 보이면 안 찍고 알려 줌", len(auto_det.views) == 1 and "손" in notes[0], notes)
    tt, notes = still(minifig(empty, 320, 240, **side), tt)
    check("자동: 손을 빼고 새 면이면 찍음", len(auto_det.views) == 2, notes)
    tt, notes = still(hand, tt, 0.2)
    tt, notes = still(minifig(empty, 320, 240, **side), tt)
    check("자동: 이미 찍은 면은 건너뜀", len(auto_det.views) == 2 and "이미" in notes[0], notes)
    wiggle = []
    for i in range(30):
        wiggle.append(auto.update(minifig(empty, 300 + i * 2, 240, torso=(60, 150, 50)), tt + i / 30))
    check("자동: 사람을 돌리는 중에는 안 찍음", len(auto_det.views) == 2 and not any(wiggle), wiggle)

    far_det = PersonDetector(lf.HORIZON_Y)
    far_auto = AutoViews(far_det)
    busy = []
    for i in range(45):     # Someone moving far behind, outside the box.
        scene = minifig(empty, 320, 240)
        cv2.rectangle(scene, (10 + 3 * i, 10), (120 + 3 * i, 120), (200, 200, 200), -1)
        busy.append(far_auto.update(scene, i / 30))
    check("자동: 네모 밖에서 움직이는 게 있어도 찍음", len(far_det.views) == 1, [n for n in busy if n])

    print("  -- 등록 조작")
    start = (0.3, 0.42, 0.7, 0.97)
    moved_box = tune_person.move_box(start, 1, 0)
    check("네모: d로 오른쪽, 크기는 그대로", moved_box[0] > start[0] and abs((moved_box[2] - moved_box[0]) - 0.4) < 1e-9, moved_box)
    check("네모: 화면 밖으로 안 나감", tune_person.move_box((0.0, 0.0, 0.4, 0.5), -1, -1) == (0.0, 0.0, 0.4, 0.5)
          and tune_person.move_box((0.6, 0.5, 1.0, 1.0), 1, 1) == (0.6, 0.5, 1.0, 1.0))
    bigger = tune_person.resize_box(start, 1)
    check("네모: x로 크게, z로 작게", bigger[2] - bigger[0] > 0.4 and tune_person.resize_box(start, -1)[2] - tune_person.resize_box(start, -1)[0] < 0.4)
    check("네모: 너무 작게는 안 줄어듦", tune_person.resize_box((0.4, 0.4, 0.55, 0.55), -1) == (0.4, 0.4, 0.55, 0.55))
    read_end, write_end = os.pipe()
    os.write(write_end, "wㅈA\n".encode())
    check("키: 한글 입력 상태와 대문자도 같은 키", tune_person.read_keys(read_end) == "wwa\n")
    check("키: 눌린 키가 없으면 안 기다림", tune_person.read_keys(read_end) == "")
    os.close(read_end)
    os.close(write_end)
    box_det = PersonDetector(lf.HORIZON_Y)
    box_det.box = tune_person.move_box(tune_person.move_box(start, -1, 0), -1, 0)
    for _ in range(3):
        box_det.box = tune_person.move_box(box_det.box, -1, 0)
    check("네모: 옮긴 네모 안의 사람을 등록", box_det.add_view(minifig(empty, 200, 280))[0], box_det.box)
    box_auto = AutoViews(box_det)
    box_auto.ready, box_auto.around = False, np.zeros(1)
    box_auto.touch()
    check("네모: 옮기면 멈춘 뒤 다시 찍을 준비", box_auto.ready and box_auto.around is None)

    print("  -- 가짜 객체")
    solid = {"torso": (150, 70, 30), "legs": (150, 70, 30), "head": (150, 70, 30)}
    only, _ = registered_figure(**solid)    # A figure of one color: color alone cannot tell it from a box.
    check("가짜: 진짜(한 색 사람)는 여러 거리에서 찾음", all(only(minifig(empty, 320, f, **solid)) for f in (200, 240, 280)))
    width = int(tall * 0.62)
    box = empty.copy()
    cv2.rectangle(box, (320 - width // 2, 240 - tall), (320 + width // 2, 240), solid["torso"], -1)
    check("가짜: 같은 색, 같은 크기의 네모 상자", only(box) is None)
    oval = empty.copy()
    cv2.ellipse(oval, (320, 240 - tall // 2), (width // 2, tall // 2), 0, 0, 360, solid["torso"], -1)
    check("가짜: 같은 색, 같은 크기의 타원", only(oval) is None)
    check("가짜: 같은 모양, 다른 색 사람", only(minifig(empty, 320, 240, torso=(60, 150, 50), legs=(60, 150, 50), head=(60, 150, 50))) is None)
    pale = (220, 180, 150)
    check("가짜: 같은 모양, 같은 색상의 옅은 복제품", only(minifig(empty, 320, 240, torso=pale, legs=pale, head=pale)) is None)
    check("가짜: 옷 색만 다른 사람", det(minifig(empty, 320, 240, torso=(60, 150, 50))) is None)
    check("가짜: 위아래 옷 색이 바뀐 사람", det(minifig(empty, 320, 240, torso=(150, 70, 30), legs=(40, 40, 200))) is None)
    shrunk = minifig(empty, 320, 240)[lf.HORIZON_Y:]    # Same outline at half size, standing at the same place on the floor.
    tiny = empty.copy()
    part = cv2.resize(shrunk, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)[(240 - lf.HORIZON_Y - tall) // 2 - 2:(240 - lf.HORIZON_Y) // 2 + 1, 140:180]
    tiny[240 - part.shape[0]:240, 300:340] = part
    check("가짜: 같은 위치의 절반 크기 복제품", det(tiny) is None)
    check("미등록: 등록 전에는 아무것도 인식 안 함", PersonDetector(lf.HORIZON_Y)(minifig(empty, 320, 240)) is None)
    old = Path(tempfile.mkdtemp()) / "old.json"
    old.write_text(json.dumps({"views": [{"hist": [0.0] * 22, "w": 1, "h": 1, "feet": 1}], "background": [0.0] * 22}))
    check("옛날 형식 저장 파일은 다시 등록하게 함", PersonDetector.load(lf.HORIZON_Y, old) is None)

    path = Path(tempfile.mkdtemp()) / "person.json"
    det.match = 0.6
    det.save(path)
    again = PersonDetector.load(lf.HORIZON_Y, path)
    check("저장/불러오기: 그대로 인식", again.match == 0.6 and again(minifig(empty, 320, 240))["feet"] == found_feet(det))
    check("저장/불러오기: 파일이 없으면 None", PersonDetector.load(lf.HORIZON_Y, path.with_name("none.json")) is None)
    det.undo()
    check("되돌리기: 시점을 지우면 인식 안 함", not det.views and det(minifig(empty, 320, 240)) is None)


def found_feet(det):
    return det(minifig(lane_frame(), 320, 240))["feet"]


def person_driver(course, feet=240, **config):
    defaults = {"PERSON_AT": None, "PERSON_STOP_DELAY": 0.0, "PERSON_RESUME_DELAY": 0.5,
                "PERSON_MIN_STOP": 0.0, "PERSON_MAX_WAIT": None, "PERSON_X": (0.1, 0.9)}
    old = {k: getattr(lf, k) for k in defaults}
    for k, v in {**defaults, **config}.items():
        setattr(lf, k, v)
    det = registered_figure(feet)[0]
    driver = lf.Driver(course, person=det)
    for k, v in old.items():
        setattr(lf, k, v)
    return driver


def run_frames(d, frame, t0, sec):
    """Tick for sec seconds of frames at 30 fps. Returns (time after, motor commands)."""
    cmds = []
    t = t0
    for _ in range(int(sec * 30)):
        with quiet():
            _, cmd, dbg = d.tick(frame, t, show=True)
            d.view(dbg, t, 30)
        cmds.append(cmd)
        t += 1 / 30
    return t, cmds


def test_person_stop():
    print("\n[레고 사람: 정지 / 출발]")
    empty, person = lane_frame(), minifig(lane_frame(), 320, 240)
    far = minifig(lane_frame(), 320, 190)

    d = person_driver([follow(2.0), finish()])
    t, cmds = run_frames(d, empty, 0.0, 0.5)
    check("사람 없으면 그대로 달림", all(c > 0 for c in cmds))
    t, cmds = run_frames(d, far, t, 0.5)
    check("멀리 있으면(정지 위치 전) 계속 달림", cmds[-1] > 0 and d.person.seen)
    t, cmds = run_frames(d, person, t, 1.0)
    check("정지 위치에 오면 멈춤", cmds[-1] == 0 and d.person.hold and d.course.paused)
    check("상태 표시: PERSON STOP", d.course.status(t) == "PERSON STOP")
    t, cmds = run_frames(d, person, t, 5.0)
    check("서 있는 동안 동작 시간이 줄지 않음 (follow 2초가 안 끝남)", d.course.idx == 0 and all(c == 0 for c in cmds))
    t, cmds = run_frames(d, empty, t, 0.3)
    check("사람이 없어져도 딜레이 동안은 서 있음", all(c == 0 for c in cmds))
    t, cmds = run_frames(d, empty, t, 0.5)
    check("딜레이가 지나면 다시 출발 (출발 킥 포함)", cmds[-1] > 0 and not d.person.hold and max(cmds) >= lf.KICK_SPEED)
    used = t - d.course.step_start
    check("멈춘 시간을 뺀 만큼만 동작 시간이 지남", 0.5 < used < 2.0, f"{used:.2f}초")
    t, _ = run_frames(d, empty, t, 2.0 - used + 0.2)
    check("남은 시간을 채우고 다음 동작으로", d.done)

    d = person_driver([follow(9), finish()], PERSON_AT=0.8)
    t, cmds = run_frames(d, person, 0.0, 1.0)
    check("PERSON_AT: 정한 위치 전에는 안 멈춤", cmds[-1] > 0 and not d.person.hold)
    t, cmds = run_frames(d, minifig(empty, 320, 300), t, 0.5)
    check("PERSON_AT: 정한 위치에 오면 멈춤", cmds[-1] == 0)

    d = person_driver([follow(9, speed=0.5), finish()], PERSON_AT=0.8)
    run_frames(d, empty, 0.0, 0.5)
    fast = d.speed
    run_frames(d, person, 0.5, 0.5)
    check("PERSON_SLOW: 사람이 보이면 줄여서 감", fast > 0.4 and abs(d.speed - lf.PERSON_SLOW) < 0.05, f"{fast:.2f} -> {d.speed:.2f}")

    d = person_driver([follow(9), finish()], PERSON_STOP_DELAY=0.5)
    t, cmds = run_frames(d, person, 0.0, 0.3)
    check("PERSON_STOP_DELAY: 닿고 바로는 안 멈춤", cmds[-1] > 0)
    t, cmds = run_frames(d, person, t, 0.6)
    check("PERSON_STOP_DELAY: 지나면 멈춤", cmds[-1] == 0)

    d = person_driver([follow(9), finish()], PERSON_MIN_STOP=1.5, PERSON_RESUME_DELAY=0.0)
    t, _ = run_frames(d, person, 0.0, 0.3)
    t, cmds = run_frames(d, empty, t, 1.0)
    check("PERSON_MIN_STOP: 금방 없어져도 최소 시간은 섬", cmds[-1] == 0)
    t, cmds = run_frames(d, empty, t, 1.0)
    check("PERSON_MIN_STOP: 지나면 출발", cmds[-1] > 0)

    d = person_driver([follow(30), finish()], PERSON_MAX_WAIT=1.0)
    t, cmds = run_frames(d, person, 0.0, 0.9)
    check("PERSON_MAX_WAIT: 그 전에는 서 있음", cmds[-1] == 0)
    t, cmds = run_frames(d, person, t, 1.0)
    check("PERSON_MAX_WAIT: 지나면 사람이 있어도 출발, 바로 다시 안 멈춤", cmds[-1] > 0 and not d.person.hold)
    t, _ = run_frames(d, empty, t, 0.5)
    t, cmds = run_frames(d, person, t, 0.5)
    check("PERSON_MAX_WAIT: 없어졌다 다시 나타나면 또 멈춤", cmds[-1] == 0)

    d = person_driver([stop(0.5), follow(9), finish()])
    t, cmds = run_frames(d, person, 0.0, 1.0)
    check("stop 동작: 시간은 그대로 흐르고, 끝나도 사람이 있으면 출발 안 함", d.course.idx == 1 and all(c == 0 for c in cmds))
    d = person_driver([straight(2.0), finish()])
    t, cmds = run_frames(d, person, 0.0, 3.0)
    check("straight 동작에서도 멈추고 시간이 안 줌", d.course.idx == 0 and cmds[-1] == 0)
    d = person_driver([cs_tr(), finish()])
    run_frames(d, empty, 0.0, 0.1)
    t, cmds = run_frames(d, person, 0.1, 1.5)
    check("회전 동작에서도 멈춤", cmds[-1] == 0 and d.course.paused)
    d = person_driver([stop(0.2), finish()])
    run_frames(d, person, 0.0, 1.0)
    check("코스 마지막 finish는 사람이 있어도 끝남", d.done)

    d = lf.Driver([follow(0.5), finish()])
    t, cmds = run_frames(d, person, 0.0, 0.7)
    check("등록 안 했으면 사람을 안 보고 달림", d.person is None and d.done)
    d = person_driver([follow(9), finish()])
    d.lost_since, d.back = 0.0, {"start": 0.0, "plan": [(1.0, 0.0)], "len": 1.0, "used_at": 0, "idx": 0}
    run_frames(d, person, 0.0, 0.5)
    check("후진 중에 사람이 나타나면 후진을 멈춤", d.back is None and d.speed == 0)


def cs_tr():
    import course as cs
    return cs.tr()


def test_set_speed():
    print("\n[코스 속도 set_speed]")
    import course as cs
    flat = cs._flat([cs.to_junction(), cs.set_speed(0.33), cs.to_junction(), cs.follow(2), cs.to_end(), cs.to_crosswalk(),
                     cs.to_gap(), cs.follow(2, speed=0.7), cs.straight(2), cs.straight(2, speed=0.6), cs.right(), cs.curve(10, 1),
                     cs.tr(), cs.stop(1), cs.set_speed(), cs.to_junction(), cs.follow(2), finish()])
    spd = [s["speed"] for s in flat]
    check("set_speed 전: 원래 속도", spd[0] == cs.SPEED)
    check("set_speed 뒤: 주행 동작이 모두 바뀜", spd[1:5 + 1] == [0.33] * 5, spd[1:6])
    check("직접 준 speed=는 그대로", spd[6] == 0.7)
    check("straight는 그대로", spd[7] is None and spd[8] == 0.6)
    check("right, curve는 그대로", spd[9] is None and spd[10] is None)
    tr_steps = cs._flat(cs.tr())
    check("tr(회전/정렬/천천히)은 그대로", spd[11:11 + len(tr_steps)] == [s["speed"] for s in tr_steps])
    check("set_speed(): 기본값으로 되돌림", spd[-3] == cs.SPEED and spd[-2] == 0.425)
    check("코스에 set_speed 동작이 남지 않음", all(s["do"] != "set_speed" for s in flat) and cs._flat(flat) == flat)
    bundles = cs._flat([cs.set_speed(0.33), cs.cross_gap(), cs.cross(), cs.traffic(), cs.corner("left"), cs.uturn("right"),
                        cs.cross(line="crosswalk"), cs.pass_crosswalk()])
    follows = [s["speed"] for s in bundles if s["do"] == "follow" and s["until"] not in ("aligned", None)]
    straights = [s["speed"] for s in bundles if s["do"] == "straight" and s["until"] == "lane"]
    check("묶음 동작: 안의 차선 주행은 바뀌고", set(follows) == {0.33}, follows)
    check("묶음 동작: 안의 straight는 그대로", set(straights) == {cs.SPEED}, straights)
    defaults = cs._flat([cs.to_end(), cs.uturn("left"), cs.follow(1)])
    check("set_speed 없으면 기존 기본값 그대로", (defaults[0]["speed"], defaults[1]["speed"], defaults[-1]["speed"]) == (0.26, 0.4, 0.425))
    d = lf.Driver([cs.set_speed(0.33), follow(9), finish()])
    with quiet():
        d.tick(lane_frame(), 0.0)
    check("Driver: set_speed가 적용되고 --start 번호에 안 잡힘", d.course.speed == 0.33 and len(d.course.course) == 2)
    d = lf.CourseRunner([follow(1), cs.set_speed(0.33), follow(1), follow(1)], start=3)
    check("--start로 중간부터 시작해도 앞의 set_speed 적용", d.speed == 0.33)
    try:
        cs.set_speed(1.5)
        ok = False
    except ValueError:
        ok = True
    check("set_speed: 범위 밖이면 오류", ok)


def main():
    for t in (test_fast_math, test_lane, test_gap, test_mat_mask, test_crosswalk, test_track_color,
              test_traffic_light, test_course_logic, test_calibration, test_motor, test_turn_align_stop, test_blind_and_bundles, test_lost_and_far, test_stripes_and_crossing,
              test_driver_smoke, test_person_detect, test_person_stop, test_set_speed):
        t()
    passed = sum(results)
    print(f"\n{passed}/{len(results)} 통과")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
