"""Check detection and course logic with synthetic images. Runs on a Mac or a Pi, with no recording and no camera.

  aicar test-vision                       on a Mac (uses .venv)
  python tests/test_vision.py             run it directly (locates the pi/ folder and imports it)

Images are drawn from the current settings (LANE_WIDTH_PX, HORIZON_Y, and the rest),
so the test still applies after the field of view or the track changes. Any failure exits with code 1.
"""
import contextlib
import io
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pi"))
import lane_follow as lf  # noqa: E402
from course import COURSE, finish, follow, stop, straight, wait_green  # noqa: E402

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
            import course as cs
            d = lf.Driver([cs.tr(), finish()], start=1)
            for i in range(5):
                d.tick(lane_frame(), i / 30, show=True)
        ok = True
    except Exception as e:  # noqa: BLE001
        ok = False
        print(e)
    check("회전 동작부터 시작(--start)해도 첫 프레임에서 오류 없음", ok)


def main():
    for t in (test_fast_math, test_lane, test_gap, test_mat_mask, test_crosswalk, test_track_color,
              test_traffic_light, test_course_logic, test_calibration, test_motor, test_turn_align_stop, test_blind_and_bundles, test_lost_and_far, test_stripes_and_crossing,
              test_driver_smoke):
        t()
    passed = sum(results)
    print(f"\n{passed}/{len(results)} 통과")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
