"""합성 영상으로 인식/코스 로직을 점검 (녹화나 카메라 없이 맥이나 파이에서 실행)

  aicar test-vision                       맥에서 (.venv 사용)
  python tests/test_vision.py             직접 (pi/ 폴더를 찾아서 import)

설정값(LANE_WIDTH_PX, HORIZON_Y 등)이 바뀌어도 그 값으로 영상을 그려서 테스트하므로
화각/트랙을 바꾼 뒤에도 그대로 쓸 수 있음. 실패가 하나라도 있으면 종료 코드 1
"""
import contextlib
import io
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pi"))
import lane_follow as lf  # noqa: E402
from course import COURSE, finish, follow, straight, wait_green  # noqa: E402

W, H = lf.FRAME_W, lf.FRAME_H
RED, GREEN = (0, 0, 255), (0, 255, 0)
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  {'OK  ' if ok else 'FAIL'} {name}" + (f"  ({detail})" if detail else ""))


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


# ---------------- 합성 영상 ----------------
def lane_x(y, shift=0.0):
    w = lf.LANE_WIDTH_PX * (y - lf.HORIZON_Y) / (342 - lf.HORIZON_Y)
    cx = W * lf.CAMERA_X + shift
    return cx - w / 2, cx + w / 2


def lane_frame(y_end=0, left=True, right=True, shift=0.0, bg=30, line=235, crosswalk=False):
    """바닥(bg) 위 차선(line). y_end보다 위(먼 쪽)는 선이 없음 (= 끊김)"""
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


# ---------------- 테스트 ----------------
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
    y4 = H - 5 * band_h                          # 아래에서 5번째 띠(4번) 영역만 비움 → 위(5번 띠)에서 선이 다시 보임
    d6, _ = run_detector(blank(lane_frame(), y4, y4 + band_h))
    check("짧은 끊김(빈 띠 1개, 위에서 선이 다시 보임) → gap", d6.gap_y is not None, f"gap_y={d6.gap_y}")
    img = lane_frame()
    blank(img, H - 4 * band_h, H - 2 * band_h, W // 2, W)     # 오른쪽 선이 먼저 끊김
    blank(img, H - 4 * band_h, H - 3 * band_h, 0, W // 2)     # 왼쪽 선은 한 띠 늦게 끊김
    d7, _ = run_detector(img)
    check("양쪽 선이 끊기는 위치가 어긋나도 gap", d7.gap_y is not None, f"gap_y={d7.gap_y}")
    d8, _ = run_detector(blank(lane_frame(), int(H * lf.ROI_TOP), int(H * lf.ROI_TOP) + band_h))
    check("맨 위 띠 하나만 빔(먼 선이 흐림) → gap 아님", d8.gap_y is None, f"why={d8.gap_why}")
    img = lane_frame(y_end=250)
    img[int(H * 0.82):, W // 2:] = 30           # 가까운 오른쪽 선이 화면 아래에서 잘려 안 보임
    d4, _ = run_detector(img)
    check("가까운 선 하나가 화면 밖으로 잘려도 끊기기 직전에 양쪽이 보이면 gap", d4.gap_y is not None,
          f"gap_y={d4.gap_y}")
    img = lane_frame(y_end=0, bg=200, line=20)  # 나무 바닥(밝음) + 검은 테이프, 아래 모서리만 어두움(비네팅)
    yy, xx = np.mgrid[0:H, 0:W]
    shade = (1 - 0.35 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H) / H) ** 2)).clip(0.6, 1)
    img = (img * shade[..., None]).astype(np.uint8)
    d5, det5 = run_detector(img, color="black")
    check("바닥 하나 + 가장자리 어두움: 가까운 양쪽 선을 지우지 않음",
          d5.left[0] is not None and d5.right[0] is not None, f"left {d5.left[0]} right {d5.right[0]}")


def test_mat_mask():
    print("[매트 마스크]")
    img = lane_frame(y_end=0, bg=25, line=235)
    img[:, :int(W * 0.12)] = 170                # 왼쪽은 나무 테두리 (매트 밖)
    cv2.line(img, (int(W * 0.06), int(H * 0.45)), (int(W * 0.06), H), (250, 250, 250), 8)   # 테두리 위 흰 것
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
        for near, n in ((True, 5), (False, 20)):     # 끊긴 구간 위: 너머 차선은 위쪽에 계속 보이지만 바로 앞 선이 없음
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
        c = lf.CourseRunner([straight(until="lane", angle="hold", max_sec=30), finish()])
        c.update(False, 0.0, None, False, {}, None)
        a1, a2 = c.target(None, 0.0, current=12.0), c.target(None, 0.1, current=3.0)
    check('straight(angle="hold"): 시작 각도 유지', a1 == 12.0 and a2 == 12.0, f"{a1}, {a2}")


def test_calibration():
    print("[자동 캘리브레이션 계산]")
    import calibrate as cal
    true_w, true_h, true_shift = 430.0, 70.0, 12.0      # 지금 설정과 다른 '실제' 트랙

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
    # 파이에서 실제로 잰 렌즈 위치별 선명도 (먼 / 가운데 / 가까운): 가까운 쪽에 비중을 둔 4.5가 골라져야 함
    table = {1.5: [349, 159, 92], 3.0: [506, 356, 216], 3.75: [495, 447, 319], 4.5: [408, 462, 418],
             5.25: [305, 388, 467], 6.0: [217, 292, 445], 7.5: [111, 133, 277]}
    check("초점: 차선 영역(가까운 쪽 비중) 기준으로 고름", cal.best_focus(table) == 4.5, cal.best_focus(table))
    check("선이 안 보이면 측정 안 하고 이유를 알려 줌",
          isinstance(cal.fit_geometry(cal.measure_lanes([np.full((H, W, 3), 30, np.uint8)] * 4)), str))


def test_motor():
    print("[모터 최소값]")
    import calibrate_motor as cm
    check("직진 최소값", abs(lf.motor_floor(0) - lf.MOTOR_MIN) < 1e-9, lf.motor_floor(0))
    check("최대로 꺾으면 꺾음 최소값", abs(lf.motor_floor(lf.STEER_LIMIT) - lf.MOTOR_MIN_TURN) < 1e-9,
          lf.motor_floor(lf.STEER_LIMIT))
    check("반쯤 꺾으면 그 사이", lf.MOTOR_MIN <= lf.motor_floor(lf.STEER_LIMIT / 2) <= lf.MOTOR_MIN_TURN)
    with quiet():
        d = lf.Driver([follow(until="junction", speed=0.1, max_sec=99), finish()])
        cmds = [d.tick(lane_frame(), i / 30)[1] for i in range(40)]
    check("아주 느린 명령(0.1)도 최소값 이상으로 보냄", min(c for c in cmds if c > 0) >= lf.MOTOR_MIN - 1e-9,
          f"최소 {min(c for c in cmds if c > 0):.2f}")
    a = cm.small_gray(lane_frame())
    b = cm.small_gray(np.roll(lane_frame(), 12, axis=0))     # 차가 움직이면 화면이 밀림
    noise = cm.frame_diff(a, cm.small_gray(lane_frame()))
    check("가만히 있으면 '움직임' 아님", noise <= cm.moved_threshold(noise), f"{noise:.1f}")
    check("화면이 밀리면 '움직임'", cm.frame_diff(a, b) > cm.moved_threshold(noise), f"{cm.frame_diff(a, b):.1f}")


def test_turn_align_stop():
    print("[tr(): 회전 → 정렬 → 정지]")
    import course as cs
    with quiet():
        c = lf.CourseRunner([cs.tr(stop_sec=1.0), finish()])
        t = feed(c, 12, 0.0, det=None)                  # 도는 중: 차선 안 보임 (0.6초)
        t = feed(c, lf.TURN_CLEAR_FRAMES, t, det=(0.4, 0.3))     # 새 차선이 비스듬히 잡힘
    check("새 차선이 잡히면 3초를 안 채우고 회전 끝", c.idx == 1 and t < 3.0, f"{t:.2f}초, 동작 {c.idx + 1}")
    with quiet():
        t = feed(c, 10, t, det=(0.3, 0.2))              # 아직 덜 정렬됨
    check("덜 정렬되면 계속 차선 따라감", c.idx == 1 and not c.stopped(t))
    with quiet():
        t = feed(c, lf.ALIGN_FRAMES, t, det=(0.05, 0.03))       # 차로 가운데, 똑바로
    check("정렬되면 그 자리에서 정지", c.idx == 2 and c.stopped(t), c.status(t))
    with quiet():
        t = feed(c, 25, t, det=(0.05, 0.03))
    check("정지 시간(1초) 뒤 다음 동작", c.idx == 3 or c.done, f"동작 {c.idx + 1}")
    with quiet():
        c = lf.CourseRunner([cs.tr(), finish()])
        t = feed(c, 70, 0.0, det=None)                  # 끝내 차선을 못 찾음
    check("차선을 못 찾으면 3초 안전장치로 회전 끝", c.idx >= 1, f"{t:.1f}초, 동작 {c.idx + 1}")
    check("tl(0): 정지 없이 회전 → 정렬만", [s["do"] for s in cs._flat(cs.tl(0))] == ["left", "follow"])


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


def main():
    for t in (test_fast_math, test_lane, test_gap, test_mat_mask, test_crosswalk, test_track_color,
              test_traffic_light, test_course_logic, test_calibration, test_motor, test_turn_align_stop, test_driver_smoke):
        t()
    passed = sum(results)
    print(f"\n{passed}/{len(results)} 통과")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
