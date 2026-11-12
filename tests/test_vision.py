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
import course as _course_settings  # noqa: E402
# The tests check the code, not the day's settings at the top of course.py, so pin those to their defaults.
_course_settings.STEER_MAX = None
_course_settings.REFLECT_MOTION_ON = True
_course_settings.SLOWDOWN_ON = True     # The slowdown code is tested as it works when switched on.
from course import COURSE, finish, follow, stop, straight, wait_green  # noqa: E402
import person  # noqa: E402
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
    print("[fast math = original math]")
    rng = np.random.default_rng(1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (lf.TOPHAT_K, lf.TOPHAT_K))
    same_t = same_p = 0
    for _ in range(20):
        img = rng.integers(0, 256, (H // 2, W // 2), dtype=np.uint8)
        img = cv2.GaussianBlur(img, (7, 7), 0)
        same_t += np.array_equal(cv2.morphologyEx(img, cv2.MORPH_TOPHAT, k), lf.tophat_ellipse(img, lf.TOPHAT_K))
        same_p += float(np.percentile(img, 99)) == lf.percentile_u8(img, 99)
    check("top-hat (rectangle split) = morphologyEx", same_t == 20, f"{same_t}/20")
    check("percentile_u8 = np.percentile", same_p == 20, f"{same_p}/20")


def test_lane():
    print("[lane]")
    _, det = run_detector(lane_frame())
    check("straight center line -> offset about 0", det is not None and abs(det[0]) < 0.05, f"{det}")
    _, det_r = run_detector(lane_frame(shift=40))
    check("lane seen to the right -> offset +", det_r is not None and det_r[0] > 0.05, f"{det_r}")
    _, det_1 = run_detector(lane_frame(right=False))
    check("center estimated from the left line alone", det_1 is not None and abs(det_1[0]) < 0.15, f"{det_1}")


def test_gap():
    print("[lane gap]")
    d, _ = run_detector(lane_frame(y_end=0))
    check("continuous lane -> no gap", d.gap_y is None, f"gap_y={d.gap_y}")
    d1, _ = run_detector(lane_frame(y_end=250))
    check("lane cut off far away -> gap", d1.gap_y is not None, f"gap_y={d1.gap_y}")
    d2, _ = run_detector(lane_frame(y_end=300))
    check("cut off nearer -> gap_y lower", d2.gap_y is not None and d2.gap_y > (d1.gap_y or 0),
          f"{d1.gap_y} → {d2.gap_y}")
    d3, _ = run_detector(lane_frame(y_end=250, right=False))
    check("one line only (sharp curve) -> not a gap", d3.gap_y is None, f"gap_y={d3.gap_y}")
    band_h = (H - int(H * lf.ROI_TOP)) // lf.SLICES
    def blank(img, y0, y1, x0=0, x1=W):
        img[y0:y1, x0:x1] = np.clip(np.random.default_rng(3).normal(30, 4, (y1 - y0, x1 - x0, 3)), 0, 255)
        return img
    y4 = H - 5 * band_h                          # Clear only band 4, the fifth from the bottom, so the line resumes in band 5 above it.
    d6, _ = run_detector(blank(lane_frame(), y4, y4 + band_h))
    check("short gap (1 empty band, line seen again above) -> gap", d6.gap_y is not None, f"gap_y={d6.gap_y}")
    img = lane_frame()
    blank(img, H - 4 * band_h, H - 2 * band_h, W // 2, W)     # The right line ends first.
    blank(img, H - 4 * band_h, H - 3 * band_h, 0, W // 2)     # The left line ends one band later.
    d7, _ = run_detector(img)
    check("gap even when the two lines end at different heights", d7.gap_y is not None, f"gap_y={d7.gap_y}")
    d8, _ = run_detector(blank(lane_frame(), int(H * lf.ROI_TOP), int(H * lf.ROI_TOP) + band_h))
    check("only the top band empty (far line faint) -> not a gap", d8.gap_y is None, f"why={d8.gap_why}")
    img = lane_frame(y_end=250)
    img[int(H * 0.82):, W // 2:] = 30           # The near right line is cut off at the bottom of the frame and is not visible.
    d4, _ = run_detector(img)
    check("gap when both lines are seen just before the cut, even if a near line is cut off by the frame", d4.gap_y is not None,
          f"gap_y={d4.gap_y}")
    img = lane_frame(y_end=0, bg=200, line=20)  # Bright wood floor plus black tape. Only the bottom corners are dark from vignetting.
    yy, xx = np.mgrid[0:H, 0:W]
    shade = (1 - 0.35 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H) / H) ** 2)).clip(0.6, 1)
    img = (img * shade[..., None]).astype(np.uint8)
    d5, det5 = run_detector(img, color="black")
    check("one floor + dark edge: near lines on both sides are kept",
          d5.left[0] is not None and d5.right[0] is not None, f"left {d5.left[0]} right {d5.right[0]}")


def test_mat_mask():
    print("[mat mask]")
    img = lane_frame(y_end=0, bg=25, line=235)
    img[:, :int(W * 0.12)] = 170                # The left side is a wooden border, outside the mat.
    cv2.line(img, (int(W * 0.06), int(H * 0.45)), (int(W * 0.06), H), (250, 250, 250), 8)   # Something white on that border.
    d, det = run_detector(img)
    top = int(H * lf.ROI_TOP)
    mask = d.line_mask(img[top:])
    check("black mat + wooden border: white things on the border are ignored", int((mask[:, :int(W * 0.1)] > 0).sum()) == 0)
    check("black mat + wooden border: the lane stays", det is not None and abs(det[0]) < 0.1, f"{det}")


def test_crosswalk():
    print("[crosswalk]")
    d, _ = run_detector(lane_frame(crosswalk=True))
    check("crosswalk stripes found", d.crosswalk_y is not None, f"crosswalk_y={d.crosswalk_y}")
    d0, _ = run_detector(lane_frame())
    check("lane only -> no crosswalk", d0.crosswalk_y is None)


def test_track_color():
    print("[track color: black lines on a white floor]")
    for name, kw in (("straight", {}), ("offset", {"shift": 40}), ("gap", {"y_end": 250}), ("crosswalk", {"crosswalk": True})):
        da, a = run_detector(lane_frame(bg=30, line=235, **kw), color="white")
        db, b = run_detector(lane_frame(bg=225, line=20, **kw), color="black")
        same = (a is not None and b is not None and np.allclose(a, b, atol=0.02)
                and (da.gap_y, da.crosswalk_y) == (db.gap_y, db.crosswalk_y))
        check(f"{name}: same result as white lines on a black floor", same, f"{a} / {b}")


def test_traffic_light():
    print("[traffic light]")
    close = np.zeros((H, W, 3), np.uint8)
    cv2.circle(close, (30, int(H * 0.57)), 8, (40, 40, 240), -1)    # Stopped just before the crosswalk: the lamp is this low in the frame.
    check("a red light seen low right in front of a crosswalk is found", lf.TrafficLight()(close) == "red")
    tl = lf.TrafficLight()
    check("red lamp on the left", tl(lamp_img(((50, 60), RED))) == "red", tl.state)
    check("green lamp on the right", tl(lamp_img(((W - 50, 60), GREEN))) == "green", tl.state)
    img = np.full((H, W, 3), 25, np.uint8)
    cv2.circle(img, (50, 60), 9, (150, 230, 150), -1)
    cv2.circle(img, (50, 60), 5, (255, 255, 255), -1)
    check("white LED center + pale green rim", tl(img) == "green", tl.state)
    img = np.full((H, W, 3), 25, np.uint8)
    cv2.rectangle(img, (10, 40), (220, 55), RED, -1)
    check("long red object -> ignored", tl(img) is None, tl.state)
    check("lamp in the middle of the frame -> out of range", tl(lamp_img(((W // 2, 60), RED))) is None, tl.state)
    check("green in the lane area (bottom) -> ignored", tl(lamp_img(((50, int(H * 0.85)), GREEN))) is None, tl.state)


def test_course_logic():
    print("[course logic]")
    with quiet():
        c = lf.CourseRunner([straight(until="lane", max_sec=30), finish()])
        t = feed(c, 40, 0.0)
    check("straight(until=lane): a lane seen from the start does not end it", c.idx == 0)
    with quiet():
        t = feed(c, 5, t, det=None)
        t = feed(c, lf.TURN_CLEAR_FRAMES + 1, t)
    check("straight(until=lane): ends when it goes and comes back", c.idx == 1)

    with quiet():
        c = lf.CourseRunner([straight(until="lane", max_sec=30), finish()])
        for near, n in ((True, 5), (False, 20)):     # Over a gap the lane beyond stays visible higher up, but there is no line just ahead.
            for _ in range(n):
                c.update(False, t, (0.0, 0.0), False, {"near": near}); t += 0.05
    check("over a gap (no line just ahead) it keeps going straight", c.idx == 0)
    with quiet():
        for _ in range(lf.TURN_CLEAR_FRAMES + 1):
            c.update(False, t, (0.0, 0.0), False, {"near": True}); t += 0.05
    check("straight ends once the line just ahead is seen again across the gap", c.idx == 1)

    with quiet():
        c = lf.CourseRunner([wait_green(max_sec=30), finish()])
        t = feed(c, 20, 0.0, light="red")
    check("wait_green: stays stopped on red", c.idx == 0 and c.stopped(t))
    with quiet():
        t = feed(c, 3, t, light="green")
    check("wait_green: red -> green for 3 frames goes", c.idx == 1)
    with quiet():
        c = lf.CourseRunner([wait_green(max_sec=30), finish()])
        t = feed(c, 40, 0.0, light="green")
    check("wait_green: green from the start does not go", c.idx == 0)
    with quiet():
        c = lf.CourseRunner([wait_green(max_sec=1), finish()])
        feed(c, 30, 0.0)
    check("wait_green: goes after max_sec when nothing is detected", c.idx == 1)
    with quiet():
        c = lf.CourseRunner([wait_green(max_sec=1), finish()])
        t = feed(c, 3, 0.0, light="red")
        t = feed(c, 200, t, light="red")
    check("wait_green: once red is seen it does not leave after max_sec", c.idx == 0 and c.stopped(t), f"{t:.0f}s")
    with quiet():
        t = feed(c, 200, t)
    check("wait_green: after red, keeps waiting even if the light is not seen", c.idx == 0, f"{t:.0f}s")
    with quiet():
        t = feed(c, 3, t, light="green")
    check("wait_green: then green goes", c.idx == 1)
    with quiet():
        c = lf.CourseRunner([wait_green(max_sec=1), finish()])
        t = feed(c, 2, 0.0, light="red")
        t = feed(c, 30, t)
    check("wait_green: red for only 1-2 frames counts as not found, goes after max_sec", c.idx == 1)
    with quiet():
        c = lf.CourseRunner([wait_green(max_sec=1), follow(1), finish()])
        t = feed(c, 3, 0.0, light="red")
        t = feed(c, 30, t, light="red")
        t = feed(c, 3, t, light="green")
        t = feed(c, 30, t, light="red")
    check("wait_green: does not affect max_sec of the next step", c.idx == 2 or c.done, f"step {c.idx + 1}")

    with quiet():
        c = lf.CourseRunner([wait_green(max_sec=30), finish()])
        t = feed_img(c, lamp_img(((50, 50), RED)), 5, 0.0)
        t = feed_img(c, lamp_img(((W - 50, 50), GREEN)), 10, t)
    check("green object away from the red light -> does not go", c.idx == 0)
    with quiet():
        t = feed_img(c, lamp_img(((50, 80), GREEN)), 5, t)
    check("green right below the red lamp -> goes", c.idx == 1)

    with quiet():
        c = lf.CourseRunner([follow(until="gap", at=0.85, max_sec=30), finish()])
        t = feed(c, 10, 0.0, gap=int(H * 0.75))
    check("until=gap at=0.85: not yet at 0.75", c.idx == 0)
    with quiet():
        feed(c, 3, t, gap=int(H * 0.85) + 1)
    check("until=gap at=0.85: ends at 0.85", c.idx == 1)
    with quiet():
        c = lf.CourseRunner([follow(until="gap", max_sec=30), finish()])
        t = feed(c, 10, 0.0, gap=int(H * 0.35))
    check("until=gap without at: a gap far away does not end it", c.idx == 0)
    with quiet():
        feed(c, 3, t, gap=int(H * lf.JUNCTION_Y) + 1)
    check("until=gap without at: ends at JUNCTION_Y", c.idx == 1)

    with quiet():
        c = lf.CourseRunner([straight(until="lane", angle="hold", max_sec=30), finish()])
        c.update(False, 0.0, None, False, {}, None)
        a1, a2 = c.target(None, 0.0, current=12.0), c.target(None, 0.1, current=3.0)
    check('straight(angle="hold"): keeps the start angle', a1 == 12.0 and a2 == 12.0, f"{a1}, {a2}")


def test_calibration():
    print("[automatic calibration math]")
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
    check("dark floor -> white line track", color == "white", color)
    color_b, _ = cal.detect_track_color([frame(225, 20)])
    check("bright floor -> black line track", color_b == "black", color_b)
    geo = cal.fit_geometry(cal.measure_lanes([frame(30, 235)] * 6))
    ok = isinstance(geo, dict)
    check("lane width measured", ok and abs(geo["lane_width_px"] - true_w) < 8, geo["lane_width_px"] if ok else geo)
    check("vanishing point measured", ok and abs(geo["horizon_y"] - true_h) < 10, geo["horizon_y"] if ok else geo)
    check("camera center measured", ok and abs(geo["camera_x"] * W - (W / 2 + true_shift)) < 3,
          geo["camera_x"] if ok else geo)
    old = lf.LINE_COLOR
    lf.LINE_COLOR = "black"
    try:
        geo_b = cal.fit_geometry(cal.measure_lanes([frame(225, 20)] * 6))
    finally:
        lf.LINE_COLOR = old
    check("same measurement on a white floor with black lines", isinstance(geo_b, dict) and abs(geo_b["lane_width_px"] - true_w) < 8,
          geo_b["lane_width_px"] if isinstance(geo_b, dict) else geo_b)
    # Sharpness measured on the Pi at each lens position (far, middle, near). The near-weighted choice must be 4.5.
    table = {1.5: [349, 159, 92], 3.0: [506, 356, 216], 3.75: [495, 447, 319], 4.5: [408, 462, 418],
             5.25: [305, 388, 467], 6.0: [217, 292, 445], 7.5: [111, 133, 277]}
    check("focus: chosen over the lane area (near side weighted)", cal.best_focus(table) == 4.5, cal.best_focus(table))
    old_lens = lf.LENS_POSITION
    with tempfile.TemporaryDirectory() as tmp:
        saved = Path(tmp) / "vision.json"
        saved.write_text(json.dumps({"lens_position": 7.5}))
        lf.load_vision(saved)
    check("focus: fixed at about 15 cm even with a saved value", lf.LENS_POSITION == old_lens and 14 <= 100 / lf.LENS_POSITION <= 16, lf.LENS_POSITION)
    check("no measurement and a reason when no lines are seen",
          isinstance(cal.fit_geometry(cal.measure_lanes([np.full((H, W, 3), 30, np.uint8)] * 4)), str))


def test_motor():
    print("[motor minimum]")
    check("straight minimum", abs(lf.motor_floor(0) - lf.MOTOR_MIN) < 1e-9, lf.motor_floor(0))
    check("full steer gives the turn minimum", abs(lf.motor_floor(lf.STEER_LIMIT) - lf.MOTOR_MIN_TURN) < 1e-9,
          lf.motor_floor(lf.STEER_LIMIT))
    check("half steer is in between", lf.MOTOR_MIN <= lf.motor_floor(lf.STEER_LIMIT / 2) <= lf.MOTOR_MIN_TURN)
    with quiet():
        d = lf.Driver([follow(until="junction", speed=0.1, max_sec=99), finish()])
        cmds = [d.tick(lane_frame(), i / 30)[1] for i in range(40)]
    check("a very slow command (0.1) is raised to the minimum", min(c for c in cmds if c > 0) >= lf.MOTOR_MIN - 1e-9,
          f"min {min(c for c in cmds if c > 0):.2f}")


def test_turn_align_stop():
    print("[tr(): turn -> align -> slow drive]")
    import course as cs
    # Frames (0.05 s each) until the turn may end: past both its blind period and its min_sec, whatever course.py sets them to.
    first = cs._flat(cs.tr())[0]
    pre = int(max(first["min"], first["blind"]) / 0.05) + 2
    with quiet():
        c = lf.CourseRunner([cs.tr(sec=1.0), finish()])
        t = feed(c, pre, 0.0, det=None)                 # Still no lane after the blind period and min_sec, so counting starts after that.
        t = feed(c, lf.TURN_END_FRAMES, t, det=(0.3, 0.3))       # New lane: both position and heading match.
    check("the turn ends soon after min_sec once a new lane is found", c.idx == 1 and t < first["min"] + 1.5, f"{t:.2f}s, step {c.idx + 1}")
    with quiet():
        t = feed(c, 10, t, det=(0.3, 0.2))              # Still not aligned enough.
    check("keeps following the lane while not yet aligned", c.idx == 1 and not c.stopped(t))
    with quiet():
        t = feed(c, lf.ALIGN_FRAMES, t, det=(0.05, 0.03))       # Lane center, pointed straight.
    check("once aligned, drives on at the set speed (no slowing in a turn) without stopping", c.idx == 2 and not c.stopped(t) and c.speed == cs.SPEED, c.status(t))
    with quiet():
        t = feed(c, 25, t, det=(0.05, 0.03))
    check("next step after sec (1 s)", c.idx == 3 or c.done, f"step {c.idx + 1}")
    with quiet():
        c = lf.CourseRunner([cs.tr(), finish()])
        t = feed(c, 190, 0.0, det=None)                 # The lane is never found (9.5 s).
        before = c.idx
        t = feed(c, 20, t, det=None)                    # 10.5 s.
    check("the turn ends by the 10 s safety limit when no lane is found", before == 0 and c.idx >= 1, f"{t:.1f}s, step {c.idx + 1}")
    check("tl(sec=0): turn -> align only, no slow drive", [s["do"] for s in cs._flat(cs.tl(sec=0))] == ["left", "follow"])
    check("tr(align=False): turn -> slow drive, no aligning", [(s["do"], s["until"]) for s in cs._flat(cs.tr(align=False))] == [("right", "lane"), ("follow", None)])
    check("tl(align=False, sec=0): the turn alone", [s["do"] for s in cs._flat(cs.tl(align=False, sec=0))] == ["left"])

    with quiet():   # A leftover lane crossing at an angle during a turn may look like a horizontal line, but a new lane still ends the turn.
        c = lf.CourseRunner([cs.tl(sec=0), finish()])
        t = 0.0
        for _ in range(pre):
            c.update(False, t, None, False, {}); t += 0.05          # The lane stays lost until the blind period and min_sec end.
        for _ in range(lf.TURN_END_FRAMES):
            c.update(True, t, (0.1, -0.3), False, {"near": True}); t += 0.05   # It looks like a junction, but it is a new lane.
    check("turn: ends on a new lane even if it looks like a line across (junction)", c.idx == 1, f"{t:.2f}s, step {c.idx + 1}")
    with quiet():   # Pointed along the exit lane but off to one side (offset 0.55): end the turn, alignment fixes the offset.
        c = lf.CourseRunner([cs.tr(sec=0), finish()])
        t = feed(c, 6, 0.0, det=(0.05, 0.05))
        t = feed(c, max(30, pre), t, det=None)
        for _ in range(lf.TURN_END_FRAMES):
            c.update(False, t, (0.55, 0.1), False, {"near": True}); t += 0.05
    check("turn: ends on a lane pointing the right way but off to the side (0.55)", c.idx == 1, f"step {c.idx + 1}")
    with quiet():
        c = lf.CourseRunner([cs.tr(sec=0), finish()])
        t = feed(c, 6, 0.0, det=(0.05, 0.05))
        t = feed(c, 30, t, det=None)
        for _ in range(lf.TURN_END_FRAMES + 2):
            c.update(False, t, (0.55, 0.45), False, {"near": True}); t += 0.05
    check("turn: does not end on a line off to the side and pointing wrong (0.55, 0.45)", c.idx == 0)
    with quiet():   # The lane already visible at the start of the turn, before the car turns, must not end the turn.
        c = lf.CourseRunner([cs.tr(sec=0), finish()])
        t = feed(c, 30, 0.0, det=(0.05, 0.05))
    check("turn: a lane seen before turning does not end it", c.idx == 0)
    with quiet():   # A gap that stayed visible during the previous action must not end the next action the moment it starts.
        c = lf.CourseRunner([stop(1.0), follow(until="gap", max_sec=30), finish()])
        t = feed(c, 5, 0.0)                           # No gap at first.
        t = feed(c, 16, t, gap=int(H * 0.8))          # A fake gap appears during the stop, and the stop ends at 1 second.
        started = c.idx
        t = feed(c, 2, t, gap=int(H * 0.8))           # Second frame of the new action.
    check("a detection from the previous step does not end it at once (3 frames again in the new step)", started == 1 and c.idx == 1,
          f"step {c.idx + 1}")
    with quiet():
        feed(c, 1, t, gap=int(H * 0.8))
    check("ends after 3 frames in a row in the new step", c.idx == 2)
    old = lf.GAP_MAX_HEADING
    lf.GAP_MAX_HEADING = -1.0       # Treat even a tiny heading error as misalignment, so the guard can be checked.
    try:
        d9, _ = run_detector(lane_frame(y_end=250))
    finally:
        lf.GAP_MAX_HEADING = old
    check("no gap when the car is far out of line with the lane", d9.gap_y is None and "aligned" in d9.gap_why, d9.gap_why)


def feed_junc(course, frames, t, on, det=(0.0, 0.0)):
    """A junction line at 0.6 of the frame height (on=True), or no line."""
    for _ in range(frames):
        course.update(on, t, det, False, {"junction": int(H * 0.6) if on else None, "near": True})
        t += 0.05
    return t


def test_blind_and_bundles():
    print("[blind_sec / grouped actions]")
    import course as cs
    first = cs._flat(cs.tr())[0]
    with quiet():   # Quick turn: the old lane goes away and the new lane comes back, all inside the blind period.
        c = lf.CourseRunner([cs.tr(sec=0), finish()])
        t = feed(c, 6, 0.0, det=(0.05, 0.05))
        t = feed(c, 10, t, det=None)
        t = feed(c, max(30, int(first["min"] / 0.05) + 4), t, det=(0.05, 0.05))     # Until past min_sec, whatever course.py sets it to.
    check("turn: a new lane found during blind ends the turn when blind is over (no 3 s wait)", c.idx >= 1 and c.course[c.idx]["until"] == "aligned"
          or c.idx >= 2, f"step {c.idx + 1}")
    with quiet():   # A near junction comes into view at 0.8 s and is still in view when the 1.5 second blind period ends.
        c = lf.CourseRunner([cs.to_junction(blind_sec=1.5), finish()])
        t = feed_junc(c, 16, 0.0, False)
        t = feed_junc(c, 20, t, True)
    check("to_junction: ends at a near junction that came in during blind (not passed)", c.idx == 1, f"{t:.2f}s, step {c.idx + 1}")
    with quiet():   # A one- or two-frame gap during blind does not end the action before blind is over.
        c = lf.CourseRunner([cs.to_junction(blind_sec=1.5), finish()])
        t = feed_junc(c, 10, 0.0, True)
        t = feed_junc(c, 2, t, False)
        while t < 1.5:
            t = feed_junc(c, 1, t, True)
        during = c.idx
        t = feed_junc(c, 3, t, True)
    check("to_junction: a flicker during blind does not end it; a line after blind does", during == 0 and c.idx == 1, f"step {c.idx + 1}")

    with quiet():   # nth=2: the first line (a bend) is passed, the second ends the action.
        c = lf.CourseRunner([cs.to_junction(nth=2, blind_sec=0), finish()])
        t = feed_junc(c, 5, 0.0, False)
        t = feed_junc(c, 8, t, True)
        first = (c.idx, c.count, c.final_approach())
        t = feed_junc(c, 8, t, False)
        before = c.final_approach()
        t = feed_junc(c, 8, t, True)
    check("to_junction(nth=2): passes the first line, ends at the second", first[:2] == (0, 1) and c.idx == 1, f"{first} -> step {c.idx + 1}")
    check("nth=2: no slowdown/steer limit while passing the first line, applied again after", not first[2] and before)

    s = cs.follow(3)
    check("follow(3): a timed step has no until", s["until"] is None)
    try:
        cs.follow(3, slow=0.2)
        ok = False
    except ValueError:
        ok = True
    check("follow(3, slow=): error together with sec", ok)
    steps = cs._flat(cs.corner("left", wide=2))
    check("corner: junction -> straight -> turn -> align -> slow drive",
          [x["do"] for x in steps] == ["follow", "straight", "left", "follow", "follow"]
          and steps[0]["shift"] == 2 and steps[2]["blind"] == round(cs.TURN_BLIND * cs.TURN_BLIND_SPEED / cs.TURN_SPEED, 2), str([x["do"] for x in steps]))
    steps = cs._flat(cs.uturn("right"))
    check("uturn: full steer, blind 1.5 s, up to aligned",
          [x["do"] for x in steps] == ["follow", "right", "follow", "follow"]
          and steps[1]["angle"] is None and steps[1]["blind"] == 1.5 and steps[1]["max"] == 5)
    check("cross(): crosses on the junction line", cs._flat(cs.cross())[0]["until"] == "junction")
    check('cross(line="crosswalk"): crosses on the crosswalk', cs._flat(cs.cross(line="crosswalk"))[0]["until"] == "crosswalk")

    with quiet():
        c = lf.CourseRunner([cs.to_end(), finish()])
        t = feed(c, 5, 0.0)
        t = feed_junc(c, 5, t, True)
    check("to_end: ends on a junction line (line across) too", c.idx == 1, f"step {c.idx + 1}")
    with quiet():
        c = lf.CourseRunner([cs.to_end(), finish()])
        t = feed(c, 5, 0.0)
        t = feed(c, 5, t, gap=int(H * 0.3))
    check("to_end: a gap far away does not end it", c.idx == 0)
    with quiet():
        t = feed(c, 5, t, gap=int(H * 0.6))
    check("to_end: ends on a lane gap that came below at (0.5)", c.idx == 1, f"step {c.idx + 1}")

    fast, slow = cs._flat(cs.tr(speed=0.7))[0]["blind"], cs._flat(cs.tr(speed=0.3))[0]["blind"]
    check("turn blind: shorter when faster, unchanged at or below the base speed", fast < cs.TURN_BLIND and slow == cs.TURN_BLIND,
          f"0.7 -> {fast}s, 0.3 -> {slow}s")


def test_lost_and_far():
    print("[back-up after a lost lane / turn angle / far junction line]")
    import course as cs
    blank = np.full((H, W, 3), 30, np.uint8)
    with quiet():
        d = lf.Driver([follow(until="junction", max_sec=99), finish()])
        for i in range(10):
            d.tick(lane_frame(), i / 30)
        cmds = [(i / 30, d.tick(blank, i / 30)[1]) for i in range(10, 160)]
    back = [t for t, c in cmds if c < 0]
    check("backs up after LOST_HOLD when the lane is lost", back and abs(back[0] - (10 / 30 + lf.LOST_HOLD)) < 0.1,
          f"back-up starts at {back[0] if back else None}")
    check("backs up only for LOST_BACK_SEC, then stops", back and back[-1] - back[0] <= lf.LOST_BACK_SEC + 0.05
          and cmds[-1][1] == 0, f"{back[-1] - back[0] if back else 0:.2f}s")
    with quiet():
        for i in range(160, 170):
            d.tick(lane_frame(), i / 30)
        second = [d.tick(blank, i / 30)[1] for i in range(170, 300)]
        for i in range(300, 310):
            d.tick(lane_frame(), i / 30)
        third = [d.tick(blank, i / 30)[1] for i in range(310, 440)]
    check(f"backs up again when the lane is lost again (up to {lf.LOST_BACK_TRIES} per step)", min(second) < 0 and min(third) >= 0,
          f"2nd {min(second):+.2f}, 3rd {min(third):+.2f}")

    with quiet():   # Retrace: drive forward steering +15, then lose the lane. Backing up replays +15.
        d = lf.Driver([follow(until="junction", max_sec=99), finish()])
        for i in range(30):
            d.tick(lane_frame(shift=60), i / 30)
        hist = [a for _, a, _ in d.history[-10:]]
        steers = [d.tick(blank, i / 30) for i in range(30, 120)]
    back = [st for st, cmd, _ in steers if cmd < 0 and st is not None]
    check("backing up replays the steering just used, in reverse", back and abs(back[0] - hist[-1]) < 3, f"before {hist[-1]:+.1f} / back-up {back[0] if back else None}")

    ym = lf.YawMeter()
    room = np.random.default_rng(3).integers(0, 255, (H, W + 80, 3), np.uint8)
    room = cv2.GaussianBlur(room, (9, 9), 0)
    for k in range(9):      # The view slides 10 px left per frame, as when turning right.
        total = ym(np.ascontiguousarray(room[:, 10 * k:10 * k + W]))
    check("camera turn measured: as far as the view slid (right turn = negative)", abs(total + 80) < 6, f"{total:+.1f}px (true -80)")
    with quiet():
        first = cs._flat(cs.tr())[0]
        c = lf.CourseRunner([cs.tr(sec=0), finish()])
        t = feed(c, int(max(first["min"], first["blind"]) / 0.05) + 2, 0.0, det=None)     # Past the blind period and min_sec.
        c.turned = lf.TURN_YAW_PX + 1
        feed(c, 1, t, det=None)
    check("turn: ends after 90 deg (TURN_YAW_PX) even with no new lane in view", c.idx == 1, f"step {c.idx + 1}")
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
    check("tr(min_sec=2.5): keeps turning until 2.5 s even after 90 deg with a new lane in view", t_short < 2.0 and 2.5 <= t_long < 2.8,
          f"min_sec 0.5 -> {t_short:.2f}s, 2.5 -> {t_long:.2f}s")
    long_turn = cs._flat(cs.tr(4))[0]
    check("tr(4): turns at least 4 s, the max time grows by itself", long_turn["min"] == 4 and long_turn["max"] >= 5,
          f"min {long_turn['min']}, max {long_turn['max']}")

    far = lane_frame()
    cv2.line(far, (W // 2 - 110, 175), (W // 2 + 110, 179), (235, 235, 235), 3)    # Thin, slightly slanted line ahead.
    with quiet():
        d = lf.Driver([finish()])
        d.tick(far, 0.0)
    y = d.detect.far_hline(far)
    check("a thin, slanted far line across inside the ROI is caught for slowing down", y is not None and 170 <= y <= 190, y)


def test_stripes_and_crossing():
    print("[crosswalk stripes / crossing / steering while stopped]")
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
    check("3 short stripes = crosswalk", len(d1.stripes) >= lf.STRIPES_MIN and d1.crosswalk_any, f"{len(d1.stripes)}")
    check("stripes are left out of the lane (same lane result as without them)",
          det0 is not None and det1 is not None and abs(det0[0] - det1[0]) < 0.05 and abs(det0[1] - det1[1]) < 0.1,
          f"{det0} / {det1}")
    check("no crosswalk on a lane without stripes", not d0.stripes and not d0.crosswalk_any)
    def stripes_at(xs_frac, ys=(230, 270, 310)):
        img = lane_frame()
        for xf, y in zip(xs_frac, ys):
            l, r = lane_x(y)
            w = int(0.12 * (r - l))
            x = int(l + (r - l) * xf)
            cv2.rectangle(img, (x, y - w // 4), (x + w, y + w // 4), (235, 235, 235), -1)
        return run_detector(img)[0]
    two = stripes_at((0.3, 0.5))
    check(f"2 stripes are not a crosswalk ({lf.STRIPES_MIN} or more needed)", not two.stripes and not two.crosswalk_any, len(two.stripes))
    out = stripes_at((-0.6, -0.4, 1.3))     # Three stripes, but outside the lane lines.
    check("3 stripes outside the lane are not a crosswalk", not out.stripes, len(out.stripes))
    far_only = np.full((H, W, 3), 30, np.uint8)
    top = int(H * lf.ROI_TOP)
    cv2.line(far_only, (0, top + 8), (40, top + 14), (235, 235, 235), 6)    # Something at the far top-left edge, no lane.
    _, det_far = run_detector(far_only)
    check("a single dot at the far end is not a lane (lane lost)", det_far is None, det_far)
    right_only = lane_frame(left=False)
    with_obj = right_only.copy()
    top = int(H * lf.ROI_TOP)
    cv2.rectangle(with_obj, (50, top + 25), (120, top + 60), (235, 235, 235), -1)   # A short object off the track, far left.
    d_o, _ = run_detector(with_obj, n=1)
    check("a short thing off the track (far edge) is left out of the lane candidates", all(x is None or x > W * 0.4 for x in d_o.left), d_o.left)
    _, det_r = run_detector(right_only, n=1)
    old_side = lf.SPECK_SIDE
    lf.SPECK_SIDE = 0.0         # Without that removal: no previous position (just after a turn), so the longer line must win.
    try:
        _, det_o = run_detector(with_obj, n=1)
    finally:
        lf.SPECK_SIDE = old_side
    check("width does not fit and no previous position: the long line (the real lane) is kept", det_r is not None and det_o is not None
          and abs(det_r[0] - det_o[0]) < 0.1, f"{det_r} / {det_o}")

    steps = cs._flat(cs.cross(line="crosswalk", approach=False))
    check('cross(line="crosswalk", approach=False): crosses at once with no approach', [x["until"] for x in steps] == ["crossed"])
    with quiet():
        c = lf.CourseRunner(steps + [finish()])
        t = 0.0
        for _ in range(20):     # Crosswalk under the car.
            c.update(False, t, (0.0, 0.0), True, {"crosswalk": 300}); t += 0.05
        mid = c.idx
        for _ in range(12):     # Gone from view.
            c.update(False, t, (0.0, 0.0), False, {}); t += 0.05
    check("crossing: goes on while the crosswalk is seen, ends once past", mid == 0 and c.idx == 1, f"{mid} → {c.idx}")

    with quiet():
        d = lf.Driver([stop(5), finish()])
        a = [d.tick(lane_frame(shift=sh), i / 30)[0] for i, sh in enumerate([0] * 5 + [80] * 30)]
    a = [x for x in a if x is not None]
    check("while stopped, steering stays even if the lane moves sideways", max(a) - min(a) < 0.5, f"{min(a):+.1f}..{max(a):+.1f}")

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
    check("align: with a line across in view, it does not steer into a large heading error (less than without the line)",
          a_line is not None and a_free is not None and abs(a_line) <= lf.ALIGN_STEER_MAX + 0.5 and abs(a_free) > abs(a_line),
          f"no line {a_free}, line {a_line}")


def test_driver_smoke():
    print("[Driver, one full run]")
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
    check("COURSE for 60 frames of tick + drawing (no errors, values in range)", ok)
    import course as cs
    events = [e for e in cs.EVENTS if e]
    missing = [e for e in events if e not in lf.Driver.EVENT_EN or e not in lf.CourseRunner.EVENT_NAMES
               or e not in lf.CourseRunner([finish()]).hits]
    check("every until in course.py has a view/log name and a counter", not missing, missing)
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
    check("drawing: no error when the next/current step crosses a crosswalk", ok)
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
    check("drawing: no error when the next step is a turn at its default angle (right(), uturn)", ok)
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
    check("no error on the first frame when starting (--start) at a turn", ok)


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


real_cut_fn = PersonDetector._marked_cutout


def _cut_top(figure):
    """A figure mask with its top third taken away."""
    rows = np.nonzero(figure.any(axis=1))[0]
    out = figure.copy()
    out[:rows[0] + len(rows) // 3] = False
    return out


def test_person_detect():
    print("\n[LEGO figure detection]")
    det, (ok, msg) = registered_figure()
    check("register: registered from the background and figure photos", ok and len(det.views) == 1, msg)
    empty = lane_frame()
    check("register: refused with no figure", not registered_figure()[0].add_view(empty)[0])
    check("detect: none on a lane with no figure", det(empty) is None)
    found = det(minifig(empty, 320, 240))
    check("detect: found where it was registered", found is not None and abs(found["feet"] - 240) <= 4, found)
    check("detect: found at several distances and positions", all(det(minifig(empty, x, f)) for x in (220, 320, 420) for f in (190, 240, 300)))
    far = det(minifig(empty, 320, 240), (0.7, 0.9))
    check("detect: ignored outside the set horizontal range", far is None)
    boxes = empty.copy()
    cv2.rectangle(boxes, (300, 180), (340, 240), (40, 40, 200), -1)
    check("false: one box of the same color is not a figure", det(boxes) is None)
    big = minifig(empty, 320, 240)
    cv2.rectangle(big, (100, 60), (520, 300), (150, 70, 30), -1)
    check("false: a large object of the wrong size is not a figure", det(big) is None)
    lamp = empty.copy()
    cv2.circle(lamp, (80, 60), 8, (0, 0, 255), -1)
    check("false: none on a crosswalk / red light view", det(lane_frame(crosswalk=True)) is None and det(lamp) is None)

    black = {"torso": (20, 20, 20), "legs": (20, 20, 20)}
    bright = lane_frame(line=20, bg=200)
    dark = registered_figure(back=bright, **black)[0]
    found = dark(minifig(bright, 320, 240, **black))
    check("black clothes, white floor: found by the head, same thresholds as registered", found is not None and abs(found["feet"] / lf.FRAME_H - dark.stop_at(lf.FRAME_H)) < 0.03, found)
    check("black clothes, white floor: black lane lines are not the figure", dark(bright) is None)
    dark = registered_figure(**black)[0]
    found = dark(minifig(empty, 320, 240, **black))
    check("black clothes, black floor: found with the same thresholds as registered", found is not None and abs(found["feet"] / lf.FRAME_H - dark.stop_at(lf.FRAME_H)) < 0.03, found)
    white, (_, msg) = registered_figure(back=lane_frame(line=20, bg=120), torso=(235, 235, 235), legs=(235, 235, 235), head=(235, 235, 235))
    check("almost white/black figure: warned at registration", "warning" in msg, msg)

    print("  -- near figures always, far ones later")
    near_y = 0.55 * lf.FRAME_H
    check("distance: a figure beyond the line is not looked at", det(minifig(empty, 320, 190), (0.0, 1.0), near_y) is None)
    check("distance: a figure past the line is found", det(minifig(empty, 320, 205), (0.0, 1.0), near_y) is not None)
    check("distance: a figure cut off at either frame side is found", all(det(minifig(empty, x, 300), (0.0, 1.0), near_y) for x in (5, 25, 615, 635)))
    check("distance: a figure with its legs cut off by the frame bottom is found", det(minifig(empty, 320, 400), (0.0, 1.0), near_y) is not None)
    check("distance: found even when much larger than at registration", all(det(minifig(empty, 320, f), (0.0, 1.0), near_y) for f in (280, 320, 350)))
    grown = empty.copy()
    grown_fig = cv2.resize(minifig(empty, 320, 240)[150:244, 290:350], None, fx=2.0, fy=1.0, interpolation=cv2.INTER_NEAREST)
    grown[220:220 + grown_fig.shape[0], 260:260 + grown_fig.shape[1]] = grown_fig
    check("distance: found by color even when outline and size do not fit", det(grown) is None and det(grown, (0.0, 1.0), near_y, photos=False) is not None)
    check("distance: none on a near lane with no figure", det(empty, (0.0, 1.0), near_y) is None and det(lane_frame(crosswalk=True), (0.0, 1.0), near_y) is None)
    check("distance: a near figure in other clothes is not it", det(minifig(empty, 320, 300, torso=(60, 150, 50)), (0.0, 1.0), near_y) is None)
    block = minifig(empty, 320, 240)[150:244, 290:350]
    huge = empty.copy()
    huge[210:350, 40:600] = cv2.resize(block, (560, 140), interpolation=cv2.INTER_NEAREST)
    check("distance: a blob of the same color much larger than a figure is not it", det(huge, (0.0, 1.0), near_y) is None)

    print("  -- registration with no background photo")
    tall = int(0.42 * (240 - lf.HORIZON_Y))     # Height minifig() draws at feet=240.
    behind = lane_frame()       # Run on the Pi: a same-colored patch behind the figure was joined to it, and the registered height was 108 px instead of 67.
    cv2.rectangle(behind, (305, 240 - tall - 35), (335, 240 - tall - 3), (150, 70, 30), -1)
    joined = registered_figure(back=behind)[0]
    check("register: measures only the figure even with the same color behind it", joined.views[0]["h"] <= tall + 6, (joined.views[0]["h"], tall))
    arm = empty.copy()      # Run on the Pi: with a background shot, an arm that moved was the largest change, and all 8 views registered the arm.
    cv2.rectangle(arm, (0, 150), (260, 230), (120, 160, 210), -1)
    armed = PersonDetector(lf.HORIZON_Y)
    added, msg = armed.add_view(minifig(arm, 400, 240))
    check("register: registers the figure, not an arm in view", added and armed.last_box[0] > 340, msg)
    check("register: after registering, the arm is not found and the figure is", armed(arm) is None and armed(minifig(empty, 400, 240)) is not None)
    check("register: refused with an arm and no figure", not PersonDetector(lf.HORIZON_Y).add_view(arm)[0])
    shifted = np.roll(minifig(empty, 320, 240), 6, axis=1)
    moved = PersonDetector(lf.HORIZON_Y)
    check("register: registered even if the car moved a little (no background comparison)", moved.add_view(shifted)[0])
    outside = PersonDetector(lf.HORIZON_Y)
    check("register: a figure outside the box is refused", not outside.add_view(minifig(empty, 80, 240))[0])
    cut = PersonDetector(lf.HORIZON_Y)
    check("register: a figure touching the box edge is refused", not cut.add_view(minifig(empty, 196, 250))[0])
    flat = empty.copy()     # Run on the Pi: 4 views of a wide dull area in the room were accepted, and nothing could ever be found with them.
    cv2.rectangle(flat, (210, 170), (340, 200), (70, 80, 90), -1)
    wide_det = PersonDetector(lf.HORIZON_Y)
    check("register: a wide object is not taken as a figure", not wide_det.add_view(flat)[0] and not wide_det.views)
    added, msg = wide_det.add_view(minifig(flat, 400, 300))
    check("register: picks the figure when a wide object is there too", added and wide_det.last_box[0] > 340, (msg, wide_det.last_box))
    painted = empty.copy()
    cv2.rectangle(painted, (0, 0), (639, 215), (40, 40, 200), -1)
    tone = PersonDetector(lf.HORIZON_Y)     # The figure's color fills the room too.
    solid_red = {"torso": (40, 40, 200), "legs": (40, 40, 200), "head": (40, 40, 200)}
    check("register: refused when the figure is the color of its surroundings", not tone.add_view(minifig(painted, 320, 330, **solid_red))[0] and not tone.views)

    # Run on the Pi: a black mat under warm light has the hue of yellow, so yellow legs counted as floor and only the shirt was left, too small to ever be found.
    warm = np.clip(lane_frame(bg=170) * np.array([0.78, 0.93, 1.0]), 0, 255).astype(np.uint8)
    yellow = {"torso": (210, 180, 40), "legs": (10, 215, 245), "head": (40, 80, 150)}
    warm_det = PersonDetector(lf.HORIZON_Y)
    warm_fig = minifig(warm, 320, 300, **yellow)
    warm_ok = warm_det.add_view(warm_fig)[0]
    check("register: yellow legs on a black mat under warm light are figure color", warm_ok and warm_det.views[0]["h"] >= 0.8 * 0.42 * (300 - lf.HORIZON_Y))
    check("detect: a figure with yellow legs on a black mat under warm light", warm_ok and warm_det(warm_fig) is not None)
    limit = person.MIN_H
    person.MIN_H = 1000
    too_small = PersonDetector(lf.HORIZON_Y)
    check("register: refused when found smaller than the minimum size", not too_small.add_view(warm_fig)[0] and not too_small.views)
    person.MIN_H = limit
    column = np.zeros((24, 12), bool)
    column[4:20, 3:9] = True
    check("outline: one row off up or down is the same outline", PersonDetector._overlap(column, np.roll(column, 1, axis=0)) == 1.0)
    check("outline: a full box differs from a figure outline", PersonDetector._overlap(column, np.ones((24, 12), bool)) < 0.4)

    print("  -- same head, other clothes")
    navy = {"torso": (110, 50, 30), "legs": (30, 60, 150), "head": (40, 90, 170)}
    other = {"torso": (210, 190, 60), "legs": (10, 215, 245), "head": (40, 90, 170)}
    navy_fig = minifig(warm, 320, 300, **navy)
    navy_det = PersonDetector(lf.HORIZON_Y)
    navy_ok = navy_det.add_view(navy_fig)[0]
    check("dark navy clothes are figure color, not colorless", navy_ok and navy_det.table[10:14].any())
    check("navy figure found; another figure with only the same head is not", navy_ok and navy_det(navy_fig) is not None and navy_det(minifig(warm, 320, 300, **other)) is None)
    tint = np.full((1, 1, 3), (20, 58, 170), np.uint8)   # HSV of the real mat.
    check("the yellowish tint of the mat is not a color", person.color_index(tint)[0, 0] >= person.HUE_BINS)

    cool = np.full((1, 1, 3), (98, 55, 55), np.uint8)   # HSV of the real mat under a cool white balance: the hue of a sky blue shirt.
    check("the bluish tint of the mat is not a color either", person.color_index(cool)[0, 0] >= person.HUE_BINS)

    print("  -- automatic thresholds")
    tuned = PersonDetector(lf.HORIZON_Y)
    tuned.add_view(navy_fig)
    box_score = PersonDetector._overlap(np.array(tuned.views[0]["outline"], dtype=bool), np.ones((24, 12), bool))
    check("auto: registering sets the thresholds within range", person.SHAPE_MIN <= tuned.shape <= person.TUNE_SHAPE_MAX and person.MATCH_MIN <= tuned.match <= person.TUNE_MATCH_MAX, (tuned.match, tuned.shape))
    check("auto: higher than a box of the same size", tuned.shape > box_score, (tuned.shape, round(box_score, 2)))
    check("auto: a fake box does not pass", box_score < tuned.shape and tuned(minifig(warm, 320, 300, **navy)) is not None)
    before = tuned.shape, tuned.match
    tuned.add_view(navy_fig)
    tuned.undo()
    check("auto: undoing the last view brings the thresholds back", (tuned.shape, tuned.match) == before)
    keys_left = tune_person.HELP
    check("auto: no help for manual threshold keys", "+ / -" not in keys_left)
    check("no view limit (None), auto capture neither", person.MAX_VIEWS is None and person.AUTO_MAX is None)

    print("  -- keeping bad registrations out")
    old_max = person.MAX_VIEWS
    person.MAX_VIEWS = 64       # A limit, when one is set, still holds.
    full = PersonDetector(lf.HORIZON_Y)
    full.views = [{}] * person.MAX_VIEWS
    check("register: refused past a set limit", not full.add_view(navy_fig)[0] and len(full.views) == person.MAX_VIEWS)
    person.MAX_VIEWS = old_max
    loose_path = Path(tempfile.mkdtemp()) / "person.json"
    navy_det.match, navy_det.shape = 0.1, 0.3
    navy_det.save(loose_path)
    loose = PersonDetector.load(lf.HORIZON_Y, loose_path)
    check("save: a saved loose threshold is ignored and set automatically", loose.match >= person.MATCH_MIN and loose.shape >= person.SHAPE_MIN and loose.shape > 0.3, (loose.match, loose.shape))
    navy_det.match, navy_det.shape = 0.95, 0.95
    navy_det.save(loose_path)
    strict = PersonDetector.load(lf.HORIZON_Y, loose_path)
    check("save: a saved strict threshold is ignored and set automatically too", strict.match <= person.TUNE_MATCH_MAX and strict.shape <= person.TUNE_SHAPE_MAX, (strict.match, strict.shape))
    navy_det.match, navy_det.shape = person.MATCH, person.SHAPE_MATCH
    data = json.loads(loose_path.read_text())
    del data["rule"]
    loose_path.write_text(json.dumps(data))
    check("save: an old registration counted another way must be registered again", PersonDetector.load(lf.HORIZON_Y, loose_path) is None)

    print("  -- dragged box")
    fw, fh = int(0.42 * (300 - lf.HORIZON_Y) * 0.62), int(0.42 * (300 - lf.HORIZON_Y))
    tight = ((320 - fw) / W, (300 - fh - 6) / H, (320 + fw) / W, 306 / H)
    drag_det = PersonDetector(lf.HORIZON_Y)
    drag_det.box, drag_det.marked = tight, True
    drag_ok = drag_det.add_view(warm_fig)
    check("drag: registers what is inside the box as the figure", drag_ok[0] and drag_det(warm_fig) is not None, drag_ok[1])
    y, h = drag_det.last_box[1], drag_det.last_box[3]
    check("drag: the whole figure is cut out", abs(h - fh) <= 8 and abs(y + h - 300) <= 8, drag_det.last_box)
    roomy = PersonDetector(lf.HORIZON_Y)
    roomy.box, roomy.marked = tight, True
    roomy._marked_cutout = lambda small, box: _cut_top(real_cut_fn(small, box))    # GrabCut leaves the top of the figure out.
    roomy_ok = roomy.add_view(warm_fig)[0]
    check("drag: registered at the whole figure size in the box even if the cutout misses the head", roomy_ok and roomy.views[0]["h"] >= 0.8 * fh, roomy.views and roomy.views[0]["h"])
    filled = PersonDetector(lf.HORIZON_Y)
    filled.box, filled.marked = (320 / W - fw / 2 / W, (300 - fh) / H, 320 / W + fw / 2 / W, 300 / H), True
    check("drag: accepted even touching the box edge", filled.add_view(warm_fig)[0])
    shadow = warm_fig.copy()
    cv2.ellipse(shadow, (320, 304), (14, 8), 0, 0, 360, (70, 82, 90), -1)
    shadow = minifig(shadow, 320, 300, **yellow)
    shaded = PersonDetector(lf.HORIZON_Y)
    shaded.box, shaded.marked = ((320 - fw) / W, (300 - fh - 6) / H, (320 + fw) / W, 320 / H), True
    check("drag: the shadow under the feet is not the feet", shaded.add_view(shadow)[0] and abs(shaded.views[0]["feet"] - 300) <= 4, shaded.views and shaded.views[0]["feet"])
    black_legs = PersonDetector(lf.HORIZON_Y)
    black_legs.box, black_legs.marked = tight, True
    black_ok = black_legs.add_view(minifig(warm, 320, 300, torso=(210, 180, 40), legs=(25, 25, 25), head=(40, 80, 150)))[0]
    check("drag: black legs are not cut off", black_ok and abs(black_legs.views[0]["feet"] - 300) <= 6, black_legs.views and black_legs.views[0]["feet"])
    check("drag: not on by default", not PersonDetector(lf.HORIZON_Y).marked)

    print("  -- auto capture")
    side = {"torso": (40, 40, 200), "legs": (40, 40, 200), "head": (60, 210, 240)}
    auto_det = PersonDetector(lf.HORIZON_Y)
    auto = AutoViews(auto_det)

    def still(img, t0, sec=1.0):
        notes = [auto.update(img, t0 + i / 30) for i in range(int(sec * 30))]
        return t0 + sec, [n for n in notes if n]

    tt, notes = still(minifig(empty, 320, 240), 0.0)
    check("auto: takes a view when still", len(auto_det.views) == 1 and len(notes) == 1, notes)
    tt, notes = still(minifig(empty, 320, 240), tt, 3.0)
    check("auto: takes no more while it stays still", len(auto_det.views) == 1 and not notes, notes)
    hand = minifig(empty, 320, 240, **side)
    cv2.rectangle(hand, (100, 150), (260, 230), (120, 160, 210), -1)
    tt, notes = still(hand, tt)
    check("auto: with a hand in view, no view and a note", len(auto_det.views) == 1 and "hand" in notes[0], notes)
    tt, notes = still(minifig(empty, 320, 240, **side), tt)
    check("auto: takes a new side once the hand is away", len(auto_det.views) == 2, notes)
    tt, notes = still(hand, tt, 0.2)
    tt, notes = still(minifig(empty, 320, 240, **side), tt)
    check("auto: skips a side already taken", len(auto_det.views) == 2 and "already" in notes[0], notes)
    wiggle = []
    for i in range(30):
        wiggle.append(auto.update(minifig(empty, 300 + i * 2, 240, torso=(60, 150, 50)), tt + i / 30))
    check("auto: takes nothing while the figure is being turned", len(auto_det.views) == 2 and not any(wiggle), wiggle)

    far_det = PersonDetector(lf.HORIZON_Y)
    far_auto = AutoViews(far_det)
    busy = []
    for i in range(45):     # Someone moving far behind, outside the box.
        scene = minifig(empty, 320, 240)
        cv2.rectangle(scene, (10 + 3 * i, 10), (120 + 3 * i, 120), (200, 200, 200), -1)
        busy.append(far_auto.update(scene, i / 30))
    check("auto: takes a view even with movement outside the box", len(far_det.views) == 1, [n for n in busy if n])

    print("  -- registration controls")
    start = (0.3, 0.42, 0.7, 0.97)
    moved_box = tune_person.move_box(start, 1, 0)
    check("box: d moves right, size unchanged", moved_box[0] > start[0] and abs((moved_box[2] - moved_box[0]) - 0.4) < 1e-9, moved_box)
    check("box: does not leave the frame", tune_person.move_box((0.0, 0.0, 0.4, 0.5), -1, -1) == (0.0, 0.0, 0.4, 0.5)
          and tune_person.move_box((0.6, 0.5, 1.0, 1.0), 1, 1) == (0.6, 0.5, 1.0, 1.0))
    bigger = tune_person.resize_box(start, 1)
    check("box: x larger, z smaller", bigger[2] - bigger[0] > 0.4 and tune_person.resize_box(start, -1)[2] - tune_person.resize_box(start, -1)[0] < 0.4)
    check("box: does not shrink too small", tune_person.resize_box((0.4, 0.4, 0.55, 0.55), -1) == (0.4, 0.4, 0.55, 0.55))
    read_end, write_end = os.pipe()
    os.write(write_end, "wㅈA\n".encode())
    check("keys: Korean input mode and capitals are the same key", tune_person.read_keys(read_end) == "wwa\n")
    check("keys: no wait when no key is pressed", tune_person.read_keys(read_end) == "")
    os.close(read_end)
    os.close(write_end)
    box_det = PersonDetector(lf.HORIZON_Y)
    box_det.box = tune_person.move_box(tune_person.move_box(start, -1, 0), -1, 0)
    for _ in range(3):
        box_det.box = tune_person.move_box(box_det.box, -1, 0)
    check("box: registers the figure in the moved box", box_det.add_view(minifig(empty, 200, 280))[0], box_det.box)
    box_auto = AutoViews(box_det)
    box_auto.ready, box_auto.around = False, np.zeros(1)
    box_auto.touch()
    check("box: once moved, ready to take again after it stops", box_auto.ready and box_auto.around is None)

    print("  -- fake objects")
    solid = {"torso": (150, 70, 30), "legs": (150, 70, 30), "head": (150, 70, 30)}
    only, _ = registered_figure(**solid)    # A figure of one color: color alone cannot tell it from a box.
    check("fake: the real one (single-color figure) is found at several distances", all(only(minifig(empty, 320, f, **solid)) for f in (200, 240, 280)))
    width = int(tall * 0.62)
    box = empty.copy()
    cv2.rectangle(box, (320 - width // 2, 240 - tall), (320 + width // 2, 240), solid["torso"], -1)
    check("fake: a box of the same color and size", only(box) is None)
    oval = empty.copy()
    cv2.ellipse(oval, (320, 240 - tall // 2), (width // 2, tall // 2), 0, 0, 360, solid["torso"], -1)
    check("fake: an ellipse of the same color and size", only(oval) is None)
    check("fake: same shape, other color", only(minifig(empty, 320, 240, torso=(60, 150, 50), legs=(60, 150, 50), head=(60, 150, 50))) is None)
    pale = (220, 180, 150)
    check("fake: same shape, faint copy of the same hue", only(minifig(empty, 320, 240, torso=pale, legs=pale, head=pale)) is None)
    check("fake: figure with only the clothes color changed", det(minifig(empty, 320, 240, torso=(60, 150, 50))) is None)
    check("fake: figure with top and bottom colors swapped", det(minifig(empty, 320, 240, torso=(150, 70, 30), legs=(40, 40, 200))) is None)
    shrunk = minifig(empty, 320, 240)[lf.HORIZON_Y:]    # Same outline at half size, standing at the same place on the floor.
    tiny = empty.copy()
    part = cv2.resize(shrunk, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)[(240 - lf.HORIZON_Y - tall) // 2 - 2:(240 - lf.HORIZON_Y) // 2 + 1, 140:180]
    tiny[240 - part.shape[0]:240, 300:340] = part
    check("fake: half-size copy at the same place", det(tiny) is None)
    check("unregistered: nothing is found before registering", PersonDetector(lf.HORIZON_Y)(minifig(empty, 320, 240)) is None)
    old = Path(tempfile.mkdtemp()) / "old.json"
    old.write_text(json.dumps({"views": [{"hist": [0.0] * 22, "w": 1, "h": 1, "feet": 1}], "background": [0.0] * 22}))
    check("an old-format saved file must be registered again", PersonDetector.load(lf.HORIZON_Y, old) is None)

    path = Path(tempfile.mkdtemp()) / "person.json"
    det.save(path)
    again = PersonDetector.load(lf.HORIZON_Y, path)
    check("save/load: found the same way with the same automatic thresholds", (again.match, again.shape) == (det.match, det.shape) and again(minifig(empty, 320, 240))["feet"] == found_feet(det))
    check("save/load: None when there is no file", PersonDetector.load(lf.HORIZON_Y, path.with_name("none.json")) is None)
    det.undo()
    check("undo: not found once the views are removed", not det.views and det(minifig(empty, 320, 240)) is None)


def found_feet(det):
    return det(minifig(lane_frame(), 320, 240))["feet"]


def person_driver(course, feet=240, **config):
    defaults = {"PERSON_AT": 0.55, "PERSON_RESUME_DELAY": 0.5,
                "PERSON_MIN_STOP": 0.0, "PERSON_MAX_WAIT": None, "PERSON_X": (0.1, 0.9), "PERSON_IN_LANE": False}
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


def test_person_near_decoys():
    print("[figure: near false detections]")
    from person import NEAR_SHAPE
    back = lane_frame()
    det = registered_figure(back=back)[0]
    fig = minifig(back, 320, 300)
    found = det(fig, (0, 1), 198)
    check("near: a standing figure is found", found is not None, found and round(found["shape"], 2))
    x, y, w, h = found["x"], found["y"], found["w"], found["h"]
    patch = fig[y:y + h, x:x + w]
    lying = back.copy()
    lying[H - w - 4:H - 4, 270:270 + h] = np.rot90(patch)
    found = det(lying, (0, 1), 198)
    check("near: a figure lying down is found too", found is not None, found and round(found["shape"], 2))
    # Same colors as the figure, stretched over shapes nothing like it: what the room shows in the screenshot.
    corner = back.copy()
    corner[H - 150:H, 0:90] = cv2.resize(patch, (90, 150), interpolation=cv2.INTER_NEAREST)
    found = det(corner, (0, 1), 198)
    check("near: a blob running to a frame corner is not a figure", found is None, found)
    thin = back.copy()      # An L of two thin bars.
    tall = cv2.resize(patch, (10, 100), interpolation=cv2.INTER_NEAREST)
    thin[230:330, 200:210] = tall
    thin[320:330, 200:300] = np.rot90(tall)
    found = det(thin, (0, 1), 198)
    check("near: a same-colored object of a wholly different shape is not a figure", found is None, found and (round(found["score"], 2), round(found["shape"], 2)))
    check("near: the shape threshold is below a real figure (0.41 and up)", NEAR_SHAPE < 0.41)


def test_person_look():
    print("[figure: look check (embedder)]")
    import person as P

    class Fake:
        """Numbers from a picture's mean color, so tests choose what looks alike."""
        calls = 0

        def __call__(self, bgr):
            Fake.calls += 1
            v = bgr.reshape(-1, 3).mean(axis=0).astype(np.float32) + 1
            return v / np.linalg.norm(v)

    old = P._embedder, P._embedder_tried
    P._embedder, P._embedder_tried = Fake(), True
    P.PHOTO_ON = False      # These are about the embedder, used when no photos are saved.
    try:
        back = lane_frame()
        det = registered_figure(back=back)[0]
        check("registering saves the look data", det.embeds is not None and det.embeds.shape == (1, 3), det.embeds is None or det.embeds.shape)
        fig = minifig(back, 320, 300)
        found = det(fig, (0, 1), 198)
        check("a look-alike candidate passes", found is not None and found["look"] >= P.EMBED_MIN, found and found["look"])
        # Two things in the figure's colors with a poor outline must not use up the looks before the figure's turn.
        crowd = fig.copy()
        for x0 in (40, 110):
            crowd[230:330, x0:x0 + 10] = cv2.resize(fig[found["y"]:found["y"] + found["h"], found["x"]:found["x"] + found["w"]], (10, 100), interpolation=cv2.INTER_NEAREST)
        Fake.calls = 0
        again = det(crowd, (0, 1), 198)
        check("the figure is checked first even with several fake candidates", again is not None and abs(again["x"] - found["x"]) < 10 and Fake.calls == 1, (again and again["x"], found["x"], Fake.calls))
        # A blob cut off by the side of the frame needs a closer likeness; the same look inside the frame is enough.
        y0, y1, x0, w = found["y"], found["y"] + found["h"], found["x"], found["w"]
        edge = back.copy()
        edge[y0:y1, :w - w // 3] = fig[y0:y1, x0 + w // 3:x0 + w]
        det.likeness = lambda *_: (P.EMBED_MIN + P.EMBED_MIN_EDGE) / 2
        check("an inner candidate passes with a middling likeness", det(fig, (0, 1), 198) is not None)
        check("a candidate cut off by the frame side is not a figure at a middling likeness", det(edge, (0, 1), 198) is None and any(n[0] <= 0 for n in det.near), det.near)
        det.likeness = lambda *_: P.EMBED_MIN_EDGE
        cut = det(edge, (0, 1), 198)
        check("a candidate cut off by the frame side is a figure when very alike", cut is not None and cut["x"] <= 0, cut)
        det.likeness = lambda *_: 1.0
        other = minifig(lane_frame(), 320, 300, torso=(40, 200, 40), legs=(40, 200, 40))
        check("a look-alike in clothes of a color not registered is not a figure", det(other, (0, 1), 198) is None and bool(det.near), det.near)
        check("the registered figure passes the foreign color check", det(fig, (0, 1), 198) is not None)
        del det.likeness
        det.embeds = -det.embeds        # Now nothing looks alike.
        Fake.calls = 0
        check("an unlike candidate is not a figure even with fitting color and outline", det(fig, (0, 1), 198) is None and bool(det.near))
        check("computed only when there are candidates, up to the set number", 1 <= Fake.calls <= P.EMBED_MAX, Fake.calls)
        Fake.calls = 0
        det(back, (0, 1), 198)
        check("not computed with no candidates", Fake.calls == 0, Fake.calls)
        det.fit()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "person.json"
            det.save(path)
            again = PersonDetector.load(lf.HORIZON_Y, path)
        check("look data kept after saving and loading", again is not None and again.embeds is not None and np.allclose(again.embeds, det.embeds, atol=1e-3))
        Fake.calls = 0
        wide = det.likeness(np.zeros((360, 640, 3), np.uint8) + 50, 100, 100, 80, 40)
        check("a wide candidate is also compared lying down", Fake.calls == 2, Fake.calls)
    finally:
        P._embedder, P._embedder_tried = old
        P.PHOTO_ON = True
    old = P._embedder, P._embedder_tried
    P._embedder, P._embedder_tried = None, True
    P.PHOTO_ON = False
    try:
        back = lane_frame()
        det = registered_figure(back=back)[0]
        found = det(minifig(back, 320, 300), (0, 1), 198)
        check("without the embedder it works on color and outline", det.embeds is None and found is not None and found["look"] is None)
    finally:
        P._embedder, P._embedder_tried = old
        P.PHOTO_ON = True
    try:
        real = P.Embedder()
        v = real(np.full((60, 40, 3), 90, np.uint8))
        check("real model: 1024 unit vectors", v.shape == (1024,) and abs(float(np.linalg.norm(v)) - 1) < 1e-3, v.shape)
        fig = minifig(lane_frame(), 320, 300)
        d = PersonDetector(lf.HORIZON_Y)
        d.add_view(fig)
        found = d(fig, (0, 1), 198)
        check("real model: the registered photo itself is almost the same", found is not None and found["look"] > 0.9, found and found["look"])
    except ImportError:
        print("  (real model skipped: no ai_edge_litert)")


def test_person_photos():
    import person as P
    print("\n[LEGO figure: registration photo comparison]")
    back = lane_frame()
    det = registered_figure(back=back)[0]
    check("registering keeps a photo of the box", len(det.photos) == len(det.views) and det.photo_sigs is not None)
    fig = minifig(lane_frame(), 320, 300)
    found = det(fig, (0, 1), 198)
    check("the registered figure matches its photos", found is not None and found["look"] >= P.PHOTO_MIN, found)
    other = minifig(lane_frame(), 320, 300, torso=(40, 200, 40), legs=(200, 200, 40))
    check("a figure with only the same head in other clothes does not match", det(other, (0, 1), 198) is None, det.found)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "person.json"
        det.save(path)
        check("saved as photo files", len(list((Path(tmp) / P.PHOTO_DIR).glob("*.jpg"))) == len(det.views))
        again = PersonDetector.load(lf.HORIZON_Y, path)
        check("compared by photo after loading too", again.photo_sigs is not None and again(fig, (0, 1), 198) is not None)
        (Path(tmp) / P.PHOTO_DIR / "000.jpg").unlink()
        old = PersonDetector.load(lf.HORIZON_Y, path)
        check("loads without error even if a photo file is gone", old is not None and old.photo_sigs is None)
    count = len(det.views)
    det.undo()
    check("undoing a view undoes its photo", len(det.photos) == len(det.views) == count - 1)
    det = registered_figure(back=back)[0]
    tiny = det.photos[0][:4, :4].copy()
    det.views.append(dict(det.views[0]))
    det.photos.append(tiny)
    before = len(det.photo_sigs)
    det.fit()
    check("a small photo holding only part of it is left out", len(det.photo_sigs) == before, (before, len(det.photo_sigs)))


def test_person_any():
    import person as P
    print("\n[LEGO figure: unregistered figures]")
    check("down to the feet, at the whole figure size, even when the colored part is small", P.figure_box(300, 100, 20, 40, 180) == (290, 100, 40, 80), P.figure_box(300, 100, 20, 40, 180))
    check("a lying figure keeps its box", P.figure_box(300, 100, 60, 30, 130) == (300, 100, 60, 30), P.figure_box(300, 100, 60, 30, 130))

    class Seen:
        near, found = [], {"x": 300, "y": 100, "w": 20, "h": 40, "feet": 180, "score": 0.9, "shape": 0.5, "look": 0.6}

    dbg = np.zeros((360, 640, 3), np.uint8)
    lf.draw_person(dbg, Seen, None)
    check("the PERSON box on screen is drawn down to the feet", dbg[180, 310].any() and not dbg[186, 310].any())

    class Fake:     # Looks alike only when the picture holds much of the figure's red.
        calls = 0
        def __call__(self, bgr):
            Fake.calls += 1
            red = float((bgr[..., 2].astype(int) > bgr[..., 0].astype(int) + 60).mean())
            return np.array([1.0, 0.0]) if red > 0.02 else np.array([0.0, 1.0])

    old = P._embedder, P._embedder_tried
    P._embedder, P._embedder_tried = Fake(), True
    try:
        weights = np.array([P.ANY_MIN + 1, 0, 0], dtype=np.float32)
        det = P.AnyFigure(lf.HORIZON_Y, weights)
        check("there is a stop position even with nothing registered", det.stop_at(360) == P.ANY_AT and det.views == [])
        check("finds an unregistered figure", det(minifig(lane_frame(), 320, 300), (0, 1), 198) is not None)
        Fake.calls = 0
        check("nothing colored: not found and not computed", det(lane_frame(), (0, 1), 198) is None and Fake.calls == 0, Fake.calls)
        check("a figure beyond the stop line is ignored", det(minifig(lane_frame(), 320, 150), (0, 1), 198) is None)
        check("ignored outside the horizontal range", det(minifig(lane_frame(), 320, 300), (0, 0.3), 198) is None)
        blue = minifig(lane_frame(), 320, 300, torso=(200, 40, 40), legs=(200, 40, 40), head=(200, 40, 40))
        check("left as a candidate only when it does not look like a figure", det(blue, (0, 1), 198) is None and bool(det.near), det.near)
        big = lane_frame()
        big[150:360, 100:540] = (40, 40, 220)
        check("an object much larger than a figure is not even a candidate", det.blobs(big, 198) == [], det.blobs(big, 198))
        cut = lane_frame()
        fig = minifig(lane_frame(), 40, 300)
        cut[:, :30] = fig[:, 30:60]
        det.weights = np.array([(P.ANY_MIN + P.ANY_MIN_EDGE) / 2, 0, 0], dtype=np.float32)
        check("stricter for a candidate cut off by the frame side", det(cut, (0, 1), 198) is None and det.near and det.near[0][0] <= 0, det.near)
        check("an inner candidate passes at the same score", det(minifig(lane_frame(), 320, 300), (0, 1), 198) is not None)
        reg = registered_figure(back=lane_frame())[0]
        both = P.AnyFigure(lf.HORIZON_Y, weights, reg)
        found = both(minifig(lane_frame(), 320, 300), (0, 1), 198)
        check("the registered figure is found first by its registration", found is not None and found["shape"] > 0 and both.views == reg.views, found)
        with tempfile.TemporaryDirectory() as tmp:
            check("only the registered figure without the weights file", P.AnyFigure.load(lf.HORIZON_Y, reg, Path(tmp) / "none.npy") is None)
        check("the shipped weights file fits the model", np.load(P.ANY_FILE).shape == (1025,), np.load(P.ANY_FILE).shape)
        d = person_driver([follow(2.0), finish()])
        d.person.detector, d.person.at = det, 0.55
        det.weights = weights
        t, cmds = run_frames(d, minifig(lane_frame(), 320, 300), 0.0, 1.0)
        check("stops for an unregistered figure while driving", cmds[-1] == 0 and d.person.hold, cmds[-1])
    finally:
        P._embedder, P._embedder_tried = old


def test_person_in_lane():
    print("\n[LEGO figure: between the lane lines only]")
    check("the course setting counts only between the lane lines", lf.PERSON_IN_LANE is True)
    frame = lane_frame()
    lane = lf.LaneDetector()
    check("no distinction when the lane is unknown", lane.lane_span(300) is None and lane.in_lane(10, 300))
    lane(frame, draw=False)
    span = lane.lane_span(300)
    check("knows the lane lines at each height", span is not None and 0 < span[0] < 320 < span[1] < 640, span)
    check("between the lines is in, outside is out", lane.in_lane(320, 300) and not lane.in_lane(span[0] - 40, 300) and not lane.in_lane(span[1] + 40, 300))
    check("judges feet below the frame bottom too", lane.lane_span(400) is not None and lane.lane_span(400)[0] < span[0])
    one = lf.LaneDetector()
    one(lane_frame(right=False), draw=False)
    check("with one line in view, the other side is one lane width away", one.lane_span(300) is not None and one.lane_span(300)[1] > 320, one.lane_span(300))
    d = person_driver([follow(30.0), finish()], feet=300, PERSON_IN_LANE=True)
    t, cmds = run_frames(d, minifig(lane_frame(), 320, 300), 0.0, 1.0)
    check("stops for a figure between the lane lines", cmds[-1] == 0 and d.person.hold, cmds[-1])
    d = person_driver([follow(30.0), finish()], feet=300, PERSON_IN_LANE=True, PERSON_X=(0.0, 1.0))
    left = d.detect.__class__()
    left(lane_frame(), draw=False)
    x = int(left.lane_span(300)[0] - 40)
    t, cmds = run_frames(d, minifig(lane_frame(), x, 300), 0.0, 1.0)
    check("does not stop for a figure outside the lane lines", not d.person.hold and cmds[-1] > 0, (x, cmds[-1]))


def test_speed_constant():
    print("\n[speed: slowing with steering]")
    frame = lane_frame()
    check("speed_for: set speed going straight, slower with more steering", lf.speed_for(0, 0.6) == 0.6 and lf.speed_for(lf.STEER_LIMIT / 2, 0.6) < 0.6
          and abs(lf.speed_for(lf.STEER_LIMIT, 0.6) - lf.MIN_SPEED) < 1e-9)
    check("speed_for: unchanged at or below the lowest speed", lf.speed_for(lf.STEER_LIMIT, 0.2) == 0.2)
    from course import to_end as cs_to_end
    d = lf.Driver([cs_to_end(speed=0.26, slow=0.18), finish()])
    d.detect.gap_y = 300
    with quiet():
        d.course.update(False, 0.0, (0.0, 0.0), False, {"near": True})
    check("slow=: does not speed up when driving below the minimum", d.approach_speed(0.26, 0.05) <= 0.26, d.approach_speed(0.26, 0.05))
    d = lf.Driver([follow(until="junction", speed=0.6, at=0.8, slow=0.3), finish()])
    d.detect.junction_y = 0.8 * lf.FRAME_H
    with quiet():
        d.course.update(False, 0.0, (0.0, 0.0), False, {"near": True})
    check("slow=: does not slow below APPROACH_MIN near a line", abs(d.approach_speed(0.6, 0.05) - lf.APPROACH_MIN) < 1e-9, d.approach_speed(0.6, 0.05))
    check("slow=: a speed already below APPROACH_MIN is kept", d.approach_speed(0.31, 0.05) == 0.31, d.approach_speed(0.31, 0.05))
    check("the lowest command holds even for a low set speed", lf.Driver([follow(30.0, speed=0.1), finish()]).tick(frame, 0.0)[1] >= lf.MOTOR_MIN - 1e-9)
    check("creeping at the lowest speed on a lost lane is kept", lf.MIN_SPEED == 0.3)


def test_end_watch():
    print("\n[measure log: end and stop positions]")
    import course as cs
    import tempfile
    w = lf.EndWatch()
    ruler = lf.EndRuler()
    ruler.speed = 10.0
    y_end, y_stop = 250.0, 262.0
    with quiet():
        w.ended(0, {"cm": 8}, ruler, y_end, 0.3, 1.0, (0.1, 0.0), 1.0, True)
        t = 1.0
        for _ in range(12):
            t += 0.05
            w.watch(y_stop, t)
    row = w.rows[0]
    check("slid after the end = end distance - stop distance", abs(row["slid_cm"] - (lf.floor_cm(y_end) - lf.floor_cm(y_stop))) < 1e-6 and w.open is None, row["slid_cm"])
    with quiet():
        w.ended(1, {"cm": None}, ruler, None, 0.3, 0.0, None, 10.0, True)
        w.watch(None, 13.5)
    check("ends as not read when the line is out of view", w.open is None and w.rows[1]["stop_z"] is None and w.rows[1]["how"] == "none")
    with tempfile.TemporaryDirectory() as folder:
        path = lf.Path(folder) / "ends.csv"
        w.save(path)
        lines = path.read_text().splitlines()
    check("ends.csv: a header and one row per action", len(lines) == 3 and lines[0] == w.HEADER and len(lines[1].split(",")) == len(lines[0].split(",")), lines)

    line = lane_frame()
    cv2.rectangle(line, (0, 225), (W, 235), (235,) * 3, -1)
    d = lf.Driver([cs.to_junction(cm=0, blind_sec=0), cs.stop(3), finish()])
    with quiet():
        t = 0.0
        for _ in range(5):
            d.tick(lane_frame(), t)
            t += 0.05
        for _ in range(40):
            d.tick(line, t)
            t += 0.05
    check("drive: logged when to_junction ends, and the stop position read during stop", len(d.ends.rows) == 1 and d.ends.rows[0]["stop_z"] is not None, d.ends.rows)


def test_person_stop():
    print("\n[LEGO figure: stop / go]")
    empty, person = lane_frame(), minifig(lane_frame(), 320, 240)
    far = minifig(lane_frame(), 320, 190)

    d = person_driver([follow(2.0), finish()])
    t, cmds = run_frames(d, empty, 0.0, 0.5)
    check("drives on with no figure", all(c > 0 for c in cmds))
    t, cmds = run_frames(d, person, t, 1.0)
    check("stops once found", cmds[-1] == 0 and d.person.hold and d.course.paused and not d.person.checking)
    check("status: PERSON STOP", d.course.status(t) == "PERSON STOP")
    t, cmds = run_frames(d, person, t, 5.0)
    check("the step time does not run down while standing (follow 2 s does not end)", d.course.idx == 0 and all(c == 0 for c in cmds))
    t, cmds = run_frames(d, empty, t, 0.3)
    check("stays during the delay after the figure is gone", all(c == 0 for c in cmds))
    t, cmds = run_frames(d, empty, t, 0.5)
    check("goes again after the delay (no jump to a large value)", cmds[-1] > 0 and not d.person.hold and max(cmds) < cmds[-1] + 0.02, (max(cmds), cmds[-1]))
    used = t - d.course.step_start
    check("the step time passes only by the time not stopped", 0.5 < used < 2.0, f"{used:.2f}s")
    t, _ = run_frames(d, empty, t, 2.0 - used + 0.2)
    check("fills the rest of the time, then the next step", d.done)

    d = person_driver([follow(30), finish()])
    t, cmds = run_frames(d, far, 0.0, 0.5)
    check("far away it is not a figure, and the car drives on", cmds[-1] > 0 and not d.person.hold and not d.person.seen)
    t, cmds = run_frames(d, minifig(lane_frame(), 320, 215), t, 0.5)
    check("stops at once once it comes past the line", cmds[-1] == 0 and d.person.hold)

    d = person_driver([follow(30), finish()], PERSON_X=(0.0, 1.0))
    t, cmds = run_frames(d, minifig(lane_frame(), 25, 300), 0.0, 0.5)
    check("stops near even at the frame side", cmds[-1] == 0 and d.person.hold)

    d = person_driver([follow(30), finish()], PERSON_RESUME_DELAY=1.0)
    t, _ = run_frames(d, empty, 0.0, 0.2)
    t, cmds = run_frames(d, person, t, 0.1)
    check("stops at once even if seen once", cmds[-1] == 0 and d.person.checking)
    t, cmds = run_frames(d, empty, t, 0.5)
    check("stands during the check time", all(c == 0 for c in cmds) and d.person.checking)
    t, cmds = run_frames(d, empty, t, 0.8)
    check("goes if not seen again within 1 s, as a misread", cmds[-1] > 0 and not d.person.hold)

    d = person_driver([follow(30), finish()], PERSON_RESUME_DELAY=1.0)
    t, _ = run_frames(d, person, 0.0, 0.1)
    for _ in range(3):      # A few short sightings with gaps, none of them three in a row.
        t, _ = run_frames(d, empty, t, 0.2)
        t, _ = run_frames(d, person, t, 0.07)
    check("seen several times within 1 s, even with gaps, is real", d.person.hold and not d.person.checking)
    waited = []
    for _ in range(6):      # Keeps flickering, with gaps shorter than the resume delay.
        t, cmds = run_frames(d, empty, t, 0.7)
        waited += cmds
        t, cmds = run_frames(d, person, t, 0.07)
        waited += cmds
    check("keeps waiting while it is seen now and then", all(c == 0 for c in waited) and d.person.hold)
    t, cmds = run_frames(d, empty, t, 0.8)
    check("stands before the wait time even when not seen at all", all(c == 0 for c in cmds))
    t, cmds = run_frames(d, empty, t, 0.5)
    check("goes when not seen for the wait time", cmds[-1] > 0 and not d.person.hold)

    d = person_driver([follow(9), finish()], PERSON_MIN_STOP=1.5, PERSON_RESUME_DELAY=0.0)
    t, _ = run_frames(d, person, 0.0, 0.3)
    t, cmds = run_frames(d, empty, t, 1.0)
    check("PERSON_MIN_STOP: stands the minimum time even if gone at once", cmds[-1] == 0)
    t, cmds = run_frames(d, empty, t, 1.0)
    check("PERSON_MIN_STOP: goes after it", cmds[-1] > 0)

    d = person_driver([follow(30), finish()], PERSON_MAX_WAIT=1.0)
    t, cmds = run_frames(d, person, 0.0, 0.9)
    check("PERSON_MAX_WAIT: stands before it", cmds[-1] == 0)
    t, cmds = run_frames(d, person, t, 1.0)
    check("PERSON_MAX_WAIT: goes after it even with the figure there, and does not stop again at once", cmds[-1] > 0 and not d.person.hold)
    t, _ = run_frames(d, empty, t, 0.7)      # Longer than the resume delay: it counts as gone only after that.
    t, cmds = run_frames(d, person, t, 0.5)
    check("PERSON_MAX_WAIT: stops again when it goes and comes back", cmds[-1] == 0)

    d = person_driver([stop(0.5), follow(9), finish()])
    t, cmds = run_frames(d, person, 0.0, 1.0)
    check("stop step: time runs as usual, then the follow stops for the figure", d.course.idx == 1 and cmds[-1] == 0)
    d = person_driver([straight(2.0), finish()])
    t, cmds = run_frames(d, person, 0.0, 3.0)
    check("stops in a straight step too, and the time does not run down", d.course.idx == 0 and cmds[-1] == 0)
    d = person_driver([cs_tr(), finish()])
    run_frames(d, empty, 0.0, 0.1)
    t, cmds = run_frames(d, person, 0.1, 1.5)
    check("does not look for the figure in a turn step", not d.course.paused and cmds[-1] > 0)
    d = person_driver([stop(0.2), finish()])
    run_frames(d, person, 0.0, 1.0)
    check("the final finish ends even with a figure there", d.done)

    d = lf.Driver([follow(0.5), finish()])
    t, cmds = run_frames(d, person, 0.0, 0.7)
    check("with nothing registered it drives without looking", d.person is None and d.done)
    d = person_driver([follow(9), finish()])
    d.lost_since, d.back = 0.0, {"start": 0.0, "plan": [(1.0, 0.0)], "len": 1.0, "used_at": 0, "idx": 0}
    run_frames(d, person, 0.0, 0.5)
    check("stops backing up when a figure appears", d.back is None and d.speed == 0)


def cs_tr():
    import course as cs
    return cs.tr()


def test_set_speed():
    print("\n[course speed set_speed]")
    import course as cs
    flat = cs._flat([cs.to_junction(), cs.set_speed(0.33), cs.to_junction(), cs.follow(2), cs.to_end(), cs.to_crosswalk(),
                     cs.to_gap(), cs.follow(2, speed=0.7), cs.straight(2), cs.straight(2, speed=0.6), cs.right(), cs.curve(10, 1),
                     cs.tr(), cs.stop(1), cs.set_speed(), cs.to_junction(), cs.follow(2), finish()])
    spd = [s["speed"] for s in flat]
    check("before set_speed: the original speed", spd[0] == cs.SPEED)
    check("after set_speed: every driving step changes", spd[1:5 + 1] == [0.33] * 5, spd[1:6])
    check("a speed= given directly is kept", spd[6] == 0.7)
    check("straight is kept", spd[7] is None and spd[8] == 0.6)
    check("right and curve are kept", spd[9] is None and spd[10] is None)
    tr_steps = cs._flat(cs.tr())
    check("tr (turn/align/slow) follows set_speed too", spd[11:11 + len(tr_steps)] == [0.33] * len(tr_steps), spd[11:11 + len(tr_steps)])
    check("tr with no set_speed runs at SPEED all through (no slowing in a turn)", [s["speed"] for s in tr_steps] == [cs.SPEED] * 3 and all(s["keep_speed"] for s in tr_steps))
    check("tr(speed=) keeps the speed given", all(s["speed"] == 0.4 for s in cs._flat([cs.set_speed(0.7), cs.tr(speed=0.4)])[:2]))
    check("set_speed(): back to the default", spd[-3] == cs.SPEED and spd[-2] == cs.SPEED)
    check("no set_speed step is left in the course", all(s["do"] != "set_speed" for s in flat) and cs._flat(flat) == flat)
    bundles = cs._flat([cs.set_speed(0.33), cs.cross_gap(), cs.cross(), cs.traffic(), cs.corner("left"), cs.uturn("right"),
                        cs.cross(line="crosswalk"), cs.pass_crosswalk()])
    follows = [s["speed"] for s in bundles if s["do"] == "follow" and s["until"] not in ("aligned", None)]
    straights = [s["speed"] for s in bundles if s["do"] == "straight" and s["until"] == "lane"]
    check("grouped action: the lane following inside changes", set(follows) == {0.33}, follows)
    gap_cross = [s["speed"] for s in cs._flat([cs.set_speed(0.33), cs.cross_gap()]) if s["do"] == "straight"]
    cross_line = [s["speed"] for s in cs._flat([cs.set_speed(0.33), cs.cross()]) if s["do"] == "straight"]
    check("grouped action: cross_gap's straight follows set_speed, cross()'s straight is kept", gap_cross == [0.33] and cross_line == [cs.SPEED], (gap_cross, cross_line))
    defaults = cs._flat([cs.to_end(), cs.uturn("left"), cs.follow(1)])
    check("to_end: time limit with max_sec", cs.to_end(max_sec=8)["max"] == 8 and cs.to_end()["max"] is None)
    check("without set_speed the default is kept", (defaults[0]["speed"], defaults[1]["speed"], defaults[-1]["speed"]) == (0.3, 0.4, cs.SPEED))
    check("after a turn: back to SPEED", cs._flat([cs.tr(), cs.follow(1)])[-1]["speed"] == cs.SPEED)
    check("after a turn: back to the set_speed value", cs._flat([cs.set_speed(0.33), cs.tl(), cs.follow(1)])[-1]["speed"] == 0.33)
    check("after a turn: a speed= given directly is kept", cs._flat([cs.tr(), cs.follow(1, speed=0.5)])[-1]["speed"] == 0.5)
    check("after a turn given speed=, the default is kept", cs._flat([cs.tr(speed=0.5), cs.follow(1)])[-1]["speed"] == cs.SPEED)
    resumed = cs._flat([cs.tr(), cs.follow(1)])
    check("back after a turn: the same when _flat is called again", cs._flat(resumed) == resumed and all("resume" not in s for s in resumed))
    d = lf.Driver([cs.set_speed(0.33), follow(9), finish()])
    with quiet():
        d.tick(lane_frame(), 0.0)
    check("Driver: set_speed applies and does not count in --start numbers", d.course.speed == 0.33 and len(d.course.course) == 2)
    d = lf.CourseRunner([follow(1), cs.set_speed(0.33), follow(1), follow(1)], start=3)
    check("set_speed before the start applies when starting midway with --start", d.speed == 0.33)
    try:
        cs.set_speed(1.5)
        ok = False
    except ValueError:
        ok = True
    check("set_speed: error out of range", ok)


def y_at(cm):
    """Screen height of a floor point this far ahead of the camera. The inverse of floor_cm."""
    return lf.HORIZON_Y + lf.floor_cm(lf.HORIZON_Y + 100) * 100 / cm


def drive_line(c, z, t, stop_z, v=10.0, fps=40, tip=False, visible_to=21.0):
    """A line comes closer at v cm/s, read every frame, until the action ends. Returns the line's real distance then."""
    t = feed(c, 5, t)       # An event counts only after it has been absent once.
    while z > stop_z:
        y = y_at(z) if z > visible_to else None
        near = int(H * 0.6)
        ys = {"tip_y": y, "gap": near, "near": True} if tip else {"junction": near, "hline_y": y, "near": True}
        c.update(not tip, t, (0.0, 0.0), False, ys)
        if c.idx != 0:
            break
        z -= v / fps
        t += 1 / fps
    return z


def junction_frame(kind, yj, gap=60, thick=6):
    """Lane meeting a junction whose near edge is at yj. kind: "L-right", "L-left", "+", "T-side-left", "curve-right".

    The lane's line ends at yj on the inside of an L and on both sides of a +. On the outside of an L it goes on to the line across, gap px farther.
    """
    img = np.clip(np.random.default_rng(0).normal(30, 4, (H, W, 3)), 0, 255).astype(np.uint8)
    yf = yj - gap
    white = (235,) * 3
    def line(side, y0, y1):
        for y in range(max(int(y0), 0), min(int(y1), H)):
            x = lane_x(y)[0 if side == "left" else 1]
            cv2.line(img, (int(x) - 5, y), (int(x) + 5, y), white, 1)
    def branch(side, y, out=True):
        x = lane_x(y)[0 if side == "left" else 1]
        x0, x1 = ((0, int(x) + 6) if side == "left" else (int(x) - 6, W - 1))
        cv2.rectangle(img, (x0, y - thick), (x1, y), white, -1)
    top = int(H * lf.ROI_TOP) - 10
    if kind in ("L-right", "L-left"):
        inner, outer = ("right", "left") if kind == "L-right" else ("left", "right")
        line(inner, yj, H)
        branch(inner, yj)
        line(outer, yf, H)
        xl, xr = lane_x(yf)
        cv2.rectangle(img, (int(xl) - 5 if outer == "left" else 0, yf - thick), (W - 1 if outer == "left" else int(xr) + 5, yf), white, -1)
    elif kind == "+":
        for side in ("left", "right"):
            line(side, yj, H)
            branch(side, yj)
            line(side, top, yf - thick)
    elif kind == "T-side-left":
        line("left", yj, H)
        branch("left", yj)
        line("left", top, yf - thick)
        line("right", top, H)
    elif kind == "curve-right":
        line("left", top, H)
        for y in range(top, H):     # The right line bends away to the right farther up.
            x = lane_x(y)[1] + 0.004 * max(250 - y, 0) ** 2
            cv2.line(img, (int(x) - 5, y), (int(x) + 5, y), white, 1)
    return img


def test_laneend():
    print("[junction: lane end]")
    old_offset = lf.LANEEND_OFFSET_CM
    lf.LANEEND_OFFSET_CM = 0.0      # Distances below are from the camera.
    try:
        _test_laneend()
    finally:
        lf.LANEEND_OFFSET_CM = old_offset


def _test_laneend():
    import course as cs
    for kind, yj, side in (("L-right", 270, "right"), ("L-left", 270, "left"), ("+", 290, None), ("T-side-left", 280, "left")):
        d, _ = run_detector(junction_frame(kind, yj))
        got = d.lane_end()
        check(f"{kind}: measures where the lane line ends (near edge of the junction)", got is not None and abs(got - (yj - 6)) <= 4 and (side is None or d.end_side == side),
              (got, d.end_side, yj - 6))
    d, _ = run_detector(junction_frame("curve-right", 0))
    check("curve: not an end when the line bends on", d.lane_end() is None, d.end_y)
    d, _ = run_detector(lane_frame())
    check("straight lane: no end", d.lane_end() is None, d.end_y)

    target = 25.0       # cm= is the LANE END distance it ends at.
    for v in (6.0, 12.0):
        with quiet():
            c = lf.CourseRunner([cs.to_junction(cm=25, blind_sec=0), finish()])
            t = feed(c, 5, 0.0)
            z = 45.0
            while z > 5 and c.idx == 0:
                c.update(False, t, (0.0, 0.0), False, {"near": True, "end_y": y_at(z)})
                z -= v / 40
                t += 1 / 40
        check(f"to_junction(cm=25): ends at LANE END {target:g}cm on screen (read in view, speed {v:g}cm/s)", c.idx == 1 and abs(z - target) < 0.6 and not c.ruler.guess, f"{z:.2f}")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        c = lf.CourseRunner([cs.to_junction(cm=2, blind_sec=0), finish()])
    check("cm= nearer than can be read: warns and ends at LANEEND_MIN_CM", "warning" in out.getvalue() and c.laneend_target(c.step()) == lf.LANEEND_MIN_CM, out.getvalue()[:80])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        c = lf.CourseRunner([cs.to_junction(cm=40, blind_sec=0), finish()])
    check("cm= farther than can be read: warns and ends at LANEEND_MAX_CM", "warning" in out.getvalue() and c.laneend_target(c.step()) == lf.LANEEND_MAX_CM, out.getvalue()[:80])
    with quiet():
        c = lf.CourseRunner([cs.to_junction(cm=25, blind_sec=0), finish()])
        z = drive_line(c, 70.0, 0.0, 0.0)
    want = 25.0 + lf.LANEEND_LINE_GAP
    check("no lane end seen: cm= still counts, the line across standing in LANEEND_LINE_GAP beyond", c.idx == 1 and abs(z - want) < 0.8, f"{z:.2f}, want {want:g}")
    with quiet():
        c = lf.CourseRunner([cs.to_junction(cm=15, blind_sec=0), finish()])
        z = drive_line(c, 70.0, 0.0, 0.0)
    check("no lane end seen: a different cm= ends at a different place", c.idx == 1 and abs(z - (15.0 + lf.LANEEND_LINE_GAP)) < 0.8, f"{z:.2f}")
    c = lf.CourseRunner([cs.to_junction(cm=20, blind_sec=0), finish()])
    for z_end, z_line in ((40.0, 61.0), (39.0, 60.0)):
        c.track_end({"end_y": y_at(z_end), "hline_y": y_at(z_line)}, 0.0)
    check("the gap to the line across is measured when both are seen", c.line_gap is not None and abs(c.line_gap - 21.0) < 0.5, c.line_gap)
    w = lf.EndWatch()
    ruler = lf.EndRuler()
    ruler.reset("laneend")
    ruler.z = 30.0
    old = lf.LANEEND_OFFSET_CM
    lf.LANEEND_OFFSET_CM = -14.5
    try:
        with quiet():
            w.ended(0, {"cm": 15}, ruler, None, 0.3, 0.0, None, 1.0, False)
    finally:
        lf.LANEEND_OFFSET_CM = old
    check("ends.csv keeps the lane end as shown on screen (car front)", w.rows[0]["ref"] == "laneend" and abs(w.rows[0]["ref_z"] - 15.5) < 1e-6, w.rows[0]["ref_z"])
    old_ref = lf.JUNCTION_REF
    lf.JUNCTION_REF = "line"
    try:
        check("JUNCTION_REF=\"line\": to_junction() gets no cm= (stops as before)", cs.to_junction()["cm"] is None and cs.to_junction(cm=5)["cm"] == 5)
    finally:
        lf.JUNCTION_REF = old_ref
    with quiet():
        c = lf.CourseRunner([cs.to_junction(cm=20, blind_sec=0, nth=2), finish()])
        t = feed(c, 5, 0.0)
        for _ in range(8):
            c.update(False, t, (0.0, 0.0), False, {"near": True, "end_y": y_at(30.0)})
            t += 0.05
        passed = c.count
    check("nth=2: the lane end of the passed junction is not counted again at once", passed == 1 and c.end_hits < lf.LANEEND_FRAMES, (passed, c.end_hits))
    old = lf.LANEEND_OFFSET_CM
    lf.LANEEND_OFFSET_CM = 2.0
    try:
        with quiet():
            c = lf.CourseRunner([cs.to_junction(cm=25, blind_sec=0), finish()])
            t = feed(c, 5, 0.0)
            z = 45.0
            while z > 5 and c.idx == 0:
                c.update(False, t, (0.0, 0.0), False, {"near": True, "end_y": y_at(z)})
                z -= 10.0 / 40
                t += 1 / 40
        check("the offset (LANEEND_OFFSET_CM) moves the shown value and the stop together", c.idx == 1 and abs(z + 2.0 - 25.0) < 0.6, f"{z:.2f}")
    finally:
        lf.LANEEND_OFFSET_CM = old


def test_end_cm():
    old_ref = lf.JUNCTION_REF
    lf.JUNCTION_REF = "line"        # These are about cm= past the line across the lane (the "line" way of measuring).
    try:
        _test_end_cm()
    finally:
        lf.JUNCTION_REF = old_ref


def _test_end_cm():
    print("[to_junction / to_end cm=]")
    import course as cs
    check("floor_cm: smaller when nearer, and fits the 20 cm lane width", lf.floor_cm(300) < lf.floor_cm(200)
          and abs(lf.LANE_WIDTH_PX / lf.CAMERA_F * lf.floor_cm(342) - lf.LANE_CM) < 0.01)

    for y in (230, 290, 330):
        img = lane_frame()
        cv2.rectangle(img, (0, y - 10), (W, y), (235,) * 3, -1)
        d, _ = run_detector(img)
        got = d.hline_edge()
        check(f"hline_edge: measures y={y}, the bottom edge of the line across (not in 36 px bands)", got is not None and abs(got - y) <= 3, got)
    d, _ = run_detector(lane_frame())
    check("hline_edge: None with no line across", d.hline_edge() is None)
    d, _ = run_detector(lane_frame(y_end=280))
    tip = d.lane_tip()
    check("lane_tip: measures where the lane ends", tip is not None and abs(tip - 280) <= 3, tip)
    d, _ = run_detector(lane_frame())
    check("lane_tip: None when the lane does not end", d.lane_tip() is None)

    with quiet():
        c = lf.CourseRunner([cs.to_junction(cm=10, blind_sec=0), finish()])
        z = drive_line(c, 45.0, 0.0, 10.0)
    target = lf.END_BASE_CM - 10
    check("to_junction(cm=10): ends exactly by the line", c.idx == 1 and abs(z - target) < 0.5, f"line {z:.2f}cm, target {target}cm")
    ends = []
    for v in (6.0, 12.0, 20.0):
        with quiet():
            c = lf.CourseRunner([cs.to_junction(cm=10, blind_sec=0), finish()])
            ends.append(drive_line(c, 45.0, 0.0, 10.0, v=v))
    check("cm=: ends at the same place at different speeds", max(ends) - min(ends) < 1.0, [round(e, 2) for e in ends])
    with quiet():
        c = lf.CourseRunner([cs.to_junction(cm=25, blind_sec=0), finish()])
        z = drive_line(c, 45.0, 0.0, 5.0)
    check("cm=: once the line leaves the view it is carried on at the measured speed and ends near the target", c.idx == 1 and abs(z - 13) < 1.0, f"line {z:.2f}cm, target 13cm")
    with quiet():
        c = lf.CourseRunner([cs.to_end(cm=5), finish()])
        z = drive_line(c, 45.0, 0.0, 10.0, tip=True)
    check("to_end(cm=5): ends by the lane end", c.idx == 1 and abs(z - (lf.END_BASE_CM - 5)) < 0.5, f"line {z:.2f}cm")
    with quiet():
        c = lf.CourseRunner([cs.to_end(cm=5), finish()])
        z = drive_line(c, 45.0, 0.0, 10.0)
    check("to_end(cm=5): ends by the junction line too", c.idx == 1 and abs(z - (lf.END_BASE_CM - 5)) < 0.5, f"line {z:.2f}cm")
    with quiet():
        c = lf.CourseRunner([cs.to_junction(cm=10, blind_sec=0), follow(9), finish()])
        drive_line(c, 45.0, 0.0, 10.0)
    check("cm=: the next step starts with the ruler cleared", c.idx == 1 and not c.run_out and c.ruler.z is None)
    with quiet():
        c = lf.CourseRunner([cs.to_junction(cm=10, nth=2, blind_sec=0), finish()])
        t = feed_junc(c, 5, 0.0, False)
        t = feed_junc(c, 5, t, True)
        first = (c.idx, c.run_out)
    check("cm= + nth=2: the extra distance does not start at the first line", first == (0, False))

    d = lf.Driver([cs.to_junction(cm=25, blind_sec=0), finish()])
    img = lane_frame()
    cv2.rectangle(img, (0, 225), (W, 235), (235,) * 3, -1)
    blank = lane_frame(left=False, right=False)
    with quiet():
        t = 0.0
        for _ in range(5):
            d.tick(lane_frame(), t)
            t += 0.05
        for _ in range(8):
            d.tick(img, t)
            t += 0.05
        started = d.course.run_out
        d.course.ruler.z, d.course.ruler.speed = 37.0, 0.1      # Line out of view and barely moving, so the end is far off.
        speeds = []
        for _ in range(40):
            d.tick(blank, t)
            speeds.append(d.speed)
            t += 0.05
    check("cm=: no back-up when the lane goes away during the extra distance", started and d.back is None and min(speeds) > 0, f"lowest speed {min(speeds)}")

    for bad, name in ((lambda: cs.to_junction(cm=-1), "negative"), (lambda: cs.to_junction(cm=61), "over 60"),
                      (lambda: cs.follow(3, cm=5), "timed step"), (lambda: cs.follow(until="lane", cm=5), "other until")):
        try:
            bad()
            ok = False
        except ValueError:
            ok = True
        check(f"cm=: error when {name}", ok)
    lf.JUNCTION_REF, ref = "laneend", lf.JUNCTION_REF
    try:
        with quiet():
            filled = [s["cm"] for s in lf.CourseRunner([cs.to_junction(), cs.to_junction(0.8), cs.to_end(), finish()]).course[:3]]
        check("to_junction(): without cm, LANEEND_BASE_CM (filled when the drive starts); with at=, as before", filled == [lf.LANEEND_BASE_CM, None, None], filled)
    finally:
        lf.JUNCTION_REF = ref
    check("describe: shows cm", "+10cm" in cs.describe(cs.to_junction(cm=10)))


def split_lane(y_left, y_right):
    """A lane whose left line ends at y_left and right line at y_right, as when a slightly turned car nears a gap."""
    img = lane_frame(y_end=y_left)
    img[:, W // 2:] = lane_frame(y_end=y_right)[:, W // 2:]
    return img


def test_gap_tip():
    print("[to_gap: lane end measurement]")
    import course as cs

    d, _ = run_detector(split_lane(320, 260))
    check("lane_tip: the middle of both ends", d.tip_y is not None and abs(d.tip_y - 290) <= 3, d.tip_y)
    d, _ = run_detector(lane_frame())
    check("gap_y: none without a gap", d.gap_y is None and d.tip_y is None, d.gap_y)

    # Runs 230925, 231241: the left line ends far before the right one, so the band test drops the gap on the way in.
    d = lf.LaneDetector()
    c = lf.CourseRunner([cs.to_gap(0.85), finish()])
    held = ended = None
    t = 0.0
    with quiet():
        for y in range(240, 352, 3):
            det, junction, _ = d(split_lane(y, y - 66), draw=False)
            if d.gap_why == "held by line tips":
                held = y
            c.update(junction, t, det, False, {"junction": d.junction_y, "gap": d.gap_y, "near": d.near_line})
            t += 0.05
            if c.idx == 1 and ended is None:
                ended = y
    check("to_gap: ends when the lane end arrives even if the band test drops out", ended is not None, f"left end y={ended}, lane end {held} instead of the band test")

    d = lf.LaneDetector()
    img = split_lane(320, 260)
    d.find_gap = lambda *a: None    # The band test never confirms it.
    for _ in range(5):
        d(img, draw=False)
    check("gap: a lane end not confirmed by the band test does not end it", d.gap_y is None, d.gap_y)
    d = lf.LaneDetector()
    d.find_gap = lambda *a: 270     # The bands call it a gap, but no line ends in view.
    d(lane_frame(), draw=False)
    check("gap: band test only, with no lane end in view, is not a gap", d.gap_y is None and d.gap_why == "no line tips", d.gap_y)
    d, _ = run_detector(lane_frame(y_end=300), n=5)
    img = lane_frame(y_end=300)
    cv2.rectangle(img, (0, 280), (W, 290), (235,) * 3, -1)
    d(img, draw=False)
    check("gap: not a gap with a line across in view", d.gap_y is None and not d.gap_held, d.gap_why)
    d, _ = run_detector(lane_frame(y_end=300), n=5)
    blank = lane_frame(left=False, right=False)
    for _ in range(lf.GAP_TIP_LOST):
        d(blank, draw=False)
    check("gap: forgets once the lane end is out of view", not d.gap_held)

    other = np.full((H, W, 3), 30, np.uint8)
    cv2.line(other, (W // 2, H - 1), (W - 1, 280), (235,) * 3, 11)
    d, _ = run_detector(other)
    check("lane_tip: a curve leaving by the frame side is not an end", d.tip_y is None, d.tip_y)
    near_center = np.full((H, W, 3), 30, np.uint8)
    cv2.line(near_center, (W // 2 + 20, H - 1), (W // 2 + 20, 260), (235,) * 3, 11)
    d, _ = run_detector(near_center)
    check("lane_tip: tape not where a lane line is is not an end", d.tip_y is None, d.tip_y)
    one = lane_frame(y_end=300, right=False)
    d, _ = run_detector(one)
    check("lane_tip: not an end when one side ends and the other was never seen", d.tip_y is None, d.tip_y)
    d, _ = run_detector(split_lane(330, 270))
    d(lane_frame(y_end=280, left=False), draw=False)
    check("lane_tip: when one side leaves first, the other side + the gap carry on", d.tip_y is not None and abs(d.tip_y - 310) <= 4, d.tip_y)


def test_end_ruler():
    old_ref = lf.JUNCTION_REF
    lf.JUNCTION_REF = "line"        # These are about cm= past the line across the lane (the "line" way of measuring).
    try:
        _test_end_ruler()
    finally:
        lf.JUNCTION_REF = old_ref


def _test_end_ruler():
    print("[cm= ruler]")
    import course as cs
    near = int(H * 0.6)
    target = lf.END_BASE_CM - 10

    def approach(c, t, z0=45.0, v=10.0):
        """A junction line in view, coming closer, until the action ends. Returns its real distance then."""
        z = z0
        while c.idx == 0 and z > 5:
            c.update(True, t, (0.0, 0.0), False, {"junction": near, "hline_y": y_at(z), "near": True})
            if c.idx != 0:
                break
            z -= v / 40
            t += 1 / 40
        return z

    with quiet():   # The old line is still in view when the action starts, and goes.
        c = lf.CourseRunner([cs.to_junction(cm=10, blind_sec=0), finish()])
        t = feed(c, 5, 0.0)
        for z in (25.0, 24.8, 24.6, 24.4):
            c.update(False, t, (0.0, 0.0), False, {"hline_y": y_at(z), "near": True})
            t += 1 / 40
        t = feed(c, 3, t)
        end = approach(c, t)
    check("cm=: not stuck on an earlier line", c.idx == 1 and abs(end - target) < 0.6, f"line {end:.2f}cm, target {target}cm")

    with quiet():
        c = lf.CourseRunner([cs.to_junction(cm=10, blind_sec=0), finish()])
        t = feed(c, 5, 0.0)
        c.update(False, t, (0.0, 0.0), False, {"hline_y": y_at(30.0), "near": True})    # One misread, closer than the line.
        t += 1 / 40
        end = approach(c, t, z0=50.0)
    check("cm=: a first misread does not block the good readings after it", c.idx == 1 and abs(end - target) < 0.6, f"line {end:.2f}cm, target {target}cm")

    with quiet():   # nth=2: the first line is still in view after it is passed.
        c = lf.CourseRunner([cs.to_junction(cm=10, nth=2, blind_sec=0), finish()])
        t = feed(c, 5, 0.0)
        for z in (40.0, 39.6, 39.2, 38.8, 38.4):
            c.update(True, t, (0.0, 0.0), False, {"junction": near, "hline_y": y_at(z), "near": True})
            t += 1 / 40
        for z in (38.0, 37.6, 37.2):
            c.update(False, t, (0.0, 0.0), False, {"hline_y": y_at(z), "near": True})
            t += 1 / 40
        t = feed(c, 3, t)
        end = approach(c, t)
    check("cm= + nth=2: measured and ended at the second line", c.idx == 1 and abs(end - target) < 0.6, f"line {end:.2f}cm, target {target}cm")

    with quiet():   # Reached, then the line leaves the frame and the next line ahead appears.
        c = lf.CourseRunner([cs.to_junction(cm=25, blind_sec=0), finish()])
        t = feed(c, 5, 0.0)
        z = 45.0
        while c.idx == 0 and z > 5:
            read = y_at(z) if z > 21 else y_at(60.0)
            c.update(True, t, (0.0, 0.0), False, {"junction": near, "hline_y": read, "near": True})
            if c.idx == 0:
                z -= 0.25
                t += 1 / 40
    check("cm=: once the line is reached it does not switch to the next one", c.idx == 1 and abs(z - 13) < 1.0, f"line {z:.2f}cm, target 13cm")

    with quiet():   # Only the bands see the line: the distance is still covered, from the band's position.
        c = lf.CourseRunner([cs.to_junction(cm=10, blind_sec=0), finish()])
        t = feed(c, 5, 0.0)
        start = None
        for _ in range(200):
            c.update(True, t, (0.0, 0.0), False, {"junction": near, "near": True})
            start = t if c.run_out and start is None else start
            if c.idx == 1:
                break
            t += 1 / 40
    want = (lf.floor_cm(near) - target) / lf.END_DEFAULT_SPEED
    check("cm=: goes the distance and ends even when seen only by the band test", c.idx == 1 and start is not None and abs(t - start - want) < 0.3, f"{t - start:.2f}s, expected {want:.2f}s")

    with quiet():   # Never reaches it: nothing moves, nothing measures.
        c = lf.CourseRunner([cs.to_junction(cm=10, blind_sec=0), finish()])
        t = feed(c, 5, 0.0)
        for _ in range(3):
            c.update(True, t, (0.0, 0.0), False, {"junction": near, "hline_y": y_at(45.0), "near": True})
            t += 1 / 40
        c.moving = False
        before = c.ruler.z
        for _ in range(20):
            c.update(True, t, (0.0, 0.0), False, {"junction": near, "near": True})
            t += 1 / 40
        still = c.ruler.z == before and c.idx == 0
        c.shift_clock(10.0)
        t += 10.0
        paused = c.idx == 0
        for _ in range(int(lf.END_MAX_SEC * 40) + 5):
            c.update(True, t, (0.0, 0.0), False, {"junction": near, "near": True})
            t += 1 / 40
    check("cm=: the distance does not shrink while the car does not move", still)
    check("cm=: time stopped for a figure does not count toward the time limit", paused)
    check("cm=: ends by the time limit when the distance does not shrink", c.idx == 1)

    with quiet():   # to_end: something read as a line earlier, but the action ends on a lane gap.
        c = lf.CourseRunner([cs.to_end(cm=5), finish()])
        t = feed(c, 5, 0.0)
        for z in (35.0, 34.8):
            c.update(False, t, (0.0, 0.0), False, {"hline_y": y_at(z), "near": True})
            t += 1 / 40
        z = 45.0
        while c.idx == 0 and z > 5:
            c.update(False, t, (0.0, 0.0), False, {"gap": int(y_at(z)), "tip_y": y_at(z), "near": True})
            if c.idx == 0:
                z -= 0.25
                t += 1 / 40
    check("to_end(cm=): measures the line that fits why it ended", c.idx == 1 and abs(z - (lf.END_BASE_CM - 5)) < 0.6, f"line {z:.2f}cm")

    r = lf.EndRuler()
    for i, z in enumerate((30.0, 29.8, 29.6)):
        r.feed("line", y_at(z), i * 0.025)
    r.speed = 8.0
    for i in range(3, 13):
        r.feed("line", None, i * 0.025)
    guessed = r.z
    r.feed("line", y_at(guessed - 0.2), 13 * 0.025)
    check("ruler: the position does not jump back when seen again", r.z <= guessed and not r.guess, f"{guessed:.2f} -> {r.z:.2f}")
    r = lf.EndRuler()
    r.feed("line", y_at(30.0), 0.0)
    r.speed = 0.0
    r.feed("line", None, 0.05)
    check("ruler: the distance shrinks for a moving car even with a measured speed of 0", r.z < 30.0, r.z)

    with quiet():
        c = lf.CourseRunner([cs.to_junction(0.6, blind_sec=0), finish()])
        t = feed(c, 5, 0.0)
        for _ in range(6):
            c.update(True, t, None, False, {"junction": 234, "near": False})
            t += 0.05
    check("to_junction(at=): a line seen without the lane is not a junction", c.idx == 0)


def test_align_steer():
    print("[align: steering limit]")
    from course import to_end as cs_to_end

    class Fixed(lf.LaneDetector):
        """The real detector, reporting a chosen lane error with a horizontal line in view."""
        err = (0.0, 0.0)

        def __call__(self, frame, side="center", draw=True):
            _, junction, dbg = super().__call__(frame, side, draw)
            self.junction_y = 234
            return self.err, junction, dbg

    def peak(off, head):
        d = lf.Driver([follow(until="aligned", max_sec=30, speed=0.4, style="soft"), finish()])
        d.detect = Fixed()
        d.detect.err = (off, head)
        with quiet():
            angles = [d.tick(lane_frame(), 0.05 * i)[0] for i in range(40)]
        return max(abs(a) for a in angles if a is not None)

    limit = lf.ALIGN_STEER_MAX + 0.5
    check("align: a curved lane passing across (heading 0.55) is steered without a limit", peak(-0.4, -0.55) > limit, f"{peak(-0.4, -0.55):.1f} deg")
    check("align: a real line across (heading 1.0) stays limited", peak(-0.4, -1.0) <= limit, f"{peak(-0.4, -1.0):.1f} deg")
    d = lf.Driver([follow(until="aligned", max_sec=30, speed=0.4, style="soft"), finish()])
    d.detect = Fixed()
    with quiet():
        d.detect.err = (-0.4, -1.0)
        for i in range(20):
            d.tick(lane_frame(), 0.05 * i)
        d.detect.err = (-0.4, -0.6)     # A real line's heading dips for a few frames.
        angles = [d.tick(lane_frame(), 1.0 + 0.05 * i)[0] for i in range(3)]
    check("align: stays limited when a real line's heading dips for a moment", max(abs(a) for a in angles) <= limit, f"{max(abs(a) for a in angles):.1f} deg")

    d = lf.Driver([cs_to_end(), finish()])
    d.detect = Fixed()
    d.detect.err = (-0.4, -1.0)
    with quiet():
        angles = [d.tick(lane_frame(), 0.05 * i)[0] for i in range(40)]
    check("to_end: steering limited with a line across in view", max(abs(a) for a in angles if a is not None) <= lf.APPROACH_STEER_MAX + 0.5,
          f"{max(abs(a) for a in angles if a is not None):.1f} deg")


    d = lf.Driver([follow(9, speed=0.1), finish()])
    with quiet():
        for i in range(5):
            d.tick(lane_frame(), 0.05 * i)
    check("back-up history: the real motor command is saved", d.history and min(h[2] for h in d.history) >= lf.MOTOR_MIN, d.history[:1])


class FakeCamera:
    def capture_array(self):
        return lane_frame()

    def stop(self):
        pass


class FakeStream:
    """Web keys for a tuning tool, with no server."""
    keys_left = ""

    def __init__(self, *a, **k):
        self.inbox = lf.KeyInbox()
        self.inbox.add(FakeStream.keys_left)

    def read_keys(self, terminal=True):
        return self.inbox.take()

    def read_box(self):
        return None

    def update(self, img):
        pass


def run_tool(tool, keys):
    """Run a tuning tool with no camera or browser, on a terminal of its own. Returns what it saved."""
    import pty
    import types
    master, slave = pty.openpty()
    saved, old = {}, (lf.open_camera, lf.Stream, lf.save_vision, sys.stdin, sys.argv, lf.WHITE_THR)
    lf.open_camera, lf.Stream = FakeCamera, FakeStream
    lf.save_vision = lambda **v: saved.update(v)
    sys.stdin, sys.argv = os.fdopen(slave, "r"), ["tool"]
    FakeStream.keys_left = keys
    try:
        with quiet():
            tool.main()
    finally:
        sys.stdin.close()
        os.close(master)
        lf.open_camera, lf.Stream, lf.save_vision, sys.stdin, sys.argv, lf.WHITE_THR = old
    return saved


def test_tools_read_web_keys():
    print("[tuning tools: web page keys]")
    import tune_light
    import tune_white
    saved = run_tool(tune_white, "WW\n")
    check("tune_white: web keys change the value, Enter saves", saved.get("white_thr") is not None, saved)
    check("tune_white: q does not save", run_tool(tune_white, "wq") == {})
    check("tune_white: Ctrl+C quits without saving", run_tool(tune_white, "w\x03") == {})
    before = lf.LIGHT_S_MIN
    saved = run_tool(tune_light, "w\n")
    check("tune_light: web keys change the value, Enter saves", saved.get("light_s_min") == before + 5, saved)
    lf.LIGHT_S_MIN = before
    check("tune_light: Ctrl+C quits without saving", run_tool(tune_light, "\x03") == {})
    check("tune_person: quits on web keys", run_tool(tune_person, "zq") == {})


def test_red_search():
    print("[wait_green: backing up to find the red light]")

    def run(d, frames, t):
        cmds = []
        for _ in range(frames):
            cmds.append(d.tick(lane_frame(), t)[1])
            t += 0.05
        return cmds, t

    def backs(cmds):
        return sum(1 for a, b in zip([0.0] + cmds, cmds) if b < 0 and a >= 0)

    d = lf.Driver([wait_green(max_sec=20), finish()])
    d.history = [(0.05, 7.0, 0.5)] * 40
    with quiet():
        cmds, t = run(d, 45, 0.0)
    first = next((i for i, c in enumerate(cmds) if c < 0), None)
    check("backs up after 2 s when no red light is seen", first is not None and abs(first * 0.05 - lf.RED_LOOK_SEC) <= 0.1, first)
    check("back-up steering retraces the way it came", d.steering.angle > 0, d.steering.angle)
    with quiet():
        more, t = run(d, 400, t)
    all_cmds = cmds + more
    time_back = sum(1 for c in all_cmds if c < 0) * 0.05
    check("at most 2 back-ups, 1 s each", backs(all_cmds) == 2 and abs(time_back - 2 * lf.RED_BACK_SEC) <= 0.15, f"{backs(all_cmds)} times, {time_back:.2f}s")
    check("goes after max_sec if still not found", d.done or d.course.idx == 1, f"step {d.course.idx + 1}")
    check("the stretch backed over is removed from the history", len(d.history) < 40, len(d.history))

    d = lf.Driver([wait_green(max_sec=20), finish()])
    with quiet():
        cmds, t = run(d, 45, 0.0)
        d.course.red_seen = True
        after, t = run(d, 60, t)
    check("stops backing up and waits once red is found", any(c < 0 for c in cmds) and all(c == 0 for c in after), after[:3])
    check("does not leave at max_sec once red is found", d.course.idx == 0 or not d.done)

    d = lf.Driver([wait_green(max_sec=20), finish()])
    with quiet():
        d.tick(lane_frame(), 0.0)
        for i in range(60):
            angle = d.search_red(0.05 * i, True)
            assert angle is None
    check("no back-up while standing for a figure", d.red["tries"] == 0)

    old = lf.RED_BACK_ON
    lf.RED_BACK_ON = False
    d = lf.Driver([wait_green(max_sec=5), finish()])
    with quiet():
        cmds, _ = run(d, 120, 0.0)
    lf.RED_BACK_ON = old
    check("RED_BACK_ON=False: goes after max_sec without backing up", all(c >= 0 for c in cmds) and (d.done or d.course.idx == 1))

    d = lf.Driver([stop(5), finish()])
    with quiet():
        cmds, _ = run(d, 60, 0.0)
    check("no back-up in other stop steps", all(c == 0 for c in cmds))


def test_key_route():
    print("[web page: /key and the page]")
    s = lf.Stream(0, "T", "w/s", confirm="\nq", key="secret", serve=False)
    web = s.app.test_client()

    def post(body, key="secret", program=None):
        headers = {"X-AICAR-Program": program or s.program}
        if key:
            headers["X-AICAR-Key"] = key
        return web.post("/key", data=body, headers=headers).status_code

    check("keys: passed on with the right secret", post("w") == 204 and s.keys.take() == "w")
    check("keys: refused with no or a wrong secret", post("w", key=None) == 403 and post("w", key="bad") == 403 and s.keys.take() == "")
    check("keys: a long request is refused", post("x" * 33) == 413 and s.keys.take() == "")
    check("keys: keys from a page for another program are refused", post("\n", program="old") == 409 and s.keys.take() == "")
    code = post("\x03", program="old")
    try:
        s.keys.take()
        stopped = False
    except KeyboardInterrupt:
        stopped = True
    check("keys: Ctrl+C stops even from an old page", code == 204 and stopped)
    html = web.get("/?key=secret").get_data(as_text=True)
    check("page: the secret address can send keys", 'KEY = "secret"' in html and s.program in html and "__" not in html.split("<script>")[1].split("KEYS_TO_PROGRAM")[0])
    bare = web.get("/").get_data(as_text=True)
    check("page: view only without the secret, and the secret not shown", 'KEY = ""' in bare and "secret" not in bare)
    check("page: keys that need confirming are passed on", 'CONFIRM = "\\nq"' in html)
    s.update(np.zeros((10, 10, 3), np.uint8))
    check("drag: refused for a program that did not turn it on", web.post("/box", data="0.1,0.1,0.2,0.2", headers={"X-AICAR-Key": "secret", "X-AICAR-Program": s.program}).status_code == 404)
    dragging = lf.Stream(0, "T", "w/s", key="secret", serve=False, drag_share=0.5)
    drag_web = dragging.app.test_client()

    def drag(body, key="secret", program=None):
        return drag_web.post("/box", data=body, headers={"X-AICAR-Key": key, "X-AICAR-Program": program or dragging.program}).status_code

    check("drag: the box reaches the program only once", drag("0.1,0.2,0.3,0.4") == 204 and dragging.read_box() == (0.1, 0.2, 0.3, 0.4) and dragging.read_box() is None)
    check("drag: refused with a wrong secret or another program's page", drag("0.1,0.2,0.3,0.4", key="bad") == 403 and drag("0.1,0.2,0.3,0.4", program="old") == 409 and dragging.read_box() is None)
    check("drag: bad values are refused", all(drag(b) == 400 for b in ("", "1,2", "a,b,c,d", "0.3,0.2,0.1,0.4", "0,0,2,1", "nan,0,1,1")) and drag("0," * 40) == 413 and dragging.read_box() is None)
    check("page: the drag setting is passed on", "DRAG = 0.5" in drag_web.get("/?key=secret").get_data(as_text=True) and "DRAG = 0;" in html)
    check("page: the program id is sent with the video", web.get("/frame.jpg").headers.get("X-AICAR-Program") == s.program)
    import shutil as sh
    import socket
    import subprocess
    if sh.which("node"):
        with tempfile.NamedTemporaryFile("w", suffix=".js") as f:
            f.write(html.split("<script>")[1].split("</script>")[0])
            f.flush()
            check("page: script syntax is fine", subprocess.run(["node", "--check", f.name], capture_output=True).returncode == 0)
    busy = socket.socket()
    busy.bind(("0.0.0.0", 0))
    busy.listen()
    try:
        lf.Stream(busy.getsockname()[1], key="x")
        refused = False
    except SystemExit:
        refused = True
    busy.close()
    check("Stream: does not pass quietly when the port cannot be opened", refused)


def test_key_inbox():
    print("[web page key input]")
    script = lf.Stream.PAGE.split("<script>")[1].split("</script>")[0]
    check("page: script strings are not broken by newlines", all(line.count('"') % 2 == 0 for line in script.splitlines()))
    box = lf.KeyInbox()
    box.add("wS \n+")
    check("keys: passed to the program in order", box.take() == "wS \n+")
    check("keys: emptied once read", box.take() == "")
    box.add("a\x1b[Aㅈ")
    check("keys: control characters and other letters are dropped", box.take() == "a[A")
    box.add("x" * 100)
    check("keys: do not pile up without end when not read", len(box.take()) == 64)
    box.add("w\x03")
    try:
        box.take()
        stopped = False
    except KeyboardInterrupt:
        stopped = True
    check("keys: Ctrl+C stops the program like in a terminal", stopped)
    check("keys: no Ctrl+C left after stopping", box.take() == "")


def test_next():
    print("[next(): web page Enter for the next step]")
    from course import next as cs_next, describe
    d = lf.Driver([stop(0.1), cs_next(), straight(1), finish()])
    run = d.course
    check("next: Enter shown in the description", "Enter" in describe(cs_next()))
    with quiet():
        run.press_enter()
        run.update(False, 0.0)
        check("next: Enter pressed before reaching it is ignored", run.idx == 0 and not run.entered)
        run.update(False, 0.2)
        check("next: waits when reached, speed 0", run.idx == 1 and run.speed == 0 and run.status(0.2) == "WAIT ENTER")
        run.update(False, 1000.0)
        check("next: does not move on as time passes", run.idx == 1)
        run.press_enter()
        run.update(False, 1000.1)
        check("next: Enter goes to the next step", run.idx == 2, run.idx)
        run.update(False, 1000.2)
        run.press_enter()
        check("next: Enter after it has no effect", run.idx == 2 and not run.entered)
    frame = lane_frame()
    d = lf.Driver([cs_next(), straight(1), finish()])
    with quiet():
        _, cmd, _ = d.tick(frame, 0.0, show=True)
        d.view(d.tick(frame, 0.1, show=True)[2], 0.1)
    check("next: motor stopped in the drive loop, no view errors", cmd == 0, cmd)


def test_saved_with_run():
    print("[settings saved with a recording / servo center / undoing a figure view]")
    import types
    import car
    saved = {k: getattr(lf, k) for k in ("LINE_COLOR", "HORIZON_Y")}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "vision.json"
        path.write_text(json.dumps({"line_color": "black", "horizon_y": 71.5}))
        lf.load_vision(path)
        check("load_vision: reads the recording's calibration", lf.LINE_COLOR == "black" and lf.HORIZON_Y == 71.5)
    for k, v in saved.items():
        setattr(lf, k, v)

    c = car.Car.__new__(car.Car)
    c.servo, c.trim, c.bias = types.SimpleNamespace(angle=0.0), 1.5, 0.0
    c.center()
    check("servo: a new center applies at once", c.servo.angle == 1.5, c.servo.angle)

    det, _ = registered_figure()
    before = det.background.copy()
    added, _ = det.add_view(minifig(lane_frame(), 320, 240))
    grew = added and not np.array_equal(det.background, before)
    det.undo()
    check("figure registration: undo also takes back that photo's background counts", grew and np.array_equal(det.background, before))
    check("figure registration: detection right after registering equals after saving", all(np.array_equal(s, np.array(v["bands"]) * det.table) for s, v in zip(det.band_sigs, det.views)))


def test_corrections():
    print("[small bugs]")
    import course as cs
    check("follow(3): a given time drops the default until", cs.follow(3)["until"] is None)
    check("right(1): a given time drops the default until", cs.right(1)["until"] is None and cs.right(1)["sec"] == 1)
    raised = False
    try:
        cs.follow(3, until="lane")
    except ValueError:
        raised = True
    check("error with both sec and until", raised)
    raised = False
    try:
        cs.follow(min_sec=5, max_sec=1)
    except ValueError:
        raised = True
    check("error when min_sec is larger than max_sec", raised)
    raised = False
    try:
        cs.follow(speed=-0.2)
    except ValueError:
        raised = True
    check("error with a negative speed", raised)
    check("to_junction(nth=None): the first line", cs.to_junction(nth=None)["nth"] is None)
    check("wait: keeps both the time and the condition", cs.wait(1)["sec"] == 1 and cs.wait(1)["until"] == "clear")
    check("describe: a wait shows time or condition", " or " in cs.describe(cs.wait(1)))

    with quiet():
        c = lf.CourseRunner([cs.wait(1.0, until="clear"), finish()])
        for i in range(lf.JUNCTION_FRAMES):
            c.update(False, i * 0.05, (0.0, 0.0), False, {}, obstacle=False)
    check("wait(sec, until): ends when the condition comes first, before the time", c.idx == 1, f"step {c.idx + 1}")
    with quiet():
        c = lf.CourseRunner([cs.wait(0.3, until="clear"), finish()])
        c.update(False, 0.0, (0.0, 0.0), False, {}, obstacle=True)
        c.update(False, 0.3, (0.0, 0.0), False, {}, obstacle=True)
    check("wait(sec, until): ends at sec when the condition never comes", c.idx == 1)

    for start in (0, -1, 99, True):
        raised = False
        try:
            lf.CourseRunner([finish()], start=start)
        except ValueError:
            raised = True
        check(f"start number {start} is an error", raised)

    old = lf.GAP_MAX_HEADING
    lf.GAP_MAX_HEADING = -1.0
    try:
        d9, _ = run_detector(lane_frame(y_end=250))
    finally:
        lf.GAP_MAX_HEADING = old
    check("the gap memory of a car out of line is dropped too", d9.gap_y is None and not d9.gap_held, d9.gap_why)

    class CW(lf.LaneDetector):
        def __call__(self, frame, side="center", draw=True):
            det, junction, dbg = super().__call__(frame, side, draw)
            self.crosswalk = True
            self.stripes = []
            return det, junction, dbg

    with quiet():
        d = lf.Driver([follow(9), finish()])
        d.detect = CW()
        d.tick(lane_frame(), 0.0)
        started = d.cw_since
        d.tick(lane_frame(), 0.05)
    check("the crosswalk steering hold counts from time 0", started == 0.0 and d.cw_since == 0.0, (started, d.cw_since))

    saved = {k: getattr(lf, k) for k in ("LINE_COLOR", "LANE_WIDTH_PX", "HORIZON_Y")}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vision.json"
            path.write_text(json.dumps({"line_color": "black", "lane_width_px": None, "horizon_y": "nope"}))
            lf.load_vision(path)
        check("null or bad values in vision.json are dropped",
              lf.LINE_COLOR == "black" and lf.LANE_WIDTH_PX == saved["LANE_WIDTH_PX"] and lf.HORIZON_Y == saved["HORIZON_Y"],
              (lf.LINE_COLOR, lf.LANE_WIDTH_PX, lf.HORIZON_Y))
    finally:
        for k, v in saved.items():
            setattr(lf, k, v)

    with quiet():
        c = lf.CourseRunner([cs.to_junction(at=0.9, cm=5, blind_sec=0), finish()])
        c.update(False, 0.0, (0.0, 0.0), False, {"near": True})
        c.ruler.kind = "laneend"
        c.update(False, 0.05, (0.0, 0.0), False, {"near": True})
    check("with at=, a junction without a line across is taken by the lane end", c.hits["junction"] >= 1, c.hits["junction"])

    with quiet():
        c = lf.CourseRunner([cs.to_junction(cm=10, blind_sec=0), finish()])
        t = feed_junc(c, 4, 0.0, False)
        ys = {"junction": int(H * 0.8), "hline_y": y_at(45), "near": True}
        for _ in range(8):
            c.update(True, t, (0.0, 0.0), False, ys)
            t += 0.05
            if c.run_out or c.idx != 0:
                break
        running = c.idx == 0 and c.run_out
        c.step_start = t - lf.STEP_MAX_SEC - 1
        c.update(True, t, (0.0, 0.0), False, ys)
    check("the safety time does not cut the cm= extra distance", running and c.idx == 0 and c.run_out, f"step {c.idx + 1} run_out {c.run_out}")


def test_junction_lost():
    print("[junction: ends when seen and then gone]")
    import course as cs
    low, high = int(H * 0.9), int(H * 0.6)

    def frames(c, n, t, on, y=low, det=(0.0, 0.0)):
        for _ in range(n):
            c.update(bool(on) if det is not None else False, t, det, False, {"junction": y if on else None, "near": True})
            t += 0.05
        return t

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        c = lf.CourseRunner([cs.follow(until="junction", blind_sec=0), finish()])     # Measured by the band test: to_junction() now measures to the lane end.
        t = frames(c, 2, 0.0, False)
        t = frames(c, 2, t, True)
        held = c.idx
        t = frames(c, 2, t, False)
    check("ends when gone for 2 frames after 2 frames near the bottom", held == 0 and c.idx == 1, f"step {c.idx + 1}")
    lost_line = [ln.strip() for ln in out.getvalue().splitlines() if "(lost)" in ln]
    check("a line that went away is logged as lost", bool(lost_line), lost_line)

    with quiet():
        c = lf.CourseRunner([cs.to_junction(blind_sec=0), finish()])
        t = frames(c, 2, 0.0, False)
        t = frames(c, 3, t, True)
    check("ends before it goes when the line is seen for 3 frames", c.idx == 1)

    with quiet():
        c = lf.CourseRunner([cs.to_junction(blind_sec=0), finish()])
        t = frames(c, 2, 0.0, False)
        t = frames(c, 1, t, True)
        t = frames(c, 3, t, False)
    check("not the end when a line seen for 1 frame goes away", c.idx == 0 and not c.junc_latch)

    with quiet():
        c = lf.CourseRunner([cs.to_junction(blind_sec=0), finish()])
        t = frames(c, 2, 0.0, False)
        t = frames(c, 2, t, True)
        t = frames(c, 1, t, False)
        gap = (c.idx, c.junc_latch, c.junc_gone)
        t = frames(c, 1, t, True)
    check("keeps going when a line misses 1 frame and comes back", gap == (0, True, 1) and c.idx == 0 and c.junc_gone == 0, gap)

    with quiet():
        c = lf.CourseRunner([cs.to_junction(blind_sec=0), finish()])
        t = frames(c, 2, 0.0, False)
        t = frames(c, 2, t, True, y=high)
        remembered = c.junc_latch
        t = frames(c, 3, t, False)
    check("a line that goes in mid frame is taken as false and does not stop it", remembered and c.idx == 0 and not c.junc_latch)

    with quiet():
        c = lf.CourseRunner([cs.to_junction(blind_sec=0), finish()])
        t = frames(c, 2, 0.0, False)
        t = frames(c, 2, t, True)
        t = frames(c, 4, t, False, det=None)
        frozen = (c.idx, c.junc_latch, c.junc_gone)
        t = frames(c, 2, t, False)
    check("frames with the lane lost do not count as the line going away", frozen == (0, True, 0) and c.idx == 1, frozen)

    with quiet():
        c = lf.CourseRunner([cs.to_junction(), finish()])
        t = frames(c, 2, 0.0, True)
        t = frames(c, 2, t, False)
        during = (c.idx, c.junc_latch)
        while t < 0.5:
            t = frames(c, 1, t, False)
        t = frames(c, 2, t, True)
        t = frames(c, 2, t, False)
    check("not remembered during the first 0.5 s, ends when it goes after that", during == (0, False) and c.idx == 1, f"blind {during} -> step {c.idx + 1}")

    with quiet():
        c = lf.CourseRunner([cs.follow(until="junction", min_sec=1, blind_sec=0), finish()])
        t = frames(c, 2, 0.0, False)
        t = frames(c, 2, t, True)
        t = frames(c, 2, t, False)
        early = c.idx == 0 and c.junction_lost()
        while c.idx == 0 and t <= 1.05:
            t = frames(c, 1, t, False)
    check("does not end by going away before the minimum time, ends after it", early and c.idx == 1, f"{t:.2f}s")

    with quiet():
        c = lf.CourseRunner([cs.to_junction(nth=2, blind_sec=0), finish()])
        t = frames(c, 2, 0.0, False)
        t = frames(c, 2, t, True)
        t = frames(c, 2, t, False)
        first = (c.idx, c.count, c.hits["junction"])
        t = frames(c, 1, t, False)          # The passed line has to be gone once more before the next one counts.
        t = frames(c, 2, t, True)
        t = frames(c, 2, t, False)
    check("nth=2: passes the first line that went away, ends at the second", first == (0, 1, 0) and c.idx == 1, first)

    with quiet():
        c = lf.CourseRunner([cs.to_junction(at=0.9, blind_sec=0), finish()])
        t = frames(c, 2, 0.0, False)
        t = frames(c, 2, t, True, y=int(H * 0.7))
        t = frames(c, 2, t, False)
        shy = (c.idx, c.junc_latch)
        t = frames(c, 2, t, True, y=int(H * 0.92))
        t = frames(c, 2, t, False)
    check("a line beyond at= is not remembered; ends when a line that reached at= goes away", shy == (0, False) and c.idx == 1, shy)

    with quiet():
        c = lf.CourseRunner([cs.to_junction(cm=5, blind_sec=0), finish()])
        t = frames(c, 2, 0.0, False)
        c.ruler.kind = "laneend"        # A + has no cross line. The lane end is already the ruler.
        c.update(False, t, (0.0, 0.0), False, {"near": True})
    check("a junction without a line across is not ended by going away", c.hits["junction"] >= 1 and not c.junc_latch and c.idx == 0,
          (c.hits["junction"], c.junc_latch, c.idx))

    with quiet():
        c = lf.CourseRunner([cs.to_junction(cm=5, blind_sec=0), finish()])
        t = frames(c, 2, 0.0, False)
        ys = {"junction": low, "hline_y": y_at(45), "near": True}
        for _ in range(2):
            c.update(True, t, (0.0, 0.0), False, ys)
            t += 0.05
        for _ in range(2):
            c.update(False, t, (0.0, 0.0), False, {"near": True})
            t += 0.05
        held_z = c.ruler.z
        running = c.idx == 0 and c.run_out and c.hits["junction"] == 0 and held_z is not None and held_z > lf.END_BASE_CM - 5
        last = held_z
        while c.idx == 0 and t < 8:
            last = c.ruler.z       # begin() of the next action replaces the ruler, so keep the last reading.
            c.update(False, t, (0.0, 0.0), False, {"near": True})
            t += 0.05
    check("cm= does not stop where the line goes away, but goes the measured distance", running and c.idx == 1 and last < held_z, f"{held_z} → {last}")

    with quiet():
        c = lf.CourseRunner([cs.to_crosswalk(), finish()])
        t = frames(c, 3, 0.0, False)
        for _ in range(2):
            c.update(False, t, (0.0, 0.0), True, {"crosswalk": low, "near": True})
            t += 0.05
        for _ in range(2):
            c.update(False, t, (0.0, 0.0), False, {"near": True})
            t += 0.05
    check("a crosswalk does not end by going away", c.idx == 0 and not c.junc_latch)


def test_set_steer():
    print("[course max steering set_steer]")
    import course as cs
    flat = cs._flat([cs.go(), cs.set_steer(15), cs.go(), cs.follow(2, steer=20), cs.tr(), cs.push(1), cs.set_steer(), cs.go()])
    steers = [(s["do"], s.get("steer")) for s in flat]
    check("set_steer: lane following after it gets the limit, an explicit steer= stays",
          steers[:3] == [("follow", None), ("follow", 15), ("follow", 20)], steers)
    check("set_steer: turns, their align, and push are not limited", all(st is None for do, st in steers[3:-1]), steers)
    check("set_steer(): back to the default", steers[-1] == ("follow", None))
    old = cs.STEER_MAX
    cs.STEER_MAX = 20
    try:
        flat = cs._flat([cs.go(), cs.set_steer(10), cs.go(), cs.set_steer(), cs.go()])
    finally:
        cs.STEER_MAX = old
    check("STEER_MAX: the default limit, and set_steer() returns to it", [s["steer"] for s in flat] == [20, 10, 20])
    with quiet():
        d = lf.Driver(cs._flat([cs.set_steer(5), cs.follow(2), finish()]))
        steer = None
        for i in range(30):
            steer, _, _ = d.tick(lane_frame(shift=60), i / 30)
    check("set_steer: the car does not steer past the limit", steer is not None and abs(steer) <= 5 + 1e-6, steer)
    with quiet():
        d = lf.Driver(cs._flat([cs.back(1.0), cs.back(0.5, speed=0.6, angle=10), cs.follow(0.5), finish()]))
        seen = []
        for i in range(60):
            out = d.tick(lane_frame(shift=40), i / 30)
            seen.append((d.course.idx, out))     # The action that made this command (after the tick).
    first = [cmd for idx, (_, cmd, _) in seen[1:25] if idx == 0]
    second = [(st, cmd) for idx, (st, cmd, _) in seen if idx == 1]
    check("back(): reverses for its seconds at LOST_BACK_SPEED, wheels straight, ignoring the lane", first and all(c == -lf.LOST_BACK_SPEED for c in first))
    check("back(speed=, angle=): that speed, wheels at that angle", second and all(c == -0.6 for _, c in second) and abs(second[-1][0] - 10) < 1e-6, second[-1:])
    check("back(): then drives on forward", any(idx == 2 and cmd > 0 for idx, (_, cmd, _) in seen))
    blank = np.full((H, W, 3), 30, np.uint8)
    speeds = {}
    for name, opt in (("none", {}), ("floor", {"min_speed": 0.425})):
        with quiet():
            d = lf.Driver(cs._flat([cs.set_speed(0.75), cs.follow(sec=8, **opt), finish()]))
            for i in range(20):
                d.tick(lane_frame() if i < 10 else blank, i / 30)
        speeds[name] = d.speed
    check("min_speed=: slowing (here for a lost lane) stops at it", abs(speeds["none"] - lf.MIN_SPEED) < 1e-9 and abs(speeds["floor"] - 0.425) < 1e-9, speeds)
    try:
        cs.set_steer(0)
        refused = False
    except ValueError:
        refused = True
    check("set_steer: a bad value is refused", refused)


def reflection_frame(blobs=((W // 2, 300),), shift=0.0):
    """Lane with crisp tape, plus soft bright ovals (ceiling-light reflections on a glossy mat) at these centers."""
    img = lane_frame(shift=shift).astype(np.float32)
    glow = np.zeros((H, W), np.float32)
    for x, y in blobs:
        cv2.ellipse(glow, (x, y), (14, 38), 0, 0, 360, 190, -1)
    glow = cv2.GaussianBlur(glow, (0, 0), 7)
    img += glow[:, :, None]
    return np.clip(img, 0, 255).astype(np.uint8)


def test_reflections():
    print("[light reflections on a glossy mat]")
    import calibrate as cal
    old = lf.REFLECT_SHARP
    img = reflection_frame()
    top = int(H * lf.ROI_TOP)
    d = lf.LaneDetector()
    mask = d.line_mask(img[top:])
    comps = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    sharp = {i: d.blob_sharpness(img[top:], comps, i) for i in d.near_blobs(comps)}
    tall = {i: v for i, v in sharp.items() if comps[2][i][3] >= 0.6 * mask.shape[0]}
    soft = {i: v for i, v in sharp.items() if i not in tall}
    check("tape edges are crisp, a reflection is soft", tall and soft and min(tall.values()) > max(soft.values()), sharp)
    try:
        lf.REFLECT_SHARP = None
        with quiet():
            off_none = [lf.LaneDetector()(img, draw=False)[0] for _ in range(3)][-1]
        lf.REFLECT_SHARP = old
        with quiet():
            d = lf.LaneDetector()
            for _ in range(3):
                det, _, dbg = d(img)
            clean = lf.LaneDetector()(lane_frame(), draw=False)[0]
        check("the reflection is left out, and the lane reads as without it",
              d.reflections and det is not None and abs(det[0] - clean[0]) < 0.05 and abs(det[1] - clean[1]) < 0.05, (off_none, det, clean))
        many = reflection_frame(blobs=[(x, 300) for x in (230, 300, 370, 440)])
        lf.REFLECT_SHARP = None
        with quiet():
            dn = lf.LaneDetector()
            for _ in range(3):
                dn(many, draw=False)
        lf.REFLECT_SHARP = old
        with quiet():
            dr = lf.LaneDetector()
            for _ in range(3):
                dr(many, draw=False)
        check("a row of reflections is not a crosswalk", not dr.crosswalk_any, (dn.crosswalk_any, dr.crosswalk_any))
        lf.REFLECT_SHARP = None
        with quiet():
            plain = [lf.LaneDetector()(lane_frame(shift=s), draw=False)[0] for s in (0, 30, -30)]
        lf.REFLECT_SHARP = old
        with quiet():
            filt = [lf.LaneDetector()(lane_frame(shift=s), draw=False)[0] for s in (0, 30, -30)]
        check("a lane with no reflection reads the same with the filter on", plain == filt)
        frames = [reflection_frame()] * 6
        cut = cal.measure_reflections(frames)
        check("calibration: the cut falls between the reflection and the tape",
              not isinstance(cut, str) and cut[2] is not None and cut[2] < cut[0] < cut[1], cut)
        none = cal.measure_reflections([lane_frame()] * 6)
        check("calibration: with no reflection, the cut sits under the tape", not isinstance(none, str) and none[2] is None and none[0] < none[1], none)
        check("calibration: no lane in view is reported", isinstance(cal.measure_reflections([np.full((H, W, 3), 30, np.uint8)] * 4), str))
        lf.REFLECT_SHARP = None
        geo_bad = cal.fit_geometry(cal.measure_lanes([reflection_frame(blobs=[(W // 2 - 40, 300)])] * 6))
        lf.REFLECT_SHARP = old
        geo = cal.fit_geometry(cal.measure_lanes([reflection_frame(blobs=[(W // 2 - 40, 300)])] * 6))
        ref = cal.fit_geometry(cal.measure_lanes([lane_frame()] * 6))
        check("calibration: lane width is measured past a reflection between the lines",
              not isinstance(geo, str) and abs(geo["lane_width_px"] - ref["lane_width_px"]) < 5, (geo_bad, geo, ref))
    finally:
        lf.REFLECT_SHARP = old


def test_glare_motion():
    print("[reflections by motion, red floor signs]")
    top = int(H * lf.ROI_TOP)

    def y_of(cm):       # Frame row of a floor point this far ahead (inverse of floor_cm).
        scale = lf.CAMERA_F * lf.LANE_CM / lf.LANE_WIDTH_PX * max(342 - lf.HORIZON_Y, 1.0)
        return lf.HORIZON_Y + scale / cm

    def scene(t, glare=True, paint=True):
        img = lane_frame().astype(np.float32)
        layer = np.zeros((H, W), np.float32)
        if glare:
            cv2.ellipse(layer, (W // 2 - 60, 300), (14, 36), 0, 0, 360, 190, -1)
        if paint:       # A soft painted bar 40 cm ahead at t=0, coming closer at 20 cm/s.
            yb = int(y_of(40 - 20 * t))
            cv2.rectangle(layer, (W // 2 + 20, yb - 14), (W // 2 + 70, yb), 200, -1)
        layer = cv2.GaussianBlur(layer, (0, 0), 6)
        return np.clip(img + layer[:, :, None], 0, 255).astype(np.uint8)

    d = lf.LaneDetector()
    glare_seen = paint_seen = 0
    with quiet():
        t = 0.0
        while t < 1.0:
            d.now, d.moving = t, True
            d(scene(t), draw=False)
            if t > 0.5:
                glare_seen += any(abs(x + w / 2 - (W // 2 - 60)) < 20 for x, y, w, h in d.reflect_boxes)
                paint_seen += any(abs(x + w / 2 - (W // 2 + 45)) < 25 for x, y, w, h in d.paint_boxes)
            t += 1 / 30
    check("driving: a reflection that stays put is dropped", glare_seen >= 10, glare_seen)
    check("driving: a soft painted bar coming closer is kept as paint", paint_seen >= 5, paint_seen)
    with quiet():
        for k in range(10):
            d.now, d.moving = 1.0 + k / 30, False
            d(scene(1.0, paint=False), draw=False)
    check("stopped: the reflection seen while driving is still dropped", len(d.reflections) >= 1)
    d2 = lf.LaneDetector()
    with quiet():
        for k in range(20):
            d2.now, d2.moving = k / 30, True
            d2(lane_frame(), draw=False)
            d2b = d2(lane_frame(), draw=False)
    check("a plain lane: nothing is dropped by the motion test", not d2.reflections, d2.reflections)

    tl = lf.TrafficLight()
    img = np.full((H, W, 3), 40, np.uint8)
    cv2.circle(img, (70, 150), 45, (40, 40, 230), -1)       # A red sign painted on the floor (BGR), with a white bit in its red:
    cv2.rectangle(img, (40, 145), (48, 153), (250, 250, 250), -1)     # white center, red around, like a lit lamp.
    sign = tl(img)
    painted = list(tl.painted)
    lamp = np.full((H, W, 3), 40, np.uint8)
    cv2.circle(lamp, (60, 100), 8, (40, 40, 255), -1)       # A small red lamp with a blown-out center.
    cv2.rectangle(lamp, (57, 97), (63, 103), (250, 250, 250), -1)
    check("light: a white bit inside a red floor sign is not a red lamp, a small lamp is", sign is None and painted and tl(lamp) == "red", (sign, painted))

    base = lane_frame()
    xs = [int(W * 0.42) + k * 26 for k in range(4)]
    stripes = base.copy()
    for x in xs:
        cv2.rectangle(stripes, (x, 250), (x + 12, 275), (235, 235, 235), -1)
    on_sign = stripes.copy()
    cv2.rectangle(on_sign, (xs[0] - 30, 230), (xs[-1] + 42, 295), (40, 40, 220), 10)     # The same white pieces inside a red border.
    with quiet():
        a, b = lf.LaneDetector(), lf.LaneDetector()
        for _ in range(3):
            a(stripes, draw=False)
            b(on_sign, draw=False)
    check("crosswalk: white pieces inside a red sign are not a crosswalk, plain ones are", a.crosswalk_any and not b.crosswalk_any, (a.crosswalk_any, b.crosswalk_any))
    odd = base.copy()       # Three pieces of very different sizes between the lines: a thin far bit, a small one, a tall soft-edged one.
    cv2.rectangle(odd, (int(W * 0.42), 200), (int(W * 0.42) + 50, 207), (235, 235, 235), -1)
    cv2.rectangle(odd, (int(W * 0.5), 215), (int(W * 0.5) + 18, 224), (235, 235, 235), -1)
    cv2.rectangle(odd, (int(W * 0.55), 270), (int(W * 0.55) + 40, 352), (235, 235, 235), -1)
    with quiet():
        c = lf.LaneDetector()
        for _ in range(3):
            c(odd, draw=False)
    check("crosswalk: pieces of very different sizes are not stripes", not c.stripes, c.stripes)
    check("light: the whole width is searched, so a light straight ahead is found", lf.TrafficLight.regions()[0][2] >= W // 2)


def test_landmarks():
    print("[registered traffic light and crosswalk]")
    import landmarks
    import course as cs
    import tune_landmark as tl
    old_light, old_xwalk = lf.LIGHT_MODEL, lf.XWALK_MODEL
    try:
        # Light: a red lamp registered at its place. A sign ring elsewhere and a hollow ring are not taken; the lamp is.
        lamp = np.full((H, W, 3), 40, np.uint8)
        cv2.circle(lamp, (520, 110), 9, (40, 40, 255), -1)
        sample, note = landmarks.LightModel.sample(lamp, (500 / W, 90 / H, 540 / W, 130 / H))
        check("light: a lit lamp in the box is registered with its color", sample is not None and sample["color"] == "red", note)
        empty, why = landmarks.LightModel.sample(np.full((H, W, 3), 40, np.uint8), (0.1, 0.1, 0.3, 0.3))
        check("light: an unlit box is refused with a reason", empty is None and "lamp" in why)
        lf.LIGHT_MODEL = landmarks.LightModel([sample])
        t = lf.TrafficLight()
        check("light: the registered lamp is found", t(lamp) == "red")
        far = np.full((H, W, 3), 40, np.uint8)
        cv2.circle(far, (90, 200), 9, (40, 40, 255), -1)          # The same lamp, far from where it was registered.
        ring = np.full((H, W, 3), 40, np.uint8)
        cv2.circle(ring, (520, 120), 30, (40, 40, 230), 6)        # A red ring (a sign) where the lamp was.
        check("light: a red spot far from the registered place, and a red ring, are not the lamp", t(far) is None and t(ring) is None)
        lf.LIGHT_MODEL = None
        check("light: with none registered, the old thresholds find a lamp anywhere", lf.TrafficLight()(far) == "red")

        # Crosswalk: stripes registered from a box. Alike stripes in a row are found; scattered pieces are not.
        def stripes(xs, y=250, h=30, w=16):
            img = lane_frame()
            for x in xs:
                cv2.rectangle(img, (x, y), (x + w, y + h), (235, 235, 235), -1)
            return img
        row = [int(W * 0.40) + k * 30 for k in range(4)]
        view, note = tl.crosswalk_view(stripes(row), (0.36, 230 / H, 0.62, 300 / H))
        check("crosswalk: the stripes in the box are registered", view is not None and len(view) == 4, note)
        lf.XWALK_MODEL = landmarks.CrosswalkModel([{"stripes": view}])
        with quiet():
            d = lf.LaneDetector()
            for _ in range(3):
                d(stripes(row), draw=False)
        found = d.xwalk2_y
        scattered = stripes([])
        for (x, y) in ((int(W * 0.40), 200), (int(W * 0.47), 260), (int(W * 0.55), 320)):     # Same size, not in a row.
            cv2.rectangle(scattered, (x, y), (x + 16, y + 30), (235, 235, 235), -1)
        with quiet():
            d2 = lf.LaneDetector()
            for _ in range(3):
                d2(scattered, draw=False)
        check("crosswalk: the registered crosswalk is found, scattered pieces are not", found is not None and d2.xwalk2_y is None, (found, d2.xwalk2_y))

        # Course: _to_crosswalk ends when the registered crosswalk reaches at=, _pass_crosswalk when it is gone.
        steps = cs._flat([cs._to_crosswalk(0.6), cs._pass_crosswalk(), finish()])
        check("course: _to_crosswalk / _pass_crosswalk use the registered crosswalk", [s["until"] for s in steps[:2]] == ["_crosswalk", "_crossed"])
        with quiet():
            c = lf.CourseRunner(steps)
            t = 0.0
            for _ in range(6):
                c.update(False, t, (0.0, 0.0), False, {"_crosswalk": None, "near": True}); t += 0.05
            for _ in range(lf.JUNCTION_FRAMES + 1):
                c.update(False, t, (0.0, 0.0), False, {"_crosswalk": int(H * 0.7), "near": True}); t += 0.05
            reached = c.idx
            for _ in range(20):
                c.update(False, t, (0.0, 0.0), False, {"_crosswalk": None, "near": True}); t += 0.05
        check("course: ends at the registered crosswalk, then once it is passed", reached == 1 and c.idx == 2, (reached, c.idx))
        try:
            cs._to_crosswalk(sec=-1)
            refused = False
        except ValueError:
            refused = True
        check("course: _to_crosswalk checks its options", refused)
    finally:
        lf.LIGHT_MODEL, lf.XWALK_MODEL = old_light, old_xwalk


def test_line_and_stop_points():
    print("[registered lane lines and stop points]")
    import landmarks
    import course as cs
    import tune_landmark as tl
    old_line, old_stop, old_on = lf.LINE_MODEL, lf.STOP_MODEL, cs.STOP_POINTS_ON
    cs.STOP_POINTS_ON = True        # The stop-point search, whatever course.py has it set to.
    try:
        img = lane_frame()
        l, r = lane_x(300)
        sample, note = tl.line_sample(img, ((l - 20) / W, 280 / H, (l + 20) / W, 330 / H))
        check("line: a painted line in the box is measured", sample is not None and sample["sharp"] >= tl.LINE_SHARP_MIN, note)
        soft = reflection_frame(blobs=((W // 2, 300),))
        refused, why = tl.line_sample(soft, ((W // 2 - 30) / W, 260 / H, (W // 2 + 30) / W, 340 / H))
        check("line: a soft reflection is refused", refused is None and "reflection" in why, why)
        m = landmarks.LineModel([sample])
        check("line: the reflection cut sits under the registered paint", m.reflect_cut() < sample["sharp"])
        lf.LINE_MODEL = m
        with quiet():
            d = lf.LaneDetector()
            for _ in range(3):
                det, _, _ = d(reflection_frame(), draw=False)
        check("line: with lines registered, the reflection is still left out", d.reflections and det is not None)
        lf.LINE_MODEL = old_line

        # Stop point: an L corner registered where the car stood; replayed as it comes closer, go_to ends there.
        def corner(y):
            img = lane_frame()
            x = int(W * 0.25)
            cv2.line(img, (x, y), (x + 120, y), (235, 235, 235), 8)
            cv2.line(img, (x, y), (x, min(y + 60, H - 1)), (235, 235, 235), 8)
            return img
        stop_y = 300
        model = landmarks.StopModel()
        ok, note = model.add("A", tl.frame_mask(corner(stop_y)), (int(W * 0.25) - 20, stop_y - 20, 160, 70))
        check("stop: a corner is registered", ok, note)
        bad, why = landmarks.StopModel().add("B", np.zeros((H, W), np.uint8), (100, 200, 80, 60))
        check("stop: an empty box is kept, with a warning (no limit on what is dragged)", bad and "warning" in why, why)
        hi = tl.frame_mask(corner(stop_y))
        ok2, why2 = landmarks.StopModel().add("C", hi, (10, 5, 200, 80))
        check("stop: a box above the line area is taken (matched on edges)", ok2, why2)
        lf.STOP_MODEL = model
        steps = cs._flat([cs.go_to("A", blind_sec=0), finish()])
        with quiet():
            dr = lf.Driver(steps)
            ended = None
            for k, y in enumerate(range(200, 330, 4)):
                dr.tick(corner(y), k / 30)
                if ended is None and dr.course.idx >= 1:
                    ended = y
        check("stop: go_to ends when the corner comes to where it was registered", ended is not None and abs(ended - stop_y) <= 16, ended)
        plus = lane_frame()
        cv2.line(plus, (int(W * 0.6), 260), (int(W * 0.6) + 80, 260), (235, 235, 235), 8)
        cv2.line(plus, (int(W * 0.6) + 40, 230), (int(W * 0.6) + 40, 300), (235, 235, 235), 8)
        model.add("B", tl.frame_mask(plus), (int(W * 0.6) - 20, 220, 120, 90))
        with quiet():
            dr = lf.Driver(cs._flat([cs.go_to("b|a", blind_sec=0), finish()]))
            either = None
            for k, y in enumerate(range(200, 330, 4)):
                dr.tick(corner(y), k / 30)
                if either is None and dr.course.idx >= 1:
                    either = y
        check('stop: go_to("b|a") stops at whichever comes first (here A)', either is not None and abs(either - stop_y) <= 16, either)
        try:
            cs.go_to("")
            refused = False
        except ValueError:
            refused = True
        check("stop: go_to needs a name", refused)
        check("stop: go_to keeps the name and offset", [(s["until"], s["stop"], s["stop_cm"]) for s in cs._flat([cs.go_to("B", offset=3)])] == [("_stop", "B", 3)])
        cs.STOP_POINTS_ON = False
        check("stop: STOP_POINTS_ON = False makes go_to() a go() (lane end)", cs._flat([cs.go_to("A|B", steer=8)])[0]["until"] == "junction")
    finally:
        lf.LINE_MODEL, lf.STOP_MODEL, cs.STOP_POINTS_ON = old_line, old_stop, old_on


def test_stop_point_details():
    print("\n[stop points: glints, near/far views, any heading, full points, the stop log]")
    import landmarks
    import tune_landmark as tl
    img = np.zeros((200, 300), np.uint8)
    cv2.line(img, (20, 150), (280, 150), 255, 8)
    cv2.ellipse(img, (150, 60), (10, 6), 0, 0, 360, 255, -1)
    out = landmarks.drop_spots(img, landmarks.STOP_SPOT_PX)
    check("glints: a small blob is dropped, a painted line is kept", out[60, 150] == 0 and out[150, 150] == 255)

    def corner(y, x0):
        f = lane_frame()
        cv2.line(f, (x0, y), (x0 + 120, y), (235, 235, 235), 8)
        cv2.line(f, (x0, y), (x0, min(y + 60, H - 1)), (235, 235, 235), 8)
        return f
    m = landmarks.StopModel()
    box = (int(W * 0.25) - 20, 280, 160, 70)
    m.add("A", tl.frame_mask(corner(300, int(W * 0.25))), box)
    top = int(H * lf.ROI_TOP)
    with quiet():
        d = lf.LaneDetector()
        for _ in range(3):
            d(corner(300, int(W * 0.25) + 70), draw=False)
    hit = m.find("A", d.mask, top, wide=False)
    check("any heading: the same mark 70 px to the side, at its height, is found by the near check", hit is not None and abs(hit["y"] + hit["h"] / 2 - m.view_y("A", 0)) < 6, hit)
    check("a clear match may count as the stop, a weak one may not",
          landmarks.StopModel.close_enough({"score": 0.9, "scale": 1.0}) and not landmarks.StopModel.close_enough({"score": 0.8, "scale": 1.0}))
    old_max = landmarks.STOP_MAX_VIEWS
    landmarks.STOP_MAX_VIEWS = 8        # A limit, when one is set: far views go first.
    far = landmarks.StopModel()
    mask = tl.frame_mask(corner(300, int(W * 0.25)))
    for k in range(landmarks.STOP_MAX_VIEWS):
        far.add("A", mask, (10 + k, 20, 200, 80))       # Far boxes (above the line area)...
    far.add("A", mask, box)                             # ... then a near one.
    far.add("A", mask, (12, 22, 200, 80))
    near = far.near_views("A")
    check("near / far: a box on the floor is near, one above the line area is far", len(near) == 1)
    check("full point: far views go first, the near one is kept", len(far.points["A"]["views"]) == landmarks.STOP_MAX_VIEWS and len(near) == 1)
    landmarks.STOP_MAX_VIEWS = old_max
    many = landmarks.StopModel()
    for k in range(20):
        many.add("A", mask, (box[0] + k % 5, box[1], box[2], box[3]))
    check("no view limit: all 20 views are kept", len(many.points["A"]["views"]) == 20 and len(many.templates("A")) == 20)
    picks = [many._home_views("A", list(range(20))) for _ in range(3)]
    check("many views: the near check tries STOP_HOME_PER_FRAME a frame, and over frames all of them",
          all(len(p) == landmarks.STOP_HOME_PER_FRAME for p in picks) and len(set(sum(picks, []))) >= 16, picks)
    note = landmarks.StopModel().add("B", mask, (10, 20, 200, 80))[1]
    check("a far box says it only shows the point is ahead", "far box" in note, note)
    d = lf.Driver([finish()])
    path = Path(tempfile.mkdtemp()) / "stops.csv"
    d.save_stop_log(path)
    check("stop log: nothing is written without go_to()", not path.exists())
    d.stop_log.append("1.000,0,A,0,1,0.900,1.000,200.0,200.0,0.0,1")
    d.save_stop_log(path)
    check("stop log: written with its header", path.read_text().startswith("t,step,name,view,near,score"))


def test_no_slowdown():
    print("\n[SLOWDOWN_ON = False: no slowing anywhere]")
    import course as cs
    cs.SLOWDOWN_ON = False
    try:
        d = lf.Driver([follow(until="junction", speed=0.6, at=0.8, slow=0.3), finish()])
        d.detect.junction_y = 280
        with quiet():
            d.course.update(False, 0.0, (0.0, 0.0), False, {"near": True})
        check("slow= is ignored: the set speed is kept near the line", d.approach_speed(0.6, 0.05) == 0.6, d.approach_speed(0.6, 0.05))
        steps = cs._flat(cs.uturn("left"))
        check("uturn runs at SPEED with no slow=", all(st.get("slow") is None for st in steps) and steps[0]["speed"] == cs.SPEED, [st.get("speed") for st in steps])
        check("to_end runs at SPEED", cs._flat([cs.to_end()])[0]["speed"] == cs.SPEED)
    finally:
        cs.SLOWDOWN_ON = True


def main():
    for t in (test_corrections, test_junction_lost, test_laneend, test_end_cm, test_gap_tip, test_end_ruler, test_align_steer, test_key_inbox, test_key_route, test_red_search, test_tools_read_web_keys, test_saved_with_run, test_fast_math, test_lane, test_gap, test_mat_mask, test_crosswalk, test_track_color,
              test_traffic_light, test_course_logic, test_calibration, test_motor, test_turn_align_stop, test_blind_and_bundles, test_lost_and_far, test_stripes_and_crossing,
              test_driver_smoke, test_person_detect, test_person_near_decoys, test_person_look, test_person_photos, test_person_any, test_person_in_lane, test_next, test_speed_constant, test_person_stop, test_end_watch, test_set_speed, test_set_steer, test_reflections, test_glare_motion, test_landmarks, test_line_and_stop_points, test_stop_point_details, test_no_slowdown):
        t()
    passed = sum(results)
    print(f"\n{passed}/{len(results)} passed")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
