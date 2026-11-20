"""Closed-loop driving simulator: draws the camera view the car would see on a track, runs the real Driver on it, moves
the car with the steering it chose, and measures how well it kept to the lane. Runs on the Mac, no Pi needed.

  python sim.py                        every track, both lane methods, a summary table
  python sim.py --track s --model track --video out.mp4      one run, saved as video (camera view + top view)

The camera is the same flat-floor pinhole model lane_follow.py uses (HORIZON_Y, LANE_WIDTH_PX, CAMERA_F, CAMERA_X), but
the simulated camera is deliberately a little off from what the car believes (horizon, lane width, center), and the picture
gets blur, noise, uneven light, glare spots and JPEG, so a method that only works on perfect pictures does not look good.
"""
import argparse
import contextlib
import io
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lane_follow as lf      # noqa: E402
import course as cs           # noqa: E402

RES = 0.5               # cm per pixel of the top-view track picture.
LINE_CM = 1.3           # Painted line width (real tape reads 20-35 px at y=300, about 1.2-1.5 cm).
WHEELBASE = 14.0        # cm, rear axle to front axle.
CAM_AHEAD = 10.0        # cm, camera ahead of the rear axle.
SPEED_CM = 32.0         # cm/s at motor command 1.0.
FPS = 30.0
PI_VISION = {"lane_width_px": 440.6, "horizon_y": 55.0, "camera_x": 0.5189}     # The Pi's calibration (vision.json).


# --- Tracks: a centerline made of straights and arcs ---
def centerline(segments, step=0.5):
    """[(x, y, heading)] every step cm. Segments: ("S", length) or ("L"/"R", radius, degrees)."""
    x, y, th = 0.0, 0.0, math.pi / 2
    pts = [(x, y, th)]
    for seg in segments:
        if seg[0] == "S":
            for _ in range(int(seg[1] / step)):
                x += step * math.cos(th)
                y += step * math.sin(th)
                pts.append((x, y, th))
        else:
            r, deg = seg[1], seg[2]
            sign = 1 if seg[0] == "L" else -1
            for _ in range(int(r * math.radians(deg) / step)):
                th += sign * step / r
                x += step * math.cos(th)
                y += step * math.sin(th)
                pts.append((x, y, th))
    return np.array(pts)


# Radii the car can drive: at 26 degrees of steering and a 14 cm wheelbase its tightest circle is about 29 cm.
TRACKS = {
    "straight": [("S", 400)],
    "oval": [("S", 120), ("R", 50, 180), ("S", 120), ("R", 50, 90)],      # Open: a closed one would run over its own start (a fork).
    "s": [("S", 60), ("L", 50, 90), ("R", 50, 180), ("L", 50, 180), ("R", 50, 90), ("S", 60)],
    "tight": [("S", 60), ("R", 36, 90), ("S", 30), ("L", 36, 90), ("S", 30), ("R", 36, 180), ("S", 60)],
    "wiggle": [("S", 40)] + [("L", 60, 25), ("R", 60, 50), ("L", 60, 25)] * 4 + [("S", 40)],
}
RUN_OUT = 150           # cm of straight lane drawn past the finish, so the car does not lose the lane just before it.
FEATURES = {      # Extra paint on some tracks, in the trouble spots of a real arena.
    "straight": {"junctions": [150, 300], "crosswalk": [220]},
    "oval": {"glare": True},
    "s": {"dashed": True},
    "tight": {},
    "wiggle": {"glare": True, "junctions": [200]},
}


class Track:
    def __init__(self, name, lane_cm=lf.LANE_CM, seed=0):
        self.name = name
        self.finish = len(centerline(TRACKS[name])) - 1        # Index of the finish line.
        self.c = centerline(TRACKS[name] + [("S", RUN_OUT)])
        self.lane = lane_cm
        feat = FEATURES.get(name, {})
        rng = np.random.default_rng(seed)
        nx, ny = -np.sin(self.c[:, 2]), np.cos(self.c[:, 2])        # Left normal.
        left = self.c[:, :2] + np.stack([nx, ny], 1) * lane_cm / 2
        right = self.c[:, :2] - np.stack([nx, ny], 1) * lane_cm / 2
        pad = 80.0
        self.x0, self.y0 = self.c[:, 0].min() - pad, self.c[:, 1].min() - pad
        w = int((self.c[:, 0].max() + pad - self.x0) / RES)
        h = int((self.c[:, 1].max() + pad - self.y0) / RES)
        img = np.full((h, w), 28, np.uint8)
        img = np.clip(img.astype(np.int16) + rng.normal(0, 3, img.shape).astype(np.int16), 0, 255).astype(np.uint8)
        thick = max(int(round(LINE_CM / RES)), 1)

        def pix(p):
            return np.stack([(p[:, 0] - self.x0) / RES, (self.y0 + h * RES - p[:, 1]) / RES], 1)

        for line in (left, right):
            p = pix(line)
            if feat.get("dashed"):
                n = len(p)
                on = (np.arange(n) * 0.5 % 24) < 14       # 14 cm of paint, 10 cm gap.
                for a in range(1, n):
                    if on[a] and on[a - 1]:
                        cv2.line(img, tuple(np.int32(p[a - 1])), tuple(np.int32(p[a])), 235, thick, cv2.LINE_AA)
            else:
                cv2.polylines(img, [np.int32(p)], False, 235, thick, cv2.LINE_AA)
        s_cm = np.arange(len(self.c)) * 0.5
        for j in feat.get("junctions", []):      # A line across the lane, and the lanes branching off to both sides.
            k = int(np.searchsorted(s_cm, j))
            cx, cy, th = self.c[k]
            f, n = np.array([math.cos(th), math.sin(th)]), np.array([-math.sin(th), math.cos(th)])
            a, b = np.array([cx, cy]) - n * 60, np.array([cx, cy]) + n * 60
            cv2.line(img, *[tuple(np.int32(pix(q[None])[0])) for q in (a, b)], 235, thick, cv2.LINE_AA)
            for side in (-1, 1):
                for off in (-lane_cm, lane_cm):
                    q0 = np.array([cx, cy]) + f * off / 2 + n * side * lane_cm / 2
                    q1 = q0 + n * side * 50
                    cv2.line(img, *[tuple(np.int32(pix(q[None])[0])) for q in (q0, q1)], 235, thick, cv2.LINE_AA)
        for j in feat.get("crosswalk", []):
            k = int(np.searchsorted(s_cm, j))
            cx, cy, th = self.c[k]
            f, n = np.array([math.cos(th), math.sin(th)]), np.array([-math.sin(th), math.cos(th)])
            for m in range(-3, 4):
                base = np.array([cx, cy]) + n * m * 2.6
                poly = np.array([base - n * 0.9, base + n * 0.9, base + n * 0.9 + f * 8, base - n * 0.9 + f * 8])
                cv2.fillPoly(img, [np.int32(pix(poly))], 235, cv2.LINE_AA)
        self.img, self.h = img, h
        self.glare = feat.get("glare", False)
        self.length = self.finish * 0.5

    def lateral(self, x, y, k_hint):
        """Signed distance from the centerline (left positive), heading of the lane there, and the index searched from."""
        lo, hi = max(k_hint - 80, 0), min(k_hint + 400, len(self.c))
        d = np.hypot(self.c[lo:hi, 0] - x, self.c[lo:hi, 1] - y)
        k = lo + int(d.argmin())
        cx, cy, th = self.c[k]
        lat = -(x - cx) * math.sin(th) + (y - cy) * math.cos(th)
        return lat, th, k


class Camera:
    """Renders what the car's camera sees. true_*: how the simulated camera really is (a little off from the car's belief)."""

    def __init__(self, horizon, lane_px, cam_x, seed=0):
        self.W, self.H = lf.FRAME_W, lf.FRAME_H
        self.horizon, self.cx = horizon, cam_x * self.W
        self.F = lf.CAMERA_F
        self.height = lf.LANE_CM * (342 - horizon) / lane_px * self.F / lf.CAMERA_F      # Camera height in cm, from the flat-floor model.
        ys, xs = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        s = np.maximum(ys - horizon, 0.5)
        self.Z = self.F * self.height / s           # Floor distance ahead of the camera for each pixel row.
        self.X = (xs - self.cx) * self.Z / self.F   # Floor distance to the right.
        self.sky = ys < horizon + 1.5
        self.rng = np.random.default_rng(seed)
        self.light = (1.0 + 0.12 * np.cos(xs / self.W * 3.0) - 0.1 * (ys / self.H)).astype(np.float32)
        self.glare_spots = [(self.rng.uniform(0.15, 0.85) * self.W, self.rng.uniform(0.5, 0.95) * self.H, self.rng.uniform(10, 26))
                            for _ in range(3)]

    def render(self, track, x, y, th, t):
        cxw, cyw = x + CAM_AHEAD * math.cos(th), y + CAM_AHEAD * math.sin(th)
        fx, fy = math.cos(th), math.sin(th)
        rx, ry = math.sin(th), -math.cos(th)
        wx = cxw + self.Z * fx + self.X * rx
        wy = cyw + self.Z * fy + self.X * ry
        mx = ((wx - track.x0) / RES).astype(np.float32)
        my = ((track.y0 + track.h * RES - wy) / RES).astype(np.float32)
        g = cv2.remap(track.img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=28)
        g = g.astype(np.float32)
        g[self.sky] = 100       # The wall above the horizon.
        g = g * self.light
        if track.glare:     # Soft ceiling-light reflections on the glossy mat: they drift slowly, they are not paint.
            yy, xx = np.ogrid[0:self.H, 0:self.W]
            for k, (gx, gy, r) in enumerate(self.glare_spots):
                gx2 = gx + 30 * math.sin(t * 0.7 + k)
                g += 150 * np.exp(-((xx - gx2) ** 2 + (yy - gy) ** 2) / (2 * r * r))
        g = cv2.GaussianBlur(g, (0, 0), 1.1)
        g += self.rng.normal(0, 4, g.shape).astype(np.float32)
        bgr = cv2.merge([np.clip(g * c, 0, 255).astype(np.uint8) for c in (1.0, 0.98, 0.95)])
        return cv2.imdecode(cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])[1], cv2.IMREAD_COLOR)


_SETTINGS = ("LANE_WIDTH_PX", "HORIZON_Y", "CAMERA_X", "LINE_COLOR", "WHITE_THR", "LINE_MODEL", "XWALK_MODEL", "STOP_MODEL",
             "LIGHT_MODEL", "LANE_MODEL")


def _keeps_settings(fn):
    """run() and accuracy() set lane_follow's camera and model for the simulator: put them back afterwards, so a caller
    (the tests) goes on with its own settings."""
    import functools

    @functools.wraps(fn)
    def wrapped(*a, **k):
        saved = {n: getattr(lf, n) for n in _SETTINGS}
        try:
            return fn(*a, **k)
        finally:
            for n, v in saved.items():
                setattr(lf, n, v)
    return wrapped


def _check_model(model, known):
    """An unknown name would quietly drive as "bands" and its numbers be read as another model's."""
    if model not in known:
        raise ValueError(f"unknown model {model!r}: use {', '.join(known)} (LANE_POINTS is a setting: --set lane_follow.LANE_POINTS=bands)")


@_keeps_settings
def run(track_name, model, speed=0.5, steer=26, seconds=None, video=None, seed=0, quiet=True, mismatch=True):
    """One drive. Returns the metrics."""
    _check_model(model, ("bands", "track", "oracle"))
    for k, v in PI_VISION.items():
        setattr(lf, {"lane_width_px": "LANE_WIDTH_PX", "horizon_y": "HORIZON_Y", "camera_x": "CAMERA_X"}[k], v)
    lf.LINE_COLOR, lf.WHITE_THR, lf.LINE_MODEL, lf.XWALK_MODEL, lf.STOP_MODEL, lf.LIGHT_MODEL = "white", None, None, None, None, None
    lf.LANE_MODEL = "track" if model == "oracle" else model
    track = Track(track_name, seed=seed)
    rng = np.random.default_rng(seed + 1)
    off = (rng.uniform(-4, 4), rng.uniform(-0.03, 0.03), rng.uniform(-0.015, 0.015)) if mismatch else (0, 0, 0)
    cam = Camera(lf.HORIZON_Y + off[0], lf.LANE_WIDTH_PX * (1 + off[1]), lf.CAMERA_X + off[2], seed=seed)
    with contextlib.redirect_stdout(io.StringIO()):
        d = lf.Driver(cs._flat([cs.follow(600, until=None, steer=steer, speed=speed), cs.finish()]), base_speed=speed)
    x, y, th = float(track.c[0, 0]), float(track.c[0, 1]), float(track.c[0, 2])
    x += 1.5 * math.cos(th + math.pi / 2)       # Start a little off center.
    k, t, dt = 0, 0.0, 1 / FPS
    lats, steers, lost, times = [], [], 0, []
    writer = None
    steer_now, cmd_now = 0.0, 0.0
    limit = seconds or track.length / (SPEED_CM * speed) * 1.6 + 2
    result = "running"
    while t < limit:
        frame = cam.render(track, x, y, th, t)
        a = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            s, cmd, dbg = d.tick(frame, t, show=video is not None)
        times.append(time.perf_counter() - a)
        if d.det is None:
            lost += 1
        if video is not None:
            view = np.asarray(d.view(dbg, t, FPS))
            top = cv2.cvtColor(cv2.resize(track.img, (int(track.img.shape[1] * view.shape[0] / track.img.shape[0]), view.shape[0]), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
            sc = view.shape[0] / track.img.shape[0]
            px, py = (x - track.x0) / RES * sc, (track.y0 + track.h * RES - y) / RES * sc
            cv2.circle(top, (int(px), int(py)), 4, (0, 0, 255), -1)
            out = np.hstack([view, top])
            if writer is None:
                writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (out.shape[1], out.shape[0]))
            writer.write(out)
        # The command reaches the wheels one frame later (camera and servo latency).
        delta = math.radians(steer_now)
        v = SPEED_CM * cmd_now
        th -= v / WHEELBASE * math.tan(delta) * dt      # Positive steering turns right.
        x += v * math.cos(th) * dt
        y += v * math.sin(th) * dt
        if model == "oracle":
            # The same pursuit steering, fed with the true lane: shows what the steering itself can do with perfect sight.
            look = lf.LOOK_LW + lf.LOOK_SPEED_LW * speed
            height = (342 - cam.horizon) / (lf.LANE_WIDTH_PX)
            row = cam.horizon + lf.CAMERA_F * height / look
            xt = truth(cam, track, x, y, th, (row,))[0]
            if xt is not None:
                Z, X = look, (xt - cam.cx) * look / lf.CAMERA_F
                a = math.atan2(X, Z + lf.CAM_AHEAD_LW)
                target = math.degrees(math.atan(lf.WHEELBASE_LW * 2 * math.sin(a) / math.hypot(X, Z + lf.CAM_AHEAD_LW)))
                s = d.steering.output(float(np.clip(target, -steer, steer)), t)
        if s is not None:
            steer_now = s
        cmd_now = cmd
        steers.append(steer_now)
        t += dt
        mx, my = x + WHEELBASE / 2 * math.cos(th), y + WHEELBASE / 2 * math.sin(th)
        lat, _, k = track.lateral(mx, my, k)
        lats.append(lat)
        if abs(lat) > track.lane / 2 + 2:
            result = "LEFT THE LANE"
            break
        if k >= track.finish:
            result = "finished"
            break
    else:
        result = "TIMED OUT"        # Stopped or crawling: never reached the end.
    if writer is not None:
        writer.release()
    lats = np.array(lats)
    st = np.array(steers)
    return {"track": track_name, "model": model, "result": result, "done": round(min(k * 0.5 / track.length, 1.0), 3),
            "rms": float(np.sqrt(np.mean(lats ** 2))), "max": float(np.abs(lats).max()),
            "jitter": float(np.mean(np.abs(np.diff(st)))) if len(st) > 1 else 0.0, "lost": lost / max(len(lats), 1),
            "ms": float(np.mean(times) * 1e3)}


def truth(cam, track, x, y, th, rows):
    """The true lane center x on screen at these rows, from the track itself (None where it is not in view)."""
    cxw, cyw = x + CAM_AHEAD * math.cos(th), y + CAM_AHEAD * math.sin(th)
    fx, fy, rx, ry = math.cos(th), math.sin(th), math.sin(th), -math.cos(th)
    dx, dy = track.c[:, 0] - cxw, track.c[:, 1] - cyw
    Z, X = dx * fx + dy * fy, dx * rx + dy * ry
    near = np.hypot(dx, dy).argmin()
    seg = np.arange(len(Z)) >= near
    ok = seg & (Z > 3) & (Z < 400)
    if ok.sum() < 3:
        return [None] * len(rows)
    idx = np.flatnonzero(ok)
    run_end = idx[np.flatnonzero(np.diff(idx) > 1)[0]] if np.any(np.diff(idx) > 1) else idx[-1]
    idx = idx[idx <= run_end]
    ys = cam.horizon + cam.F * cam.height / Z[idx]
    xs = cam.cx + cam.F * X[idx] / Z[idx]
    out = []
    for r in rows:
        hit = np.flatnonzero((ys[:-1] - r) * (ys[1:] - r) <= 0)
        if not len(hit):
            out.append(None)
            continue
        j = hit[0]
        a = (r - ys[j]) / (ys[j + 1] - ys[j]) if ys[j + 1] != ys[j] else 0.0
        out.append(float(xs[j] + a * (xs[j + 1] - xs[j])))
    return out


@_keeps_settings
def accuracy(track_name, model, seed=0, step_cm=1.0, wobble_cm=4.0, wobble_deg=8.0):
    """Open loop: the car moves along the track with a known wobble; how close each frame's (offset, heading) is to the truth."""
    _check_model(model, ("bands", "track"))
    for k, v in PI_VISION.items():
        setattr(lf, {"lane_width_px": "LANE_WIDTH_PX", "horizon_y": "HORIZON_Y", "camera_x": "CAMERA_X"}[k], v)
    lf.LINE_COLOR, lf.WHITE_THR, lf.LINE_MODEL, lf.XWALK_MODEL, lf.STOP_MODEL, lf.LIGHT_MODEL = "white", None, None, None, None, None
    lf.LANE_MODEL = model
    track = Track(track_name, seed=seed)
    rng = np.random.default_rng(seed + 1)
    off = (rng.uniform(-4, 4), rng.uniform(-0.03, 0.03), rng.uniform(-0.015, 0.015))
    cam = Camera(lf.HORIZON_Y + off[0], lf.LANE_WIDTH_PX * (1 + off[1]), lf.CAMERA_X + off[2], seed=seed)
    det_ = lf.LaneDetector()
    top = int(lf.FRAME_H * lf.ROI_TOP)
    h = (lf.FRAME_H - top) // lf.SLICES
    near_y, far_y = lf.FRAME_H - h // 2, top + h // 2
    cx0 = int(lf.FRAME_W * lf.CAMERA_X)
    half = lf.FRAME_W / 2 * lf.LANE_WIDTH_PX / lf.REF_LANE_PX
    errs_o, errs_h, lost, bad, n, times = [], [], 0, 0, 0, []
    t = 0.0
    for k in range(0, track.finish, int(step_cm / 0.5)):
        cx_, cy_, th0 = track.c[k]
        ph = k * 0.5 / 37.0
        lat = wobble_cm * math.sin(ph)
        th = th0 + math.radians(wobble_deg * math.sin(ph * 1.7 + 1.0))
        x, y = cx_ - lat * math.sin(th0), cy_ + lat * math.cos(th0)
        frame = cam.render(track, x, y, th, t)
        det_.now, det_.moving = t, True
        a = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            d, _, _ = det_(frame, draw=False)
        times.append(time.perf_counter() - a)
        t += step_cm / 16.0
        xn, xf = truth(cam, track, x, y, th, (near_y, far_y))
        if xn is None or xf is None or not (0 <= xn < lf.FRAME_W and 0 <= xf < lf.FRAME_W):
            continue        # The true lane center is off screen (a tight curve): no reading can be right or wrong there.
        n += 1
        to, th_ = float(np.clip((xn - cx0) / half, -1, 1)), float(np.clip((xf - xn) / half, -1, 1))
        if d is None:
            lost += 1
            continue
        eo, eh = abs(d[0] - to), abs(d[1] - th_)
        errs_o.append(eo)
        errs_h.append(eh)
        bad += (eo > 0.15) or (eh > 0.25)
    return {"track": track_name, "model": model, "frames": n, "lost": lost / max(n, 1), "bad": bad / max(n, 1),
            "offset_err": float(np.mean(errs_o)) if errs_o else 1.0, "heading_err": float(np.mean(errs_h)) if errs_h else 1.0,
            "offset_p95": float(np.percentile(errs_o, 95)) if errs_o else 1.0, "ms": float(np.mean(times) * 1e3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", nargs="*", default=list(TRACKS))
    ap.add_argument("--model", nargs="*", default=["bands", "track"])
    ap.add_argument("--speed", type=float, default=0.5)
    ap.add_argument("--steer", type=float, default=26)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--video")
    ap.add_argument("--accuracy", action="store_true", help="open loop: detection error against the true lane, no steering")
    args = ap.parse_args()
    if args.accuracy:
        for model in args.model:
            rs = [accuracy(t, model, seed) for t in args.track for seed in range(args.seeds)]
            for r in rs:
                print(f"{r['track']:9s} {model:6s} offset err {r['offset_err']:.3f} (p95 {r['offset_p95']:.3f})  heading err {r['heading_err']:.3f}  "
                      f"bad {r['bad']:.0%}  lost {r['lost']:.0%}  {r['ms']:.1f}ms", flush=True)
            print(f"== {model:6s} offset err {np.mean([r['offset_err'] for r in rs]):.3f}  heading err {np.mean([r['heading_err'] for r in rs]):.3f}  "
                  f"bad {np.mean([r['bad'] for r in rs]):.0%}  lost {np.mean([r['lost'] for r in rs]):.0%}\n", flush=True)
        return
    rows = []
    for name in args.track:
        for model in args.model:
            for seed in range(args.seeds):
                r = run(name, model, args.speed, args.steer, video=args.video, seed=seed)
                rows.append(r)
                print(f"{name:9s} {model:6s} seed {seed}  {r['result']:14s} done {r['done']:.0%}  lateral rms {r['rms']:.2f}cm  max {r['max']:.2f}cm  "
                      f"steer jitter {r['jitter']:.2f}deg  lane lost {r['lost']:.0%}  {r['ms']:.1f}ms", flush=True)
    print()
    for model in args.model:
        mine = [r for r in rows if r["model"] == model]
        ok = sum(r["result"] == "finished" for r in mine)
        print(f"{model:6s}  finished {ok}/{len(mine)}  mean rms {np.mean([r['rms'] for r in mine]):.2f}cm  worst max {max(r['max'] for r in mine):.2f}cm  "
              f"mean jitter {np.mean([r['jitter'] for r in mine]):.2f}deg  mean {np.mean([r['ms'] for r in mine]):.1f}ms")


if __name__ == "__main__":
    main()
