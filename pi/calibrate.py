"""Automatic venue calibration. Place the car in the center of a straight lane, pointing along the lane, then run.

  python calibrate.py           measure, then save to vision.json (the previous file is kept as vision.json.bak)
  python calibrate.py --dry     measure only, do not save
  python calibrate.py --focus-only   only measure where focus would be sharpest. Saves nothing: focus stays at lf.LENS_POSITION

Steps
  1. Track color     a bright floor is a white floor with a black line; a dark floor is a black floor with a white line
  2. Focus           fixed at lf.LENS_POSITION (about 15 cm), not measured
  3. Exposure        settle auto exposure and white balance, then lock them. Shorten exposure so the white line does not blow out, and keep the exposure time at or below MAX_EXPOSURE_US so the line does not blur while driving. Gain makes up any missing brightness.
  4. Lane width and vanishing point   measure both line positions in each band and fit the perspective model (LANE_WIDTH_PX, HORIZON_Y)
  5. Camera center   assume the car is in the middle of the lane and record where the lane center falls at the bottom of the frame (CAMERA_X)
  6. Check           detect again with the values to be saved and write the result view to calib_view.jpg

Servo center and the traffic light cannot be set here. Servo center needs the car to move (Tune Center). The traffic light needs a lit signal (Tune Traffic Light).
"""
import argparse
import json
import shutil
import time

import cv2
import numpy as np

import lane_follow as lf

MAX_EXPOSURE_US = 8000      # Exposure-time cap, in microseconds. A longer exposure blurs the line while driving, so the rest of the brightness comes from gain.
MAX_GAIN = 8.0
CLIP_LEVEL = 250            # Brightness at or above this counts as blown out to white.
CLIP_RATIO = 0.003          # If blown-out pixels in the lane region exceed this fraction, reduce the exposure.
FRAMES = 30                 # Frames used for the lane measurement.
VIEW_FILE = lf.VISION_FILE.with_name("calib_view.jpg")


def roi_gray(frame):
    top = int(lf.FRAME_H * lf.ROI_TOP)
    return cv2.cvtColor(frame[top:], cv2.COLOR_BGR2GRAY)


def detect_track_color(frames):
    """Judge track color from floor brightness, the median of the lane region: "white" (black floor, white line) or "black" (white floor, black line)."""
    med = float(np.median([np.median(roi_gray(f)) for f in frames]))
    return ("black" if med > 128 else "white"), med


def measure_lanes(frames):
    """Median left and right line x in each band, measured from the frame center: [(y, left x, right x)].

    The lines are measured straight from the mask, independent of the lane-width model, so a wrong model still measures.
    """
    det = lf.LaneDetector()
    top = int(lf.FRAME_H * lf.ROI_TOP)
    samples = {}
    for f in frames:
        mask = det.line_mask(f[top:])
        h = mask.shape[0] // lf.SLICES
        for i in range(lf.SLICES):
            y1 = mask.shape[0] - (i + 1) * h
            cy = top + y1 + h // 2
            lines, hrun, crosswalk = det.find_lines(mask[y1:y1 + h])
            if hrun is not None or crosswalk:
                continue
            cx = lf.FRAME_W / 2
            lefts, rights = [x for x in lines if x < cx], [x for x in lines if x >= cx]
            if lefts and rights:
                samples.setdefault(cy, []).append((max(lefts), min(rights)))
    bands = []
    for cy, pairs in sorted(samples.items()):
        if len(pairs) >= len(frames) // 2:      # Keep only bands where both lines were visible in at least half the frames.
            arr = np.array(pairs)
            bands.append((cy, float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))))
    return bands


FIT_TOLERANCE = 6.0     # A band farther than this many px from the lane-width line is dropped and the fit is redone, such as a bend far away.


def fit_geometry(bands):
    """Fit both lines in each band to the lane-width model (width at y=342 and vanishing-point y), plus camera center and car tilt.

    Start with the three most reliable near bands, then add a farther band only while it stays within FIT_TOLERANCE of that line, so a bent or broken far band cannot drag the whole line. Returns a dict, or an error message (str).
    """
    bands = sorted(bands, key=lambda b: -b[0])     # Nearest (lowest) band first.
    used = bands[:3]
    for b in bands[3:]:
        ys = np.array([u[0] for u in used], float)
        coef = np.polyfit(ys, [u[2] - u[1] for u in used], 1)
        if abs(np.polyval(coef, b[0]) - (b[2] - b[1])) > FIT_TOLERANCE:
            break                                   # From here upward the bands leave the straight model.
        used.append(b)
    bands = used
    if len(bands) < 3:
        return f"Only {len(bands)} bands showed both lines (3 or more needed). Check the car is in the middle of a straight lane"
    ys = np.array([b[0] for b in bands], float)
    widths = np.array([b[2] - b[1] for b in bands])
    centers = np.array([(b[1] + b[2]) / 2 for b in bands])
    a, b = np.polyfit(ys, widths, 1)        # Width = a*y + b. Perspective makes the lane wider toward the bottom.
    if a <= 0:
        return "The lane does not narrow with distance (wrong lines may have been read). Point the car straight along the lane"
    horizon = -b / a
    width_342 = a * 342 + b
    resid = float(np.max(np.abs(np.polyval((a, b), ys) - widths)))
    k, c0 = np.polyfit(ys, centers, 1)      # Lane-center x = k*y + c0.
    bottom_x = k * lf.FRAME_H + c0
    return {"lane_width_px": round(float(width_342), 1), "horizon_y": round(float(horizon), 1),
            "camera_x": round(float(bottom_x / lf.FRAME_W), 4),
            "tilt": float(k), "fit_error_px": resid, "bands": len(bands)}


def capture(cam, n=1, settle=0.0):
    time.sleep(settle)
    return [cam.capture_array() for _ in range(n)]


# Focus is chosen by sharpness where the lane is read. Autofocus locks on the center of the frame, the far floor, and leaves the near lane soft.
FOCUS_REGIONS = ((0.0, 0.3, 0.2), (0.3, 0.7, 0.4), (0.7, 1.0, 0.4))   # (top, bottom, weight) inside the ROI: far, middle, near.


def sharpness(frame):
    """Sharpness of each ROI region, as the variance of the Laplacian."""
    top = int(lf.FRAME_H * lf.ROI_TOP)
    lap = cv2.Laplacian(cv2.cvtColor(frame[top:], cv2.COLOR_BGR2GRAY), cv2.CV_64F)
    n = lap.shape[0]
    return [float(lap[int(a * n):int(b * n)].var()) for a, b, _ in FOCUS_REGIONS]


def best_focus(table):
    """Pick a lens position from {lens position: per-region sharpness}. Each region is divided by its peak, then the weighted average decides."""
    peaks = [max(v[i] for v in table.values()) or 1.0 for i in range(len(FOCUS_REGIONS))]
    score = {lp: sum(w * v[i] / peaks[i] for i, (_, _, w) in enumerate(FOCUS_REGIONS)) for lp, v in table.items()}
    return max(score, key=score.get)


def calibrate_focus(cam, controls):
    """Sweep the lens from near to far and return the position, in diopters, that is sharpest on the lane region, weighted toward the near side. None on failure."""
    table = {}

    def measure(lp):
        cam.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": float(lp)})
        time.sleep(0.35)
        cam.capture_array()
        table[round(float(lp), 2)] = sharpness(cam.capture_array())

    for lp in np.arange(1.5, 12.01, 0.75):      # 67 cm to 8 cm.
        measure(lp)
    coarse = best_focus(table)
    for lp in np.arange(coarse - 0.5, coarse + 0.51, 0.25):     # Finer steps around that position.
        if 0.5 <= lp <= 12 and round(float(lp), 2) not in table:
            measure(lp)
    if max(sum(v) for v in table.values()) < 1.0:
        return None     # The frame is completely uniform, for example because the camera is covered.
    return best_focus(table)


def calibrate_exposure(cam):
    """Lock the settled auto exposure and white balance, cap exposure so the image does not blur, and reduce it so highlights do not blow out."""
    cam.set_controls({"AeEnable": True, "AwbEnable": True})
    time.sleep(2.5)
    md = cam.capture_metadata()
    exp, gain = float(md["ExposureTime"]), float(md["AnalogueGain"])
    gains = [float(g) for g in md["ColourGains"]]
    if exp > MAX_EXPOSURE_US:       # Keep the same brightness with a shorter exposure and a higher gain.
        gain, exp = min(MAX_GAIN, gain * exp / MAX_EXPOSURE_US), MAX_EXPOSURE_US
    for _ in range(8):              # If the white line, the bright side, blows out, reduce a little at a time.
        cam.set_controls({"AeEnable": False, "AwbEnable": False, "ExposureTime": int(exp),
                          "AnalogueGain": gain, "ColourGains": tuple(gains)})
        frame = capture(cam, 1, settle=0.4)[0]
        clipped = float(np.mean(roi_gray(frame) >= CLIP_LEVEL))
        if clipped <= CLIP_RATIO:
            break
        if gain > 1.05:
            gain = max(1.0, gain * 0.8)
        else:
            exp *= 0.8
    return {"ExposureTime": int(exp), "AnalogueGain": round(gain, 3),
            "ColourGains": [round(g, 4) for g in gains]}, clipped


def check_result(frames):
    """Detect again with the values about to be saved: position error, fraction of frames with both lines, and the last frame."""
    det = lf.LaneDetector()
    offsets, both, dbg = [], 0, None
    for f in frames:
        d, _, dbg = det(f)
        if d is not None:
            offsets.append(d[0])
        both += det.left[0] is not None and det.right[0] is not None
    return (float(np.median(offsets)) if offsets else None), both / max(len(frames), 1), dbg


def annotate(dbg, lines):
    y = 24
    for text in lines:
        cv2.putText(dbg, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 4, cv2.LINE_AA)
        cv2.putText(dbg, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    return dbg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="measure only, save nothing")
    ap.add_argument("--focus-only", action="store_true", help="only measure the sharpest focus, save nothing (car on the track)")
    args = ap.parse_args()
    from libcamera import controls

    if args.focus_only:
        cam = lf.open_camera()      # Keep the saved locked exposure.
        try:
            lens = calibrate_focus(cam, controls)
        finally:
            cam.stop()
            cam.close()
        if lens is None:
            print("Focus: measuring failed (check nothing blocks the camera) -> kept as it was")
            return
        print(f"Focus: sharpest at lens {lens:.2f} (about {100 / lens:.0f} cm). Not saved: driving stays at lens {lf.LENS_POSITION} (about {100 / lf.LENS_POSITION:.0f} cm)")
        return

    print(__doc__.split("\n\n")[0])
    cam = lf.open_camera(locked=False)
    saved, notes = {}, []
    try:
        print("\n[1/6] Track color ...", flush=True)
        frames = capture(cam, 5, settle=1.5)
        color, med = detect_track_color(frames)
        lf.LINE_COLOR = color
        saved["line_color"] = color
        print(f"      floor brightness {med:.0f} -> {'black lines on a white floor' if color == 'black' else 'white lines on a black floor'}")

        print("[2/6] Focus ...", flush=True)
        cam.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": lf.LENS_POSITION})
        print(f"      lens {lf.LENS_POSITION} (fixed at about {100 / lf.LENS_POSITION:.0f} cm)")

        print("[3/6] Light (exposure / gain / white balance) ...", flush=True)
        lock, clipped = calibrate_exposure(cam)
        saved["camera_lock"] = lock
        print(f"      exposure {lock['ExposureTime']} us  gain {lock['AnalogueGain']}  "
              f"white balance R{lock['ColourGains'][0]} B{lock['ColourGains'][1]}  (clipped pixels {clipped:.1%})")

        print("[4/6] Lane width / vanishing point ...", flush=True)
        frames = capture(cam, FRAMES, settle=0.3)
        geo = fit_geometry(measure_lanes(frames))
        geo_ok = not isinstance(geo, str)
        if not geo_ok:
            notes.append("Lane: " + geo)
            print("      " + geo)
        else:
            saved["lane_width_px"], saved["horizon_y"] = geo["lane_width_px"], geo["horizon_y"]
            print(f"      {geo['bands']} bands measured  lane width {lf.LANE_WIDTH_PX} -> {geo['lane_width_px']}px (y=342)  "
                  f"vanishing point {lf.HORIZON_Y} -> {geo['horizon_y']}  (max error {geo['fit_error_px']:.0f}px)")

            print("[5/6] Camera center ...", flush=True)
            if abs(geo["tilt"]) > 0.25:
                notes.append("Camera center: skipped, the car is at an angle to the lane (point it straight along the lane and run again)")
                print("      the car is at an angle to the lane -> skipped")
            elif not 0.4 <= geo["camera_x"] <= 0.6:
                notes.append(f"Camera center: {geo['camera_x']} is too far off (put the car in the middle of the lane and run again)")
                print(f"      {geo['camera_x']} -> too far off, skipped")
            else:
                saved["camera_x"] = geo["camera_x"]
                print(f"      {lf.CAMERA_X} → {geo['camera_x']}")
        saved["white_thr"] = None       # The white-line threshold stays automatic. Locked exposure keeps it stable.

        print("[6/6] Check ...", flush=True)
        for key, value in saved.items():        # Apply this run's values immediately and detect again.
            name = {"line_color": "LINE_COLOR", "lane_width_px": "LANE_WIDTH_PX", "horizon_y": "HORIZON_Y",
                    "camera_x": "CAMERA_X", "white_thr": "WHITE_THR", 
                    "camera_lock": "CAMERA_LOCK"}[key]
            setattr(lf, name, value)
        frames = capture(cam, 20, settle=0.3)
        offset, both, dbg = check_result(frames)
        ok = geo_ok and offset is not None and both >= 0.8 and abs(offset) < 0.1
        off_txt = "-" if offset is None else f"{offset:+.3f}"
        print(f"      both lines seen {both:.0%}  position error {off_txt}  -> {'OK' if ok else 'check'}")
        if not ok:
            notes.append("Check: detection is unsteady -> see calib_view.jpg and check the car position and light")
        annotate(dbg, [f"CALIBRATION {'OK' if ok else 'CHECK'}  both lines {both:.0%}  offset {off_txt}",
                       f"track {color}  lane {saved.get('lane_width_px', lf.LANE_WIDTH_PX)}px  "
                       f"horizon {saved.get('horizon_y', lf.HORIZON_Y)}  cam_x {saved.get('camera_x', lf.CAMERA_X)}",
                       f"exp {lock['ExposureTime']}us  gain {lock['AnalogueGain']}  "
                       f"lens {lf.LENS_POSITION}"])
        cv2.imwrite(str(VIEW_FILE), dbg)
    finally:
        cam.stop()
        cam.close()

    if args.dry:
        print("\n--dry: nothing saved")
    else:
        if lf.VISION_FILE.exists():
            shutil.copy(lf.VISION_FILE, lf.VISION_FILE.with_suffix(".json.bak"))
        lf.save_vision(**saved)
        print(f"\nSaved: {lf.VISION_FILE}  (previous: vision.json.bak)")
    print(json.dumps(saved, ensure_ascii=False))
    for n in notes:
        print("[warning] " + n)
    print("Not done automatically: servo center (Tune Center), traffic light (Tune Traffic Light)")


if __name__ == "__main__":
    main()
