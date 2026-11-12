"""Register the traffic light, crosswalk, lane lines or stop points of the day on the web page, like the LEGO figure.
Normally started by aicar (light / crosswalk / line / stoppoint).

  python tune_landmark.py light|crosswalk|line|stop          new (the saved one stays until Enter)
  python tune_landmark.py light|crosswalk|line|stop --add    add to the saved one

Light: put the car where it stops for the light. Turn the red lamp on and drag a box around it on
       http://<pi-ip>:8000. Then turn the green lamp on and drag around it. wait_green then looks only for
       lamps like these, near where they were. With none registered, the color thresholds are used as before.
Crosswalk: put the car on the lane before the crosswalk and drag a box around its stripes. Push the car a
       little nearer and drag again, two or three times. _to_crosswalk() and _pass_crosswalk() use them.
       to_crosswalk() and pass_crosswalk() stay as they are.
Line: drag boxes around lane lines, in a few places (straight, curve, under bright lights). The reflection
       cut is then set to how crisp this venue's paint is. The other filters stay on.
Stop: stand the car exactly where it should stop, and drag a box around what marks the place ahead (the
       lane's line end, the line across, a corner). That is stop point A. Move it 1-2 cm and drag again for a
       surer match. n starts the next point (B, C, ...). In course.py: go_to("A").

Left: camera with what is found now. Right: the registered views so far.
"""
import argparse
import sys
import termios
import tty

import cv2
import numpy as np

import landmarks
import lane_follow as lf

VIEW_FRAMES = 5
LINE_SHARP_MIN = 0.3    # A "line" softer than this is a ceiling-light reflection (they measure 0.17-0.33, paint 0.37-0.5).
HELP = {
    "light": """
  Turn the red lamp on and drag a box tightly around it on the web page. Then the green lamp, the same way.
  Each box is one lamp. More boxes (from where the car may stop) make it surer.

    Enter (twice)   save and quit         u   undo the last lamp
    q (twice)       quit without saving
""",
    "line": """
  Drag a box around a piece of lane line on the web page. Do it in a few places: straight, a curve, and
  where ceiling lights reflect on the mat. Each box is one sample.

    Enter (twice)   save and quit         u   undo the last sample
    q (twice)       quit without saving
""",
    "stop": """
  Stand the car exactly where it should stop. Drag a box around what marks the place ahead:
  the lane's line end, the line across, or a corner. Move the car 1-2 cm and drag again (2-3 views).
  Then n for the next stop point, and so on. Use them in course.py as go_to("A").
  Or press Space: the whole view is registered (no box). Good where no one mark stands out.

    Space           register the whole view as a view of this point
    n               next stop point (A, B, C, ...)
    Enter (twice)   save and quit         u   undo the last view
    q (twice)       quit without saving
""",
    "crosswalk": """
  Drag a box around the crosswalk stripes on the web page (not the lane lines beside them).
  Push the car a little nearer and drag again, two or three times, so near and far both match.

    Enter (twice)   save and quit         u   undo the last view
    q (twice)       quit without saving
""",
}


def grab(cam, n):
    frames = [cam.capture_array() for _ in range(n)]
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def crosswalk_view(frame, box):
    """Stripes inside the dragged box, as [[w, h, fill], ...] in lane widths, or (None, reason)."""
    det = lf.LaneDetector()
    for _ in range(3):      # A few passes so the lane lines (left / right) settle.
        det(frame, draw=False)
    top = int(lf.FRAME_H * lf.ROI_TOP)
    roi = frame[top:]
    mask = det.line_mask(roi)
    signs = det.sign_area(roi)
    if signs is not None:
        mask[signs] = 0
    comps = det.drop_reflections(roi, mask)
    n, labels, stats, _ = comps
    H, W = labels.shape
    bx0, by0 = int(box[0] * lf.FRAME_W), int(box[1] * lf.FRAME_H) - top
    bx1, by1 = int(box[2] * lf.FRAME_W), int(box[3] * lf.FRAME_H) - top
    stripes = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < lf.STRIPE_MIN_AREA:
            continue
        ix = max(0, min(x + w, bx1) - max(x, bx0))
        iy = max(0, min(y + h, by1) - max(y, by0))
        if ix * iy < 0.7 * w * h:
            continue        # Mostly outside the box: a lane line running through, not a stripe.
        w_l, h_l, fill = det.stripe_size(stats, i, top)
        cy = top + y + h / 2
        lane_w = max(lf.LANE_WIDTH_PX * (cy - lf.HORIZON_Y) / max(342 - lf.HORIZON_Y, 1.0), 1.0)
        sharp = det.blob_sharpness(roi, comps, i) or 0.0
        stripes.append([round(float(v), 4) for v in (w_l, h_l, fill, sharp, (x + w / 2) / lane_w)])
    if len(stripes) < 2:
        return None, f"{len(stripes)} stripe(s) in the box: drag around the crosswalk stripes (2 or more)"
    return stripes, f"{len(stripes)} stripes"


def settled(frame):
    """A detector that has looked at this frame a few times, so the lane lines (left / right) have settled."""
    det = lf.LaneDetector()
    for _ in range(3):
        det(frame, draw=False)
    return det


def line_sample(frame, box):
    """The lane line in the dragged box: {"sharp", "width", "contrast"}, or (None, reason)."""
    det = settled(frame)
    top = int(lf.FRAME_H * lf.ROI_TOP)
    roi = frame[top:]
    mask = det.line_mask(roi)
    signs = det.sign_area(roi)
    if signs is not None:
        mask[signs] = 0
    comps = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    n, labels, stats, _ = comps
    bx0, by0 = int(box[0] * lf.FRAME_W), max(int(box[1] * lf.FRAME_H) - top, 0)
    bx1, by1 = int(box[2] * lf.FRAME_W), int(box[3] * lf.FRAME_H) - top
    if by1 <= by0 + 2:
        return None, "the box is above the area used for lines: drag lower"
    inside = labels[by0:by1, bx0:bx1]
    ids, counts = np.unique(inside[inside > 0], return_counts=True)
    if not len(ids):
        return None, "no line in the box"
    i = int(ids[np.argmax(counts)])
    sharp = det.blob_sharpness(roi, comps, i)
    rows = [(r, np.flatnonzero(labels[r, bx0:bx1] == i)) for r in range(by0, by1)]
    widths = []
    for r, xs in rows:
        if len(xs):
            lane_w = max(lf.LANE_WIDTH_PX * (r + top - lf.HORIZON_Y) / max(342 - lf.HORIZON_Y, 1.0), 1.0)
            widths.append(len(xs) / lane_w)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if lf.LINE_COLOR == "black":
        gray = 255 - gray
    blob = labels[by0:by1, bx0:bx1] == i
    contrast = float(np.percentile(gray[by0:by1, bx0:bx1][blob], 90) - np.median(gray[by0:by1, bx0:bx1][~blob])) if (~blob).any() else 0.0
    if sharp is None or not widths:
        return None, "the line in the box could not be measured"
    if sharp < LINE_SHARP_MIN:
        return None, f"what is in the box has soft edges ({sharp:.2f}): a light reflection, not a line. Drag around a painted line"
    return {"sharp": round(float(sharp), 4), "width": round(float(np.median(widths)), 4), "contrast": round(contrast, 1)}, \
        f"line: edge {sharp:.2f}, width {np.median(widths):.2f} lane, contrast {contrast:.0f}"


def frame_mask(frame):
    """The detector's line mask for this frame, frame size (0 above the area used for lines)."""
    det = settled(frame)
    top = int(lf.FRAME_H * lf.ROI_TOP)
    full = np.zeros(frame.shape[:2], np.uint8)
    full[top:] = det.mask
    return full


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def next_name(model):
    """The first letter A-Z not registered yet, or None when all 26 are."""
    names = set(model.names())
    return next((c for c in LETTERS if c not in names), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=("light", "crosswalk", "line", "stop"))
    ap.add_argument("--add", action="store_true", help="add to the saved one")
    ap.add_argument("--point", help="stop: add views to this registered point (A-Z)")
    args = ap.parse_args()
    kind = args.kind
    cls = {"light": landmarks.LightModel, "crosswalk": landmarks.CrosswalkModel, "line": landmarks.LineModel, "stop": landmarks.StopModel}[kind]
    path = {"light": landmarks.LIGHT_FILE, "crosswalk": landmarks.XWALK_FILE, "line": landmarks.LINE_FILE, "stop": landmarks.STOP_FILE}[kind]
    saved = cls.load() if args.add else None
    model = saved or cls()
    count = lambda: (len(model.samples) if kind in ("light", "line") else len(model.views) if kind == "crosswalk"
                     else sum(len(p["views"]) for p in model.points.values()))
    name = next_name(model) if kind == "stop" else None
    if kind == "stop" and args.point:
        name = args.point.strip().upper()
        if len(name) != 1 or name not in LETTERS:
            sys.exit(f"  --point must be one letter A-Z ({args.point})")
        have = len(model.points.get(name, {}).get("views", []))
        print(f"\n  More views for stop point {name} ({have} so far)." if have else f"\n  Stop point {name} is new.")
    elif kind == "stop" and name is None:
        sys.exit("  All 26 stop points (A-Z) are registered. Add views to one with: aicar stoppoint more")
    added = []          # Stop points: the names in the order views were added, for undo.
    print(f"\n  {'Adding to the saved' if saved else 'New'} {kind}{f' ({count()} saved)' if saved else ''}. Enter saves, q leaves the saved one as it is.")
    print(HELP[kind])
    if lf.CAMERA_LOCK is None:
        print("  [warning] Exposure is not locked. Run aicar calibrate first for steadier results.\n")

    cam = lf.open_camera()
    keys = "drag a box or Space whole view · n next point · u undo · Enter twice save · q twice quit" if kind == "stop" else "drag a box · u undo · Enter twice save · q twice quit without saving"
    stream = lf.Stream(8000, f"AICAR {kind}", keys, confirm="\nq", drag_share=0.5)
    traffic, det = lf.TrafficLight(), lf.LaneDetector()

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    note = None
    try:
        while True:
            # The model being made is what the view uses, so each box shows its effect at once.
            if kind == "light":
                lf.LIGHT_MODEL = model if model.samples else None
            elif kind == "crosswalk":
                model.fit()
                lf.XWALK_MODEL = model if model.views else None
            elif kind == "line":
                lf.LINE_MODEL = model if model.samples else None
            frame = cam.capture_array()
            if kind == "light":
                view = frame.copy()
                state = traffic(frame)
                traffic.draw(view)
                found = f"light {state or '-'}"
            else:
                _, _, view = det(frame, draw=True)
                if kind == "crosswalk":
                    found = f"crosswalk {'FOUND' if det.xwalk2_y is not None else '-'}"
                elif kind == "line":
                    found = f"reflection cut {model.reflect_cut():.2f}" if model.samples else "reflection cut (default)"
                else:
                    found = f"now registering {name}"
                    top = int(lf.FRAME_H * lf.ROI_TOP)
                    for nm in model.names():
                        hit = model.find(nm, det.mask, top)
                        if hit:
                            cv2.rectangle(view, (hit["x"], hit["y"]), (hit["x"] + hit["w"], hit["y"] + hit["h"]), lf.C_AT, 2)
                            lf.chip(view, f"{nm} {hit['score']:.2f}", hit["x"], max(hit["y"] - 22, 0), lf.C_AT, 0.4)
                        lf.dashed_hline(view, int(model.stop_y(nm)), lf.C_AT, dash=10, gap=6)
            side = np.zeros_like(frame)
            if kind == "light":
                lines = [f"{s['color']}  area {s['area']}  hue {s['h'][0]}..{s['h'][1]}" for s in model.samples]
            elif kind == "crosswalk":
                lines = [f"view {k + 1}: {len(v['stripes'])} stripes" for k, v in enumerate(model.views)]
            elif kind == "line":
                lines = [f"sample {k + 1}: edge {s['sharp']:.2f} width {s['width']:.2f} contrast {s['contrast']:.0f}" for k, s in enumerate(model.samples)]
            else:
                lines = [f"{nm}: {len(model.points[nm]['views'])} views, stops at y {model.stop_y(nm):.0f}" for nm in model.names()] + [f"(next box: {name})"]
            for k, text in enumerate(lines[-10:]):
                cv2.putText(side, text, (10, 30 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(view, f"{count()} registered  |  {found}", (10, lf.FRAME_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            stream.update(np.hstack([view, side]))
            print(f"\r  {count()} registered  |  {found}      ", end="", flush=True)
            if note:
                print(f"\n  {note}")
                note = None

            box = stream.read_box()
            if box:
                shot = grab(cam, VIEW_FRAMES)
                if kind == "light":
                    sample, note = landmarks.LightModel.sample(shot, box)
                    if sample:
                        model.samples.append(sample)
                        note = f"Added: {note}"
                elif kind == "crosswalk":
                    stripes, note = crosswalk_view(shot, box)
                    if stripes:
                        model.views.append({"stripes": stripes})
                        note = f"Added: {note}"
                elif kind == "line":
                    sample, note = line_sample(shot, box)
                    if sample:
                        model.samples.append(sample)
                        note = f"Added: {note}"
                else:
                    x0, y0 = int(box[0] * lf.FRAME_W), int(box[1] * lf.FRAME_H)
                    x1, y1 = int(box[2] * lf.FRAME_W), int(box[3] * lf.FRAME_H)
                    top = int(lf.FRAME_H * lf.ROI_TOP)
                    if y1 <= top + 8:
                        note = "The box is above the area used for lines: drag around something lower in the view"
                    else:
                        y0 = max(y0, top)
                        ok, note = model.add(name, frame_mask(shot), (x0, y0, x1 - x0, y1 - y0))
                        if ok:
                            added.append(name)
                            note = f"Added: {note}  (stop where its bottom is at y {model.stop_y(name):.0f})"
            for key in stream.read_keys(terminal=False).lower() + (sys.stdin.read(1).lower() if lf.select.select([sys.stdin], [], [], 0)[0] else ""):
                if key == " " and kind == "stop":
                    shot = grab(cam, VIEW_FRAMES)
                    mask = frame_mask(shot)
                    ok, note = model.add(name, mask, landmarks.StopModel.whole_box(mask.shape, int(lf.FRAME_H * lf.ROI_TOP)), whole=True)
                    if ok:
                        added.append(name)
                        note = f"Added whole view: {note}"
                elif key == "n" and kind == "stop":
                    if name in model.points:
                        nxt = next_name(model)
                        if nxt is None:
                            note = "All 26 stop points (A-Z) are registered"
                            continue
                        name = nxt
                    note = f"Next stop point: {name}"
                elif key == "u":
                    if kind == "stop":
                        if added:
                            model.undo(added.pop())
                            note = "Last view undone"
                    elif kind in ("light", "line") and model.samples:
                        model.samples.pop()
                        note = f"Last one undone ({count()} left)"
                    elif kind == "crosswalk" and model.views:
                        model.views.pop()
                        note = f"Last one undone ({count()} left)"
                elif key in ("\n", "\r"):
                    if not count():
                        note = "Nothing registered yet, nothing to save"
                        continue
                    model.save()
                    extra = f" ({', '.join(model.names())})" if kind == "stop" else ""
                    print(f"\n\n  Saved: {count()} in {path.name}{extra}. Used from the next drive.")
                    if kind == "line" and len(model.samples) < 2:
                        print("  [warning] Only one place registered: add one or two more (a curve, under bright lights) with --add.")
                    return
                elif key == "q":
                    print("\n\n  Quit without saving.")
                    return
    except KeyboardInterrupt:
        print("\n\n  Quit without saving.")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        cam.stop()


if __name__ == "__main__":
    main()
