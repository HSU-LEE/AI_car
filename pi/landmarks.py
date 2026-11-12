"""Traffic light and crosswalk registered on the day, from boxes dragged on the web page (aicar light / aicar crosswalk).

Light: each registration is one lit lamp (red or green). Its color range, size and place in the view are kept.
       Once registered, wait_green looks only for lamps like these, near where they were (traffic light: TrafficLight).
Crosswalk: each registration is the crosswalk's stripes seen from one place. Their size and shape, in lane widths,
       are kept. _to_crosswalk() and _pass_crosswalk() use them. The older crosswalk test is not changed.

Files sit beside lane_follow.py (light.json, crosswalk.json), and a recording keeps a copy.
"""
import json
from pathlib import Path

import cv2
import numpy as np

LIGHT_FILE = Path(__file__).with_name("light.json")
XWALK_FILE = Path(__file__).with_name("crosswalk.json")

LIGHT_HUE_PAD = 5       # Hue margin around what was registered.
LIGHT_S_PAD, LIGHT_V_PAD = 30, 40   # Saturation / value margin below what was registered (a lamp looks a little dimmer from farther away).
LIGHT_AREA_RANGE = (0.2, 5.0)       # A lamp this many times smaller / larger than registered still counts (nearer or farther).
LIGHT_REACH = (0.22, 0.18)          # How far from the registered place (share of frame width, height) the lamp may be: the car stops a little differently.

XWALK_SIZE_RANGE = (0.6, 1.7)       # A stripe this many times smaller / larger (in lane widths) than registered still matches.
XWALK_FILL_PAD = 0.15               # How much less filled than registered a stripe may be.
XWALK_COUNT_SHARE = 0.6             # This share of the registered stripe count must be seen (at least 2).
XWALK_SHARP_SHARE = 0.8             # Stripes must be at least this share as crisp as registered (a ceiling-light reflection is soft).
XWALK_PITCH_RANGE = (0.6, 1.7)      # Stripe to next stripe (center to center, lane widths): this many times the registered spacing.
XWALK_ROW = 0.6                     # Stripes of one crosswalk sit side by side: centers within this share of a stripe height of each other.


def red_hue(h):
    """Hue as distance around red: red sits at both ends of OpenCV's 0-180, so 175 becomes -5."""
    h = np.asarray(h, np.int16)
    return np.where(h > 90, h - 180, h)


class LightModel:
    """Registered lamps. samples: [{"color", "h": [lo, hi], "s_min", "v_min", "area", "box": [x0, y0, x1, y1] (0-1)}]."""

    def __init__(self, samples=None):
        self.samples = list(samples or [])

    @classmethod
    def load(cls, path=LIGHT_FILE):
        try:
            data = json.loads(Path(path).read_text())
            return cls(data.get("samples", [])) if data.get("samples") else None
        except (OSError, ValueError, AttributeError):
            return None

    def save(self, path=LIGHT_FILE):
        Path(path).write_text(json.dumps({"samples": self.samples}))

    def colors(self):
        return {s["color"] for s in self.samples}

    @staticmethod
    def sample(frame, box):
        """A lamp from the box (fractions of the frame) dragged around a lit lamp, or (None, reason)."""
        H, W = frame.shape[:2]
        x0, y0, x1, y1 = int(box[0] * W), int(box[1] * H), int(box[2] * W), int(box[3] * H)
        if x1 - x0 < 3 or y1 - y0 < 3:
            return None, "the box is too small"
        hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        lit = (sat >= 60) & (val >= 120)
        red = lit & ((hue < 12) | (hue > 168))
        green = lit & (hue >= 35) & (hue <= 100)
        if max(red.sum(), green.sum()) < 6:
            return None, "no lit red or green lamp in the box (is the lamp on?)"
        color, mask = ("red", red) if red.sum() >= green.sum() else ("green", green)
        n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        area = int(st[1:, 4].max()) if n > 1 else int(mask.sum())
        keep = lab == (1 + int(np.argmax(st[1:, 4]))) if n > 1 else mask     # The lamp itself: its biggest blob of that color.
        h = red_hue(hue[keep]) if color == "red" else hue[keep].astype(np.int16)
        s, v = sat[keep], val[keep]
        lo, hi = int(np.percentile(h, 5)) - LIGHT_HUE_PAD, int(np.percentile(h, 95)) + LIGHT_HUE_PAD
        lo, hi = (max(lo, -15), min(hi, 12)) if color == "red" else (max(lo, 38), min(hi, 100))     # Never orange or yellow (a lamp housing).
        return {"color": color,
                "h": [lo, hi],
                "s_min": int(max(40, np.percentile(s, 10) - LIGHT_S_PAD)),
                "v_min": int(max(60, np.percentile(v, 10) - LIGHT_V_PAD)),
                "area": area, "box": [round(float(b), 4) for b in box]}, f"{color} lamp ({area} px)"

    def color_mask(self, color, hue, sat, val):
        """Pixels that look like a registered lamp of this color, or None if none of this color was registered."""
        out = None
        for s in self.samples:
            if s["color"] != color:
                continue
            h = red_hue(hue) if color == "red" else hue.astype(np.int16)
            m = (h >= s["h"][0]) & (h <= s["h"][1]) & (sat >= s["s_min"]) & (val >= s["v_min"])
            out = m if out is None else (out | m)
        return out

    def fits(self, color, x, y, w, h, area, frame_w, frame_h, check_area=True):
        """Whether a lamp candidate (frame px) is the size of a registered one and near where it was."""
        for s in self.samples:
            if color is not None and s["color"] != color:
                continue
            if check_area and not LIGHT_AREA_RANGE[0] * s["area"] <= area <= LIGHT_AREA_RANGE[1] * s["area"]:
                continue
            bx0, by0, bx1, by1 = s["box"]
            cx, cy = (x + w / 2) / frame_w, (y + h / 2) / frame_h
            if (bx0 - LIGHT_REACH[0] <= cx <= bx1 + LIGHT_REACH[0]) and (by0 - LIGHT_REACH[1] <= cy <= by1 + LIGHT_REACH[1]):
                return True
        return False


class CrosswalkModel:
    """Registered crosswalk stripes. views: [{"stripes": [[w, h, fill, sharp, cx], ...]}].

    w, h: size in lane widths at the stripe's height. fill: share of its box it fills. sharp: edge steepness.
    cx: center across the frame in lane widths, for the spacing between stripes. (Older files have the first three only.)
    """

    def __init__(self, views=None):
        self.views = list(views or [])
        self.fit()

    def fit(self):
        rows = [s for v in self.views for s in v["stripes"]]
        if not rows:
            self.w = self.h = self.fill = None
            self.count = 0
            return
        arr = np.array([r[:3] for r in rows], float)
        self.w = (float(arr[:, 0].min()) * XWALK_SIZE_RANGE[0], float(arr[:, 0].max()) * XWALK_SIZE_RANGE[1])
        self.h = (float(arr[:, 1].min()) * XWALK_SIZE_RANGE[0], float(arr[:, 1].max()) * XWALK_SIZE_RANGE[1])
        self.fill = max(0.2, float(arr[:, 2].min()) - XWALK_FILL_PAD)
        self.count = max(2, int(round(XWALK_COUNT_SHARE * min(len(v["stripes"]) for v in self.views))))
        sharp = [r[3] for r in rows if len(r) > 3]
        self.sharp = min(sharp) * XWALK_SHARP_SHARE if sharp else None
        pitches = []
        for v in self.views:
            cxs = sorted(r[4] for r in v["stripes"] if len(r) > 4)
            pitches += [b - a for a, b in zip(cxs, cxs[1:])]
        self.pitch = (min(pitches) * XWALK_PITCH_RANGE[0], max(pitches) * XWALK_PITCH_RANGE[1]) if pitches else None

    @classmethod
    def load(cls, path=XWALK_FILE):
        try:
            data = json.loads(Path(path).read_text())
            return cls(data.get("views", [])) if data.get("views") else None
        except (OSError, ValueError, AttributeError):
            return None

    def save(self, path=XWALK_FILE):
        Path(path).write_text(json.dumps({"views": self.views}))

    def like(self, w, h, fill):
        """Whether one blob (sizes in lane widths) looks like a registered stripe."""
        return (self.w is not None and self.w[0] <= w <= self.w[1] and self.h[0] <= h <= self.h[1] and fill >= self.fill)

    def row(self, stripes):
        """The largest set of stripes lying side by side like the registered ones.

        stripes: [(id, cx, cy, h)] with cx in lane widths (spacing) and cy, h in px. A stripe joins the row when it sits at the
        height of the others and the spacing to its neighbor is like the registered spacing.
        """
        best = []
        for _, _, cy0, h0 in stripes:
            same = sorted((s for s in stripes if abs(s[2] - cy0) <= XWALK_ROW * max(h0, s[3])), key=lambda s: s[1])
            run = same[:1]
            for s in same[1:]:
                if self.pitch is None or self.pitch[0] <= s[1] - run[-1][1] <= self.pitch[1]:
                    run.append(s)
                elif len(run) > len(best):
                    best, run = run, [s]
                else:
                    run = [s]
            if len(run) > len(best):
                best = run
        return [s[0] for s in best]


# --- Lane lines (aicar line) ---

LINE_FILE = Path(__file__).with_name("line.json")
LINE_SHARP_SHARE = 0.85     # Reflections are cut below this share of the softest registered line.
LINE_SHARP_RANGE = (0.2, 0.45)


class LineModel:
    """Registered lane lines. samples: [{"sharp", "width", "contrast"}]: edge steepness, width in lane widths, brightness over the floor.

    Used with the other filters, not instead of them: it sets the reflection cut (lane_follow REFLECT_SHARP) to this venue's paint.
    """

    def __init__(self, samples=None):
        self.samples = list(samples or [])

    @classmethod
    def load(cls, path=LINE_FILE):
        try:
            data = json.loads(Path(path).read_text())
            return cls(data.get("samples", [])) if data.get("samples") else None
        except (OSError, ValueError, AttributeError):
            return None

    def save(self, path=LINE_FILE):
        Path(path).write_text(json.dumps({"samples": self.samples}))

    @property
    def sharp(self):
        """The softest registered line: paint is at least this crisp here."""
        return min(s["sharp"] for s in self.samples) if self.samples else None

    def reflect_cut(self):
        """Edge steepness under which a blob is a reflection, for this venue's paint."""
        return float(np.clip(self.sharp * LINE_SHARP_SHARE, *LINE_SHARP_RANGE)) if self.samples else None


# --- Stop points (aicar stoppoint) ---

STOP_FILE = Path(__file__).with_name("stops.json")
STOP_SCALES = (0.45, 0.55, 0.65, 0.75, 0.85, 1.0, 1.15)   # Sizes the registered shape is looked for at: from farther away it looks smaller.
STOP_MATCH = 0.78           # How well (normalized correlation of the line masks) the shape must match.
STOP_VIEWS_PER_FRAME = 2    # Not found yet: this many views are tried per frame, in turn, so many views do not slow the drive.
STOP_MAX_VIEWS = 8          # Views kept per point (the newest). More only cost time.
STOP_WIDE = 0.6             # A box wider than this share of the frame is a road area, not one mark: it gets the whole-view rules below.
STOP_WHOLE_ROWS = (0.30, 0.85)  # Whole-view stop point (Space): this band of the line area (0 = its top, 1 = the frame bottom).
STOP_WHOLE_COLS = (0.04, 0.96)  # ... across this much of the width. The room left above and at the sides lets it be found while still coming.
STOP_WHOLE_MATCH = 0.9         # A road area is followed from afar like a box, but counts as reached only matching this well
STOP_WHOLE_SCALE = 0.95         # ... and at about its registered size (farther, it looks smaller): a plain lane looks much the same all along.
STOP_TRACK_PX = 40          # Once found, the next frame looks only this far (half-size px) around it, at nearby sizes.


def _encode(img):
    return cv2.imencode(".png", img)[1].tobytes().hex()


def _decode(text):
    return cv2.imdecode(np.frombuffer(bytes.fromhex(text), np.uint8), cv2.IMREAD_GRAYSCALE)


class StopModel:
    """Registered stop points. points: {name: {"views": [{"patch": line mask (half size, png hex), "box": [x, y, w, h] frame px}]}}.

    The car was standing where it should stop when each view was taken, so the box's center is where the shape must come to.
    (The center, not the bottom edge: a box drawn down to the frame bottom has its bottom cut off there.)
    """

    def __init__(self, points=None):
        self.points = dict(points or {})
        self._patches = {}
        self._turn = {}         # Per point: which views the next search tries.

    @classmethod
    def load(cls, path=STOP_FILE):
        try:
            data = json.loads(Path(path).read_text())
            return cls(data.get("points", {})) if data.get("points") else None
        except (OSError, ValueError, AttributeError):
            return None

    def save(self, path=STOP_FILE):
        Path(path).write_text(json.dumps({"points": self.points}))

    def names(self):
        return sorted(self.points)

    @staticmethod
    def whole_box(shape, top):
        """The box a whole-view stop point uses: most of the line area, frame px (x, y, w, h)."""
        H, W = shape[:2]
        y0, y1 = int(top + (H - top) * STOP_WHOLE_ROWS[0]), int(top + (H - top) * STOP_WHOLE_ROWS[1])
        x0, x1 = int(W * STOP_WHOLE_COLS[0]), int(W * STOP_WHOLE_COLS[1])
        return x0, y0, x1 - x0, y1 - y0

    def add(self, name, mask, box, whole=False):
        """Add a view of point name: mask is the frame-size line mask (0/255), box (x, y, w, h) in frame px.

        whole: a road area (a wide box drawn over the road): matched more strictly, and only near full size.
        A box wider than STOP_WIDE is taken as one.
        """
        x, y, w, h = box
        whole = whole or w > STOP_WIDE * mask.shape[1]
        patch = cv2.resize(mask[y:y + h, x:x + w], (max(w // 2, 4), max(h // 2, 4)), interpolation=cv2.INTER_AREA)
        if patch.std() < 8:
            return False, "no line inside the box (draw it around the line end, line across or corner)"
        views = self.points.setdefault(name, {"views": []})["views"]
        views.append({"patch": _encode(patch), "box": [int(v) for v in box], **({"whole": True} if whole else {})})
        del views[:-STOP_MAX_VIEWS]
        self._patches.pop(name, None)
        return True, f"{name}: view {len(self.points[name]['views'])}"

    @staticmethod
    def close_enough(found):
        """A find that may count as the stop: a road area must match strictly and be at about full size
        (it is followed loosely while still coming, but a plain lane looks much the same all along)."""
        return not (found.get("whole") and (found["score"] < STOP_WHOLE_MATCH or found["scale"] < STOP_WHOLE_SCALE))

    def undo(self, name):
        views = self.points.get(name, {}).get("views", [])
        if views:
            views.pop()
        if not views:
            self.points.pop(name, None)
        self._patches.pop(name, None)

    def stop_y(self, name):
        """Frame y the shape's center had when the car stood at the stop (median over views)."""
        return float(np.median([v["box"][1] + v["box"][3] / 2 for v in self.points[name]["views"]]))

    def templates(self, name):
        if name not in self._patches:
            views = self.points[name]["views"][-STOP_MAX_VIEWS:]     # A file from before the limit may hold more: the newest count.
            self._patches[name] = [(cv2.GaussianBlur(_decode(v["patch"]), (5, 5), 0).astype(np.float32), v["box"], bool(v.get("whole")))
                                   for v in views]
        return self._patches[name]

    def find(self, name, mask, top, last=None):
        """Look for point name in the ROI line mask (0/255, ROI size). Returns {"x","y","w","h","score","scale"} in frame px, or None.

        last: the previous find, to look only around it at nearby sizes (cheaper, and it cannot jump to a look-alike elsewhere).
        With no last, a coarse search at quarter size over the whole ROI picks the place, then it is refined there at half size.
        """
        if name not in self.points:
            return None
        small = cv2.GaussianBlur(cv2.resize(mask, (mask.shape[1] // 2, mask.shape[0] // 2), interpolation=cv2.INTER_AREA), (5, 5), 0).astype(np.float32)
        if last is None:
            coarse = cv2.resize(small, (small.shape[1] // 2, small.shape[0] // 2), interpolation=cv2.INTER_AREA)
            n = len(self.templates(name))
            k = self._turn.get(name, 0) % max(n, 1)
            self._turn[name] = k + STOP_VIEWS_PER_FRAME
            views = [(k + j) % n for j in range(min(STOP_VIEWS_PER_FRAME, n))]
            guess = self._search(name, coarse, STOP_SCALES, None, factor=4, top=top, min_score=STOP_MATCH - 0.15, views=views)
            if guess is None:
                return None
            last = guess
        return self._search(name, small, None, last, factor=2, top=top, min_score=STOP_MATCH)

    def _search(self, name, img, scales, last, factor, top, min_score, views=None):
        """Best match of point name in img (the line mask at 1/factor size). last: look only around it, at sizes near its scale,
        with the view it was found with. views: which views to try (all if None)."""
        SH, SW = img.shape
        best = None
        tpls = self.templates(name)
        if last is not None and "view" in last:
            views = [last["view"]]
        for vi in (range(len(tpls)) if views is None else views):
            tpl = tpls[vi][0]
            for sc in (scales if last is None else [last["scale"] * f for f in (0.92, 1.0, 1.08)]):
                k = sc * 2 / factor             # Templates are kept at half size.
                th, tw = int(tpl.shape[0] * k), int(tpl.shape[1] * k)
                if th < 5 or tw < 5 or th > SH or tw > SW:
                    continue
                t = cv2.resize(tpl, (tw, th), interpolation=cv2.INTER_AREA)
                if t.std() < 4:
                    continue
                if last is None:
                    rx0, ry0, region = 0, 0, img
                else:
                    pad = STOP_TRACK_PX * 2 // factor
                    cx, cy = (last["x"] + last["w"] / 2) / factor, (last["y"] - top + last["h"] / 2) / factor
                    rx0, ry0 = int(max(cx - tw / 2 - pad, 0)), int(max(cy - th / 2 - pad, 0))
                    region = img[ry0:int(min(cy + th / 2 + pad, SH)), rx0:int(min(cx + tw / 2 + pad, SW))]
                    if region.shape[0] < th or region.shape[1] < tw:
                        continue
                res = cv2.matchTemplate(region, t, cv2.TM_CCOEFF_NORMED)
                _, score, _, loc = cv2.minMaxLoc(res)
                if tpls[vi][2] and score < min_score + (STOP_WHOLE_MATCH - STOP_MATCH):
                    continue        # Road area: the stricter match (a plain lane looks much the same all along).
                if best is None or score > best["score"]:
                    best = {"score": float(score), "scale": sc, "x": factor * (rx0 + loc[0]), "y": top + factor * (ry0 + loc[1]),
                            "w": factor * tw, "h": factor * th, "view": vi, "whole": tpls[vi][2]}
        return best if best is not None and best["score"] >= min_score else None
