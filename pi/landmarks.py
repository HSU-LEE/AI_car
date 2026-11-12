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
STOP_MAX_VIEWS = None       # Views kept per point: no limit (None). A number keeps only that many (far ones dropped first).
STOP_HOME_PER_FRAME = 8     # The every-frame near check tries this many views a frame: the best of the last frames, then the rest in turn.
STOP_WIDE = 0.6             # A box wider than this share of the frame is a road area, not one mark: it gets the whole-view rules below.
STOP_WHOLE_ROWS = (0.30, 0.85)  # Whole-view stop point (Space): this band of the line area (0 = its top, 1 = the frame bottom).
STOP_WHOLE_COLS = (0.04, 0.96)  # ... across this much of the width. The room left above and at the sides lets it be found while still coming.
STOP_WHOLE_MATCH = 0.9         # A road area is followed from afar like a box, but counts as reached only matching this well
STOP_WHOLE_SCALE = 0.95         # ... and at about its registered size (farther, it looks smaller): a plain lane looks much the same all along.
STOP_AROUND = (0.92, 1.0, 1.08)    # Sizes tried around a known one.
STOP_HOME_PX = 24          # ... looked for this far (frame px) around where each view sits at the stop.
STOP_REFINE_PX = 10        # The every-frame check near each view's place: after a quarter-size look, refined this close (frame px).
STOP_TRACK_PX = 40          # Once found, the next frame looks only this far (half-size px) around it, at nearby sizes.


STOP_NEAR_Y = 144          # Frame y where the line area starts (lane_follow ROI_TOP 0.4 of 360): views centered below it are near.
STOP_EDGES = (60, 150)      # Above the line area there is no line mask: a box drawn up there is matched on the picture's edges (Canny).
STOP_ABOVE_PAD = 40         # ... from this many px above the highest such box.


def edges_above(frame, y0, y1):
    """Edge picture (0/255) of frame rows y0..y1: what a stop point box above the line area is matched on."""
    gray = cv2.cvtColor(frame[y0:y1], cv2.COLOR_BGR2GRAY)
    return cv2.dilate(cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), *STOP_EDGES), np.ones((3, 3), np.uint8))


def stop_view(frame, roi_mask, top, from_y=None):
    """What stop points are found in: the line mask, plus the edges above it when a box reaches up there (from_y).

    Returns (picture, its top row in the frame). Boxes all inside the line area use the line mask only, as before.
    """
    if from_y is None or from_y >= top:
        return roi_mask, top
    y0 = max(int(from_y) - STOP_ABOVE_PAD, 0)
    return np.vstack([edges_above(frame, y0, top), roi_mask]), y0


STOP_SPOT_PX = 34          # Blobs smaller than this every way (frame px) are dropped from stop-point pictures: ceiling-light glints
                            # on a glossy mat, which a car standing still to register cannot tell from paint, and which move when it drives.
STOP_DECIDE = 0.85          # A match this good (a mark box) is needed for the stop itself; weaker ones only follow the point while coming.


def drop_spots(img, px):
    """img (0/255 or blurred) without its small blobs (smaller than px every way): glints, not painted lines."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats((img > 127).astype(np.uint8), connectivity=8)
    small = np.zeros(n, bool)
    small[1:] = np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]) < px
    if not small.any():
        return img
    out = img.copy()
    out[small[labels]] = 0
    return out


def _kept(views):
    """The views that count: all, or the newest STOP_MAX_VIEWS when there is a limit."""
    return views if STOP_MAX_VIEWS is None else views[-STOP_MAX_VIEWS:]


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
        self._prep = None       # (picture, half, quarter) of the last picture looked in.
        self._rank, self._home_turn = {}, {}
        self._sizes = {}        # (point, view, w, h): that view resized, made once.    # Near check with many views: last scores per view, and whose turn it is.

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
        plain = patch.std() < 8
        views = self.points.setdefault(name, {"views": []})["views"]
        views.append({"patch": _encode(patch), "box": [int(v) for v in box], **({"whole": True} if whole else {})})
        while STOP_MAX_VIEWS is not None and len(views) > STOP_MAX_VIEWS:      # Full: the oldest far view goes first (near ones place the stop), else the oldest.
            far = [i for i, v in enumerate(views[:-1]) if v["box"][1] + v["box"][3] / 2 < STOP_NEAR_Y]
            del views[far[0] if far else 0]
        self._patches.pop(name, None)
        self._sizes = {k: v for k, v in self._sizes.items() if k[0] != name}
        note = f"{name}: view {len(self.points[name]['views'])}"
        if plain:
            note += "  [warning] almost nothing to see in this box (no line or edge): it may never be found"
        if y + h / 2 < STOP_NEAR_Y:
            note += ("  [note] far box (above the lane area): far things hardly change as the car comes, so it only shows the point is ahead."
                     " A box on the floor near the car decides the exact stop")
        return True, note

    @staticmethod
    def close_enough(found):
        """A find that may count as the stop: a road area must match strictly and be at about full size
        (it is followed loosely while still coming, but a plain lane looks much the same all along)."""
        if found["score"] < STOP_DECIDE:
            return False
        return not (found.get("whole") and (found["score"] < STOP_WHOLE_MATCH or found["scale"] < STOP_WHOLE_SCALE))

    def undo(self, name):
        views = self.points.get(name, {}).get("views", [])
        if views:
            views.pop()
        if not views:
            self.points.pop(name, None)
        self._patches.pop(name, None)
        self._sizes = {k: v for k, v in self._sizes.items() if k[0] != name}

    def top_y(self, names):
        """Highest frame y any view of these points reaches (for stop_view)."""
        ys = [v["box"][1] for n in names if n in self.points for v in _kept(self.points[n]["views"])]
        return min(ys) if ys else None

    def near_views(self, name, top=None):
        """Views whose box center is in the line area (the floor near the car): they place the stop to the cm.
        Views of far things (boxes drawn above it) hardly change as the car comes closer."""
        top = STOP_NEAR_Y if top is None else top
        return {vi for vi, (_, box, _) in enumerate(self.templates(name)) if box[1] + box[3] / 2 >= top}

    def view_y(self, name, view):
        """Frame y the center of this view's box had when the car stood at the stop."""
        box = _kept(self.points[name]["views"])[view]["box"]
        return box[1] + box[3] / 2

    def stop_y(self, name):
        """Frame y the shape's center had when the car stood at the stop (median over views)."""
        return float(np.median([v["box"][1] + v["box"][3] / 2 for v in self.points[name]["views"]]))

    def templates(self, name):
        if name not in self._patches:
            views = _kept(self.points[name]["views"])
            self._patches[name] = [(cv2.GaussianBlur(drop_spots(_decode(v["patch"]), STOP_SPOT_PX // 2), (5, 5), 0).astype(np.float32),
                                    v["box"], bool(v.get("whole")))
                                   for v in views]
        return self._patches[name]

    def find(self, name, mask, top, last=None, wide=True, key=None):
        """Look for point name in the picture from stop_view (mask) whose first row is frame row top. Returns {"x","y","w","h","score","scale"} in frame px, or None.

        last: the previous find, to look only around it at nearby sizes (cheaper, and it cannot jump to a look-alike elsewhere).
        With no last, a coarse search at quarter size over the whole ROI picks the place, then it is refined there at half size.
        """
        if name not in self.points:
            return None
        small, coarse = self._prepare(mask, key)
        # Every view, every frame, only around where it sits when the car is at the stop, at about its size: cheap, and the
        # moment the car passes over the place is never missed (at speed that is 1-2 frames), whichever view has its turn.
        home = None
        tpls = self.templates(name)
        near = self.near_views(name)
        # Far views (above the line area) say little about where exactly the car is: the near ones decide.
        for vi in self._home_views(name, [vi for vi in range(len(tpls)) if not near or vi in near]):
            box = tpls[vi][1]
            spot = {"x": box[0], "y": box[1], "w": box[2], "h": box[3], "scale": 1.0, "view": vi}
            hit = self._search(name, coarse, None, spot, factor=4, top=top, min_score=STOP_MATCH - 0.25, pad=STOP_HOME_PX, x_free=True)
            self._rank.setdefault(name, {})[vi] = 0.0 if hit is None else hit["score"]
            if hit is not None and hit["score"] >= STOP_MATCH - 0.1 and (home is None or hit["score"] > home["score"]):
                home = hit
        if home is not None:        # Refine the best at half size, close around it.
            home = self._search(name, small, None, home, factor=2, top=top, min_score=STOP_MATCH, pad=STOP_REFINE_PX, around=(1.0,))
        if home is not None and (last is None or home["score"] >= last.get("score", 0)):
            return home
        if last is None and not wide:
            return home     # wide=False: only the every-frame check near each view's place (the drive does the wide look every other frame).
        if last is None:
            n = len(self.templates(name))
            k = self._turn.get(name, 0) % max(n, 1)
            self._turn[name] = k + STOP_VIEWS_PER_FRAME
            views = [(k + j) % n for j in range(min(STOP_VIEWS_PER_FRAME, n))]
            guess = self._search(name, coarse, STOP_SCALES, None, factor=4, top=top, min_score=STOP_MATCH - 0.15, views=views)
            if guess is None:
                return home
            last = guess
        tracked = self._search(name, small, None, last, factor=2, top=top, min_score=STOP_MATCH)
        if home is not None and (tracked is None or home["score"] > tracked["score"]):
            return home     # The near check found it even though following the last place did not.
        return tracked

    def _home_views(self, name, views):
        """Which views the near check tries this frame. All when few; with many, those that came closest lately
        (as the car nears the place they climb), then the others in turn, STOP_HOME_PER_FRAME in all."""
        if len(views) <= STOP_HOME_PER_FRAME:
            return views
        rank = self._rank.get(name, {})
        best = sorted(views, key=lambda v: -rank.get(v, 0.0))[:STOP_HOME_PER_FRAME // 2]
        rest = [v for v in views if v not in best]
        k = self._home_turn.get(name, 0) % len(rest)
        self._home_turn[name] = k + STOP_HOME_PER_FRAME - len(best)
        return best + [rest[(k + j) % len(rest)] for j in range(min(STOP_HOME_PER_FRAME - len(best), len(rest)))]

    def _sized(self, name, vi, tpl, tw, th):
        """View vi resized to tw x th (None when it has too little in it), made once and kept: the same few sizes come up
        every frame."""
        key = (name, vi, tw, th)
        if key not in self._sizes:
            if len(self._sizes) > 4000:
                self._sizes.clear()
            t = cv2.resize(tpl, (tw, th), interpolation=cv2.INTER_AREA)
            self._sizes[key] = None if t.std() < 4 else t
        return self._sizes[key]

    def _prepare(self, mask, key=None):
        """The picture at half and quarter size, glints off. key (the frame's number): kept for the same frame, so several
        points, or a second look in one frame, do not redo it. No key: always made fresh."""
        if key is not None and self._prep is not None and self._prep[0] == key:
            return self._prep[1], self._prep[2]
        clean = drop_spots(mask, STOP_SPOT_PX)      # Glints off, as in the saved views.
        small = cv2.GaussianBlur(cv2.resize(clean, (clean.shape[1] // 2, clean.shape[0] // 2), interpolation=cv2.INTER_AREA), (5, 5), 0).astype(np.float32)
        coarse = cv2.resize(small, (small.shape[1] // 2, small.shape[0] // 2), interpolation=cv2.INTER_AREA)
        self._prep = (key, small, coarse)
        return small, coarse

    def _search(self, name, img, scales, last, factor, top, min_score, views=None, pad=None, around=STOP_AROUND, x_free=False):
        """Best match of point name in img (the line mask at 1/factor size). last: look only around it, at sizes near its scale,
        with the view it was found with. views: which views to try (all if None)."""
        SH, SW = img.shape
        best = None
        tpls = self.templates(name)
        if last is not None and "view" in last:
            views = [last["view"]]
        for vi in (range(len(tpls)) if views is None else views):
            tpl = tpls[vi][0]
            for sc in (scales if last is None else [last["scale"] * f for f in around]):
                k = sc * 2 / factor             # Templates are kept at half size.
                th, tw = int(tpl.shape[0] * k), int(tpl.shape[1] * k)
                if th < 5 or tw < 5 or th > SH or tw > SW:
                    continue
                t = self._sized(name, vi, tpl, tw, th)
                if t is None:
                    continue
                if last is None:
                    rx0, ry0, region = 0, 0, img
                else:
                    p = (STOP_TRACK_PX if pad is None else pad) * 2 // factor
                    cx, cy = (last["x"] + last["w"] / 2) / factor, (last["y"] - top + last["h"] / 2) / factor
                    rx0, ry0 = (0 if x_free else int(max(cx - tw / 2 - p, 0))), int(max(cy - th / 2 - p, 0))
                    # x_free: any place across (the car may come to the stop pointed a little differently), only the height is held.
                    region = img[ry0:int(min(cy + th / 2 + p, SH)), rx0:(SW if x_free else int(min(cx + tw / 2 + p, SW)))]
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
