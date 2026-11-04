"""Find the LEGO figure registered on the day (tune_person.py), and decide when the car stands still for it.

The figure is not known in advance, so nothing is trained ahead of time. Registration takes views of the figure standing inside a guide box:
what lies around the box is floor and room, and whatever inside the box has colors not found around it is the figure. No empty background
shot is needed, since taking the figure away and putting it back moves the car and changes the light. Every color outside the figure counts against it.

Each frame then goes through:
  1. Color: a pixel counts if its color is common on the figure and rare in the background.
  2. Parts: head, torso, and legs can come apart (a black torso on a black mat), so pieces stacked vertically are also tried together.
  3. Shape: on the floor, a figure's height on screen grows in step with how far its feet are below the horizon. A box or a wall does not fit that.
  4. Colors in that blob compared with a registered view, overall and from top to bottom. A red box does not match a figure that is only
     partly red, and a figure with its shirt and trousers colors swapped does not match either.
  5. Outline compared with the same view. A box of the figure's color does not have a head, arms, and legs.
Only a blob that passes all of these counts. Anything else may be a decoy, so a miss is preferred to a false match.

Saved to person.json on the Pi. `aicar push` does not overwrite it.
"""
import json
from pathlib import Path

import cv2
import numpy as np

PERSON_FILE = Path(__file__).with_name("person.json")

HUE_BINS = 18           # Colored pixels: this many hue steps. Brightness and saturation are compared separately (LAB_*), so no bin edge falls inside a lighting change.
GRAY_LEVELS = 4         # Colorless pixels: this many brightness steps.
N_BINS = HUE_BINS + GRAY_LEVELS
GRAY_S = 50             # Saturation below this has no reliable hue, so only brightness is used.
BLACK_V = 50            # Value below this has no reliable hue either.
TRACK_ONLY = 0.1        # Colorless pixels count only when real colors cover less than this share of the figure. Black and white are the floor and the lane line,
                        # so a stretch of line looks like a figure in those clothes, and gray is mostly the soft edge between them, which changes with distance.

ALPHA = 2.0             # A color counts as the figure when its share of the figure is at least this times its share of the background.
FIG_MIN = 0.01          # A color must cover at least this share of the figure. Smaller is noise from the cutout.
MATCH = 0.7             # Minimum color match with a registered view (0 to 1). + and - in tune_person.py change it.
SHAPE_MATCH = 0.82      # Minimum outline overlap with the same view (0 to 1). On the real figure: itself 0.88-1.0, an oval 0.78, a box 0.63. + and - in tune_person.py change it.
LAB_L = 18              # Largest difference from the view's mean lightness (Lab L, 0 to 100). Real figure at +-15% brightness: about 10.
LAB_AB = 14             # Largest difference from the view's mean color (Lab a, b). Tells a pale or dull copy of the same hue.
SILHOUETTE = (12, 24)   # Outlines are compared at this size (width, height).
BANDS = 3               # Colors are also compared in this many slices from top to bottom (head, torso, legs).
GUIDE = (0.3, 0.42, 0.7, 0.97)  # Registration: guide box the figure stands in (left, top, right, bottom), as fractions of the frame.
RING = 8                # Registration: width of the strip around the box that shows what the floor and room look like, in half-size px.
RING_SHARE = 0.002      # Registration: a color covering this share of the strip counts as floor or room.
UPRIGHT = 0.9           # Registration: a standing figure is about this tall for its width or more (the real one: 1.6, only its square head: 1.0). Flatter is something else on the floor.
VIEW_AREA = 60          # Registration: smallest cutout, in half-size px.
BOX_FILL = 0.5          # Registration: a cutout filling more of the box than this is not a figure.
DIFF_THR = 30           # Auto capture: per-channel difference between frames that counts as a change.
MIN_PART = 10           # Smallest color piece, in half-size px.
MAX_PARTS = 12          # Only this many of the largest pieces are compared, so a busy frame costs the same.
MIN_H = 7               # Smallest figure height, in half-size px. A figure this far away is ignored.
MIN_AREA = 25           # Smallest total figure area, in half-size px.
MIN_FILL = 0.2          # Color pixels must fill at least this much of the box.
SCALE = (0.7, 1.4)      # Allowed height against the height expected at that floor position.
ASPECT = (0.7, 1.4)     # Allowed height/width against the registered view.
PART_GAP = 0.8          # Stacked pieces are joined when the gap is at most this times the taller piece.
CLOSE = np.ones((5, 3), np.uint8)   # Bridges thin gaps such as a neck or a belt. Taller than wide. 5 rows, 3 columns, half-size px.
AUTO_STILL = 0.7        # Auto capture: the view must stay still this long before a shot, in seconds.
AUTO_MOVE_PX = 0.002    # Auto capture: share of pixels around the guide box that may change and still count as still.
AUTO_SAME = 0.92        # Auto capture: a side that already matches this well, in both color and outline, is not taken again.
AUTO_MAX = 12           # Auto capture stops at this many views.
AUTO_HAND = 0.08        # Auto capture: this share of new colors in the strip around the box means a hand or an arm is reaching in.
HITS = 3                # Checked frames in a row before the figure counts as appeared or gone.


def color_index(hsv):
    """Color bin of every pixel of an HSV image."""
    h, s, v = cv2.split(hsv)
    colored = (s >= GRAY_S) & (v >= BLACK_V)
    return np.where(colored, h.astype(np.intp) // 10, HUE_BINS + v // (256 // GRAY_LEVELS))


def normalized(hist):
    total = hist.sum()
    return hist / total if total > 0 else hist


def smooth(hist):
    """Let neighboring colors share, so a small lighting shift across a bin edge does not break a match. Hue wraps around (red)."""
    out = hist.astype(np.float64)
    part = hist[:HUE_BINS]
    out[:HUE_BINS] = (np.roll(part, 1) + 2 * part + np.roll(part, -1)) / 4
    return out


def band_hists(q, mask):
    """Color histogram of the masked pixels in each slice from top to bottom."""
    h = q.shape[0]
    return np.array([np.bincount(q[h * i // BANDS:h * (i + 1) // BANDS][mask[h * i // BANDS:h * (i + 1) // BANDS]], minlength=N_BINS)
                     for i in range(BANDS)], dtype=np.float64)


def mean_lab(bgr, mask):
    """Mean Lab color of the masked pixels."""
    lab = cv2.cvtColor(bgr.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
    return lab[mask].mean(axis=0)


def guide_px(shape, box):
    """Guide box in pixels of an image of this shape, clipped so the strip around it stays in the frame."""
    h, w = shape[:2]
    return (max(int(box[0] * w), 1), max(int(box[1] * h), 1), min(int(box[2] * w), w - 1), min(int(box[3] * h), h - 1))


def ring_colors(small, box):
    """Fine color bin of every pixel, the strip around the box, and which fine bins that strip holds (with their near neighbors, so soft shading counts)."""
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2Lab).astype(np.intp)
    q = (lab[..., 0] // 32) * 32 * 32 + (lab[..., 1] // 8) * 32 + lab[..., 2] // 8
    x0, y0, x1, y1 = guide_px(small.shape, box)
    ring = np.zeros(q.shape, bool)
    ring[max(y0 - RING, 0):y1 + RING, max(x0 - RING, 0):x1 + RING] = True
    ring[y0:y1, x0:x1] = False
    counts = np.bincount(q[ring], minlength=8 * 32 * 32)
    grid = (counts >= max(3, RING_SHARE * ring.sum())).reshape(8, 32, 32).astype(np.uint8)
    seen = np.stack([cv2.dilate(g, np.ones((3, 3), np.uint8)) for g in grid]).ravel() > 0
    return q, ring, seen


class PersonDetector:
    """The registered figure. Call it with a frame to get the best match, or None."""

    def __init__(self, horizon, match=MATCH, shape=SHAPE_MATCH):
        self.horizon = horizon          # y where the floor would meet the lane vanishing point (lane_follow.HORIZON_Y).
        self.match, self.shape = match, shape
        self.views = []                 # {"hist", "w", "h", "feet", "below", "outline", "lab", "bands"}: colors, size of the color blob, real feet y, how far the feet are
                                        # below that blob in blob heights (black legs on a black mat are not part of the blob), its outline, its mean color, and its colors from top to bottom. Full-frame px.
        self.background = np.zeros(N_BINS)
        self.table = None               # Per color bin: whether it counts as the figure.
        self.sigs = []                  # Per view: the colors that count, for comparing with a blob.
        self.band_sigs = []             # Per view: the same, per slice from top to bottom.
        self.found = None               # Last match: {"x", "y", "w", "h", "feet", "score"} in full-frame px.
        self.near = []                  # Blobs of plausible size that failed: [(x, y, w, h, color match, outline match)]. Shown while tuning.
        self.fg = None                  # Last color mask at half size. Shown while tuning.
        self.last_box = None            # What the last view cut out, in full-frame px. Shown while tuning.
        self.box = GUIDE                # Guide box the next view looks in. Not saved.
        self.added = []                 # Background each view of this session added, so taking the view back takes that back too. Not saved.

    # --- Registration ---
    def add_background(self, frames):
        for frame in frames:
            q = color_index(cv2.cvtColor(self._half(frame), cv2.COLOR_BGR2HSV))
            self.background += np.bincount(q.ravel(), minlength=N_BINS)
        self.fit()

    def add_view(self, frame):
        """Add one view of the figure standing in the guide box. Returns (added, message)."""
        small = self._half(frame)
        mask = self._cutout(small, self.box)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        x0, y0, x1, y1 = guide_px(small.shape, self.box)
        parts = [i for i in range(1, n) if stats[i, 4] >= VIEW_AREA]
        # A figure stands fully inside the box. What reaches its edge is a part of the room or a hand.
        inside = [i for i in parts if stats[i, 0] > x0 and stats[i, 1] > y0
                  and stats[i, 0] + stats[i, 2] < x1 and stats[i, 1] + stats[i, 3] < y1]
        if not inside:
            return False, ("사람이 네모 테두리에 걸렸습니다. 사람 전체가 네모 안에 들어오게 세우고 손을 빼세요" if parts
                           else "네모 안에 사람이 안 보입니다. 네모 안에 세우세요 (주변과 색이 같으면 구분하지 못합니다)")
        upright = [i for i in inside if stats[i, 3] >= UPRIGHT * stats[i, 2]]
        if not upright:
            return False, "서 있는 사람 모양이 안 보입니다. 사람을 세우고, 네모 안의 다른 물건은 치우세요"
        best = max(upright, key=lambda i: stats[i, 4])
        if stats[best, 4] > BOX_FILL * (x1 - x0) * (y1 - y0):
            return False, "네모를 거의 채우는 물체라 사람이 아닙니다"
        x, y, w, h = (int(v) for v in stats[best, :4])
        self.last_box = (x * 2, y * 2, w * 2, h * 2)
        q = color_index(cv2.cvtColor(small, cv2.COLOR_BGR2HSV))
        hist = np.bincount(q[labels == best], minlength=N_BINS)
        around = cv2.dilate((labels == best).astype(np.uint8), np.ones((5, 5), np.uint8)) == 0     # Everything else in the frame counts against the figure.
        added = np.bincount(q[around], minlength=N_BINS)
        self.background += added
        self.added.append(added)
        view = {"hist": normalized(hist.astype(np.float64)).tolist(), "w": w * 2, "h": h * 2, "feet": (y + h) * 2, "below": 0.0,
                "outline": self._outline(labels[y:y + h, x:x + w] == best).tolist(),
                "lab": mean_lab(small[y:y + h, x:x + w], labels[y:y + h, x:x + w] == best).tolist(),
                "bands": band_hists(q[y:y + h, x:x + w], labels[y:y + h, x:x + w] == best).tolist()}
        self.views.append(view)
        self.fit()
        # Measure the shape as the color mask sees it, the same way it is seen while driving: black legs on a black mat are not part of it.
        # Only inside the cutout's box: a same-colored thing behind the figure must not be joined to it and inflate its size.
        # The box, not the cutout itself, because the cutout leaves out the soft edges of the figure that the color mask does see.
        inside = np.zeros_like(q, dtype=bool)
        inside[max(y - 2, 0):y + h + 2, max(x - 2, 0):x + w + 2] = True
        fg = cv2.morphologyEx((self.table[q] & inside).astype(np.uint8), cv2.MORPH_CLOSE, CLOSE)
        groups = self._groups(fg)
        if groups:
            gx, gy, gw, gh, glabels, _, ids = max(groups, key=lambda g: g[5])
            part = np.isin(glabels[gy:gy + gh, gx:gx + gw], ids)
            view.update(w=gw * 2, h=gh * 2, below=max(y + h - (gy + gh), 0) / gh, outline=self._outline(part).tolist(),
                        bands=band_hists(q[gy:gy + gh, gx:gx + gw], part).tolist(),
                        lab=mean_lab(small[gy:gy + gh, gx:gx + gw], part & self.table[q[gy:gy + gh, gx:gx + gw]]).tolist())
            self.fit()      # Those are what detection compares with, here as after saving.
        if view["feet"] <= self.horizon:
            self.undo()
            return False, "발이 수평선보다 위에 있습니다. 사람을 바닥에 세우세요"
        if not self.table.any():
            self.undo()
            return False, "사람 색이 주변과 같아 구분할 수 없습니다. 색이 다른 곳에 세워 보세요"
        message = f"등록 {len(self.views)}번 ({view['w']}x{view['h']}px, 발 위치 {view['feet'] / frame.shape[0]:.2f})"
        if self.table[HUE_BINS:].any():
            message += " [주의] 사람이 거의 검정/흰색이라 차선을 사람으로 착각할 수 있습니다"
        return True, message

    def undo(self):
        if self.views:
            self.views.pop()
            if self.added:
                self.background -= self.added.pop()
            self.fit()

    def clear(self):
        self.views, self.background, self.added = [], np.zeros(N_BINS), []
        self.fit()

    def stop_at(self, frame_h):
        """Feet height where the figure was registered, as a fraction of the frame. Used when PERSON_AT is None."""
        return float(np.median([v["feet"] for v in self.views])) / frame_h if self.views else None

    @staticmethod
    def _half(frame):
        return cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _outline(mask):
        """A blob's filled area at the comparison size, as a boolean array."""
        return cv2.resize(mask.astype(np.float32), SILHOUETTE, interpolation=cv2.INTER_AREA) >= 0.5

    @staticmethod
    def _cutout(small, box):
        """Figure mask: start from the colors inside the box that are not around it, then let GrabCut settle its edges."""
        q, _, seen = ring_colors(small, box)
        x0, y0, x1, y1 = guide_px(small.shape, box)
        known = seen[q[y0:y1, x0:x1]]
        if (~known).sum() < MIN_PART:
            return np.zeros(small.shape[:2], np.uint8)
        mask = np.full(small.shape[:2], cv2.GC_BGD, np.uint8)
        mask[y0:y1, x0:x1] = np.where(known, cv2.GC_PR_BGD, cv2.GC_PR_FGD)
        cv2.grabCut(small, mask, None, np.zeros((1, 65)), np.zeros((1, 65)), 3, cv2.GC_INIT_WITH_MASK)
        figure = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
        return cv2.morphologyEx(figure, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    def fit(self):
        """Rebuild the color table and the per-view colors from the views and the background."""
        if not self.views:
            self.table, self.sigs, self.band_sigs = None, [], []
            return
        hists = [np.array(v["hist"]) for v in self.views]
        fig = smooth(np.max(hists, axis=0))     # Largest over views, so a color seen from only one side still counts.
        back = smooth(normalized(self.background))
        self.table = (fig >= ALPHA * back) & (fig >= FIG_MIN)
        if fig[:HUE_BINS].sum() >= TRACK_ONLY * fig.sum():
            self.table[HUE_BINS:] = False
        self.sigs = [smooth(normalized(h * self.table)) for h in hists]
        self.band_sigs = [np.array(v["bands"]) * self.table for v in self.views]

    # --- Detection ---
    @staticmethod
    def _groups(fg):
        """Candidate blobs: each color piece, and pieces stacked on each other. Returns (x, y, w, h, labels, area, ids)."""
        n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        ids = [i for i in range(1, n) if stats[i, 4] >= MIN_PART]
        ids = sorted(ids, key=lambda i: -stats[i, 4])[:MAX_PARTS]
        out = [(*(int(v) for v in stats[i, :4]), labels, int(stats[i, 4]), (i,)) for i in ids]
        by_top = sorted(ids, key=lambda i: stats[i, 1])
        for a, first in enumerate(by_top):
            x0, y0, w0, h0, area = (int(v) for v in stats[first])
            x1, y1, members = x0 + w0, y0 + h0, [first]
            for other in by_top[a + 1:]:
                ox, oy, ow, oh, oa = (int(v) for v in stats[other])
                overlap = min(x1, ox + ow) - max(x0, ox)
                if overlap >= 0.5 * min(x1 - x0, ow) and oy - y1 <= PART_GAP * max(y1 - y0, oh):
                    x0, x1, y1 = min(x0, ox), max(x1, ox + ow), max(y1, oy + oh)
                    members.append(other)
                    area += oa
            if len(members) > 1:
                out.append((x0, y0, x1 - x0, y1 - y0, labels, area, tuple(members)))
        return out

    def _fits(self, view, y, w, h):
        """Real feet y if this blob (half-size px) has the shape of this view at its place on the floor, otherwise None."""
        feet = (y + h * (1 + view["below"])) * 2
        floor, view_floor = feet - self.horizon, view["feet"] - self.horizon
        if floor <= 0 or view_floor <= 0:
            return None
        scale_ok = SCALE[0] <= h * 2 / (view["h"] * floor / view_floor) <= SCALE[1]
        return feet if scale_ok and ASPECT[0] <= (h / w) / (view["h"] / max(view["w"], 1)) <= ASPECT[1] else None

    @staticmethod
    def _band_match(bands, view_bands):
        """Color match slice by slice, weighted by how much of the view each slice holds."""
        weights = normalized(view_bands.sum(axis=1))
        return float(sum(w * np.minimum(smooth(normalized(a)), smooth(normalized(b))).sum()
                         for w, a, b in zip(weights, bands, view_bands)))

    def _score(self, small, q, group):
        """(color match, outline match, real feet y) with the view it matches best, or None if no registered view fits its size."""
        x, y, w, h, labels, area, ids = group
        if h < MIN_H or area < MIN_AREA or area < MIN_FILL * w * h:
            return None
        fits = [(i, feet) for i, feet in ((i, self._fits(v, y, w, h)) for i, v in enumerate(self.views)) if feet is not None]
        if not fits:
            return None
        inside, qcrop = np.isin(labels[y:y + h, x:x + w], ids), q[y:y + h, x:x + w]
        colored = inside & self.table[qcrop]
        pixels = qcrop[colored]
        if not len(pixels):
            return None
        lab = mean_lab(small[y:y + h, x:x + w], colored)
        outline = self._outline(inside)
        bands = band_hists(qcrop, colored)
        sig = smooth(normalized(np.bincount(pixels, minlength=N_BINS).astype(np.float64)))
        best = None
        for i, feet in fits:
            view = np.array(self.views[i]["outline"], dtype=bool)
            view_lab = self.views[i]["lab"]
            color = min(float(np.minimum(sig, self.sigs[i]).sum()), self._band_match(bands, self.band_sigs[i]))
            if abs(lab[0] - view_lab[0]) > LAB_L or np.hypot(lab[1] - view_lab[1], lab[2] - view_lab[2]) > LAB_AB:
                color = 0.0     # Same hue, but a paler, duller, or darker object.
            shape = float((view & outline).sum() / max((view | outline).sum(), 1))
            margin = min(color / self.match, shape / self.shape)    # Both must pass with the same view.
            if best is None or margin > best[0]:
                best = (margin, color, shape, feet)
        return best[1:]

    def __call__(self, frame, x_range=(0.0, 1.0)):
        """Best match inside x_range (fractions of the frame width, by the blob's center), or None."""
        self.found, self.near = None, []
        if self.table is None:
            return None
        small = self._half(frame)
        q = color_index(cv2.cvtColor(small, cv2.COLOR_BGR2HSV))
        self.fg = cv2.morphologyEx(self.table[q].astype(np.uint8), cv2.MORPH_CLOSE, CLOSE)
        best = 0.0
        for group in self._groups(self.fg):
            x, y, w, h = group[:4]
            if not x_range[0] <= (x + w / 2) / small.shape[1] <= x_range[1]:
                continue
            result = self._score(small, q, group)
            if result is None:
                continue
            color, shape, feet = result
            margin = min(color / self.match, shape / self.shape)
            if margin < 1:
                self.near.append((x * 2, y * 2, w * 2, h * 2, color, shape))
            elif margin > best:
                best = margin
                self.found = {"x": x * 2, "y": y * 2, "w": w * 2, "h": h * 2, "feet": int(feet), "score": color, "shape": shape}
        return self.found

    # --- Saved File ---
    def save(self, path=PERSON_FILE):
        Path(path).write_text(json.dumps({"match": self.match, "shape": self.shape, "views": self.views, "background": self.background.tolist()}))

    @classmethod
    def load(cls, horizon, path=PERSON_FILE):
        """The saved figure, or None if nothing usable is saved."""
        try:
            data = json.loads(Path(path).read_text())
            det = cls(horizon, data.get("match", MATCH), data.get("shape", SHAPE_MATCH))
            det.views = data["views"]
            if not all("outline" in v and "lab" in v and "bands" in v for v in det.views):      # Saved before outlines existed: must be registered again.
                return None
            det.background = np.array(data["background"], dtype=np.float64)
            if not det.views or det.background.shape != (N_BINS,):
                return None
        except (OSError, ValueError, KeyError):
            return None
        det.fit()
        return det


class AutoViews:
    """Take a view by itself each time the figure has been turned and the view has gone still again, so both hands stay free."""

    def __init__(self, detector):
        self.detector = detector
        self.ref = None         # Frame the view has stayed still against. Not the previous frame: a slow turn changes little between two frames.
        self.still_since = None
        self.ready = True       # Something moved since the last try. Standing still alone does not try twice.
        self.around = None      # Colors around the box when the first view was taken. A hand reaching in brings new ones.

    def touch(self):
        """The guide box moved: wait for it to settle, then look again."""
        self.still_since, self.ready, self.around = None, True, None

    def update(self, frame, now):
        """Look at this frame. Returns a message when it tried a shot, otherwise None."""
        small = PersonDetector._half(frame)
        # Only the box and its strip: someone walking around far behind the figure must not hold the shot back.
        x0, y0, x1, y1 = guide_px(small.shape, self.detector.box)
        gray = cv2.cvtColor(small[max(y0 - RING, 0):y1 + RING, max(x0 - RING, 0):x1 + RING], cv2.COLOR_BGR2GRAY)
        if self.ref is None or self.ref.shape != gray.shape or (cv2.absdiff(gray, self.ref) > DIFF_THR).mean() > AUTO_MOVE_PX:
            self.ref, self.still_since, self.ready = gray, None, True
            return None
        self.still_since = now if self.still_since is None else self.still_since
        if not self.ready or now - self.still_since < AUTO_STILL:
            return None
        self.ready = False
        if len(self.detector.views) >= AUTO_MAX:
            return f"시점 {AUTO_MAX}개를 다 찍었습니다. 엔터로 저장하세요"
        q, ring, seen = ring_colors(small, self.detector.box)
        if self.around is not None and (~self.around[q[ring]]).mean() > AUTO_HAND:
            return "팔이나 손이 보여서 안 찍었습니다. 손을 빼면 찍습니다"
        found = self.detector(frame)
        if found and found["score"] >= AUTO_SAME and found["shape"] >= AUTO_SAME:
            return "이미 등록된 면입니다. 사람을 더 돌리세요"
        added, message = self.detector.add_view(frame)
        if added and self.around is None:
            self.around = seen
        return message


class PersonWatch:
    """Whether the car must stand still for the figure. Driving stops once the feet reach the at height, and starts again once the figure is gone.

    hold is the answer. ahead means the figure is in view, stopping or not. A figure that outlasts max_wait is
    ignored until it has gone away, so the car does not stop for it again the moment it starts.
    """

    def __init__(self, detector, frame_h, at=None, x_range=(0.0, 1.0), stop_delay=0.0, resume_delay=1.0,
                 min_stop=0.0, max_wait=None, every=1):
        self.detector, self.frame_h = detector, frame_h
        self.at = at if at is not None else detector.stop_at(frame_h)
        self.x_range = x_range
        self.stop_delay, self.resume_delay = stop_delay, resume_delay
        self.min_stop, self.max_wait = min_stop, max_wait
        self.every = max(1, every)
        self.tick = 0
        self.hits = self.misses = 0
        self.seen = False               # Figure counts as present (HITS checked frames in a row).
        self.box = None                 # Last match.
        self.hold = False
        self.ignored = False            # max_wait ran out: drive on until this figure has gone away.
        self.close_since = None         # Time the feet reached the at height (for stop_delay).
        self.hold_since = None
        self.gone_since = None          # Time the figure went away while holding (for resume_delay).

    @property
    def ahead(self):
        return self.seen and not self.ignored

    def update(self, frame, now):
        """Look at this frame (every `every` frames) and return whether to stand still."""
        if self.tick % self.every == 0:
            self.box = self.detector(frame, self.x_range)
            self.hits, self.misses = (self.hits + 1, 0) if self.box else (0, self.misses + 1)
            if self.hits >= HITS:
                self.seen = True
            elif self.misses >= HITS:
                self.seen = self.ignored = False
        self.tick += 1

        if self.hold:
            self.gone_since = None if self.seen else (self.gone_since if self.gone_since is not None else now)
            gone = self.gone_since is not None and now - self.gone_since >= self.resume_delay
            timed_out = self.max_wait is not None and now - self.hold_since >= self.max_wait
            if timed_out or (gone and now - self.hold_since >= self.min_stop):
                self.hold = False
                self.ignored = self.seen
                print("\n[사람] " + ("최대 대기 시간이 지나 출발" if self.seen else "없어져서 출발"))
        else:
            feet = self.box["feet"] if self.box else None
            close = self.ahead and feet is not None and feet >= self.at * self.frame_h - 1
            self.close_since = (self.close_since if self.close_since is not None else now) if close else None
            if close and now - self.close_since >= self.stop_delay:
                self.hold, self.hold_since, self.gone_since = True, now, None
                print(f"\n[사람] 정지 (발 위치 화면 {feet / self.frame_h:.2f}, 일치도 {self.box['score']:.2f})")
        return self.hold
