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

While driving, only how close it is decides whether these steps apply. Feet above the near line (PERSON_AT) are too far to judge, so that figure is ignored until the car
has come closer: at that size a real figure passes in one frame out of ten, and it stopped and started the car over and over. Feet below the line must never be missed, so only
its colors are compared (NEAR_COLOR), and only something far bigger than a figure is left out (NEAR_BIG). Outline, brightness, and place in the frame change or get cut off up close, and none of them can reject it there.

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
GRAY_C = 60             # A warm color (WARM_H) of strength (saturation x value / 255) below this is the tint of the light, not a color. Real black mat under warm light: strength 30-45, hue of yellow. Yellow legs: 120-250.
WARM_H = (5, 35)        # Hues the warm light tints colorless things with.
GRAY_C_COOL = 20        # Other hues need much less, because a navy jacket has strength 50-60 and is what tells it from a figure with the same hair. Black mat under a cool white balance: at most 10-12, with the hue of a sky blue shirt.
TRACK_ONLY = 0.1        # Colorless pixels count only when real colors cover less than this share of the figure. Black and white are the floor and the lane line,
                        # so a stretch of line looks like a figure in those clothes, and gray is mostly the soft edge between them, which changes with distance.

ALPHA = 2.0             # A color counts as the figure when its share of the figure is at least this times its share of the background.
FIG_MIN = 0.01          # A color must cover at least this share of the figure. Smaller is noise from the cutout.
NEAR_COLOR = 0.7        # Minimum color match past the near line. Real figure there, under other light than at registration, even lying down: 0.78-0.80.
NEAR_BIG = 2.5          # Past the near line, a blob wider or taller than this times the registered figure at that place on the floor is not a figure, however close it is. Real figure: at most 1.5. A yellow block joined to the wooden edge beside the track, which also matches at 0.75: 3 and up.
MATCH = 0.7             # Minimum color match with a registered view (0 to 1), until registration sets it (auto_tune).
SHAPE_MATCH = 0.82      # Minimum outline overlap with the same view (0 to 1), until registration sets it (auto_tune). On the real figure: itself 0.88-1.0, an oval 0.78, a box 0.63.
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
ASPECT = (0.7, 1.7)     # Allowed height/width against the registered view.
PART_GAP = 0.8          # Stacked pieces are joined when the gap is at most this times the taller piece.
CLOSE = np.ones((5, 3), np.uint8)   # Bridges thin gaps such as a neck or a belt. Taller than wide. 5 rows, 3 columns, half-size px.
AUTO_STILL = 0.7        # Auto capture: the view must stay still this long before a shot, in seconds.
AUTO_MOVE_PX = 0.002    # Auto capture: share of pixels around the guide box that may change and still count as still.
AUTO_SAME = 0.92        # Auto capture: a side that already matches this well, in both color and outline, is not taken again.
MAX_VIEWS = 64          # No view is taken past this many, by any means. 63 views cost the real drive about 4 ms a frame.
AUTO_MAX = MAX_VIEWS    # Auto capture stops at this many views.
MATCH_MIN, SHAPE_MIN = 0.5, 0.7     # auto_tune never goes looser. At 0.1 and 0.3 on the real track: a figure with the same hair, a desk, and a drawer all matched.
TUNE_COLOR, TUNE_MATCH_MAX = 0.25, 0.75     # auto_tune: the color threshold is this far below the lowest match between views, and at most this. The real figure elsewhere on the mat: 0.71 and up.
TUNE_BOX, TUNE_SHAPE_MAX = 0.1, 0.78        # auto_tune: the outline threshold is this far above a box of the figure's size, and at most this. The real figure elsewhere: 0.75 and up, far away less.
RULE = 3                # How colors are counted. A saved figure counted another way would compare wrongly, so it must be registered again.
PART_ONLY = 1.0         # Registration: refused when the figure below the color blob is taller than this times the blob. Only a piece of the figure was seen. Black legs: about 0.5.
AUTO_HAND = 0.08        # Auto capture: this share of new colors in the strip around the box means a hand or an arm is reaching in.
SHADOW_H = 0.25         # Registration in a dragged box: colorless rows at the bottom, up to this share of the figure's height, are its shadow, not its feet. Black legs are about a third of a figure, so they stay.
SHADOW_C = 0.2          # A row with less than this share of colored pixels counts as colorless.
CHECK_SEC = 1.0         # After any sighting the car stands this long to see whether it is real.
CONFIRM_HITS = 3        # This many sightings in that time, in a row or not, mean the figure is there. About 13 frames are checked in a second.
HITS = 3                # Checked frames in a row before the figure counts as appeared or gone.


def color_index(hsv):
    """Color bin of every pixel of an HSV image."""
    h, s, v = cv2.split(hsv)
    need = np.where((h >= WARM_H[0]) & (h <= WARM_H[1]), GRAY_C, GRAY_C_COOL)
    colored = (s >= GRAY_S) & (v >= BLACK_V) & (s.astype(np.int32) * v >= need * 255)
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
        self.upper_sigs = []            # Per view: the colors without the legs, for a figure so close that the frame cuts them off.
        self.found = None               # Last match: {"x", "y", "w", "h", "feet", "score"} in full-frame px.
        self.near = []                  # Blobs of plausible size that failed: [(x, y, w, h, color match, outline match)]. Shown while tuning.
        self.fg = None                  # Last color mask at half size. Shown while tuning.
        self.last_box = None            # What the last view cut out, in full-frame px. Shown while tuning.
        self.box = GUIDE                # Guide box the next view looks in. Not saved.
        self.marked = False             # The box was dragged around the figure: all of it is taken as the figure, whatever its colors. Not saved.
        self.added = []                 # Background each view of this session added, so taking the view back takes that back too. Not saved.

    # --- Registration ---
    def add_background(self, frames):
        for frame in frames:
            q = color_index(cv2.cvtColor(self._half(frame), cv2.COLOR_BGR2HSV))
            self.background += np.bincount(q.ravel(), minlength=N_BINS)
        self.fit()

    def add_view(self, frame):
        """Add one view of the figure standing in the guide box. Returns (added, message)."""
        if len(self.views) >= MAX_VIEWS:
            return False, f"이미 {MAX_VIEWS}장입니다. 더 찍지 말고 엔터로 저장하세요 (앞, 옆, 뒤면 충분합니다)"
        small = self._half(frame)
        if self.marked:
            figure = self._marked_cutout(small, self.box)
            if figure.sum() < MIN_AREA:      # Smaller than VIEW_AREA is fine here: someone pointed at it, so a far figure can be taken too.
                return False, "드래그한 네모에서 레고를 찾지 못했습니다. 레고 전체를 조금 여유 있게 감싸 보세요"
            ys, xs = np.nonzero(figure)
            x, y, w, h = int(xs.min()), int(ys.min()), int(xs.max() - xs.min()) + 1, int(ys.max() - ys.min()) + 1
            return self._add_figure(frame, small, figure, x, y, w, h, guide_px(small.shape, self.box))
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
        return self._add_figure(frame, small, labels == best, *(int(v) for v in stats[best, :4]))

    def _add_figure(self, frame, small, figure, x, y, w, h, marked=None):
        """Add the view whose figure is this mask, with this bounding box (half-size px). Returns (added, message).

        marked: the box dragged around it, which holds all of the figure. GrabCut can leave dark hair out of the mask, and driving sees that hair.
        """
        self.last_box = (x * 2, y * 2, w * 2, h * 2)
        q = color_index(cv2.cvtColor(small, cv2.COLOR_BGR2HSV))
        hist = np.bincount(q[figure], minlength=N_BINS)
        around = cv2.dilate(figure.astype(np.uint8), np.ones((5, 5), np.uint8)) == 0     # Everything else in the frame counts against the figure.
        added = np.bincount(q[around], minlength=N_BINS)
        self.background += added
        self.added.append(added)
        view = {"hist": normalized(hist.astype(np.float64)).tolist(), "w": w * 2, "h": h * 2, "feet": (y + h) * 2, "below": 0.0,
                "outline": self._outline(figure[y:y + h, x:x + w]).tolist(),
                "lab": mean_lab(small[y:y + h, x:x + w], figure[y:y + h, x:x + w]).tolist(),
                "bands": band_hists(q[y:y + h, x:x + w], figure[y:y + h, x:x + w]).tolist()}
        self.views.append(view)
        self.fit()
        # Measure the shape as the color mask sees it, the same way it is seen while driving: black legs on a black mat are not part of it.
        # Only inside the cutout's box: a same-colored thing behind the figure must not be joined to it and inflate its size.
        # The box, not the cutout itself, because the cutout leaves out the soft edges of the figure that the color mask does see.
        inside = np.zeros_like(q, dtype=bool)
        if marked:
            inside[marked[1]:marked[3], marked[0]:marked[2]] = True
        else:
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
        if view["below"] > PART_ONLY:
            self.undo()
            return False, "사람의 일부만 색으로 잡혔습니다. 드래그로 사람 전체를 감싸 다시 찍으세요"
        if view["h"] < MIN_H * 2:
            self.undo()     # Detection ignores a blob this small, so this view could never be found.
            return False, "사람 색으로 잡힌 부분이 너무 작아 인식할 수 없습니다. 차에 더 가까이 세우세요"
        self.auto_tune()
        message = f"등록 {len(self.views)}번 ({view['w']}x{view['h']}px, 발 위치 {view['feet'] / frame.shape[0]:.2f}, 자동 기준 색 {self.match:.2f} 윤곽 {self.shape:.2f})"
        if self.table[HUE_BINS:].any():
            message += " [주의] 사람이 거의 검정/흰색이라 차선을 사람으로 착각할 수 있습니다"
        return True, message

    def auto_tune(self):
        """Set both thresholds from the registered views, so nobody has to.

        Outline: above how a box of the figure's size matches it. Color: below how the views match each other. Views taken at one spot look more alike
        than the figure does at another spot, so neither follows the views all the way up (TUNE_*_MAX).
        """
        if not self.views:
            return
        outlines = [np.array(v["outline"], dtype=bool) for v in self.views]
        box = max(self._overlap(o, np.ones_like(o)) for o in outlines)
        self.shape = round(float(np.clip(box + TUNE_BOX, SHAPE_MIN, TUNE_SHAPE_MAX)), 2)
        # Each view against the view most like it: the front and the back of a figure need not match, because a blob is compared with one view at a time.
        colors = min(max((float(np.minimum(a, b).sum()) for j, b in enumerate(self.sigs) if j != i), default=1.0) for i, a in enumerate(self.sigs))
        self.match = round(float(np.clip(colors - TUNE_COLOR, MATCH_MIN, TUNE_MATCH_MAX)), 2)

    def undo(self):
        if self.views:
            self.views.pop()
            if self.added:
                self.background -= self.added.pop()
            self.fit()
            self.auto_tune()

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

    @staticmethod
    def _marked_cutout(small, box):
        """Figure mask inside a box dragged around it. Only the floor close by is shown to GrabCut as background: the yellow tape or the room far away may share the figure's colors."""
        x0, y0, x1, y1 = guide_px(small.shape, box)
        side = max(x1 - x0, y1 - y0)
        cx0, cy0, cx1, cy1 = max(x0 - side, 0), max(y0 - side, 0), min(x1 + side, small.shape[1]), min(y1 + side, small.shape[0])
        mask = np.full((cy1 - cy0, cx1 - cx0), cv2.GC_BGD, np.uint8)
        q, _, seen = ring_colors(small, box)
        known = seen[q[y0:y1, x0:x1]]       # A small figure in a roomy box: starting from all of it, the floor inside would take over. Colors not around the box are the figure.
        mask[y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0] = np.where(known, cv2.GC_PR_BGD, cv2.GC_PR_FGD) if (~known).sum() >= MIN_PART else cv2.GC_PR_FGD
        cv2.grabCut(small[cy0:cy1, cx0:cx1], mask, None, np.zeros((1, 65)), np.zeros((1, 65)), 3, cv2.GC_INIT_WITH_MASK)
        figure = np.zeros(small.shape[:2], bool)
        figure[cy0:cy1, cx0:cx1] = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(figure.astype(np.uint8), connectivity=8)
        figure = np.isin(labels, [i for i in range(1, n) if stats[i, 4] >= MIN_PART])      # Head and body may come apart. Specks are dropped.
        rows = np.nonzero(figure.any(axis=1))[0]
        if len(rows):
            colored = color_index(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)) < HUE_BINS
            share = (figure & colored).sum(axis=1)[rows] / figure.sum(axis=1)[rows]
            solid = np.nonzero(share >= SHADOW_C)[0]
            if len(solid) and len(rows) - 1 - solid[-1] <= SHADOW_H * len(rows):
                figure[rows[solid[-1]] + 1:] = False    # The feet set where the car stops, and a soft shadow would put them too low.
        return figure

    def fit(self):
        """Rebuild the color table and the per-view colors from the views and the background."""
        if not self.views:
            self.table, self.sigs, self.band_sigs, self.upper_sigs = None, [], [], []
            return
        hists = [np.array(v["hist"]) for v in self.views]
        fig = smooth(np.max(hists, axis=0))     # Largest over views, so a color seen from only one side still counts.
        back = smooth(normalized(self.background))
        self.table = (fig >= ALPHA * back) & (fig >= FIG_MIN)
        if fig[:HUE_BINS].sum() >= TRACK_ONLY * fig.sum():
            self.table[HUE_BINS:] = False
        self.sigs = [smooth(normalized(h * self.table)) for h in hists]
        self.band_sigs = [np.array(v["bands"]) * self.table for v in self.views]
        self.upper_sigs = [smooth(normalized(b[:BANDS - 1].sum(axis=0))) for b in self.band_sigs]

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

    @staticmethod
    def _overlap(view, outline):
        """Outline overlap, at the best of three heights: one stray pixel above a still figure moves its whole outline down a row."""
        best = 0.0
        for dy in (-1, 0, 1):
            moved = np.roll(outline, dy, axis=0)
            moved[0 if dy > 0 else -1] &= dy == 0   # What rolled around the end is not part of the outline.
            best = max(best, float((view & moved).sum() / max((view | moved).sum(), 1)))
        return best

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
            shape = self._overlap(view, outline)
            margin = min(color / self.match, shape / self.shape)    # Both must pass with the same view.
            if best is None or margin > best[0]:
                best = (margin, color, shape, feet)
        return best[1:]

    def _not_too_big(self, view, feet, w, h):
        """Whether a blob (half-size px) could be this view at this place on the floor. The expected size grows as it comes closer, so closeness alone never fails this."""
        size = (feet - self.horizon) / max(view["feet"] - self.horizon, 1)
        return w * 2 <= NEAR_BIG * view["w"] * size and h * 2 <= NEAR_BIG * view["h"] * size

    def _score_near(self, small, q, group, near_y):
        """(color match, outline match, real feet y) of a blob standing past near_y, or None if it is farther or is not on the floor.

        The top of a figure stays below the horizon at any distance, because the camera is higher than the figure. What reaches above it is the room.
        """
        x, y, w, h, labels, area, ids = group
        if h < MIN_H or area < MIN_AREA or area < MIN_FILL * w * h or y * 2 <= self.horizon:
            return None
        close = [(i, (y + h * (1 + v["below"])) * 2) for i, v in enumerate(self.views)]
        close = [(i, feet) for i, feet in close if feet >= near_y and self._not_too_big(self.views[i], feet, w, h)]
        if not close:
            return None
        inside, qcrop = np.isin(labels[y:y + h, x:x + w], ids), q[y:y + h, x:x + w]
        pixels = qcrop[inside & self.table[qcrop]]
        if not len(pixels):
            return None
        sig = smooth(normalized(np.bincount(pixels, minlength=N_BINS).astype(np.float64)))
        cut = y + h >= small.shape[0]       # The frame ends at its legs.
        outline = self._outline(inside)
        best = None
        for i, feet in close:
            color = float(np.minimum(sig, self.sigs[i]).sum())
            if cut:
                color = max(color, float(np.minimum(sig, self.upper_sigs[i]).sum()))
            if best is None or color > best[0]:
                best = (color, self._overlap(np.array(self.views[i]["outline"], dtype=bool), outline), feet)
        return best

    def __call__(self, frame, x_range=(0.0, 1.0), near_y=None):
        """Best match inside x_range (fractions of the frame width, by the blob's center), or None.

        near_y: feet height in px where the figure counts as close. Only blobs past it count, and only their colors are judged. None judges every blob by every check.
        """
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
            result = self._score(small, q, group) if near_y is None else self._score_near(small, q, group, near_y)
            if result is None:
                continue
            color, shape, feet = result
            margin = min(color / self.match, shape / self.shape) if near_y is None else color / NEAR_COLOR
            if margin < 1:
                self.near.append((x * 2, y * 2, w * 2, h * 2, color, shape))
            elif margin > best:
                best = margin
                self.found = {"x": x * 2, "y": y * 2, "w": w * 2, "h": h * 2, "feet": int(feet), "score": color, "shape": shape}
        return self.found

    # --- Saved File ---
    def save(self, path=PERSON_FILE):
        Path(path).write_text(json.dumps({"rule": RULE, "match": self.match, "shape": self.shape, "views": self.views, "background": self.background.tolist()}))

    @classmethod
    def load(cls, horizon, path=PERSON_FILE):
        """The saved figure, or None if nothing usable is saved."""
        try:
            data = json.loads(Path(path).read_text())
            if data.get("rule") != RULE:        # Counted another way: must be registered again.
                return None
            det = cls(horizon)
            det.views = data["views"]
            if not all("outline" in v and "lab" in v and "bands" in v for v in det.views):      # Saved before outlines existed: must be registered again.
                return None
            det.background = np.array(data["background"], dtype=np.float64)
            if not det.views or det.background.shape != (N_BINS,):
                return None
        except (OSError, ValueError, KeyError):
            return None
        det.fit()
        det.auto_tune()
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
    """Whether the car must stand still for the figure.

    Only a figure with its feet past the near line (at) counts. Farther away it is not looked at, and the car drives on towards it.
    Any sighting stops the car, wherever it is across the frame, for check_sec. A few sightings in that time, in a row or not, mean it is there. One alone was a misread, and the car goes on.
    Once it is there the car waits until it has not been seen at all for resume_delay: a sighting that flickers keeps it waiting.

    hold is the answer. A figure that outlasts max_wait is ignored until it has gone away, so the car does not stop for it again the moment it starts.
    """

    def __init__(self, detector, frame_h, at=None, x_range=(0.0, 1.0), resume_delay=1.0, min_stop=0.0, max_wait=None, every=1,
                 check_sec=CHECK_SEC, confirm=CONFIRM_HITS):
        self.detector, self.frame_h = detector, frame_h
        self.at = at if at is not None else detector.stop_at(frame_h)   # Near line: feet height as a fraction of the frame.
        self.x_range = x_range
        self.resume_delay, self.min_stop, self.max_wait = resume_delay, min_stop, max_wait
        self.check_sec, self.confirm = check_sec, confirm
        self.every = max(1, every)
        self.tick = 0
        self.misses = 0
        self.seen = False               # Seen lately, for the display (not missed HITS checked frames in a row).
        self.box = None                 # Last match.
        self.hold = False
        self.checking = False           # Holding to see whether the first sighting was real.
        self.check_hits = 0
        self.ignored = False            # max_wait ran out: drive on until this figure has gone away.
        self.last_seen = None           # Time of the last sighting.
        self.hold_since = None
        self.gone_since = None          # While holding and not seeing it: when it was last seen. For the display.

    @property
    def ahead(self):
        return self.seen and not self.ignored

    def update(self, frame, now):
        """Look at this frame (every `every` frames) and return whether to stand still."""
        hit = False
        if self.tick % self.every == 0:
            self.box = self.detector(frame, self.x_range, self.at * self.frame_h if self.at is not None else None)
            hit = bool(self.box)
            self.misses = 0 if hit else self.misses + 1
            self.seen = hit or (self.seen and self.misses < HITS)
            if hit:
                self.last_seen = now
        self.tick += 1

        gone = self.last_seen is None or now - self.last_seen >= self.resume_delay
        self.gone_since = self.last_seen if self.hold and self.misses else None
        if self.ignored:
            self.ignored = not gone
        elif not self.hold:
            if hit:
                self.hold, self.checking, self.check_hits, self.hold_since = True, True, 1, now
        elif self.checking:
            self.check_hits += hit
            if hit and self.check_hits >= self.confirm:
                self.checking = False
                print(f"\n[사람] 정지 (발 위치 화면 {self.box['feet'] / self.frame_h:.2f}, 일치도 {self.box['score']:.2f})")
            elif now - self.hold_since >= self.check_sec:
                self.hold = self.checking = False
                print(f"\n[사람] {self.check_sec:g}초 동안 {self.check_hits}번만 보여서 출발")
        else:
            timed_out = self.max_wait is not None and now - self.hold_since >= self.max_wait
            if timed_out or (gone and now - self.hold_since >= self.min_stop):
                self.hold = False
                self.ignored = not gone
                print("\n[사람] " + ("최대 대기 시간이 지나 출발" if not gone else "없어져서 출발"))
        return self.hold
