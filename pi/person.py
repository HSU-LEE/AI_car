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
  6. What it looks like, by a small image model (Embedder), only for a blob that passed the rest. It needs no training on the figure: the model turns a picture into numbers, and
     pictures that look alike get similar numbers. Without the model or a view that holds these numbers, this check is skipped and the rest decide.
Only a blob that passes all of these counts. Anything else may be a decoy, so a miss is preferred to a false match.

While driving, only how close it is decides whether these steps apply. Feet above the near line (PERSON_AT) are too far to judge, so that figure is ignored until the car
has come closer: at that size a real figure passes in one frame out of ten, and it stopped and started the car over and over. Feet below the line must never be missed, so only
its colors are compared (NEAR_COLOR), and only something far bigger than a figure (NEAR_BIG), an outline nothing like one standing or lying (NEAR_SHAPE), or a blob running off a frame corner is left out. Outline, brightness, and place in the frame change or get cut off up close, and none of them can reject it there.

Saved to person.json on the Pi. `aicar push` does not overwrite it.
"""
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

PERSON_FILE = Path(__file__).with_name("person.json")
PHOTO_DIR = "person_photos"     # Beside person.json: the picture of each registered view.
EMBED_MODEL = Path(__file__).with_name("mobilenet_v3_small.tflite")

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
ANY_FILE = Path(__file__).with_name("lego_any.npy")     # Any figure: weights over the embedder's numbers, and last the offset. From photos of 14 other figures against 459 things in old recordings.
ANY_MIN = 1.17          # Any figure: lowest score. Figures left out of the weights: 87% of real frames, 82% of composites. Things in other recordings: 1%, and no stops.
ANY_MIN_EDGE = 1.7      # Same, for a blob cut off by the left or right edge of the frame. Blocks beside the track there: up to 1.04.
ANY_MAX = 3             # Any figure: at most this many blobs are looked at per frame, each about 13 ms on the Pi.
ANY_AT = 0.55           # Any figure: where the feet must come to when the course sets no PERSON_AT and nothing is registered.
ANY_HEIGHT = 0.33       # A standing figure is this many px tall per px its feet are below the horizon. The real one: 48 px at feet 208, horizon 65.
ANY_SIZE = (0.35, 1.7)  # The blob's longer side against that height. Low, since dark clothes do not count as color. High, for one lying down or cut off by the frame.
ANY_THIN = 0.15         # Its shorter side must be at least this times that height.
ANY_RATIO = 0.2         # And at least this times its longer side.
ANY_S, ANY_V, ANY_C = 70, 45, 25    # A pixel has color above this saturation, value and strength (saturation x value / 255).
ANY_JOIN = np.ones((9, 5), np.uint8)    # Joins a figure's colored parts across its dark ones. Taller than wide, half-size px.
ANY_MIN_PX = 12         # Fewest colored half-size px in a blob.
NEAR_COLOR = 0.7        # Minimum color match past the near line. Real figure there, under other light than at registration, even lying down: 0.78-0.80.
EMBED_MIN = 0.6         # Lowest likeness (0 to 1) to a registered view. The figure registered at two places, seen elsewhere: 0.74-0.82. From one place only: 0.25-0.52, so register it at two. Another figure: 0.41-0.59. Off-track things: 0.42-0.49.
EMBED_MIN_EDGE = 0.7    # Same, for a blob cut off by the left or right edge of the frame, which is off the car's path anyway.
PHOTO_ON = True         # Compare with the photos taken at registration (person_photos). False, or views with no photo, uses EMBED_MIN and FOREIGN_MAX instead.
PHOTO_MIN = 0.55        # Lowest overlap (0 to 1) between the colors of a whole figure and those of a photo. Things in the room can pass it, at 0.58-0.81 in recordings: wood and tape.
PHOTO_MIN_EDGE = 0.65   # Same, for a blob cut off by the left or right edge of the frame, which is off the car's path anyway.
PHOTO_FLOOR = 16        # A pixel is the floor within this Lab distance of the color of the picture's edges.
PHOTO_PARTS = 2         # The figure is compared by this many parts, top and bottom. Own figure 0.56-0.78 on the live mat, others 0.22-0.49. 3 parts brought those closer: 0.59-0.69 against up to 0.51.
PHOTO_WHOLE = 0.7       # Only photos within this ratio of the usual height for their place are used.
PHOTO_REACH = 1.2       # How far above and below a blob its figure is looked for, in figure heights.
PHOTO_COLUMN = 0.4      # Half the width of the column it is looked for in, in figure heights.
PHOTO_FILL = 0.2        # A row or column of it has this share standing out from the floor.
PHOTO_GAP = 3           # A gap of this many px is still the same figure.
PHOTO_SIZE = 1.1        # At most this tall or wide against a figure at its place.
PHOTO_SMALL = 0.6       # At least this long against a figure at its place.
PHOTO_DARK = 0.3        # A figure whose colors end this far above its feet, as a share of the colored part, has dark clothes. Its colored part may lie anywhere in it.
PHOTO_SLIDES = 5        # Then this many positions of the figure around the blob are tried.
FOREIGN_MAX = 0.3       # A blob is not the figure when more than this share of the real colors at its place are ones no view has. A black figure matches anything dark by color, so a figure in yellow trousers is told apart by them. Real figure: 0.00.
EMBED_MARGIN = 0.25     # The picture compared is the blob's box and this much around it, as a share of its longer side.
EMBED_MAX = 2           # At most this many blobs are looked at per frame. Each costs about 13 ms on the Pi.
NEAR_SHAPE = 0.25       # Past the near line, the outline must overlap a view at least this much, standing or lying. A room object in the figure's colors: 0.14. Real figure, cut off or not: 0.43 and up.
NEAR_SMALL = 0.5        # Past the near line, a blob shorter and narrower than this times the registered figure at that place on the floor is too small to be it. A small thing off to the side otherwise passes on color alone, being too small to look at.
NEAR_BIG = 2.0          # Past the near line, a blob wider or taller than this times the registered figure at that place on the floor is not a figure, however close it is. Real figure: at most 1.5. A yellow block joined to the wooden edge beside the track, which also matches at 0.75: 3 and up.
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


_HUE_BIN = (np.arange(256) // 10).astype(np.uint8)
_GRAY_BIN = (HUE_BINS + np.arange(256) // (256 // GRAY_LEVELS)).astype(np.uint8)


def color_index(hsv):
    """Color bin of every pixel of an HSV image. Runs on every checked frame, so it works in small integer types and lookup tables."""
    h, s, v = cv2.split(hsv)
    warm = (h >= WARM_H[0]) & (h <= WARM_H[1])
    need = np.where(warm, np.uint16(GRAY_C * 255), np.uint16(GRAY_C_COOL * 255))
    colored = (s >= GRAY_S) & (v >= BLACK_V) & (s.astype(np.uint16) * v >= need)
    return np.where(colored, cv2.LUT(h, _HUE_BIN), cv2.LUT(v, _GRAY_BIN))


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


class Embedder:
    """Numbers for what a picture looks like: the MediaPipe image embedder model, run by LiteRT, which the Pi has. Nothing is downloaded while running."""

    def __init__(self, path=EMBED_MODEL):
        from ai_edge_litert.interpreter import Interpreter
        self.net = Interpreter(model_path=str(path), num_threads=2)
        self.net.allocate_tensors()
        self.input, self.output = self.net.get_input_details()[0], self.net.get_output_details()[0]
        self.size = int(self.input["shape"][1])

    def __call__(self, bgr):
        picture = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (self.size, self.size), interpolation=cv2.INTER_AREA).astype(np.float32) / 127.5 - 1
        self.net.set_tensor(self.input["index"], picture[None])
        self.net.invoke()
        numbers = self.net.get_tensor(self.output["index"]).ravel().astype(np.float32)
        return numbers / (np.linalg.norm(numbers) + 1e-9)


_embedder, _embedder_tried = None, False


def get_embedder():
    """The embedder, or None with one message if it cannot run here (no model file, or no LiteRT, as on a Mac)."""
    global _embedder, _embedder_tried
    if not _embedder_tried:
        _embedder_tried = True
        try:
            _embedder = Embedder()
        except (ImportError, OSError, ValueError, RuntimeError) as e:
            print(f"[주의] 레고 모양 확인(임베더)을 쓸 수 없어 색과 윤곽만 봅니다: {e}")
    return _embedder


def photo_sig(bgr):
    """Colors of a picture of a whole figure, by its upper and lower half: hue and saturation of its colored pixels, brightness of the others.

    Without the floor, whose color is that of the picture's edges: two figures on one mat would otherwise match by the mat. By part, because white trousers and a brown cap say more in their place.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    edges = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    figure = np.linalg.norm(lab - np.median(edges, axis=0), axis=2) >= PHOTO_FLOOR
    if figure.mean() < 0.25:    # The edges are mostly the figure itself, as in a tight picture of one color: it is not known which part is the floor.
        figure[:] = True
    parts = []
    rows = bgr.shape[0]
    for i in range(PHOTO_PARTS):
        y0, y1 = rows * i // PHOTO_PARTS, max(rows * (i + 1) // PHOTO_PARTS, rows * i // PHOTO_PARTS + 1)
        part, inside = hsv[y0:y1], figure[y0:y1]
        colored = (part[..., 1] >= 50) & (part[..., 2] >= 40) & inside
        parts.append(cv2.calcHist([part], [0, 1], colored.astype(np.uint8), [12, 2], [0, 180, 50, 256]).ravel() * 2)
        parts.append(cv2.calcHist([part], [2], (inside & ~colored).astype(np.uint8), [6], [0, 256]).ravel() * 0.25)
    sig = np.concatenate(parts)
    return sig / (sig.sum() + 1e-6)


def figure_box(x, y, w, h, feet):
    """The whole figure around a blob that holds only its colored part: down to its feet, and wide enough for a figure that tall. A wide blob is one lying down, with nothing below it."""
    if w > h:
        return x, y, w, h
    tall = max(int(feet) - y, h)
    wide = max(w, tall // 2)
    return x - (wide - w) // 2, y, wide, tall


def embed_crop(frame, x, y, w, h):
    """The picture of a blob and what is around it. Full-frame px."""
    m = int(EMBED_MARGIN * max(w, h))
    return frame[max(y - m, 0):min(y + h + m, frame.shape[0]), max(x - m, 0):min(x + w + m, frame.shape[1])]


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
        self.embeds = None              # Per view that has them: numbers for what it looks like.
        self.upper_sigs = []            # Per view: the colors without the legs, for a figure so close that the frame cuts them off.
        self.found = None               # Last match: {"x", "y", "w", "h", "feet", "score"} in full-frame px.
        self.near = []                  # Blobs of plausible size that failed: [(x, y, w, h, color match, outline match)]. Shown while tuning.
        self.fg = None                  # Last color mask at half size. Shown while tuning.
        self.last_box = None            # What the last view cut out, in full-frame px. Shown while tuning.
        self.box = GUIDE                # Guide box the next view looks in. Not saved.
        self.photos = []                # Picture of the figure in each view, in the same order. None for a view saved before photos existed.
        self.photo_sigs = None
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
        self.photos.append(None)
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
        # The photo holds both what was cut out and what the colors see, because either can miss a dark part of the figure.
        px0, py0, px1, py1 = x, y, x + w, y + h
        if groups:
            px0, py0, px1, py1 = min(px0, gx), min(py0, gy), max(px1, gx + gw), max(py1, gy + gh)
        self.photos[-1] = frame[py0 * 2:py1 * 2, px0 * 2:px1 * 2].copy()
        view["photo_feet"] = py1 * 2
        self.fit()
        if view["feet"] <= self.horizon:
            self.undo()
            return False, "발이 수평선보다 위에 있습니다. 사람을 바닥에 세우세요"
        if not self.table.any():
            self.undo()
            return False, "사람 색이 주변과 같아 구분할 수 없습니다. 색이 다른 곳에 세워 보세요"
        embedder = get_embedder()
        if embedder is not None:
            bx, by = max(groups, key=lambda g: g[5])[:2] if groups else (x, y)     # The same blob detection will see.
            view["embed"] = [round(float(n), 4) for n in embedder(embed_crop(frame, bx * 2, by * 2, view["w"], view["h"]))]
            self.fit()
        if view["below"] > PART_ONLY and not PHOTO_ON:     # With photos the whole figure is compared, so dark clothes that the colors miss do no harm.
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
            self.photos.pop()
            if self.added:
                self.background -= self.added.pop()
            self.fit()
            self.auto_tune()

    def clear(self):
        self.views, self.background, self.added, self.photos = [], np.zeros(N_BINS), [], []
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
            self.table, self.sigs, self.band_sigs, self.upper_sigs, self.embeds, self.photo_sigs = None, [], [], [], None, None
            return
        pics = [(p, p.shape[0] / max(v.get("photo_feet", v["feet"]) - self.horizon, 1)) for v, p in zip(self.views, self.photos) if p is not None and p.size]
        self.photo_sigs = None
        if pics:
            usual = sorted(t for _, t in pics)[len(pics) // 2]     # One of them, so at least that photo stays.
            pics = [(p, t) for p, t in pics if PHOTO_WHOLE * usual <= t <= usual / PHOTO_WHOLE]     # Much shorter than the others at its place holds only a part of the figure, whose colors other figures may share. Much taller holds a hand as well.
            self.photo_sigs = np.array([photo_sig(p) for p, _ in pics])
            self.photo_tall = float(np.median([t for _, t in pics]))
            self.photo_wide = float(np.median([p.shape[1] / p.shape[0] for p, _ in pics]))
            self.photo_dark = float(np.median([v["below"] for v in self.views])) >= PHOTO_DARK
        hists = [np.array(v["hist"]) for v in self.views]
        seen = np.max(hists, axis=0)[:HUE_BINS] >= 0.02
        self.seen_colors = seen | np.roll(seen, 1) | np.roll(seen, -1)      # A neighboring hue is the same color under other light.
        self.tall = float(np.median([v["h"] * (1 + v["below"]) / max(v["feet"] - self.horizon, 1) for v in self.views]))    # Height of the figure per px its feet are below the horizon.
        fig = smooth(np.max(hists, axis=0))     # Largest over views, so a color seen from only one side still counts.
        back = smooth(normalized(self.background))
        self.table = (fig >= ALPHA * back) & (fig >= FIG_MIN)
        if fig[:HUE_BINS].sum() >= TRACK_ONLY * fig.sum():
            self.table[HUE_BINS:] = False
        self.sigs = [smooth(normalized(h * self.table)) for h in hists]
        self.band_sigs = [np.array(v["bands"]) * self.table for v in self.views]
        self.upper_sigs = [smooth(normalized(b[:BANDS - 1].sum(axis=0))) for b in self.band_sigs]
        embeds = [v["embed"] for v in self.views if v.get("embed")]
        self.embeds = np.array(embeds, dtype=np.float32) if embeds else None

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
        return (w * 2 <= NEAR_BIG * view["w"] * size and h * 2 <= NEAR_BIG * view["h"] * size
                and max(w * 2 / (view["w"] * size), h * 2 / (view["h"] * size)) >= NEAR_SMALL)

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
        corner = cut and (x <= 0 or x + w >= small.shape[1])     # Runs off two edges: something bigger than the view, not a figure in front of the car.
        outlines = (self._outline(inside), self._outline(np.rot90(inside)))     # Standing, or lying on its side.
        best = None
        for i, feet in close:
            color = float(np.minimum(sig, self.sigs[i]).sum())
            if cut:
                color = max(color, float(np.minimum(sig, self.upper_sigs[i]).sum()))
            view = np.array(self.views[i]["outline"], dtype=bool)
            shape = max(self._overlap(view, o) for o in outlines)
            if corner or shape < NEAR_SHAPE:
                color = 0.0
            if best is None or color > best[0]:
                best = (color, shape, feet)
        return best

    def foreign(self, q, x, y, w, h):
        """Share of the real colors around this blob (half-size px) that no view of the figure has.

        Looks a figure's height above and below it, since the blob may be only its head or only its legs. A wide blob is one lying down, and is all there is.
        """
        H, W = q.shape
        if w <= h:
            tall = max(h, int(self.tall * (y + h - self.horizon / 2)))
            wide = max(w, tall // 2)
            x, y, w, h = x - (wide - w) // 2, max(y + h - tall, int(self.horizon / 2)), wide, 2 * tall - h
        crop = q[max(y, 0):min(y + h, H), max(x, 0):min(x + w, W)]
        colored = crop[crop < HUE_BINS]
        return float((~self.seen_colors[colored]).mean()) if len(colored) else 0.0

    def figure_extent(self, frame, x, y, w, h):
        """The box of what stands out from the floor in a figure-wide column through this blob, full-frame px. The blob may be only its head or its legs."""
        if w > h:       # Lying down: the blob is all of it that can be told from the floor around it.
            return x, y, w, h
        tall = self.photo_tall * (y + h - self.horizon)
        half = int(max(w / 2, PHOTO_COLUMN * tall))
        center = x + w // 2
        x0, x1 = max(center - half, 0), min(center + half, frame.shape[1])
        y0, y1 = max(int(y - PHOTO_REACH * tall), 0), min(int(y + h + PHOTO_REACH * tall), frame.shape[0])
        lab = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).astype(np.float32)
        # The mat gets lighter with distance, so the floor is measured in each row, at both sides of the column.
        rows = lab.shape[0]
        left_floor = cv2.blur(lab[:, :3].mean(axis=1, keepdims=True), (1, 5)).reshape(rows, 1, 3)
        right_floor = cv2.blur(lab[:, -3:].mean(axis=1, keepdims=True), (1, 5)).reshape(rows, 1, 3)
        away_left, away_right = np.linalg.norm(lab - left_floor, axis=2), np.linalg.norm(lab - right_floor, axis=2)
        if x0 == 0 or x1 == frame.shape[1]:     # Cut off by the frame: that side of the column is the figure itself.
            mask = (away_right if x0 == 0 else away_left) >= PHOTO_FLOOR
        else:
            mask = np.minimum(away_left, away_right) >= PHOTO_FLOOR

        def run(fill, start):
            """The stretch through start where this much stands out, across gaps of a few px."""
            lo = hi = start
            while lo > 0 and fill[max(lo - PHOTO_GAP, 0):lo].any():
                lo -= 1
            while hi < len(fill) - 1 and fill[hi + 1:hi + 1 + PHOTO_GAP].any():
                hi += 1
            return lo, hi

        top, bottom = run(mask.mean(axis=1) >= PHOTO_FILL, min(y + h // 2 - y0, mask.shape[0] - 1))
        top, bottom = min(top, y - y0), max(bottom, y + h - 1 - y0)     # The blob is part of it, however dark.
        # Its shadow and reflection on the mat stand out too, and always lie below it, so the top is what can be trusted. A figure whose top is there is this tall:
        most = int(PHOTO_SIZE * self.photo_tall * (y0 + top - self.horizon) / max(1 - self.photo_tall, 0.1))
        bottom = min(bottom, top + max(most, 1) - 1)
        if bottom < y + h // 2 - y0:    # Something above it was taken for its top. The blob itself is the top then.
            top = y - y0
            bottom = min(top + int(PHOTO_SIZE * self.photo_tall * (y - self.horizon) / max(1 - self.photo_tall, 0.1)), mask.shape[0] - 1)
        left, right = run(mask[top:bottom + 1].mean(axis=0) >= PHOTO_FILL, min(center - x0, mask.shape[1] - 1))
        left, right = min(left, max(x - x0, 0)), max(right, min(x + w - 1 - x0, mask.shape[1] - 1))    # The blob is part of it here too.
        wide = max(int(PHOTO_SIZE * self.photo_wide * (bottom - top + 1)), w)
        if right - left + 1 > wide:     # A line beside it.
            left = max(min(center - x0 - wide // 2, right - wide + 1), left)
            right = left + wide - 1
        return x0 + left, y0 + top, right - left + 1, bottom - top + 1

    def photo_match(self, frame, x, y, w, h):
        """(overlap with the closest photo, whole-figure box) for this blob, full-frame px."""
        def score(box):
            bx, by, bw, bh = box
            crop = frame[by:by + bh, bx:bx + bw]
            turns = (np.rot90(crop).copy(), np.rot90(crop, 3).copy()) if bw > bh else (crop,)   # Lying down: its head may be on either side.
            return max(float(np.minimum(photo_sig(c), self.photo_sigs).sum(axis=1).max()) for c in turns), box

        H, W = frame.shape[:2]
        extent = self.figure_extent(frame, x, y, w, h)
        whole = max(extent[2], extent[3]) >= PHOTO_SMALL * self.photo_tall * (extent[1] + extent[3] - self.horizon) and min(extent[2], extent[3]) >= 6
        if w > h or (whole and not self.photo_dark):
            return score(extent) if whole else (-1.0, extent)
        # Dark clothes on a dark mat do not stand out, so where the blob lies in the figure is not known: its head, its middle or its legs. Each is tried, at the size a figure has there.
        options = [extent] if whole else []
        k = self.photo_tall
        reach = int(k * (y + h - self.horizon) / max(1 - k, 0.1))
        for i in range(PHOTO_SLIDES):
            top = y - max(reach - h, 0) * i // max(PHOTO_SLIDES - 1, 1)
            tall = max(int(k * (top - self.horizon) / max(1 - k, 0.1)), h)
            wide = max(int(tall * self.photo_wide), w)
            x0, y0, x1, y1 = max(x + (w - wide) // 2, 0), max(top, 0), min(x + (w + wide) // 2, W), min(top + tall, H)
            if x1 - x0 >= 6 and y1 - y0 >= 6:
                options.append((x0, y0, x1 - x0, y1 - y0))
        return max((score(box) for box in options), key=lambda r: r[0], default=(-1.0, extent))

    def likeness(self, frame, x, y, w, h):
        """How much this box looks like a registered view, 0 to 1, or None if that cannot be told. Standing or on its side."""
        embedder = get_embedder() if self.embeds is not None else None
        if embedder is None:
            return None
        crop = embed_crop(frame, x, y, w, h)
        best = float((self.embeds @ embedder(crop)).max())
        if w > h:       # Lying down. A blob can only be this wide if it does.
            best = max(best, float((self.embeds @ embedder(np.rot90(crop).copy())).max()))
        return best

    def __call__(self, frame, x_range=(0.0, 1.0), near_y=None, photos=True):
        """Best match inside x_range (fractions of the frame width, by the blob's center), or None.

        near_y: feet height in px where the figure counts as close. Only blobs past it count, and only their colors are judged. None judges every blob by every check.
        photos: compare with the registration photos. Auto capture asks only whether a side looks like one already taken.
        """
        self.found, self.near = None, []
        if self.table is None:
            return None
        small = self._half(frame)
        q = color_index(cv2.cvtColor(small, cv2.COLOR_BGR2HSV))
        self.fg = cv2.morphologyEx(self.table[q].astype(np.uint8), cv2.MORPH_CLOSE, CLOSE)
        passed = []
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
            else:
                passed.append((margin, {"x": x * 2, "y": y * 2, "w": w * 2, "h": h * 2, "feet": int(feet), "score": color, "shape": shape}))
        # Most figure-like first, by color and outline together: past the near line color alone passes, and things in the figure's colors can score higher on it
        # than the figure does (1.00 for a chair leg against 0.99, outline 0.4 against 0.9), which would use up the looks before the figure's turn.
        if photos and PHOTO_ON and self.photo_sigs is not None:
            passed.sort(key=lambda p: -p[1]["score"] * p[1]["shape"])
            for _, blob in passed[:MAX_PARTS]:
                look, box = self.photo_match(frame, blob["x"], blob["y"], blob["w"], blob["h"])
                blob["look"] = look
                edge = blob["x"] <= 0 or blob["x"] + blob["w"] >= frame.shape[1]
                if look >= (PHOTO_MIN_EDGE if edge else PHOTO_MIN):
                    blob.update(x=box[0], y=box[1], w=box[2], h=box[3], feet=box[1] + box[3])
                    self.found = blob
                    break
                self.near.append((blob["x"], blob["y"], blob["w"], blob["h"], blob["score"], blob["shape"]))
            return self.found
        keep = []
        for margin, blob in passed:
            blob["foreign"] = self.foreign(q, blob["x"] // 2, blob["y"] // 2, blob["w"] // 2, blob["h"] // 2)
            if blob["foreign"] > FOREIGN_MAX:
                self.near.append((blob["x"], blob["y"], blob["w"], blob["h"], blob["score"], blob["shape"]))
            else:
                keep.append((margin, blob))
        passed = keep
        passed.sort(key=lambda p: -p[1]["score"] * p[1]["shape"])
        for _, blob in passed[:EMBED_MAX]:      # The costly look only for what passed everything else.
            look = self.likeness(frame, blob["x"], blob["y"], blob["w"], blob["h"])
            blob["look"] = look
            edge = blob["x"] <= 0 or blob["x"] + blob["w"] >= frame.shape[1]
            if look is None or look >= (EMBED_MIN_EDGE if edge else EMBED_MIN):
                self.found = blob
                break
            self.near.append((blob["x"], blob["y"], blob["w"], blob["h"], blob["score"], blob["shape"]))
        return self.found

    # --- Saved File ---
    def save(self, path=PERSON_FILE):
        folder = Path(path).with_name(PHOTO_DIR)
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir()
        for i, (view, photo) in enumerate(zip(self.views, self.photos)):
            view.pop("photo", None)
            if photo is not None and photo.size:
                view["photo"] = f"{i:03d}.jpg"
                cv2.imwrite(str(folder / view["photo"]), photo, [cv2.IMWRITE_JPEG_QUALITY, 100, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])
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
            det.photos = [cv2.imread(str(Path(path).with_name(PHOTO_DIR) / v["photo"])) if v.get("photo") else None for v in det.views]
        except (OSError, ValueError, KeyError):
            return None
        det.fit()
        det.auto_tune()
        return det


class AnyFigure:
    """Any LEGO figure, registered or not: the registered ones first, then anything colored of a figure's size on the floor that looks like a figure.

    Finds with the same calls as PersonDetector, so PersonWatch and the views take either.
    """

    def __init__(self, horizon, weights, registered=None):
        self.horizon, self.weights, self.registered = horizon, weights, registered
        self.found, self.near, self.fg = None, [], None

    @classmethod
    def load(cls, horizon, registered=None, path=ANY_FILE):
        """None if the weights or the embedder are missing, since color alone cannot tell a figure from a block."""
        try:
            weights = np.load(path)
        except (OSError, ValueError):
            print(f"[주의] {Path(path).name}이 없어 등록한 레고만 찾습니다")
            return None
        return cls(horizon, weights, registered) if get_embedder() is not None else None

    @property
    def views(self):
        return self.registered.views if self.registered else []

    def stop_at(self, frame_h):
        return self.registered.stop_at(frame_h) if self.registered else ANY_AT

    def blobs(self, frame, near_y):
        """Boxes of a figure's size past near_y, largest first. Full-frame px."""
        small = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2), interpolation=cv2.INTER_AREA)
        _, s, v = cv2.split(cv2.cvtColor(small, cv2.COLOR_BGR2HSV))
        colored = (s >= ANY_S) & (v >= ANY_V) & (s.astype(np.uint16) * v >= ANY_C * 255)
        colored[:int(self.horizon / 2)] = False
        self.fg = colored.astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(self.fg, ANY_JOIN), connectivity=8)
        boxes = []
        for i in range(1, count):
            x, y, w, h = stats[i][:4]
            ys, xs = np.nonzero(colored[y:y + h, x:x + w] & (labels[y:y + h, x:x + w] == i))
            if len(ys) < ANY_MIN_PX:
                continue
            X, Y, W, H = int(x + xs.min()) * 2, int(y + ys.min()) * 2, int(xs.max() - xs.min() + 1) * 2, int(ys.max() - ys.min() + 1) * 2
            feet, tall = Y + H, ANY_HEIGHT * (Y + H - self.horizon)
            short, long = min(W, H), max(W, H)
            if feet >= near_y and Y > self.horizon and ANY_SIZE[0] * tall <= long <= ANY_SIZE[1] * tall and short >= ANY_THIN * tall and short >= ANY_RATIO * long:
                boxes.append((X, Y, W, H))
        boxes.sort(key=lambda b: -b[2] * b[3])
        return boxes

    def __call__(self, frame, x_range=(0.0, 1.0), near_y=None):
        self.found, self.near = None, []
        if self.registered is not None:
            self.found = self.registered(frame, x_range, near_y)
            self.near = list(self.registered.near)
            if self.found:
                return self.found
        near_y = ANY_AT * frame.shape[0] if near_y is None else near_y
        boxes = [b for b in self.blobs(frame, near_y) if x_range[0] <= (b[0] + b[2] / 2) / frame.shape[1] <= x_range[1]]
        for x, y, w, h in boxes[:ANY_MAX]:
            score = float(self.weights[:-1] @ get_embedder()(embed_crop(frame, x, y, w, h)) - self.weights[-1])
            edge = x <= 0 or x + w >= frame.shape[1]
            if score >= (ANY_MIN_EDGE if edge else ANY_MIN):
                self.found = {"x": x, "y": y, "w": w, "h": h, "feet": y + h, "score": score, "shape": 0.0, "look": None}
                break
            self.near.append((x, y, w, h, score, 0.0))
        return self.found


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
        found = self.detector(frame, photos=False)
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
                look = self.box.get("look")
                print(f"\n[사람] 정지 (발 위치 화면 {self.box['feet'] / self.frame_h:.2f}, 일치도 {self.box['score']:.2f}"
                      + (f", 모습 {look:.2f})" if look is not None else ")"))
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
