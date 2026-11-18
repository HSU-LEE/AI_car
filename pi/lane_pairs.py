"""Lane position from both lane lines fitted as one perspective pair (LANE_POINTS = "pairs" in lane_follow.py).

The band method finds one point per line in each of 6 equal-height bands, then fits a straight line through the band
centers. This reads the same line mask (reflections, crosswalk stripes, signs and specks already removed) differently:

  1. Sample rows instead of bands: PAIR_ROWS thin strips (PAIR_STRIP rows each), spaced by PAIR_SPACING between the frame
     bottom and the top of the band area. Their places come from the calibration (HORIZON_Y), so they move with it.
  2. Each row of a strip gives its own line pieces. A slanted line stays narrow on one row, so a curve or a car at an angle
     keeps its points (a whole band summed up made such a line too wide, and it was dropped).
  3. On a flat floor a straight lane is two straight lines on screen that meet at the vanishing point. So both lines are
     fitted at once: left x = xv + bL * s, right x = xv + bR * s, with s = y - HORIZON_Y and xv shared.
       - Each piece goes to the nearer of the two lines, measured in lane widths at its own row, and pieces far from both
         are left out (a stripe between the lines, the next lane's line). The gate scales with the lane at that row.
       - The lane width (bR - bL) is held near the calibrated one, so one line alone still gives the whole lane: the other
         side is where the width puts it, in perspective, not one fixed horizontal half width.
       - Rows nearer the car count more (PAIR_NEAR_POW): far pieces are few pixels and bend first in a curve.
       - The last frame's lane is a weak prior, so nothing jumps; a few robust passes leave out pieces far from the fit.
  4. Position and heading are given on the same rows and in the same unit as the band method, so the steering gains and
     every threshold stay as they are.

Junctions, lane ends, gaps and crosswalks are still judged by the bands: only the lane position and heading come from here.
With too little to fit, or a result that is not a lane, the band answer stands for that frame and the next frame starts
again from it.
"""
import numpy as np

PAIR_ROWS = 8               # Sample rows.
PAIR_STRIP = 6              # Rows read around each sample row.
PAIR_SPACING = "screen"     # "screen": evenly on screen. "ground": evenly in floor distance (crowds far). "log": in between.
                            # Simulator, 5 tracks x 8 seeds: "screen" the best overall (oval done 100%, tight 79%).
PAIR_LINE_W = (0.012, 0.22)  # A line piece is this wide, in lane widths at its row (narrower: a speck, wider: a line across or a blob).
PAIR_GATE = (0.3, 0.15)     # A piece farther than this from its line (lane widths at its row) is left out: first pass, later passes.
PAIR_ITER = 3               # Fitting passes.
PAIR_HUBER = 0.05           # Pieces beyond this (lane widths) count less in each pass.
PAIR_NEAR_POW = 1.0         # Weight of a piece: (s / s at the frame bottom) to this power.
PAIR_WIDTH_PRIOR = 0.3      # Pull of the calibrated lane width, as a share of all the pieces' weight.
PAIR_SMOOTH = 0.05          # Pull toward the last frame's lane, likewise.
PAIR_WIDTH_RANGE = (0.75, 1.33)  # A fitted width outside this share of the calibrated one is not a lane: the band answer stands.
PAIR_MIN_PIECES = 6         # Pieces needed in all.
PAIR_MIN_ROWS = 3           # Sample rows that must give at least one piece.
PAIR_VIEW_GROW = 0.12       # The drawn curve takes in pieces this near it (lane widths), row by row up the line (view only).
PAIR_VIEW_SMOOTH = 0.5      # The drawn curves keep this share of the last frame's (view only, never the steering).
PAIR_RELOCK = (0.4, 3)      # The band method sure (both lines in 3+ bands) and this far off (position) this many frames in a row:
                            # start again from it.


class LanePairs:
    def __init__(self):
        self.p = None           # [xv, bL, bR] in screen pixels: x of both lines at the horizon, and their slopes per row below it.
        self.pieces = None      # Last frame's pieces (y, x, side: -1 left, 1 right, 0 left out), for the view.
        self.rows_y = None      # Last frame's sample rows (frame y).
        self.off = 0            # Frames in a row far from a sure band answer.
        self.used = False       # This frame's answer came from the pair fit.
        self.drawn = {}         # The view's last drawn curve of each line (curve_points).

    def reset(self):
        self.p, self.off, self.drawn = None, 0, {}

    # --- where to read ---
    @staticmethod
    def sample_rows(lf, top):
        """Frame rows of the sample strips' centers, nearest first."""
        H = lf.HORIZON_Y
        half = PAIR_STRIP // 2
        s0 = lf.FRAME_H - 1 - half - H          # Nearest: the frame bottom.
        s1 = top + half - H                     # Farthest: the top of the band area.
        if s1 < 4 or s0 <= s1:
            return np.zeros(0)
        t = np.linspace(0.0, 1.0, PAIR_ROWS)
        if PAIR_SPACING == "screen":
            s = s0 + (s1 - s0) * t
        elif PAIR_SPACING == "ground":
            s = 1.0 / (1.0 / s0 + (1.0 / s1 - 1.0 / s0) * t)
        else:
            s = s0 * (s1 / s0) ** t
        return np.round(H + s).astype(int)

    def read_pieces(self, lf, mask, top):
        """Line pieces in the sample strips: arrays y, x center (frame px), and the row index each came from."""
        H = lf.HORIZON_Y
        key = (lf.HORIZON_Y, lf.FRAME_H, top)
        if getattr(self, "_rows_key", None) != key:     # The rows only move with the calibration.
            self._rows_key, self.rows_y = key, self.sample_rows(lf, top)
        rows = self.rows_y
        lane_at = lf.LANE_WIDTH_PX / max(342 - H, 1.0)
        half = PAIR_STRIP // 2
        # Every strip's rows (mask rows, inside the mask) in one array: one pass of numpy calls for all of them.
        r0 = np.clip(rows - half - top, 0, max(mask.shape[0] - PAIR_STRIP, 0))
        idx = (r0[:, None] + np.arange(PAIR_STRIP)[None, :]).ravel()
        idx = idx[idx < mask.shape[0]]
        k_of = np.repeat(np.arange(len(rows)), PAIR_STRIP)[:len(idx)]
        on = np.zeros((len(idx), mask.shape[1] + 2), np.int8)
        on[:, 1:-1] = mask[idx] > 0
        d = on[:, 1:] - on[:, :-1]
        rr, c0 = np.nonzero(d == 1)
        c1 = np.nonzero(d == -1)[1]
        w = c1 - c0
        y = idx[rr] + top
        lane = lane_at * np.maximum(y - H, 1.0)
        keep = (w >= np.maximum(2, PAIR_LINE_W[0] * lane)) & (w <= PAIR_LINE_W[1] * lane)
        return y[keep].astype(float), (c0[keep] + c1[keep] - 1) / 2.0, k_of[rr[keep]]

    # --- start ---
    @staticmethod
    def from_bands(lf, band_line, near_y, far_y):
        """[xv, bL, bR] from the band method's center line (x at near_y and far_y) and the calibrated width."""
        H = lf.HORIZON_Y
        x_near, x_far = band_line
        if near_y == far_y:
            return None
        bc = (x_near - x_far) / (near_y - far_y)
        xv = x_near - bc * (near_y - H)
        D = lf.LANE_WIDTH_PX / max(342 - H, 1.0)
        return np.array([xv, bc - D / 2, bc + D / 2], float)

    # --- one frame ---
    def fit(self, lf, ys, xs, prior):
        """The pair fit from the prior. Returns (p, side per piece) or (None, sides) when there is too little."""
        H = lf.HORIZON_Y
        D = lf.LANE_WIDTH_PX / max(342 - H, 1.0)
        s = np.maximum(ys - H, 1.0)
        s_near = max(lf.FRAME_H - 1 - H, 1.0)
        base_w = (s / s_near) ** PAIR_NEAR_POW
        p = prior.copy()
        sides = np.zeros(len(xs), np.int8)
        # The priors, as parts of the normal equations per unit of the pieces' weight.
        # Width at the frame bottom, in px: (bR - bL) * s_near = D * s_near.
        g = np.array([0.0, -s_near, s_near])
        smooth_M = PAIR_SMOOTH * (lambda G: G.T @ G)(np.array([[1.0, 0.0, 0.0], [1.0, s_near, 0.0], [1.0, 0.0, s_near]]))
        prior_M = PAIR_WIDTH_PRIOR * np.outer(g, g) + smooth_M      # Last frame's lane: x at the horizon, both lines at the bottom.
        prior_v = PAIR_WIDTH_PRIOR * g * (D * s_near) + smooth_M @ prior
        ss, sx = s * s, s * xs
        for it in range(PAIR_ITER):
            lane = np.maximum((p[2] - p[1]) * s, 1.0)
            rl = (xs - (p[0] + p[1] * s)) / lane
            rr = (xs - (p[0] + p[2] * s)) / lane
            left = np.abs(rl) <= np.abs(rr)
            r = np.where(left, rl, rr)
            ok = np.abs(r) < PAIR_GATE[0 if it == 0 else 1]
            sides = np.where(ok, np.where(left, -1, 1), 0).astype(np.int8)
            if np.count_nonzero(ok) < PAIR_MIN_PIECES:
                return None, sides
            # Pieces left out get no weight: the same as dropping them, without copying every array.
            w = np.where(ok, base_w * np.minimum(1.0, PAIR_HUBER / np.maximum(np.abs(r), 1e-9)), 0.0)
            wl = np.where(left, w, 0.0)
            wr = w - wl
            ws, wls, wrs = w.sum(), wl @ s, wr @ s
            # Normal equations of left x = xv + bL * s, right x = xv + bR * s, written out (rows [1, s, 0] and [1, 0, s]).
            M = np.array([[ws, wls, wrs],
                          [wls, wl @ ss, 0.0],
                          [wrs, 0.0, wr @ ss]]) + prior_M * ws
            v = np.array([w @ xs, wl @ sx, wr @ sx]) + prior_v * ws
            try:
                p = np.linalg.solve(M, v)
            except np.linalg.LinAlgError:
                return None, sides
        return p, sides

    def __call__(self, lf, det, mask, top, near_y, far_y, cx0, band_line, sure_bands):
        """(offset, heading) from the pair fit, or the band answer det when the fit cannot be trusted this frame."""
        self.used = False
        self.pieces = None
        H = lf.HORIZON_Y
        D = lf.LANE_WIDTH_PX / max(342 - H, 1.0)
        start = self.from_bands(lf, band_line, near_y, far_y) if band_line is not None else None
        prior = self.p if self.p is not None else start
        if prior is None:
            return det
        ys, xs, ks = self.read_pieces(lf, mask, top)
        if len(xs) < PAIR_MIN_PIECES:
            self.p = None
            return det
        p, sides = self.fit(lf, ys, xs, prior)
        self.pieces = (ys, xs, sides)
        if p is None or len(np.unique(ks[sides != 0])) < PAIR_MIN_ROWS \
                or not PAIR_WIDTH_RANGE[0] <= (p[2] - p[1]) / D <= PAIR_WIDTH_RANGE[1]:
            self.p = None
            return det
        half = lf.FRAME_W / 2 * lf.LANE_WIDTH_PX / lf.REF_LANE_PX
        c = (p[1] + p[2]) / 2
        x_near, x_far = p[0] + c * (near_y - H), p[0] + c * (far_y - H)
        out = (float(np.clip((x_near - cx0) / half, -1, 1)), float(np.clip((x_far - x_near) / half, -1, 1)))
        # Held on to a misread while the band method clearly sees both lines elsewhere: start again from it.
        if det is not None and sure_bands and abs(out[0] - det[0]) > PAIR_RELOCK[0]:
            self.off += 1
            if self.off >= PAIR_RELOCK[1]:
                self.p, self.off = start, 0
                return det
        else:
            self.off = 0
        self.p = p
        self.used = True
        return out

    def curve_points(self, lf, which, top, n=16):
        """For the view only: the left (-1) or right (1) line as it really runs, so a curve shows as one.

        Starts from the pieces the pair fit gave this line, then grows up the line: a piece in the next sample row above
        that lies near the curve so far (PAIR_VIEW_GROW lane widths) joins it, even if it is far from the straight pair
        line (a bend), but never one the fit gave the other line. A 2nd-degree fit of x over y, from the frame bottom up
        to the farthest piece. The straight pair line when there are too few pieces or the curve leaves the lane."""
        straight = self.line_points(lf, which, top)
        if self.pieces is None or straight is None:
            return straight
        H = lf.HORIZON_Y
        ys, xs, sides = self.pieces
        use = sides == which
        if len(np.unique(ys[use])) < 3:
            return straight
        lane = (self.p[2] - self.p[1]) * np.maximum(ys - H, 1.0)
        rows = self.rows_y if self.rows_y is not None and len(self.rows_y) > 1 else np.array([lf.FRAME_H, top])
        step = float(np.abs(np.diff(rows)).max()) * 1.5 + PAIR_STRIP     # From one sample row to the next one up.
        free = sides != -which
        for _ in range(PAIR_ROWS):
            c = np.polyfit(ys[use], xs[use], 2 if len(np.unique(ys[use])) >= 4 else 1)
            reach = ys >= ys[use].min() - step
            grow = use | (free & reach & (np.abs(xs - np.polyval(c, ys)) < PAIR_VIEW_GROW * lane))
            if np.array_equal(grow, use):
                break
            use = grow
        c = np.polyfit(ys[use], xs[use], 2 if len(np.unique(ys[use])) >= 4 else 1)
        y_top = max(float(ys[use].min()), top)
        y = np.linspace(lf.FRAME_H - 1, y_top, n)
        x = np.polyval(c, y)
        b = self.p[1] if which < 0 else self.p[2]
        if np.any(np.abs(x - (self.p[0] + b * (y - H))) > (self.p[2] - self.p[1]) * np.maximum(y - H, 1.0)):
            return straight     # More than a lane width off the pair line: a misread, not this line bending.
        pts = np.stack([x, y], axis=1)
        last = self.drawn.get(which)
        if last is not None and last.shape == pts.shape and abs(last[-1, 1] - pts[-1, 1]) < 20:
            pts = PAIR_VIEW_SMOOTH * last + (1 - PAIR_VIEW_SMOOTH) * pts     # Steadier on screen: a piece more or less moves it less.
        self.drawn[which] = pts
        return pts

    def line_points(self, lf, which, top):
        """Screen points of the left (-1), right (1) or center (0) line over the band area, for the view."""
        if self.p is None:
            return None
        H = lf.HORIZON_Y
        y = np.array([lf.FRAME_H - 1, top], float)
        b = {-1: self.p[1], 1: self.p[2], 0: (self.p[1] + self.p[2]) / 2}[which]
        x = self.p[0] + b * (y - H)
        return np.stack([x, y], axis=1)
