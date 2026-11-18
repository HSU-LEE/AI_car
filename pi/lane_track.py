"""Lane tracking with a lane model fitted to every line pixel in view (LANE_MODEL = "track" in lane_follow.py).

The band method (LaneDetector) finds one point per line in each of 6 bands and draws a straight line through the centers.
This tracks one model of the whole lane instead, fitted to all the line pixels from the bottom of the frame up to near the
horizon, starting from where the lane was in the last frame.

The model, in screen pixels (no cm, no camera height). With s = y - HORIZON_Y (rows below the vanishing point), a flat floor
seen by the camera puts a lane line at exactly
        x(s) = B + A*s + C/s
  B  where the lane points (its vanishing x): the heading
  A  how the line spreads out toward the car: together with the other line, the position and the lane width
  C  how it bends: the curvature (straight lane: 0)
Both lines share B and C and differ in A by the lane width D: left = center - D*s/2, right = center + D*s/2, with the
center x(s) = B + M*s + C/s + E/s^2. E is how the bend changes with distance (a curve that straightens out, an S-bend):
with one bend only, a road that switches from left to right was forced into one compromise bend, wildly wrong far away.
Five numbers, [B, M, D, C, E], describe the whole lane, near and far, and the model is linear in them, so a fit is a small
weighted least squares, solved a few times while pixels far from the model lose weight (robust).

Each frame:
  1. The line pixels (the band method's mask, plus a strip above it up to near the horizon) become runs: one per row and line piece.
     In the strip only the pieces that reach down to the band area count: a lane line runs on from the near lines, while the
     edges of walls, boxes and chair legs near the horizon stand apart, and the bend terms (largest so far up) would follow them.
  2. Runs near the left or the right model line (in lane widths) are given to that line. Wide runs are lines across the lane
     (a junction) and are left out; so are pieces between the lines (crosswalk stripes, reflections) and far outside them.
  3. Weighted least squares for [B, M, D, C, E], with the last frame's lane as a soft prior: one line alone, or a gap, still
     gives the whole lane, and nothing jumps.
  4. Confidence from how much of each line was found. With too little for a while, it starts again from the band method
     (from its line points in the bands it kept on its center line).
Position and heading come out exactly as the band method defines them (the same rows), so steering and every course test work as before.
"""
import cv2
import numpy as np

TRACK_FAR_ROWS = 18         # The tracker reads lines down from this many rows below the horizon (the band method starts much lower).
TRACK_GATE = 0.3            # A run counts for a line within this many lane widths of it (wider while unsure).
TRACK_GATE_LOST = 0.4       # Never half a lane width or more: a line could then be taken for the other one (the lane jumps a whole width).
TRACK_WIDE = 0.45           # A run wider than this share of the lane width at its row is a line across the lane, not a lane line.
TRACK_ITER = 2              # Robust re-weighting passes per frame.
TRACK_HUBER = 0.06          # Residual (lane widths) beyond which a run counts less.
TRACK_PRIOR = (0.2, 0.2, 5.0, 0.1, 0.1)   # Pull of the last frame's lane on B, M, D, C, E, against the line pixels (larger = moves less per frame).
TRACK_E_ZERO = 0.02         # Extra pull of E toward 0 (an even bend), so it only grows when the pixels clearly show the bend changing.
                                        # The lane width D is the same everywhere, so it is held firmly: one line alone cannot change it.
TRACK_C_MAX = 1.0e5         # Curvature limit: about a 15 cm radius (C = curvature x F^2 x camera height / 2; a 50 cm curve is about 31000).
TRACK_LOST_FRAMES = 6       # Frames of too little support before the tracker starts again from the band method.
TRACK_RELOCK_DIFF = 0.3     # The band method sees both lines, a plausible width apart, and its lane position differs from the
TRACK_RELOCK_FRAMES = 4     # tracker's by more than this (in the steering position unit) this many frames in a row: the tracker
                            # is the one that is wrong (it held on to a misread), so it starts again from the band method.
TRACK_RELOCK_BANDS = 3      # Bands with both lines needed for the band method to count as sure.
TRACK_EMPTY_FRAMES = 2      # Frames with no line pixel near either line at all: the lane is gone, not in a gap (a dashed line still shows pieces).
TRACK_FAR_EVERY = 2         # The far strip is read every this many frames (it changes slowly; the last reading is used in between).
TRACK_MIN_SUPPORT = 0.12    # Share of rows with a line found, below which the frame does not count as tracked.
TRACK_FAR_WEIGHT = 0.5      # Weight of rows above the band area (thin, far, less sure).
TRACK_FAR_LINK = 2          # A piece of line above the band area counts only if it reaches this close to the band area (half-size rows).
TRACK_ROW_STEP = 3          # Every this many rows of the band area are read: a smooth model needs no more, and the Pi saves time.
TRACK_INSIDE = 0.6          # The car is inside the lane: |M| below this share of D. Beyond it the lines were taken for each other.
TRACK_D_RANGE = (0.85, 1.18)    # The lane width D stays within this share of the calibrated width (LANE_WIDTH_PX).
TRACK_D_RETURN = 0.05       # Each frame D moves this share of the way back to the calibrated width: the lane is the same width
                            # everywhere, so a width learned from a misread (one line, a reflection) cannot stay.
TRACK_BOTH_MIN = 25         # Runs each line needs before the lane width D may change. With one line only, nothing measures the width:
                            # a wider lane and a sharper bend explain one line equally well, so D is held and only the bend moves.
TRACK_ASSIST_CONF = 0.6     # Junction check: the tracked lane stands in for missing line memory only when the tracker is at least this sure.
TRACK_ASSIST_FRAMES = 10    # ...and has held the lane this many frames since it started or re-locked (a fresh model can be the next lane).
TRACK_END_CONTINUE = 0.3    # Farther above a reported lane end, the line still on the tracked lane over this share of rows: it did not end.
TRACK_END_NEAR = 0.5        # The same, in the rows just above the end. A curve leaves the straight fit but is still on the tracked line there.
TRACK_ASSIST_SUPPORT = 0.12  # ...and both lines were found last frame, each at least this share of the rows (TRACK_MIN_SUPPORT). One line only
                            # and the model can sit a whole lane off while looking sure (run 20261110-183806 frame 316: support 0.55 and 0.01).


class LaneTracker:
    def __init__(self):
        self.p = None           # [B, M, D, C, E]
        self.bad = 0
        self.empty = 0
        self.disagree = 0       # Frames in a row the sure band method has disagreed with the tracker.
        self.relocked_at = None
        self.frame_no = 0
        self.far_cache = (np.zeros((0, 4)), 0)
        self.support = (0.0, 0.0)
        self.conf = 0.0
        self.age = 0            # Frames the current lane has been held since it started or re-locked.
        self.runs = None        # Last frame's runs and which line each went to, for the view.
        self.far_mask = None

    # --- geometry ---
    @staticmethod
    def lines_at(p, s):
        B, M, D, C, E = p
        c = B + M * s + C / s + E / (s * s)
        return c - D * s / 2, c + D * s / 2, c

    def reset(self):
        self.p, self.bad, self.age = None, 0, 0

    @staticmethod
    def inside(p):
        """Put the car back inside the lane if the model has it outside: that only happens when one line was taken for the other.

        While the car is in the lane, the left line widens toward the bottom left and the right one toward the bottom right on
        screen: |M| < D/2. A model with |M| beyond that is the same visible line given to the wrong side, so it is moved by
        whole lane widths (the lines change names) until the car is inside again.
        """
        p = p.copy()
        for _ in range(3):
            if abs(p[1]) <= TRACK_INSIDE * p[2]:
                break
            p[1] -= np.sign(p[1]) * p[2]
        return p

    # --- evidence ---
    @staticmethod
    def runs_of(mask, y0, step=1, weight=1.0, scale=1):
        """Runs of line pixels: arrays (y, x center, width, weight). Every step-th row. scale: the mask is this many times
        smaller than the frame (both ways), and the runs are given in frame pixels."""
        m = mask[::step]
        if not m.any():
            return np.zeros((0, 4))
        on = np.zeros((m.shape[0], m.shape[1] + 2), np.int8)
        on[:, 1:-1] = m > 0
        d = on[:, 1:] - on[:, :-1]      # +1 where a run starts, -1 just past its end.
        r0, c0 = np.nonzero(d == 1)
        c1 = np.nonzero(d == -1)[1]     # Same rows in the same order: every start has its end.
        out = np.empty((len(r0), 4))
        out[:, 0] = y0 + r0 * step * scale
        out[:, 1] = (c0 + c1 - 1) * scale / 2.0
        out[:, 2] = (c1 - c0) * scale
        out[:, 3] = weight
        return out

    def far_lines(self, lf, frame, thr, top):
        """Line pixels between near the horizon and the band area, the same way the band method finds them (top-hat, threshold)."""
        y0 = int(max(lf.HORIZON_Y + TRACK_FAR_ROWS, 0))
        if top - y0 < 8:
            self.far_mask = None
            return np.zeros((0, 4)), y0
        gray = cv2.GaussianBlur(cv2.cvtColor(frame[y0:top], cv2.COLOR_BGR2GRAY), (5, 5), 0)
        if lf.LINE_COLOR == "black":
            gray = cv2.bitwise_not(gray)
        half = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2), interpolation=cv2.INTER_AREA)
        tophat = lf.tophat_ellipse(half, max(lf.TOPHAT_K // 2, 3))
        mask = cv2.threshold(tophat, int(np.ceil(thr)) - 1, 255, cv2.THRESH_BINARY)[1]
        # A lane line reaches the band area; a chair leg or box edge near the horizon does not (run 20261110-191610).
        # Joining gaps let those edges back in, so only pieces that already touch the bottom rows are kept.
        r0 = max(mask.shape[0] - TRACK_FAR_LINK - 1, 0)
        ys, xs = np.nonzero(mask[r0:])
        for y, x in zip((ys + r0).tolist(), xs.tolist()):
            if mask[y, x] == 255:
                cv2.floodFill(mask, None, (x, y), 128, flags=8)
        mask = mask == 128
        self.far_mask = (y0, mask)
        return self.runs_of(mask, y0, 1, TRACK_FAR_WEIGHT, scale=2), y0       # Read at half size: far lines are thin anyway.

    # --- start from the band method ---
    def init_from_bands(self, lf, det):
        """A first lane from the band method's line points (left / right per band), with the calibrated lane width.

        Only the bands whose center lies on the band method's final center line (det.band_line): it has dropped the
        bands that do not line up (another lane's line in one band). A fit through every band put the whole lane one
        lane width off (run 20261112-193530 started on the next lane and, seeing one line only, never re-locked).
        No answer from the band method: no start."""
        H = lf.HORIZON_Y
        D0 = lf.LANE_WIDTH_PX / max(342 - H, 1.0)
        top = int(lf.FRAME_H * lf.ROI_TOP)
        h = (lf.FRAME_H - top) // lf.SLICES
        if det.det_bands is None:
            return False
        line = getattr(det, "band_line", None)
        near_y, far_y = lf.FRAME_H - h // 2, top + h // 2
        rows, xs, sides = [], [], []
        for i in range(lf.SLICES):
            cy = lf.FRAME_H - (i + 1) * h + h // 2
            if line is not None and det.center[i] is not None and \
                    abs(det.center[i] - line[0] - (line[1] - line[0]) * (cy - near_y) / (far_y - near_y)) >= lf.OUTLIER_PX:
                continue
            for side, arr in ((-1, det.left), (1, det.right)):
                if arr[i] is not None:
                    rows.append(cy - H)
                    xs.append(arr[i])
                    sides.append(side)
        if len(rows) < 2:
            return False
        s, x, sd = np.array(rows, float), np.array(xs, float), np.array(sides, float)
        # Fit B, M with D fixed and no bend: two numbers from a few points is all the band method can give.
        A = np.stack([np.ones_like(s), s], 1)
        y = x - sd * D0 * s / 2
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        self.p = self.inside(np.array([sol[0], sol[1], D0, 0.0, 0.0]))
        self.bad = 0
        self.age = 0
        return True

    # --- one frame ---
    def update(self, lf, det, frame, now=None):
        """Track the lane in this frame. det is the LaneDetector after its own pass (mask, line points, thr).
        Returns (offset, heading) as the band method defines them, or None when the lane is not tracked."""
        if self.p is not None:      # init_from_bands resets age; this counts frames held since then.
            self.age += 1
        H = lf.HORIZON_Y
        top = int(lf.FRAME_H * lf.ROI_TOP)
        if self.p is None and not self.init_from_bands(lf, det):
            self.runs, self.conf = None, 0.0
            return None
        near = self.runs_of(det.mask, top, TRACK_ROW_STEP, float(TRACK_ROW_STEP))     # Weighted back up: the near rows keep their share against the far strip.
        self.frame_no += 1
        if self.frame_no % max(TRACK_FAR_EVERY, 1) == 0 or self.far_cache[1] == 0:
            self.far_cache = self.far_lines(lf, frame, det.thr, top)
        far, y0 = self.far_cache
        if not len(near):           # Nothing near the car: an old far reading must not keep a lane alive that is gone.
            far = far[:0]
        runs = np.vstack([near, far]) if len(far) else near
        s_all = runs[:, 0] - H
        keep = s_all > 4
        runs, s_all = runs[keep], s_all[keep]
        D0 = lf.LANE_WIDTH_PX / max(342 - H, 1.0)
        p_prior = self.p.copy()
        p_prior[2] += TRACK_D_RETURN * (D0 - p_prior[2])
        p_prior = self.inside(p_prior)
        p = p_prior.copy()
        gate = TRACK_GATE_LOST if self.bad else TRACK_GATE
        side = np.zeros(len(runs), np.int8)
        # Everything that depends only on the row, made once (numpy calls are dear on the Pi, the arrays are small).
        x, width, wt = runs[:, 1], runs[:, 2], runs[:, 3]
        inv_s = 1.0 / s_all
        basis = np.stack([np.ones_like(s_all), s_all, s_all / 2, inv_s, inv_s * inv_s], 1)    # Column 2 gets the side sign.
        sm = float(np.median(s_all)) if len(s_all) else 100.0
        prior_base = np.array(TRACK_PRIOR)
        for _ in range(TRACK_ITER):
            B, M, D, C, E = p
            center = B + M * s_all + C * inv_s + E * inv_s * inv_s
            lane_px = np.maximum(D * s_all, 4.0)
            rc = (x - center) / lane_px             # Residual to the center line, in lane widths.
            to_left = rc < 0
            r = rc + np.where(to_left, 0.5, -0.5)  # ... and to the nearer of the two lines.
            ar = np.abs(r)
            ok = (ar < gate) & (width < TRACK_WIDE * lane_px)
            side = np.where(ok, np.where(to_left, -1, 1), 0).astype(np.int8)
            n_l, n_r = int((side < 0).sum()), int((side > 0).sum())
            if n_l + n_r < 3:
                break
            # Residuals in lane widths at their own row, with less weight beyond TRACK_HUBER (robust), times the run's weight.
            # The weights are scaled to sum to 1, so the data term is the mean squared error in lane widths, however many runs there are.
            hub = np.minimum(1.0, TRACK_HUBER / np.maximum(ar[ok], 1e-9))
            base = wt[ok] * hub
            ww = base / (base.sum() + 1e-12) / lane_px[ok] ** 2
            A = basis[ok].copy()
            A[:, 2] *= side[ok]
            Aw = A * ww[:, None]
            AtA = Aw.T @ A
            Atb = Aw.T @ x[ok]
            # The last frame's lane as a soft prior. Each number's pull is scaled by how much it moves the lines (in lane widths),
            # so TRACK_PRIOR is a plain ratio against the pixels.
            sens = np.array([1.0 / (D * sm), 1.0 / D, 0.5 / D, 1.0 / (D * sm * sm), 1.0 / (D * sm ** 3)])
            prior_w = prior_base * sens ** 2
            if min(n_l, n_r) < TRACK_BOTH_MIN:
                prior_w[2] *= 1e4           # One line only: keep the lane width.
            AtA[np.diag_indices(5)] += prior_w
            Atb += prior_w * p_prior
            AtA[4, 4] += TRACK_E_ZERO * sens[4] ** 2       # E toward 0 as well (Atb gains nothing: the target is 0).
            try:
                p = np.linalg.solve(AtA, Atb)
            except np.linalg.LinAlgError:
                break
            p[3] = min(max(p[3], -TRACK_C_MAX), TRACK_C_MAX)
            p[2] = min(max(p[2], TRACK_D_RANGE[0] * D0), TRACK_D_RANGE[1] * D0)
            p = self.inside(p)
        # Support: share of the rows read with a run found for each line.
        rows_total = max((lf.FRAME_H - top) / TRACK_ROW_STEP + (top - y0) / 2, 1)
        seen = np.zeros((2, lf.FRAME_H + 1), bool)
        rows = runs[:, 0].astype(int)
        seen[0, rows[side == -1]] = True
        seen[1, rows[side == 1]] = True
        sl, sr = seen[0].sum() / rows_total, seen[1].sum() / rows_total
        self.support = (sl, sr)
        self.conf = float(min(1.0, (sl + sr) / 0.6))
        self.runs = (runs, side)
        sane = bool(np.all(np.isfinite(p)))
        self.empty = self.empty + 1 if sl == 0 and sr == 0 else 0
        if self.empty >= TRACK_EMPTY_FRAMES:
            self.reset()
            self.empty = 0
            if det.det_bands is not None and self.init_from_bands(lf, det):
                return self.measure(lf)
            return None
        if max(sl, sr) < TRACK_MIN_SUPPORT or not sane:
            self.bad += 1
            if not sane or (self.bad >= TRACK_LOST_FRAMES and max(sl, sr) < TRACK_MIN_SUPPORT / 3):
                # Lost for good (nothing near either line). Starting again from the band method is a last resort: in a tight curve,
                # with one line in view, it often takes that line for the wrong side, and the tracker would inherit that.
                self.reset()
                if det.det_bands is not None and self.init_from_bands(lf, det):
                    return self.measure(lf)
                return None
            return self.measure(lf, p_prior)
        self.bad = 0
        self.p = p
        out = self.measure(lf)
        # Cross-check with the band method. It finds the lane afresh every frame, so it recovers at once where the tracker
        # can hold on to a misread; when it is sure (both lines in several bands) and keeps disagreeing, it wins.
        bands = det.det_bands
        both = sum(1 for a, b in zip(det.left, det.right) if a is not None and b is not None)
        if bands is not None and both >= TRACK_RELOCK_BANDS and abs(bands[0] - out[0]) > TRACK_RELOCK_DIFF:
            self.disagree += 1
            if self.disagree >= TRACK_RELOCK_FRAMES:
                self.reset()
                if self.init_from_bands(lf, det):
                    self.relocked_at = now
                    return self.measure(lf)
                return bands
        else:
            self.disagree = 0
        return out

    def ready(self, turning):
        """The tracked lane is fit to judge a detection by: sure, held since it started, not a turn's old lane, both lines found."""
        return not (turning or self.p is None or self.conf < TRACK_ASSIST_CONF or self.age < TRACK_ASSIST_FRAMES
                    or min(self.support) < TRACK_ASSIST_SUPPORT)

    def assist_lane(self, lf, y, turning):
        """(left x, right x) at frame row y from the last frame's model, or None when ready() is not."""
        if not self.ready(turning):
            return None
        s = y - lf.HORIZON_Y
        if s <= 4:
            return None
        left, right, _ = self.lines_at(self.p, s)
        if right - left <= 1:
            return None
        return float(left), float(right)

    def measure(self, lf, p=None):
        """(offset, heading) on the band method's rows and scale."""
        p = self.p if p is None else p
        top = int(lf.FRAME_H * lf.ROI_TOP)
        h = (lf.FRAME_H - top) // lf.SLICES
        near_y, far_y = lf.FRAME_H - h // 2, top + h // 2
        H = lf.HORIZON_Y
        _, _, xn = self.lines_at(p, near_y - H)
        _, _, xf = self.lines_at(p, far_y - H)
        cx0 = int(lf.FRAME_W * lf.CAMERA_X)
        half = lf.FRAME_W / 2 * lf.LANE_WIDTH_PX / lf.REF_LANE_PX
        offset = float(np.clip((xn - cx0) / half, -1, 1))
        heading = float(np.clip((xf - xn) / half, -1, 1))
        return offset, heading

    def pursuit(self, lf, look_lw, wheelbase_lw, cam_ahead_lw, shift_lw=0.0):
        """Steering angle (degrees, + = right) that puts the car on an arc through the lane center look_lw lane widths ahead
        (shift_lw lane widths to the right of it: the course's shift=).

        Flat floor, everything in lane widths: the camera is (342 - HORIZON_Y) / LANE_WIDTH_PX lane widths above the floor,
        so a screen row s below the horizon is F * height / s ahead. The arc starts at the rear axle, cam_ahead_lw behind the
        camera: angle a to the target, distance d, curvature 2 sin(a) / d, wheel angle atan(wheelbase * curvature).
        Returns (angle, the target on screen) or None.
        """
        if self.p is None:
            return None
        H = lf.HORIZON_Y
        height = (342 - H) / max(lf.LANE_WIDTH_PX, 1.0)
        F = lf.CAMERA_F
        s = F * height / max(look_lw, 0.05)
        _, _, xc = self.lines_at(self.p, s)
        xc += shift_lw * self.p[2] * s
        cx = lf.FRAME_W * lf.CAMERA_X
        Z = look_lw
        X = (xc - cx) * Z / F
        a = np.arctan2(X, Z + cam_ahead_lw)
        d = np.hypot(X, Z + cam_ahead_lw)
        delta = np.degrees(np.arctan(wheelbase_lw * 2 * np.sin(a) / d))
        return float(delta), (float(xc), float(H + s))

    def curve_points(self, lf, which, y_from=None, step=6):
        """Screen points of a model line (-1 left, 0 center, 1 right) for drawing: from the bottom up to the farthest row
        where line pixels were found (beyond that the model is only a guess, and its bend can swing wide)."""
        if self.p is None:
            return None
        H = lf.HORIZON_Y
        y0 = int(H + TRACK_FAR_ROWS) if y_from is None else y_from
        if y_from is None and self.runs is not None and (self.runs[1] != 0).any():
            y0 = max(y0, int(self.runs[0][self.runs[1] != 0, 0].min()))
        ys = np.arange(lf.FRAME_H - 1, y0, -step, dtype=float)
        xl, xr, xc = self.lines_at(self.p, ys - H)
        xs = {-1: xl, 0: xc, 1: xr}[which]
        ok = (xs > -lf.FRAME_W) & (xs < 2 * lf.FRAME_W)
        return np.stack([xs[ok], ys[ok]], 1).astype(np.int32)
