"""Save recording frames where the tracker's lane disagrees with the band method, with both drawn, to look at who is right.

  .venv/bin/python tools/far_bend_frames.py                          every recording, the worst frames of each
  .venv/bin/python tools/far_bend_frames.py --frames 20261112-193530:66 20261112-193530:120
  .venv/bin/python tools/far_bend_frames.py --code /tmp/old/pi --out snapshots/far_bend_before

Drawn on each frame:
  band method: its line points per band (left blue, right red), its lane center points (white) and its straight center line (white)
  tracker:     its left / right / center lines (yellow), solid over the rows that had line pixels and thin beyond them
  runs:        the line pieces the tracker read (left blue dot, right red dot, unused gray dot)
  text:        the tracker's [B, M, D, C, E], both lane positions and the bend terms at the far end of the data
Also writes <out>/<recording>_params.csv: every frame's [B, M, D, C, E], confidence and both lane positions.
Frames are picked by the lane position difference (the measure compare_recordings.py counts), or with --by bend by the
far bend (compare_recordings.py: how far the drawn center line bends away from straight at its top, in lane widths). The curve difference, the
widest gap (lane widths at that row) between the tracker's center and the band method's center points, is shown too; it
also counts the band method's own bad points, so it is a hint, not a verdict.
"""
import argparse
import contextlib
import csv
import io
import multiprocessing as mp
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _replay_common import ROOT, all_runs, far_bend, frames, setup  # noqa: E402


def band_rows(lf):
    top = int(lf.FRAME_H * lf.ROI_TOP)
    h = (lf.FRAME_H - top) // lf.SLICES
    return [lf.FRAME_H - (i + 1) * h + h // 2 for i in range(lf.SLICES)]


def score(lf, detect):
    tr, det, bands = detect.tracker, detect.det_bands, None
    if tr is None or tr.p is None:
        return 0.0, None
    H = lf.HORIZON_Y
    worst = 0.0
    for cy, c in zip(band_rows(lf), detect.center):
        if c is None:
            continue
        _, _, xc = tr.lines_at(tr.p, cy - H)
        worst = max(worst, abs(xc - c) / max(tr.p[2] * (cy - H), 4.0))
    if det is not None:
        bands = det[0]
    return worst, bands


def draw(lf, frame, detect, driver, title):
    img = frame.copy()
    tr = detect.tracker
    H = lf.HORIZON_Y
    top = int(lf.FRAME_H * lf.ROI_TOP)
    cv2.line(img, (0, top), (lf.FRAME_W, top), (90, 90, 90), 1)
    import lane_track
    y_far = int(H + lane_track.TRACK_FAR_ROWS)
    cv2.line(img, (0, y_far), (lf.FRAME_W, y_far), (60, 60, 60), 1)
    if tr is not None and tr.runs is not None:
        runs, side = tr.runs
        for (y, x, w, _), sd in zip(runs, side):
            col = (255, 128, 0) if sd < 0 else (0, 0, 255) if sd > 0 else (140, 140, 140)
            cv2.circle(img, (int(x), int(y)), 2, col, -1)
    rows = band_rows(lf)
    for cy, l, r, c in zip(rows, detect.left, detect.right, detect.center):
        for x, col in ((l, (255, 80, 0)), (r, (0, 0, 230))):
            if x is not None:
                cv2.circle(img, (int(x), cy), 6, (20, 20, 20), -1)
                cv2.circle(img, (int(x), cy), 5, col, -1)
        if c is not None:
            cv2.circle(img, (int(c), cy), 4, (255, 255, 255), -1)
    det = detect.det_bands
    cx0 = int(lf.FRAME_W * lf.CAMERA_X)
    half = lf.FRAME_W / 2 * lf.LANE_WIDTH_PX / lf.REF_LANE_PX
    if det is not None:
        near_y, far_y = rows[0], rows[-1]
        xn = cx0 + det[0] * half
        xf = xn + det[1] * half
        cv2.line(img, (int(xn), near_y), (int(xf), far_y), (255, 255, 255), 2, cv2.LINE_AA)
    if tr is not None and tr.p is not None:
        ys = np.arange(lf.FRAME_H - 1, y_far, -3, dtype=float)
        xl, xr, xc = tr.lines_at(tr.p, ys - H)
        y_data = y_far
        if tr.runs is not None and (tr.runs[1] != 0).any():
            y_data = int(tr.runs[0][tr.runs[1] != 0, 0].min())
        for xs, thick in ((xl, 2), (xr, 2), (xc, 3)):
            pts = np.stack([xs, ys], 1)
            ok = (xs > -lf.FRAME_W) & (xs < 2 * lf.FRAME_W)
            inside = ok & (ys >= y_data)
            beyond = ok & (ys <= y_data)
            if inside.sum() > 1:
                cv2.polylines(img, [pts[inside].astype(np.int32)], False, (0, 230, 255), thick, cv2.LINE_AA)
            if beyond.sum() > 1:
                cv2.polylines(img, [pts[beyond].astype(np.int32)], False, (0, 160, 200), 1, cv2.LINE_AA)
        B, M, D, C, E = tr.p
        s_d = max(y_data - H, 1.0)
        txt = [title,
               f"B {B:.1f}  M {M:.3f}  D {D:.3f}  C {C:.0f}  E {E:.0f}  conf {tr.conf:.2f}",
               f"at data top s={s_d:.0f}: C/s {C / s_d:.1f}px  E/s^2 {E / s_d ** 2:.1f}px  (lane {D * s_d:.0f}px)",
               f"pos track {driver.det[0] if driver.det else float('nan'):+.2f}  bands {det[0] if det else float('nan'):+.2f}"]
        for i, t in enumerate(txt):
            cv2.putText(img, t, (8, 16 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, t, (8, 16 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def work(job):
    run, code, out, wanted, top_n, thr, sets, by = job
    lf, rows, driver = setup(run, code, "track", sets)
    detect = driver.detect
    name = Path(run).name
    out = Path(out)
    picked = []
    params = []
    with contextlib.redirect_stdout(io.StringIO()):
        for row, frame, now in frames(run, rows, driver):
            driver.tick(frame, now, show=False)
            fno = int(row["frame"])
            curve, bpos = score(lf, detect)
            tr = detect.tracker
            pos = driver.det[0] if driver.det is not None else None
            diff = abs(pos - bpos) if bpos is not None and pos is not None else 0.0
            bend = far_bend(lf, tr) or 0.0
            sc = bend if by == "bend" else diff
            params.append([fno] + (list(tr.p) if tr is not None and tr.p is not None else [None] * 5)
                          + [tr.conf if tr is not None else None, pos, bpos, diff, curve, bend])
            if wanted is not None:
                if fno in wanted:
                    picked.append((sc, fno, draw(lf, frame, detect, driver, f"{name} #{fno}  pos diff {diff:.2f}  far bend {bend:.2f}")))
            elif sc > thr:
                if picked and fno - picked[-1][1] < 15:     # One per stretch: keep the worse of two close frames.
                    if sc <= picked[-1][0]:
                        continue
                    picked.pop()
                picked.append((sc, fno, draw(lf, frame, detect, driver, f"{name} #{fno}  pos diff {diff:.2f}  far bend {bend:.2f}")))
    if wanted is None:
        picked = sorted(picked, key=lambda p: -p[0])[:top_n]
    out.mkdir(parents=True, exist_ok=True)
    for sc, fno, img in picked:
        cv2.imwrite(str(out / f"{name}_{fno:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    with open(out / f"{name}_params.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "B", "M", "D", "C", "E", "conf", "pos_track", "pos_bands", "pos_diff", "curve_diff", "far_bend"])
        w.writerows(params)
    n_bad = sum(1 for p in params if p[-3 if by == 'diff' else -1] > thr)
    return name, len(params), n_bad, [(fno, round(sc, 2)) for sc, fno, _ in picked]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default=str(ROOT / "pi"))
    ap.add_argument("--out", default=str(ROOT / "snapshots" / "far_bend"))
    ap.add_argument("--runs", nargs="*")
    ap.add_argument("--frames", nargs="*", help="RECORDING:FRAME, only these frames")
    ap.add_argument("--top", type=int, default=6, help="frames saved per recording")
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--set", nargs="*", default=[], help="MODULE.NAME=VALUE overrides, for experiments")
    ap.add_argument("--by", choices=("diff", "bend"), default="diff", help="pick frames by lane position difference or by far bend")
    ap.add_argument("--jobs", type=int, default=max(mp.cpu_count() - 1, 1))
    args = ap.parse_args()
    wanted = {}
    if args.frames:
        for f in args.frames:
            r, n = f.split(":")
            wanted.setdefault(r, set()).add(int(n))
    runs = [ROOT / "runs" / r for r in (wanted or args.runs or [])] or all_runs()
    jobs = [(str(r), args.code, args.out, wanted.get(Path(r).name) if wanted else None, args.top, args.thr, args.set, args.by) for r in runs]
    with mp.get_context("spawn").Pool(args.jobs, maxtasksperchild=1) as pool:
        for name, n, n_bad, picked in pool.imap_unordered(work, jobs):
            print(f"{name}: {n} frames, {n_bad} ({n_bad / max(n, 1):.0%}) over {args.thr}  saved {picked}")
    print(f"Images and params: {args.out}")


if __name__ == "__main__":
    main()
