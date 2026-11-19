"""Replay every recording in runs/ with both lane methods and compare them. Mac only, nothing runs on the Pi.

  .venv/bin/python tools/compare_recordings.py                     both methods, every recording, a table
  .venv/bin/python tools/compare_recordings.py --code /tmp/old/pi  the same with another copy of pi/ (before / after)
  .venv/bin/python tools/compare_recordings.py --models bands --dump /tmp/a     save every frame's decisions
  .venv/bin/python tools/compare_recordings.py --same /tmp/a /tmp/b             are two dumps identical, frame by frame

Per recording and method:
  lost     share of frames with no lane (det None)
  jump     share of frame pairs (both with a lane) whose lane position moved more than 0.3
  jitter   mean steering change per frame (degrees)
  diff     track only: share of frames (both with a lane) where the tracker and the band method differ by more than 0.3
           in lane position. The band method's answer comes from the same frame (LaneDetector.det_bands).
  relocks  track only: times the tracker started again from the band method because they kept disagreeing
  bend     track only: share of frames whose drawn center line, at its top (the farthest row with line pixels), bends
           more than 0.5 lane widths away from a straight lane (the bend terms C/s + E/s^2 over the lane width there)
Frames are handled as in pi/replay.py: each recording's own vision.json, landmarks, course.py and run.json settings.
"""
import argparse
import json
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _replay_common import ROOT, RUNS, all_runs, far_bend, frames, setup  # noqa: E402

DIFF = 0.3
BEND = 0.5      # far bend: share of frames whose drawn center line bends more than this many lane widths away from straight


def replay(job):
    run, code, model, dump, sets = job
    lf, rows, driver = setup(run, code, model, sets)
    import contextlib
    import io
    detect, course, steering = driver.detect, driver.course, driver.steering
    rec = []
    t0 = time.perf_counter()
    n = 0
    last_relock = None
    relocks = 0
    with contextlib.redirect_stdout(io.StringIO()):
        for row, frame, now in frames(run, rows, driver):
            s, cmd, _ = driver.tick(frame, now, show=True)
            n += 1
            det, bands = driver.det, detect.det_bands
            tr = detect.tracker
            if tr is not None and tr.relocked_at is not None and tr.relocked_at != last_relock:
                relocks += 1
                last_relock = tr.relocked_at
            rec.append({
                "frame": int(row["frame"]), "det": det, "bands": bands, "steer": s, "angle": steering.angle, "cmd": cmd,
                "step": course.idx, "junction_y": detect.junction_y, "gap_y": detect.gap_y, "edge": (detect.edge, driver.edge),
                "mat_top": detect.mat_top, "p": None if tr is None or tr.p is None else tr.p.tolist(),
                "conf": None if tr is None else tr.conf, "bend": far_bend(lf, tr),
                "do": (course.step() or {}).get("do"), "turning": course.turning(now),
            })
    ms = (time.perf_counter() - t0) / max(n, 1) * 1e3
    if dump:
        Path(dump).mkdir(parents=True, exist_ok=True)
        with open(Path(dump) / f"{Path(run).name}_{model}.pkl", "wb") as f:
            pickle.dump(rec, f)
    return Path(run).name, model, metrics(rec, model), relocks, ms


def metrics(rec, model):
    off = [r["det"][0] if r["det"] is not None else None for r in rec]
    lost = sum(o is None for o in off) / max(len(off), 1)
    pairs = [(a, b) for a, b in zip(off, off[1:]) if a is not None and b is not None]
    jump = sum(abs(b - a) > DIFF for a, b in pairs) / max(len(pairs), 1)
    ang = np.array([r["angle"] for r in rec])
    jitter = float(np.mean(np.abs(np.diff(ang)))) if len(ang) > 1 else 0.0
    out = {"frames": len(rec), "lost": lost, "jump": jump, "jitter": jitter}
    if model == "track":
        both = [(r["det"][0], r["bands"][0]) for r in rec if r["det"] is not None and r["bands"] is not None]
        out["diff"] = sum(abs(a - b) > DIFF for a, b in both) / max(len(both), 1)
        # The same, split: frames during a turn, frames where the band method's answer is clipped to +-1 (it saw the lane
        # center beyond the edge of its range), and the rest (the share of all compared frames each).
        kinds = {"turn": 0, "clipped": 0, "other": 0}
        for r in rec:
            if r["det"] is None or r["bands"] is None or abs(r["det"][0] - r["bands"][0]) <= DIFF:
                continue
            kind = "turn" if r.get("turning") else "clipped" if max(abs(r["bands"][0]), abs(r["bands"][1])) >= 1.0 else "other"
            kinds[kind] += 1
        out.update({f"diff_{k}": v / max(len(both), 1) for k, v in kinds.items()})
        bends = [r["bend"] for r in rec if r.get("bend") is not None]
        out["bend"] = float(np.mean([b > BEND for b in bends])) if bends else 0.0
    return out


def same(a, b):
    """Compare two dumps frame by frame: every decision must be equal."""
    da, db = Path(a), Path(b)
    bad = 0
    for fa in sorted(da.glob("*.pkl")):
        fb = db / fa.name
        if not fb.exists():
            print(f"{fa.name}: missing in {db}")
            bad += 1
            continue
        ra, rb = pickle.load(open(fa, "rb")), pickle.load(open(fb, "rb"))
        keys = ("frame", "det", "bands", "steer", "angle", "cmd", "step", "junction_y", "gap_y", "edge", "mat_top")
        diffs = [(i, k) for i, (x, y) in enumerate(zip(ra, rb)) for k in keys if x[k] != y[k]]
        if len(ra) != len(rb) or diffs:
            bad += 1
            print(f"{fa.name}: DIFFERENT  frames {len(ra)} vs {len(rb)}  first {diffs[:3]}")
        else:
            print(f"{fa.name}: identical ({len(ra)} frames)")
    print("ALL IDENTICAL" if not bad else f"{bad} recording(s) differ")
    return bad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default=str(ROOT / "pi"), help="the pi/ folder to test (default: this repository's)")
    ap.add_argument("--models", nargs="*", default=["bands", "track"])
    ap.add_argument("--runs", nargs="*", help="recording names (default: every one in runs/)")
    ap.add_argument("--jobs", type=int, default=max(mp.cpu_count() - 1, 1))
    ap.add_argument("--dump", help="save every frame's decisions here (one .pkl per recording and method)")
    ap.add_argument("--json", help="also write the table as JSON here")
    ap.add_argument("--set", nargs="*", default=[], help="MODULE.NAME=VALUE overrides, for experiments")
    ap.add_argument("--brief", metavar="LABEL", help="print one summary line per method, labeled LABEL")
    ap.add_argument("--same", nargs=2, metavar=("DUMP_A", "DUMP_B"), help="compare two dumps and exit")
    args = ap.parse_args()
    if args.same:
        sys.exit(0 if same(*args.same) else 1)
    runs = [RUNS / r for r in args.runs] if args.runs else all_runs()
    jobs = [(str(r), args.code, m, args.dump, args.set) for r in runs for m in args.models]
    jobs.sort(key=lambda j: -len(list(Path(j[0]).glob("*.jpg"))))      # Longest first: the pool finishes sooner.
    with mp.get_context("spawn").Pool(args.jobs, maxtasksperchild=1) as pool:
        res = pool.map(replay, jobs, chunksize=1)
    table = {}
    for name, model, m, relocks, ms in res:
        table.setdefault(name, {})[model] = dict(m, relocks=relocks, ms=ms)
    if args.json:
        Path(args.json).write_text(json.dumps(table, indent=1))
    if args.brief:      # One line per method, for quick experiments.
        for m in args.models:
            rs = [table[n][m] for n in sorted(table)]
            line = (f"{args.brief:14s} {m:5s} jump {np.mean([r['jump'] for r in rs]):5.1%} jit {np.mean([r['jitter'] for r in rs]):.2f}"
                    f" lost {np.mean([r['lost'] for r in rs]):4.1%}")
            if m == "track":
                d = [r["diff"] for r in rs]
                line += (f" | diff mean {np.mean(d):5.1%} max {max(d):5.1%} >10%: {sum(x > 0.1 for x in d):2d}"
                         f" relock {sum(r['relocks'] for r in rs):4d} bend {np.mean([r['bend'] for r in rs]):5.1%} | " + " ".join(f"{n[-6:]}:{r['diff'] * 100:.0f}" for n, r in zip(sorted(table), rs)))
            print(line)
        return
    hdr = f"{'recording':17s} {'frames':>6s}"
    for m in args.models:
        hdr += f" | {m}: {'lost':>5s} {'jump':>5s} {'jitter':>6s}" + (f" {'diff':>5s} {'relock':>6s} {'bend':>5s}" if m == "track" else "")
    print(hdr)
    for name in sorted(table):
        line = f"{name:17s} {next(iter(table[name].values()))['frames']:6d}"
        for m in args.models:
            r = table[name][m]
            line += f" | {' ' * len(m)}  {r['lost']:5.1%} {r['jump']:5.1%} {r['jitter']:6.2f}"
            if m == "track":
                line += f" {r['diff']:5.1%} {r['relocks']:6d} {r['bend']:5.1%}"
        print(line)
    line = f"{'mean':17s} {'':6s}"
    for m in args.models:
        rs = [table[n][m] for n in table]
        line += f" | {' ' * len(m)}  {np.mean([r['lost'] for r in rs]):5.1%} {np.mean([r['jump'] for r in rs]):5.1%} {np.mean([r['jitter'] for r in rs]):6.2f}"
        if m == "track":
            line += f" {np.mean([r['diff'] for r in rs]):5.1%} {sum(r['relocks'] for r in rs):6d} {np.mean([r['bend'] for r in rs]):5.1%}"
    print(line)
    for m in args.models:
        print(f"{m}: mean {np.mean([table[n][m]['ms'] for n in table]):.1f} ms/frame on this Mac (drawing included, parallel load)")


if __name__ == "__main__":
    main()
