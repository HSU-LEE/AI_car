"""Simulator evaluation: pi/sim.py's open-loop accuracy and closed-loop drive, every track x several seeds, in parallel.

  .venv/bin/python tools/sim_eval.py                          both methods, 5 tracks x 5 seeds
  .venv/bin/python tools/sim_eval.py --models track --seeds 3
  .venv/bin/python tools/sim_eval.py --code /tmp/old/pi       another copy of pi/ (before / after)

Accuracy (sim.accuracy): offset / heading error against the true lane, share of bad frames, share lost.
Drive (sim.run): share of the track driven before leaving the lane (done), lateral rms, steering jitter, finished runs.
"""
import argparse
import ast
import contextlib
import importlib
import io
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def work(job):
    kind, code, track, model, seed, sets = job
    sys.path.insert(0, str(Path(code).resolve()))
    with contextlib.redirect_stdout(io.StringIO()):
        import sim
        for kv in sets:     # MODULE.NAME=VALUE, for experiments.
            key, val = kv.split("=", 1)
            mod, name = key.rsplit(".", 1)
            setattr(importlib.import_module(mod), name, ast.literal_eval(val))
        if kind == "acc":
            return kind, model, sim.accuracy(track, model, seed)
        return kind, model, sim.run(track, model, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default=str(ROOT / "pi"))
    ap.add_argument("--models", nargs="*", default=["bands", "track"])
    ap.add_argument("--tracks", nargs="*")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=max(mp.cpu_count() - 1, 1))
    ap.add_argument("--set", nargs="*", default=[], help="MODULE.NAME=VALUE overrides, for experiments")
    ap.add_argument("--no-drive", action="store_true", help="accuracy only")
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.code).resolve()))
    with contextlib.redirect_stdout(io.StringIO()):
        import sim
    tracks = args.tracks or list(sim.TRACKS)
    kinds = ["acc"] if args.no_drive else ["acc", "run"]
    jobs = [(k, args.code, t, m, s, args.set) for k in kinds for t in tracks for m in args.models for s in range(args.seeds)]
    with mp.get_context("spawn").Pool(args.jobs, maxtasksperchild=1) as pool:
        res = pool.map(work, jobs, chunksize=1)
    print(f"{len(tracks)} tracks x {args.seeds} seeds: {', '.join(tracks)}\n")
    print(f"{'model':6s} | {'offset err':>10s} {'heading err':>11s} {'bad':>5s} {'lost':>5s} | {'done':>5s} {'finished':>8s} {'rms cm':>6s} {'jitter':>6s} {'lost':>5s}")
    for model in args.models:
        acc = [r for k, m, r in res if k == "acc" and m == model]
        line = (f"{model:6s} | {np.mean([r['offset_err'] for r in acc]):10.3f} {np.mean([r['heading_err'] for r in acc]):11.3f} "
                f"{np.mean([r['bad'] for r in acc]):5.1%} {np.mean([r['lost'] for r in acc]):5.1%}")
        drv = [r for k, m, r in res if k == "run" and m == model]
        if drv:
            line += (f" | {np.mean([r['done'] for r in drv]):5.1%} {sum(r['result'] == 'finished' for r in drv):4d}/{len(drv):<3d} "
                     f"{np.mean([r['rms'] for r in drv]):6.2f} {np.mean([r['jitter'] for r in drv]):6.2f} {np.mean([r['lost'] for r in drv]):5.1%}")
        print(line)
    print("\nPer track (offset err / heading err / done):")
    for model in args.models:
        for t in tracks:
            acc = [r for k, m, r in res if k == "acc" and m == model and r["track"] == t]
            drv = [r for k, m, r in res if k == "run" and m == model and r["track"] == t]
            print(f"  {model:6s} {t:9s} {np.mean([r['offset_err'] for r in acc]):.3f} / {np.mean([r['heading_err'] for r in acc]):.3f}"
                  + (f" / {np.mean([r['done'] for r in drv]):.0%}" if drv else ""))


if __name__ == "__main__":
    main()
