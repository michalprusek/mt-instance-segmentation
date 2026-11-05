#!/usr/bin/env python3
"""How far can the ASSEMBLY go when the foreground is perfect? A ceiling probe.

The 0.8807 measured on the clean benchmark is "oracle mask + the instancer parameters tuned for the
v5H model". That conflates the assembly's real limit with one parameter set, and the whole plan
branches on which it is:

  reaches ~0.95  -> assembly is not the wall; better annotation and a better foreground buy the target
  stays  ~0.88   -> assembly IS the wall, and no amount of annotation substitutes for changing it

THIS RESULT IS A CEILING, NOT AN OPERATING POINT. The parameters are fitted on VAL against the
ground truth itself; they describe what the assembly could do, and must never be used to score any
arm. Random search rather than Optuna on purpose: Optuna is in no environment on this host, and
installing into the shared `seg`/`nnunet` envs could disturb a running campaign -- a ceiling probe
does not need a clever optimiser, it needs an honest maximum over enough samples.

Search is on a stratified SUBSET (cost), the winner is then re-measured on ALL frames, and both
numbers are printed: tuning and reporting on the same frames would overstate the ceiling.

    python scripts/oracle_ceiling_probe.py --data data/real/roi303_val121 --trials 200
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from instance.instancer_a import instance_a  # noqa: E402
from instance.metrics import (aggregate_benchmark, centerline_f1,  # noqa: E402
                              rasterise_polyline)
from mt_bench.cvat_import import read_frame_h5  # noqa: E402

UP, KAPPA_MAX = 1.5, 0.25

# the same ranges tune_instancer.py searches, so the probe is comparable to how the shipped
# parameters were found
SPACE = {
    "merge_radius": (2.0, 10.0), "bridge_max_len": (6.0, 50.0), "window": (6.0, 36.0),
    "w_theta": (0.2, 5.0), "w_kappa": (0.0, 40.0), "w_gap": (0.0, 0.2),
    "c_open": (0.3, 4.0), "min_length": (6.0, 50.0), "gap_floor": (2.0, 12.0),
    "w_ori": (0.0, 6.0), "link_max_gap": (0.0, 40.0), "c_open_link": (0.3, 4.0),
    "bridge_thr": (0.0, 0.5),
}
SMOOTH = [3, 5, 7, 9, 11, 13, 15]

_FRAMES = None


def _load(data_dir, split, pred_dir=None):
    out = []
    for p in sorted(glob.glob(os.path.join(data_dir, "*.h5"))):
        fr = read_frame_h5(p)
        if str(fr["attrs"].get("split")) != split:
            continue
        gt = [np.asarray(q, float) * UP for q in fr["polylines"] if len(q) >= 2]
        if not gt:
            continue
        img = fr["image"]
        shape = (int(round(img.shape[0] * UP)), int(round(img.shape[1] * UP)))
        if pred_dir:
            # a MODEL's probability map instead of the ground truth: the same search, asking how
            # much of the gap is in the instancer's parameters rather than in the mask
            f = os.path.join(pred_dir, os.path.basename(p).replace(".h5", ".npz"))
            if not os.path.isfile(f):
                continue
            pr = np.load(f)["prob"].astype(np.float32)
            out.append((pr.max(0) if pr.ndim == 3 else pr, gt, len(gt)))
            continue
        m = np.zeros(shape, bool)
        for q in gt:
            m |= rasterise_polyline(q, shape)
        out.append((m, gt, len(gt)))
    return out


def _init(data_dir, split, pred_dir=None):
    global _FRAMES
    _FRAMES = _load(data_dir, split, pred_dir)


def _score(arg):
    params, idxs = arg
    rows = []
    for k in idxs:
        src, gt, _ = _FRAMES[k]
        prob = src.astype(np.float32)
        mask = prob > params.get("prob_thr", 0.5)
        _p, masks = instance_a(mask, KAPPA_MAX, params=params, channels=prob[None], prob=prob)
        r = centerline_f1(masks, gt, tol=5.0, length_coverage=0.95, precision_coverage=0.95)
        r["n_gt"] = len(gt)
        rows.append(r)
    return aggregate_benchmark(rows)["mean_f1"]


def sample(rng, base):
    p = dict(base)
    for k, (lo, hi) in SPACE.items():
        p[k] = float(rng.uniform(lo, hi))
    p["smooth_size"] = int(rng.choice(SMOOTH))
    # for a binary oracle map the threshold is not a knob; for a model map it is the single most
    # important one and must be searched with the rest, or the result is a shared operating point
    p["prob_thr"] = 0.5 if base.get("_oracle", True) else float(rng.uniform(0.1, 0.999))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/roi303_val121")
    ap.add_argument("--split", default="val")
    ap.add_argument("--base", default="src/instance/params_a_model_v5H.json")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--tune-frames", type=int, default=30,
                    help="stratified subset the search runs on; the winner is re-measured on all")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260831)
    ap.add_argument("--pred-dir", default=None,
                    help="tune against a MODEL's probability maps instead of the oracle; the "
                         "threshold is then searched alongside the other parameters")
    ap.add_argument("--out", default="oracle_ceiling.json")
    args = ap.parse_args()

    base = json.load(open(args.base))
    base["_oracle"] = args.pred_dir is None
    frames = _load(args.data, args.split, args.pred_dir)
    n = len(frames)
    if not n:
        raise SystemExit(f"no {args.split} frames under {args.data}")
    # stratify the tuning subset by crowding, so the search is not fitted to sparse frames alone
    order = np.argsort([f[2] for f in frames])
    step = max(n // args.tune_frames, 1)
    tune_idx = [int(order[i]) for i in range(0, n, step)][:args.tune_frames]
    all_idx = list(range(n))
    print(f"{n} frames; searching on {len(tune_idx)} stratified frames, "
          f"re-measuring the winner on all {n}\n")

    rng = np.random.default_rng(args.seed)
    cands = [sample(rng, base) for _ in range(args.trials)]

    t0 = time.time()
    with Pool(args.jobs, initializer=_init, initargs=(args.data, args.split, args.pred_dir)) as pool:
        ship_p = dict(base) if args.pred_dir else dict(base, prob_thr=0.5)
        shipped = pool.map(_score, [(ship_p, tune_idx)])[0]
        print(f"  shipped v5H parameters, tuning subset : {shipped:.4f}")
        scores = []
        for i, s in enumerate(pool.imap(_score, [(c, tune_idx) for c in cands]), 1):
            scores.append(s)
            if i % 20 == 0:
                print(f"  {i}/{len(cands)} trials, best so far {max(scores):.4f} "
                      f"({time.time()-t0:.0f} s)", flush=True)
        best_i = int(np.argmax(scores))
        print(f"\n  best on the tuning subset: {scores[best_i]:.4f}")
        full_best, full_ship = pool.map(_score, [(cands[best_i], all_idx), (ship_p, all_idx)])

    print(f"\n{'MODEL MASK: ' + args.pred_dir if args.pred_dir else 'ORACLE MASK'}, ALL {n} FRAMES")
    print(f"  shipped v5H instancer parameters : {full_ship:.4f}")
    print(f"  best of {args.trials} random trials      : {full_best:.4f}")
    print(f"\n  This is a CEILING fitted on VAL against the ground truth, not an operating point.")
    for _c in cands:
        _c.pop("_oracle", None)
    json.dump(dict(n_frames=n, n_tune=len(tune_idx), trials=args.trials,
                   shipped_all=full_ship, best_all=full_best,
                   best_tune=float(scores[best_i]), params=cands[best_i]),
              open(args.out, "w"), indent=1)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
