#!/usr/bin/env python3
"""Tune the only two instancer knobs that transfer, per foreground model. RUNS ON TULEN.

Why two and not seventeen. A knob-sensitivity study over 511 configurations x 25 exact-GT
synthetic frames found that of the 14 assembly knobs exactly ONE has headroom that survives to
held-out frames -- ``merge_radius``, +0.042 [+0.004, +0.087], p=0.026 -- that the next three
largest-spread knobs gain on the tuning split and LOSE on held-out, and that composing the
top-15 one-at-a-time optima is worse than moving ``merge_radius`` alone (0.406 against 0.472).
``prob_thr`` is the second dimension because it is not a shared constant at all: the measured
optimum is 0.44 for v4b and 0.995 for nnU-Net, so a model scored at another model's threshold
is measuring calibration rather than quality.

Why on synthetic data. Protocol 17q measured that tuning on synthetic exact GT BEATS tuning on
real VAL (TEST 0.416 -> 0.457, +0.041 [+0.018, +0.065]): human GT is incomplete on sparse
frames and therefore rewards permissive settings, i.e. real-VAL tuning fits annotation noise.

Why stratified. Real MT-34 crossings are bimodal -- 10 of 34 frames carry 93 % of all crossings
while 24 frames have six or fewer -- so a tuning set of uniform 768 px crops optimises for a
regime holding 6 % of the benchmark. TUNE is half dense (crop 1344) and half sparse.

SELECT is a disjoint set of fresh frames: the grid argmax on TUNE is optimistic by construction,
so the incumbent is re-ranked on frames the search never saw, and both numbers are reported.

    PYTHONPATH=src:third_party python scripts/tune_two_knobs.py \
        --pred-dense .../pred_H_tune_dense --pred-sparse .../pred_H_tune_sparse \
        --pred-select .../pred_H_sel_dense --tag a_model_v5H
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "third_party"))

from instance.instancer_a import instance_a  # noqa: E402
from instance.metrics import aggregate_benchmark, centerline_f1  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402

UP, KAPPA_MAX, K_ORI = 1.5, 0.25, 6


def load(data_dir: str, pred_dir: str) -> list[dict]:
    out = []
    for p in sorted(glob.glob(os.path.join(data_dir, "*.h5"))):
        fr = read_frame_h5(p)
        if not fr["polylines"]:
            continue
        f = os.path.join(pred_dir, os.path.basename(p).replace(".h5", ".npz"))
        if not os.path.exists(f):
            continue
        ch = np.load(f)["prob"].astype(np.float32)
        out.append({"chans": ch, "prob2d": ch[:K_ORI].max(0),
                    "gt": [np.asarray(q, float) * UP for q in fr["polylines"]]})
    return out


_F: list[dict] = []


def _score_one(args):
    i, params, thr = args
    f = _F[i]
    _, masks = instance_a(f["prob2d"] > thr, KAPPA_MAX, params=params,
                          channels=f["chans"], prob=f["prob2d"])
    return centerline_f1(masks, f["gt"], tol=5.0, length_coverage=0.95,
                         precision_coverage=0.95)


def score(frames, params, thr, workers):
    global _F
    _F = frames
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_score_one, [(i, params, thr) for i in range(len(frames))]))
    return aggregate_benchmark(rows)["mean_f1"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dense", required=True)
    ap.add_argument("--pred-sparse", required=True)
    ap.add_argument("--pred-select", required=True)
    ap.add_argument("--data-dense", default="data/synth_tune_dense")
    ap.add_argument("--data-sparse", default="data/synth_tune_sparse")
    ap.add_argument("--data-select", default="data/synth_sel_dense")
    ap.add_argument("--base", default="src/instance/params_a_model_synthtuned.json")
    ap.add_argument("--merge", default="1.5,2.5,3.5,5.0,7.0,9.0")
    ap.add_argument("--thrs", default="0.85,0.93,0.97,0.99")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--tag", required=True, help="written as src/instance/params_<tag>.json")
    ap.add_argument("--out-dir", default="src/instance")
    args = ap.parse_args()

    base = json.load(open(args.base))
    base.pop("kappa_max", None)          # derived from the data; never a search dimension
    tune = load(args.data_dense, args.pred_dense) + load(args.data_sparse, args.pred_sparse)
    sel = load(args.data_select, args.pred_select)
    print(f"TUNE {len(tune)} frames (stratified)   SELECT {len(sel)} frames", flush=True)

    grid = [(m, t) for m in [float(x) for x in args.merge.split(",")]
            for t in [float(x) for x in args.thrs.split(",")]]
    res = []
    for m, t in grid:
        p = dict(base, merge_radius=m)
        f1 = score(tune, p, t, args.workers)
        res.append((f1, m, t))
        print("  merge_radius=%4.1f thr=%.2f  TUNE F1=%.4f" % (m, t, f1), flush=True)

    res.sort(reverse=True)
    print("\ntop 5 on TUNE, re-ranked on the disjoint SELECT set:", flush=True)
    best = None
    for f1, m, t in res[:5]:
        s = score(sel, dict(base, merge_radius=m), t, args.workers)
        print("  merge_radius=%4.1f thr=%.2f  TUNE %.4f -> SELECT %.4f" % (m, t, f1, s),
              flush=True)
        if best is None or s > best[0]:
            best = (s, f1, m, t)

    shipped = score(sel, base, base.get("prob_thr", 0.35), args.workers)
    s, f1, m, t = best
    print("\nSELECT: shipped params %.4f   tuned (merge_radius=%.1f thr=%.2f) %.4f   "
          "delta %+.4f" % (shipped, m, t, s, s - shipped), flush=True)
    print("optimism (TUNE - SELECT) = %+.4f" % (f1 - s), flush=True)

    out = os.path.join(args.out_dir, f"params_{args.tag}.json")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; params files are write-once (protocol 17n)")
    with open(out, "w") as fh:
        json.dump(dict(base, merge_radius=m, prob_thr=t), fh, indent=1)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
