#!/usr/bin/env python3
"""A locally NOISE-ADAPTIVE decode threshold, against a foreground-matched global one.

Measured 2026-09-09: the 4.8 % of GT curves the model never finds at any threshold are NOT fainter
(median contrast 0.059 vs 0.067) but sit in NOISIER neighbourhoods (median SNR 0.939 vs 2.325).
And 95.2 % of curves are recoverable at SOME threshold while only 66.2 % are recovered at the best
SINGLE one, with a per-FRAME oracle threshold worth only +0.048 -- so the variation that matters is
WITHIN a frame and its measured discriminator is local noise.

The 11-variant decoder screen tested global thresholds, hysteresis, valley cuts and watershed. It
did not test a threshold that FOLLOWS the local noise. This does:

    t(x) = clip(t0 - k * (sigma(x) - median sigma) / median sigma, t_lo, t0)

with sigma the high-pass std of the image in local windows, bilinearly interpolated. Lower the bar
exactly where the evidence is noisy.

**The comparator is not the baseline.** It is a GLOBAL threshold lowered until it paints the SAME
foreground, computed per frame by bisection -- the control that killed the overlap-conditioned
fill at every setting. Beating the baseline only proves that more foreground was painted.

    PYTHONPATH=src:scripts python scripts/read_adaptive_thr.py --data data/real/roi303_eval \
        --pred-dir <dump> --params src/instance/params_a.json --thr 0.98 --k 0.02,0.05,0.10 --out a.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("src", "scripts"):
    sys.path.insert(0, os.path.join(ROOT, sub))

from instance.instancer_a import default_params, instance_a  # noqa: E402
from instance.metrics import centerline_f1  # noqa: E402
from instance.oracle import oracle_instance_masks  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402

UP, KAPPA_MAX = 1.5, 0.25


def local_sigma(img01, win=64, grid=16):
    """High-pass std on a coarse grid, bilinearly interpolated to full resolution."""
    hp = img01 - gaussian_filter(img01, 3.0)
    h, w = img01.shape
    ys = np.linspace(0, max(h - win, 0), grid).astype(int)
    xs = np.linspace(0, max(w - win, 0), grid).astype(int)
    S = np.array([[hp[y:y + win, x:x + win].std() for x in xs] for y in ys])
    return zoom(S, (h / grid, w / grid), order=1)[:h, :w]


def matched_global(prob, target_px, t0):
    """The global threshold that paints `target_px` foreground -- the honest comparator."""
    lo, hi = 0.0, t0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if int((prob > mid).sum()) >= target_px:
            lo = mid
        else:
            hi = mid
    return lo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--thr", type=float, default=0.98)
    ap.add_argument("--k", default="0.02,0.05,0.10")
    ap.add_argument("--t-lo", type=float, default=0.70)
    ap.add_argument("--cov", type=float, default=0.80)
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1",
                    help="i/n -- process every n-th frame starting at i, so the read can be "
                         "split across cores. Shards are merged by summing the per-frame lists.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    params = {**default_params(), **json.load(open(args.params))}
    params.pop("kappa_max", None); params.pop("prob_thr", None)
    ks = [float(x) for x in args.k.split(",")]
    acc = {f"k{k:g}": [] for k in ks}
    acc["base"] = []; acc.update({f"match{k:g}": [] for k in ks})

    si, sn = (int(v) for v in args.shard.split("/"))
    frames = 0
    for fi, p in enumerate(sorted(glob.glob(os.path.join(args.data, "*.h5")))):
        if fi % sn != si:
            continue
        stem = os.path.basename(p)[:-3]
        npz = os.path.join(args.pred_dir, stem + ".npz")
        if not os.path.exists(npz):
            continue
        fr = read_frame_h5(p)
        if str(fr["attrs"].get("split")) != args.split:
            continue
        prob = np.load(npz)["prob"].astype(np.float32)[0]
        gt = [np.asarray(q, float) * UP for q in fr["polylines"]]
        if not gt:
            continue
        img = np.asarray(fr["image"], float)
        lo_, hi_ = np.percentile(img, [1, 99])
        img_e = zoom(np.clip((img - lo_) / (hi_ - lo_ + 1e-9), 0, 1), UP, order=1)
        img_e = img_e[:prob.shape[0], :prob.shape[1]]
        if img_e.shape != prob.shape:
            continue
        sig = local_sigma(img_e)
        med = float(np.median(sig)) + 1e-9

        def score(mask, tag):
            polys, _ = instance_a(mask, KAPPA_MAX, params, channels=prob[None], prob=prob)
            pm = oracle_instance_masks(polys, prob.shape, half_width=1.0, up=1.0)
            r = centerline_f1(pm, gt, tol=5.0, length_coverage=args.cov, precision_coverage=args.cov)
            acc[tag].append((r["f1"], r["tp"], r["fp"], r["fn"]))

        base = prob > args.thr
        score(base, "base")
        for k in ks:
            t = np.clip(args.thr - k * (sig - med) / med, args.t_lo, args.thr)
            m = prob > t
            score(m, f"k{k:g}")
            score(prob > matched_global(prob, int(m.sum()), args.thr), f"match{k:g}")
        frames += 1
        print(f"{stem}", flush=True)
        if args.limit and frames >= args.limit:
            break

    summ = {"frames": frames, "thr": args.thr, "cov": args.cov, "t_lo": args.t_lo}
    for tag, v in acc.items():
        if not v:
            continue
        f1 = np.array([x[0] for x in v]); tp = sum(x[1] for x in v)
        fp = sum(x[2] for x in v); fn = sum(x[3] for x in v)
        summ[tag] = {"macro_f1": round(float(f1.mean()), 4),
                     "micro_f1": round(2 * tp / max(2 * tp + fp + fn, 1), 4),
                     "tp": tp, "fp": fp, "fn": fn}
    for k in ks:
        a_, b_ = summ.get(f"k{k:g}"), summ.get(f"match{k:g}")
        if a_ and b_:
            summ[f"k{k:g}_minus_matched"] = {"macro": round(a_["macro_f1"] - b_["macro_f1"], 4),
                                             "micro": round(a_["micro_f1"] - b_["micro_f1"], 4)}
    summ["_per_frame"] = {t: v for t, v in acc.items()}     # kept so shards can be merged
    json.dump(summ, open(args.out, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
