#!/usr/bin/env python3
"""Paired per-frame bootstrap between two trained arms, each at ITS OWN best operating point.

A difference of 0.025 pooled on 49 frames is a point estimate, not a result. This scores both
arms frame by frame on the same frames, then bootstraps the paired difference. Each arm must be
given the (prob_thr, min_length) it won with -- comparing at a shared operating point measures
probability calibration rather than model quality, which has already manufactured a phantom
0.47 F1 gap once in this project.

    python scripts/paired_arm_test.py \
        --arm control preds/LEN_control 0.995 12 \
        --arm treatment preds/LEN_treatment 0.997 20 \
        --data-dir data/real/mt34_htw_eval --split val
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.ndimage import zoom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from instance.instancer_a import instance_a  # noqa: E402
from instance.metrics import aggregate_benchmark, centerline_f1  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.fov import fov_mask  # noqa: E402

UP = 1.5
KAPPA_MAX = 0.25


def source_of(name: str) -> str:
    """Block from the file name: htw_* / alice_* (task 585) / training_img_* (task 586)."""
    b = os.path.basename(name)
    if b.startswith("htw"):
        return "htw"
    return "585" if b.startswith("alice") else "586"


def score_arm(pred_dir, thr, ml, frames, base_params):
    params = dict(base_params)
    params["min_length"] = ml
    params["prob_thr"] = thr
    out = {}
    for path, fr in frames:
        name = os.path.basename(path).replace(".h5", ".npz")
        npz = os.path.join(pred_dir, name)
        gt = [np.asarray(p, float) * UP for p in fr["polylines"] if len(p) >= 2]
        if not os.path.isfile(npz):
            if not gt:
                continue                      # zero-GT frame, legitimately absent
            raise SystemExit(f"missing prediction {npz} for a frame that has ground truth")
        prob = np.load(npz)["prob"].astype(np.float32)
        if prob.ndim == 3:
            prob = prob.max(axis=0)
        img = fr["image"]
        hi = (int(round(img.shape[0] * UP)), int(round(img.shape[1] * UP)))
        fv = zoom(fov_mask(img).astype(np.uint8), UP, order=0).astype(bool)
        fv = np.pad(fv, ((0, max(0, hi[0] - fv.shape[0])), (0, max(0, hi[1] - fv.shape[1]))),
                    constant_values=True)[:hi[0], :hi[1]]
        _polys, masks = instance_a((prob > thr) & fv, KAPPA_MAX, params=params,
                                   channels=prob[None], prob=prob)
        r = centerline_f1(masks, gt, tol=5.0, length_coverage=0.95, precision_coverage=0.95)
        r["n_gt"] = len(gt)
        out[name] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs=4, action="append", required=True,
                    metavar=("NAME", "PRED_DIR", "PROB_THR", "MIN_LENGTH"),
                    help="repeat twice; each arm carries its OWN operating point")
    ap.add_argument("--data-dir", default="data/real/mt34_htw_eval")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--params", default="src/instance/params_a_model_v5H.json")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out", default="paired_arms.json")
    args = ap.parse_args()
    if len(args.arm) != 2:
        raise SystemExit("give exactly two --arm blocks")

    base = json.load(open(args.params))
    frames = []
    for p in sorted(glob.glob(os.path.join(args.data_dir, "*.h5"))):
        fr = read_frame_h5(p)
        if str(fr["attrs"].get("split")) == args.split:
            frames.append((p, fr))
    if not frames:
        raise SystemExit(f"no {args.split} frames under {args.data_dir}")

    scored, names = {}, None
    for name, pred_dir, thr, ml in args.arm:
        scored[name] = score_arm(pred_dir, float(thr), float(ml), frames, base)
        print(f"scored {name}: {len(scored[name])} frames "
              f"(thr={thr}, min_length={ml})", flush=True)
        keys = set(scored[name])
        names = keys if names is None else (names & keys)

    a, b = [x[0] for x in args.arm]
    # Only frames with ground truth enter the macro mean, matching aggregate_benchmark.
    common = sorted(n for n in names if scored[a][n]["n_gt"] > 0)
    fa = np.array([scored[a][n]["f1"] for n in common])
    fb = np.array([scored[b][n]["f1"] for n in common])
    src = np.array([source_of(n) for n in common])

    agg = {k: aggregate_benchmark([scored[k][n] for n in common]) for k in (a, b)}
    print(f"\n{'arm':12s} {'mean-F1':>9s} {'micro':>8s} {'TP':>6s} {'FP':>6s} {'FN':>6s}")
    for k in (a, b):
        v = agg[k]
        print(f"{k:12s} {v['mean_f1']:9.4f} {v['micro_f1']:8.4f} "
              f"{v['total_tp']:6d} {v['total_fp']:6d} {v['total_fn']:6d}")

    d = fb - fa
    rng = np.random.default_rng(20260828)
    idx = rng.integers(0, len(d), size=(args.boot, len(d)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_two = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    wins = int((d > 0).sum())
    print(f"\nPAIRED, n={len(d)} frames, {args.boot} bootstrap resamples")
    print(f"  {b} - {a} = {d.mean():+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   "
          f"p={p_two:.4f}   {b} better on {wins}/{len(d)} frames")
    print(f"  verdict: {'SEPARATED (CI excludes 0)' if lo > 0 or hi < 0 else 'NOT separated'}")

    print(f"\n{'source':8s} {'n':>4s} {a[:10]:>11s} {b[:10]:>11s} {'delta':>8s}   paired 95% CI")
    per_src = {}
    for s in ("585", "586", "htw"):
        m = src == s
        if not m.any():
            continue
        # Per-source CI too: the block an intervention was AIMED at deserves its own interval,
        # and a pooled null can hide a real effect that another block cancels.
        ds = d[m]
        bi = rng.integers(0, len(ds), size=(args.boot, len(ds)))
        bs = ds[bi].mean(axis=1)
        slo, shi = np.percentile(bs, [2.5, 97.5])
        sp = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
        per_src[s] = dict(n=int(m.sum()), a=float(fa[m].mean()), b=float(fb[m].mean()),
                          delta=float(ds.mean()), ci=[float(slo), float(shi)], p=float(sp))
        print(f"{s:8s} {m.sum():4d} {fa[m].mean():11.4f} {fb[m].mean():11.4f} {ds.mean():+8.4f}"
              f"   CI [{slo:+.4f}, {shi:+.4f}]  p={sp:.4f}"
              f"{'  SEPARATED' if slo > 0 or shi < 0 else ''}")

    json.dump(dict(arms={k: dict(mean_f1=agg[k]["mean_f1"], micro_f1=agg[k]["micro_f1"],
                                 tp=agg[k]["total_tp"], fp=agg[k]["total_fp"],
                                 fn=agg[k]["total_fn"]) for k in (a, b)},
                   paired=dict(delta=float(d.mean()), ci=[float(lo), float(hi)],
                               p=float(p_two), n=len(d), wins=wins),
                   per_source=per_src, config=args.arm, split=args.split),
              open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
