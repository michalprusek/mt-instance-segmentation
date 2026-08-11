#!/usr/bin/env python3
"""VAL ablation of instancer A's three added levers. RUNS ON TULEN.

Each lever is turned off by a PARAMETER, not a code path, so the ablation exercises exactly
the shipped implementation:

* ``displacement`` -- ``gap_floor`` set very large forces every pair back to the direct
  ``|theta_a + pi - theta_b|`` turn, i.e. the pre-17k cost that let two parallel microtubules
  4 px apart look like a perfect through-path;
* ``gap_link``     -- ``link_max_gap = 0``;
* ``orientation``  -- ``w_ori = 0``.

VAL only. TEST stays sealed until all levers are frozen.

    PYTHONPATH=src ~/dinov3_env/bin/python scripts/ablate_levers.py --masks oracle
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import zoom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from instance.instancer_a import instance_a  # noqa: E402
from instance.metrics import (aggregate_benchmark, bundle_recovery,  # noqa: E402
                              centerline_f1, fragmentation, junction_identity)
from instance.oracle import (oracle_instance_masks, oracle_ori_channels)  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.fov import fov_mask  # noqa: E402
from mt_bench.gt_stats import count_crossings  # noqa: E402

UP = 1.5
KAPPA_MAX = 0.25


def parallel_pairs(polylines, gap_lo=2.0 * UP, gap_hi=6.0 * UP,
                   min_len=20.0 * UP, ds=2.0):
    from scipy.spatial import cKDTree

    from instance.geometry import resample
    res = [resample(np.asarray(p, float), ds=ds) for p in polylines]
    trees = [cKDTree(p) if len(p) > 1 else None for p in res]
    out = []
    for i in range(len(res)):
        if trees[i] is None:
            continue
        for j in range(i + 1, len(res)):
            if trees[j] is None:
                continue
            d, _ = trees[j].query(res[i], k=1)
            if float(np.sum((d >= gap_lo) & (d <= gap_hi)) * ds) >= min_len:
                out.append((i, j))
    return out


def load(data_dir, split, masks, pred_dir, prob_thr):
    frames = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.h5"))):
        fr = read_frame_h5(path)
        a = fr["attrs"]
        if str(a.get("split")) != split:
            continue
        shape = (int(a["height"]), int(a["width"]))
        hi = (int(round(shape[0] * UP)), int(round(shape[1] * UP)))
        if masks == "oracle":
            chans = oracle_ori_channels(fr["polylines"], shape, K=6, half_width=1.0, up=UP)
        else:
            chans = np.load(os.path.join(
                pred_dir, os.path.basename(path).replace(".h5", ".npz")))["prob"]
            chans = np.asarray(chans, dtype=np.float32)
            fv = zoom(fov_mask(fr["image"]).astype(np.uint8), UP, order=0).astype(bool)
            fv = np.pad(fv, ((0, max(0, hi[0] - fv.shape[0])),
                             (0, max(0, hi[1] - fv.shape[1]))),
                        constant_values=True)[:hi[0], :hi[1]]
            chans = chans * fv[None, :, :]
        gt = [np.asarray(p, float) * UP for p in fr["polylines"]]
        frames.append({
            "name": os.path.basename(path), "chans": chans, "hi": hi, "gt": gt,
            "thr": 0.5 if masks == "oracle" else prob_thr,
            "cross": count_crossings(gt, tol=2.0 * UP, endpoint_margin=8.0 * UP),
            "par": parallel_pairs(gt),
        })
    return frames


def run(frames, params) -> dict:
    rows, jr, bd, fg = [], [], [], []
    for f in frames:
        thr = params.get("prob_thr", f["thr"])
        mask = f["chans"].max(axis=0) > thr
        pls, _ = instance_a(mask, KAPPA_MAX, params,
                            channels=f["chans"], prob=f["chans"].max(axis=0))
        masks = oracle_instance_masks(pls, f["hi"], half_width=1.0, up=1.0)
        rows.append({**centerline_f1(masks, f["gt"], tol=5.0, length_coverage=0.95,
                                     precision_coverage=0.95),
                     "n_gt": len(f["gt"])})
        j = junction_identity(masks, f["gt"], f["cross"], tol=5.0, arm_offset=15.0 * UP)
        if j["n_crossings"]:
            jr.append(j["rate"])
        b = bundle_recovery(masks, f["gt"], f["par"], tol=1.5 * UP)
        if b == b:
            bd.append(b)
        fr_ = fragmentation(masks, f["gt"], tol=5.0)
        if fr_ == fr_:
            fg.append(fr_)
    agg = aggregate_benchmark(rows)
    return {"mean_f1": agg["mean_f1"], "micro_f1": agg["micro_f1"],
            "junction": float(np.mean(jr)) if jr else float("nan"),
            "bundle": float(np.mean(bd)) if bd else float("nan"),
            "frag": float(np.mean(fg)) if fg else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/mt34_eval")
    ap.add_argument("--split", default="val")
    ap.add_argument("--masks", default="oracle", choices=["oracle", "model"])
    ap.add_argument("--pred-dir", default=None)
    ap.add_argument("--prob-thr", type=float, default=0.35)
    ap.add_argument("--params", default=None, help="tuned params json for the FULL config")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    frames = load(args.data, args.split, args.masks, args.pred_dir, args.prob_thr)
    base = json.load(open(args.params)) if args.params else {}
    base.pop("kappa_max", None)

    variants = {
        "FULL": {},
        "-displacement (direct turn)": {"gap_floor": 1e6},
        "-gap_link": {"link_max_gap": 0.0},
        "-orientation": {"w_ori": 0.0},
        "-all three": {"gap_floor": 1e6, "link_max_gap": 0.0, "w_ori": 0.0},
    }

    print(f"\n=== instancer A lever ablation | {args.masks} masks | "
          f"{args.split.upper()} ({len(frames)} frames) ===")
    print(f"{'variant':30s} {'mean F1':>8s} {'micro':>7s} {'junction':>9s} "
          f"{'bundle':>7s} {'frag':>6s}")
    results = {}
    for name, override in variants.items():
        r = run(frames, {**base, **override})
        results[name] = r
        print(f"{name:30s} {r['mean_f1']:8.3f} {r['micro_f1']:7.3f} "
              f"{r['junction']:9.3f} {r['bundle']:7.3f} {r['frag']:6.2f}", flush=True)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
