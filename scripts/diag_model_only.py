#!/usr/bin/env python3
"""Measure the MODEL, with no instancer and no post-processing anywhere.

Every instance number in this project passes through instancer A, whose constants were fitted on
long-filament blocks; on 2026-09-11 that was shown to delete 49 % of the clean block's ground truth
before matching, so an instance F1 there measures the filter as much as the network. This script
touches none of it: it reads the cached probability map and asks only what the NETWORK put where.

Three assembler-free readings, per source block and per GT length bin:

  1. per-curve centerline coverage -- for each GT polyline, the fraction of its length whose
     probability is above tau; reported as the share of curves covered >=95 % and >=80 %. This is
     the quantity a perfect decoder could convert into a true positive, and it is the honest
     ceiling of the map.
  2. per-curve probability profile -- median of each curve's MINIMUM and MEAN probability, which
     separates "the filament is faint everywhere" from "the filament has a hole".
  3. pixel-level precision / recall / Dice of the thresholded map against the GT centerline
     dilated to the metric's own tolerance, plus the foreground fraction.

The length bins are the point: the clean block's median GT is 29 native px while the blocks the
pipeline was fitted on sit at 107-358, so every aggregate over a block is also an average over a
length distribution, and two blocks can differ entirely because their filaments differ in length.

    PYTHONPATH=src:scripts python scripts/diag_model_only.py \
        --data data/real/mt34_htw_eval --split val \
        --pred-dir /disk2/prusek/mt_work/preds/SPARSE35_htw_native --out model_only_htw.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from mt_bench.cvat_import import read_frame_h5  # noqa: E402

UP = 1.5
# 0.40 and 0.45 are not decoration: the clean block reaches 80 % curve coverage at thr ~0.43, and
# without nodes there the precision at matched coverage would be an interpolation inside an
# interpolation across a 0.06 precision step.
THRS = (0.30, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.97, 0.98, 0.99)
# Eval-frame px. The first edge sits just below the shipped instancer's min_length (44.74) so the
# block that constant deletes is visible as its own row.
LEN_BINS = ((0, 30), (30, 45), (45, 80), (80, 160), (160, 320), (320, 10 ** 9))


def source_of(name: str) -> str:
    """The four provenance blocks, which must never be pooled: see docs/TODO.md on contamination."""
    if name.startswith("htw_"):
        return "htw"
    if name.startswith("alice_"):
        return "alice"
    if name.startswith("training_img_"):
        return "roi303-derived"
    if name.startswith("roi_img_"):
        return "roi303"
    return "other"


def densify(poly: np.ndarray, step: float = 1.0) -> np.ndarray:
    """Resample an (x, y) polyline to ~`step` px spacing so coverage is a length fraction."""
    d = np.hypot(*np.diff(poly, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 0:
        return poly[:1]
    t = np.arange(0.0, s[-1] + 1e-9, step)
    return np.stack([np.interp(t, s, poly[:, 0]), np.interp(t, s, poly[:, 1])], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--tol", type=float, default=5.0, help="eval px; the metric's own tolerance")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    curves: list[dict] = []          # one record per GT polyline
    pix = {}                         # (source, thr) -> [tp, fp, fn, fg_px, total_px]
    n_frames = {}

    paths = sorted(glob.glob(os.path.join(args.data, "*.h5")))
    for path in paths:
        name = os.path.basename(path)
        fr = read_frame_h5(path)
        if args.split != "all" and str(fr["attrs"].get("split", "")) != args.split:
            continue
        npz = os.path.join(args.pred_dir, name.replace(".h5", ".npz"))
        if not os.path.exists(npz):
            continue
        prob = np.load(npz)["prob"].astype(np.float32)
        prob = prob[0] if prob.ndim == 3 else prob
        H, W = prob.shape
        src = source_of(name)
        n_frames[src] = n_frames.get(src, 0) + 1

        gt = [np.asarray(p, float) * UP for p in fr["polylines"]]

        # --- 1 & 2: per-curve, straight off the map -------------------------------------------
        for poly in gt:
            if len(poly) < 2:
                continue
            dense = densify(poly)
            xs = np.clip(dense[:, 0], 0, W - 1)      # polylines are (x=col, y=row)
            ys = np.clip(dense[:, 1], 0, H - 1)
            p = map_coordinates(prob, [ys, xs], order=1, mode="nearest")
            length = float(np.hypot(*np.diff(poly, axis=0).T).sum())
            curves.append({
                "src": src, "len": length,
                "p_min": float(p.min()), "p_mean": float(p.mean()), "p_med": float(np.median(p)),
                "cov": {f"{t:.2f}": float((p >= t).mean()) for t in THRS},
            })

        # --- 3: pixel-level, against the GT centerline dilated to tol -------------------------
        core = np.zeros((H, W), bool)
        for poly in gt:
            d = densify(poly, 0.5)
            core[np.clip(d[:, 1].astype(int), 0, H - 1), np.clip(d[:, 0].astype(int), 0, W - 1)] = True
        near = distance_transform_edt(~core) <= args.tol if core.any() else np.zeros((H, W), bool)
        for t in THRS:
            m = prob >= t
            k = (src, f"{t:.2f}")
            acc = pix.setdefault(k, [0, 0, 0, 0, 0])
            acc[0] += int((m & near).sum())
            acc[1] += int((m & ~near).sum())
            acc[2] += int((~m & near).sum())
            acc[3] += int(m.sum())
            acc[4] += H * W
        if args.limit and sum(n_frames.values()) >= args.limit:
            break

    # ---------------------------------------------------------------- report
    out = {"data": args.data, "pred_dir": args.pred_dir, "split": args.split,
           "tol": args.tol, "n_frames": n_frames, "blocks": {}}
    srcs = sorted({c["src"] for c in curves})
    for src in srcs:
        cs = [c for c in curves if c["src"] == src]
        L = np.array([c["len"] for c in cs])
        pmin = np.array([c["p_min"] for c in cs])
        block = {"n_curves": len(cs),
                 "len_eval_px_p25_p50_p75": [float(np.percentile(L, q)) for q in (25, 50, 75)],
                 # The TAIL of the per-curve minimum, not just its median: a worst-point training
                 # term has gradient only where the worst point is not already confident, so this
                 # is what decides whether such a term can bite in a given domain.
                 "p_min_q": {f"p{q:02d}": float(np.percentile(pmin, q)) for q in (5, 10, 25, 50)},
                 "p_min_frac_below": {f"{t:.2f}": float((pmin < t).mean())
                                      for t in (0.5, 0.9, 0.97, 0.99)},
                 "per_thr": {}, "by_length": {}}
        for t in THRS:
            key = f"{t:.2f}"
            cov = np.array([c["cov"][key] for c in cs])
            tp, fp, fn, fg, tot = pix[(src, key)]
            block["per_thr"][key] = {
                "curves_cov95": float((cov >= 0.95).mean()),
                "curves_cov80": float((cov >= 0.80).mean()),
                "cov_median": float(np.median(cov)),
                "pixel_precision": tp / max(tp + fp, 1),
                "pixel_recall": tp / max(tp + fn, 1),
                "pixel_dice": 2 * tp / max(2 * tp + fp + fn, 1),
                "fg_frac": fg / max(tot, 1),
            }
        for lo, hi in LEN_BINS:
            sel = [c for c in cs if lo <= c["len"] < hi]
            if not sel:
                continue
            block["by_length"][f"{lo}-{hi if hi < 10**9 else 'inf'}"] = {
                "n": len(sel),
                "p_min_median": float(np.median([c["p_min"] for c in sel])),
                "p_mean_median": float(np.median([c["p_mean"] for c in sel])),
                **{f"cov95@{t:.2f}": float(np.mean([c["cov"][f"{t:.2f}"] >= 0.95 for c in sel]))
                   for t in (0.50, 0.90, 0.97, 0.99)},
                **{f"cov80@{t:.2f}": float(np.mean([c["cov"][f"{t:.2f}"] >= 0.80 for c in sel]))
                   for t in (0.50, 0.90, 0.97, 0.99)},
            }
        out["blocks"][src] = block

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)

    for src, b in out["blocks"].items():
        print(f"\n=== {src}  (frames {n_frames.get(src, 0)}, GT curves {b['n_curves']}, "
              f"len eval-px p25/p50/p75 "
              f"{b['len_eval_px_p25_p50_p75'][0]:.0f}/{b['len_eval_px_p25_p50_p75'][1]:.0f}/"
              f"{b['len_eval_px_p25_p50_p75'][2]:.0f}) ===")
        q, fb = b["p_min_q"], b["p_min_frac_below"]
        print(f"  per-curve MINIMUM p: p05 {q['p05']:.3f} p10 {q['p10']:.3f} p25 {q['p25']:.3f} "
              f"p50 {q['p50']:.3f} | share below 0.5/0.9/0.97/0.99: "
              f"{fb['0.50']:.3f}/{fb['0.90']:.3f}/{fb['0.97']:.3f}/{fb['0.99']:.3f}")
        print(f"{'thr':>6} {'cov>=95%':>9} {'cov>=80%':>9} {'pixP':>6} {'pixR':>6} "
              f"{'pixDice':>8} {'fg%':>6}")
        for t in THRS:
            r = b["per_thr"][f"{t:.2f}"]
            print(f"{t:6.2f} {r['curves_cov95']:9.3f} {r['curves_cov80']:9.3f} "
                  f"{r['pixel_precision']:6.3f} {r['pixel_recall']:6.3f} {r['pixel_dice']:8.3f} "
                  f"{100 * r['fg_frac']:6.2f}")
        print(f"  by GT length (eval px)   {'n':>5} {'p_min':>6} {'p_mean':>7} "
              f"{'c95@.50':>8} {'c95@.97':>8} {'c95@.99':>8} {'c80@.97':>8}")
        for k, r in b["by_length"].items():
            print(f"  {k:>22} {r['n']:5d} {r['p_min_median']:6.3f} {r['p_mean_median']:7.3f} "
                  f"{r['cov95@0.50']:8.3f} {r['cov95@0.97']:8.3f} {r['cov95@0.99']:8.3f} "
                  f"{r['cov80@0.97']:8.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
