#!/usr/bin/env python3
"""Read one ARM against the baseline on VAL, on the declared instrument, paired per frame.

Both models: 13-node grid sweeps of the ep040 native dump (`sweep_<tag>_t<thr>.json` shards or one
json), threshold chosen PER MODEL on the block's own VAL by pooled micro (ties to the lower node),
exactly step1_read.choose_thr. Reports, per block:
  - pooled micro / P / R / macro>=10 at each model's own threshold, the paired per-frame F1
    difference (frames with >= min_gt GT) with a sign count and a Wilcoxon p,
  - the arm's precision at MATCHED RECALL: the arm's grid node whose micro recall is closest to
    the baseline's recall at its own threshold (the "matched foreground" reading, grid-resolved),
  - the 100-frame block also per source (htw / alice / 586).
VAL only: it refuses a path containing 'TEST'.

    python scripts/read_arm.py --arm-dir runs/fov50read --arm FOV50 --base-dir runs/step1 --base SPARSE35
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_read import choose_thr, load, rows_by_thr, source_of, stats  # noqa: E402

BLOCKS = (("roi303", "roiVAL"), ("100-frame", "hbVAL"))


def paired(base_rows, arm_rows, min_gt):
    b = {f["name"]: f for f in base_rows if f["tp"] + f["fn"] >= min_gt}
    a = {f["name"]: f for f in arm_rows}
    d = np.array([a[n]["f1"] - b[n]["f1"] for n in b if n in a])
    if d.size == 0:
        return "no paired frames"
    try:
        from scipy.stats import wilcoxon
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    except Exception:
        p = float("nan")
    return (f"paired dF1 on {d.size} frames>= {min_gt} GT: mean {d.mean():+.4f} median {np.median(d):+.4f} "
            f"arm better {int((d > 0).sum())} / worse {int((d < 0).sum())} / tie {int((d == 0).sum())}  wilcoxon p={p:.3g}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-dir", required=True); ap.add_argument("--arm", required=True)
    ap.add_argument("--base-dir", required=True); ap.add_argument("--base", required=True)
    ap.add_argument("--params", default="der")
    ap.add_argument("--min-gt", type=int, default=10)
    args = ap.parse_args()
    for d in (args.arm_dir, args.base_dir):
        if "TEST" in d.upper():
            raise SystemExit("VAL only")
    for label, blk in BLOCKS:
        arm = load(f"{args.arm_dir}/sweep_{blk}_{args.arm}_{args.params}.json")
        base = load(f"{args.base_dir}/sweep_{blk}_{args.base}_{args.params}.json")
        if arm is None or base is None:
            print(f"== {label}: missing sweep ({'arm' if arm is None else 'base'})"); continue
        src_rule = "htw" if blk == "hbVAL" else None   # the declared rule: htw's threshold is chosen on htw VAL frames only
        print(f"== {label} VAL, {args.params}, threshold chosen per model on {'htw VAL frames' if src_rule else 'this VAL'} (pooled micro)"
              + ("; numbers below are htw-source only, then the whole block" if src_rule else ""))
        (tb, sb), (ta, sa) = choose_thr(base, src_rule, args.min_gt), choose_thr(arm, src_rule, args.min_gt)
        for name, t, s in ((args.base, tb, sb), (args.arm, ta, sa)):
            print(f"  {name:<10} thr {t:<6} micro {s['micro']:.4f}  P {s['micro_p']:.3f} R {s['micro_r']:.3f}  "
                  f"macro>=10 {s['macro_big']:.4f} (n={s['n_big']})  tp/fp/fn {s['tp']}/{s['fp']}/{s['fn']}")
        print(f"  d micro {sa['micro'] - sb['micro']:+.4f}   " + paired(rows_by_thr(base, src_rule)[tb], rows_by_thr(arm, src_rule)[ta], args.min_gt))
        # matched recall: among the arm's grid nodes whose recall is not below the baseline's by more than
        # 0.005, the HIGHEST precision (a flat recall curve makes "closest recall" pick a low-precision node
        # by a 0.001 accident -- ARM 52 read, 2026-09-17; the rule change is applied to every arm alike).
        grid = {t: stats(r, args.min_gt) for t, r in rows_by_thr(arm, src_rule).items()}
        cand = [t for t in grid if grid[t]["micro_r"] >= sb["micro_r"] - 0.005]
        tm = max(cand, key=lambda t: grid[t]["micro_p"]) if cand else min(grid, key=lambda t: abs(grid[t]["micro_r"] - sb["micro_r"]))
        print(f"  {args.arm} at matched recall (best-P node with R >= base-0.005: node {tm}, R {grid[tm]['micro_r']:.3f} vs base R {sb['micro_r']:.3f}): "
              f"P {grid[tm]['micro_p']:.3f} vs base P {sb['micro_p']:.3f}  (dP {grid[tm]['micro_p'] - sb['micro_p']:+.3f})")
        if blk == "hbVAL":
            s1, s2 = stats(rows_by_thr(base, None)[tb], args.min_gt), stats(rows_by_thr(arm, None)[ta], args.min_gt)
            print(f"    whole block (49 frames) at those thresholds: base micro {s1['micro']:.4f}  arm micro {s2['micro']:.4f}  d {s2['micro'] - s1['micro']:+.4f}")
            for src in ("alice", "586"):
                rb = [f for f in rows_by_thr(base, None)[tb] if source_of(f["name"]) == src]
                ra = [f for f in rows_by_thr(arm, None)[ta] if source_of(f["name"]) == src]
                s1, s2 = stats(rb, args.min_gt), stats(ra, args.min_gt)
                print(f"    {src:<6} base micro {s1['micro']:.4f} (P {s1['micro_p']:.3f} R {s1['micro_r']:.3f})   "
                      f"arm micro {s2['micro']:.4f} (P {s2['micro_p']:.3f} R {s2['micro_r']:.3f})   d {s2['micro'] - s1['micro']:+.4f}")
        print()


if __name__ == "__main__":
    main()
