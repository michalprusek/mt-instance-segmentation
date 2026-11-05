#!/usr/bin/env python3
"""Headline of the model-only anatomy (`diag_model_only.py`) on the TEST dumps, write-up only.

For each block and seed: the pixel precision at the HIGHEST threshold that still carries >= 80 %
of GT curves over >= 95 % of their length (the "precision at matched coverage" statistic the
campaign uses), the per-curve minimum-probability median, and the share of curves whose
minimum is below 0.5.

    python scripts/step1_anat_read.py --dir runs/step1
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def matched(per_thr: dict, target: float = 0.80) -> tuple[float, dict] | None:
    rows = sorted(((float(t), r) for t, r in per_thr.items()), key=lambda x: x[0])
    keep = [(t, r) for t, r in rows if r["curves_cov95"] >= target]
    return keep[-1] if keep else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    acc: dict[str, list] = {}
    print(f"{'file':<26}{'block':<8}{'curves':>7}{'thr@cov80':>10}{'pixP':>7}{'pixR':>7}{'p_min p50':>10}{'<0.5':>7}{'<0.9':>7}")
    for path in sorted(glob.glob(os.path.join(args.dir, "anat_*.json"))):
        d = json.load(open(path))
        for blk, b in d["blocks"].items():
            m = matched(b["per_thr"])
            if m is None:
                print(f"{os.path.basename(path):<26}{blk:<8}{b['n_curves']:>7}   (no threshold reaches 80 % coverage)")
                continue
            thr, r = m
            q = b["p_min_q"]; fb = b["p_min_frac_below"]
            p50 = q.get("p50", q.get("0.5"))
            lt5 = fb.get("0.5", fb.get("0.50")); lt9 = fb.get("0.9", fb.get("0.90"))
            print(f"{os.path.basename(path):<26}{blk:<8}{b['n_curves']:>7}{thr:>10.2f}{r["pixel_precision"]:>7.3f}{r["pixel_recall"]:>7.3f}{p50:>10.3f}{lt5:>7.3f}{lt9:>7.3f}")
            acc.setdefault(blk, []).append((r["pixel_precision"], p50, lt5))
    print("--- mean over seeds")
    for blk, lst in acc.items():
        a = np.array(lst)
        print(f"{blk:<8} pixP@cov80 {a[:,0].mean():.3f} +- {a[:,0].std(ddof=1) if len(a)>1 else 0:.3f}   p_min p50 {a[:,1].mean():.3f}   share<0.5 {a[:,2].mean():.3f}   (n={len(a)})")


if __name__ == "__main__":
    main()
