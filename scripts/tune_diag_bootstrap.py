#!/usr/bin/env python3
"""How many frames must a trial score before it can rank two parameter sets? RUNS ANYWHERE.

Consumes the per-frame rows `tune_diag_oat.py` already wrote, so this costs no compute. For a
pair of configurations whose full-split mean-F1 difference is delta, it resamples n frames with
replacement (paired -- the same frames for both configurations, which is how a tuner sees them)
and reports the fraction of resamples in which the sign of the observed difference agrees with
the full-split sign. "Stable enough to rank" is defined as that fraction reaching 0.95.

Every frame in these splits has n_gt > 0, so `aggregate_benchmark`'s macro mean_f1 is exactly
the mean of the per-frame f1 column; the resampling therefore reproduces the tuning objective
rather than a proxy for it.

    python scripts/tune_diag_bootstrap.py --oat tune_diag_out/oat21.json
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np

BINS = [(0.005, 0.015), (0.015, 0.030), (0.030, 0.060), (0.060, 0.200)]
NS = [5, 8, 10, 12, 15, 17, 20, 25, 30, 40, 50, 75, 100]


def sign_agreement(a: np.ndarray, b: np.ndarray, n: int, B: int, rng) -> float:
    idx = rng.integers(0, len(a), size=(B, n))
    d = (a[idx].mean(axis=1) - b[idx].mean(axis=1))
    truth = np.sign(a.mean() - b.mean())
    return float(np.mean(np.sign(d) == truth))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oat", required=True)
    ap.add_argument("--B", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260816)
    a = ap.parse_args()
    with open(a.oat) as fh:
        d = json.load(fh)
    rng = np.random.default_rng(a.seed)

    cfgs = {k: np.array(v["f1"], float) for k, v in d["results"].items()}
    means = {k: float(v.mean()) for k, v in cfgs.items()}
    keys = sorted(cfgs)
    print(f"{len(keys)} configurations, {d['n_frames']} frames, "
          f"macro mean_f1 of BASE = {means['BASE']:.4f}\n")

    pairs = list(itertools.combinations(keys, 2))
    rng.shuffle(pairs)
    print(f"{'|delta| bin':>16} {'pairs':>6} " + " ".join(f"{n:>6}" for n in NS))
    for lo, hi in BINS:
        sel = [p for p in pairs if lo <= abs(means[p[0]] - means[p[1]]) < hi][:60]
        if not sel:
            continue
        row = []
        for n in NS:
            row.append(np.mean([sign_agreement(cfgs[x], cfgs[y], n, a.B, rng)
                                for x, y in sel]))
        print(f"{f'[{lo:.3f},{hi:.3f})':>16} {len(sel):>6} "
              + " ".join(f"{v:6.3f}" for v in row))

    print("\nper-frame F1 spread at BASE: sd = "
          f"{cfgs['BASE'].std(ddof=1):.4f}, range "
          f"{cfgs['BASE'].min():.3f}-{cfgs['BASE'].max():.3f}")
    print("smallest n reaching 0.95 sign agreement, per bin:")
    for lo, hi in BINS:
        sel = [p for p in pairs if lo <= abs(means[p[0]] - means[p[1]]) < hi][:60]
        if not sel:
            continue
        got = "not reached by n=100"
        for n in NS:
            v = np.mean([sign_agreement(cfgs[x], cfgs[y], n, a.B, rng) for x, y in sel])
            if v >= 0.95:
                got = f"n = {n}"
                break
        print(f"  |delta| in [{lo:.3f},{hi:.3f}): {got}")


if __name__ == "__main__":
    main()
