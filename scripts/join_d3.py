#!/usr/bin/env python3
"""Join two D3 sweeps (synth VAL, real VAL) by setting name and test for CONSISTENCY.

79 settings were scored on each split. With that many comparisons an uncorrected p<0.05 on
one split means nothing; the useful filter is whether a setting is a top performer on BOTH
VALs at once, which is what this prints.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from instance.metrics import paired_bootstrap  # noqa: E402


def load(path, baseline):
    rep = json.load(open(path))["settings"]
    base = rep[baseline]
    out = {}
    for name, e in rep.items():
        d = {"f1": e["pooled"]["mean_f1"], "gate": e["gate"],
             "fg": 100 * e["props"]["fg"], "rec2": e["props"]["rec2"],
             "by_task": {t: v["mean_f1"] for t, v in e["by_task"].items()}}
        if name != baseline:
            bs = paired_bootstrap(e["frames"], base["frames"], stat="mean_f1", n_boot=5000,
                                  stratum_key="task", frame_key="name", seed=0)
            d.update(diff=bs["diff"], lo=bs["lo"], hi=bs["hi"], p=bs["p_two_sided"])
        else:
            d.update(diff=0.0, lo=0.0, hi=0.0, p=1.0)
        out[name] = d
    return out, base["pooled"]["mean_f1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", required=True)
    ap.add_argument("--real", required=True)
    ap.add_argument("--baseline", default="flat/0.4418")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    S, sb = load(args.synth, args.baseline)
    R, rb = load(args.real, args.baseline)
    common = [k for k in S if k in R]
    print(f"{len(common)} settings on both splits; baseline {args.baseline}: "
          f"synth {sb:.3f}  real {rb:.3f}")

    sv = np.array([S[k]["f1"] for k in common])
    rv = np.array([R[k]["f1"] for k in common])
    rho, pv = spearmanr(sv, rv)
    print(f"spearman(synth F1, real F1) across settings = {rho:+.3f} (p={pv:.3g})")

    ks = sorted(common, key=lambda k: -S[k]["f1"])
    kr = sorted(common, key=lambda k: -R[k]["f1"])
    rank_s = {k: i + 1 for i, k in enumerate(ks)}
    rank_r = {k: i + 1 for i, k in enumerate(kr)}
    both = [k for k in common if rank_s[k] <= 10 and rank_r[k] <= 10]
    print(f"settings in the top 10 of BOTH splits: {both if both else 'NONE'}")

    print(f"\n{'setting':28s} {'synthF1':>7s} {'dF1':>7s} {'p':>5s} {'rk':>3s} | "
          f"{'realF1':>6s} {'585':>6s} {'586':>6s} {'dF1':>7s} {'p':>5s} {'rk':>3s} | gates")
    order = sorted(common, key=lambda k: -(S[k]["f1"] / max(sb, 1e-9)
                                           + R[k]["f1"] / max(rb, 1e-9)))
    for k in order[:args.top] + [args.baseline]:
        s, r = S[k], R[k]
        print(f"{k:28s} {s['f1']:7.3f} {s['diff']:+7.3f} {s['p']:5.3f} {rank_s[k]:3d} | "
              f"{r['f1']:6.3f} {r['by_task'].get('585', float('nan')):6.3f} "
              f"{r['by_task'].get('586', float('nan')):6.3f} "
              f"{r['diff']:+7.3f} {r['p']:5.3f} {rank_r[k]:3d} | "
              f"synth {'PASS' if s['gate'] else 'FAIL'} real {'PASS' if r['gate'] else 'FAIL'}")


if __name__ == "__main__":
    main()
