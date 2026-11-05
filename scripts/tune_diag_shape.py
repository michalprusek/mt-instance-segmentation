#!/usr/bin/env python3
"""Is the instancer objective smooth, or flat-with-cliffs? RUNS ANYWHERE.

The 95 %-length / 95 %-precision coverage rule makes a single frame's strict F1 a function of
two integers (matched microtubules, predicted polylines), so it can only take a handful of
values however finely a knob is swept. Averaging many frames hides that. This reports both
levels from a dense `tune_diag_oat.py` sweep:

* per frame -- the number of DISTINCT F1 values the whole dense sweep produces, and the largest
  plateau (consecutive grid points with an identical score);
* on the split mean -- distinct values, largest single-step jump between adjacent grid points,
  and the number of local maxima, which is what a gradient-free optimiser has to climb.

    python scripts/tune_diag_shape.py --oat tune_diag_out/oat_dense.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def longest_plateau(x: np.ndarray) -> int:
    best = run = 1
    for i in range(1, len(x)):
        run = run + 1 if abs(x[i] - x[i - 1]) < 1e-12 else 1
        best = max(best, run)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oat", required=True)
    a = ap.parse_args()
    with open(a.oat) as fh:
        d = json.load(fh)
    R = d["results"]
    print(f"BASE mean_f1 = {d['base_f1']:.4f}, {d['n_frames']} frames\n")
    hdr = (f"{'knob':>12} {'pts':>4} {'distinct(mean)':>14} {'max step':>9} {'local max':>9} "
           f"{'plateau(mean)':>13} | per-frame distinct: {'min':>3} {'med':>4} {'max':>3} "
           f"{'  plateau med':>12}")
    print(hdr)
    print("-" * len(hdr))
    for k, vals in d["grids"].items():
        f1 = np.array([R[f"{k}#{i}"]["mean_f1"] for i in range(len(vals))])
        per = np.array([R[f"{k}#{i}"]["f1"] for i in range(len(vals))])
        dstep = np.abs(np.diff(f1))
        nloc = int(sum(1 for i in range(1, len(f1) - 1)
                       if f1[i] >= f1[i - 1] and f1[i] > f1[i + 1]))
        dfr = [len(set(np.round(per[:, j], 9))) for j in range(per.shape[1])]
        plfr = [longest_plateau(per[:, j]) for j in range(per.shape[1])]
        print(f"{k:>12} {len(vals):>4} {len(set(np.round(f1, 9))):>14} {dstep.max():9.4f} "
              f"{nloc:>9} {longest_plateau(f1):>13} | {'':19} {min(dfr):>3} "
              f"{int(np.median(dfr)):>4} {max(dfr):>3} {int(np.median(plfr)):>12}")

    print("\nper-frame step ladders (dense sweep, first widest knob):")
    k = list(d["grids"])[0]
    vals = d["grids"][k]
    per = np.array([R[f"{k}#{i}"]["f1"] for i in range(len(vals))])
    names = R[f"{k}#0"]["frames"]
    for j in range(min(8, per.shape[1])):
        lev = sorted(set(np.round(per[:, j], 4)))
        print(f"  {names[j]:>16} {len(lev):>2} levels: "
              + " ".join(f"{v:.3f}" for v in lev[:10]))


if __name__ == "__main__":
    main()
