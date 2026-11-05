#!/usr/bin/env python3
"""Turn a `tune_diag_oat.py` result file into the sensitivity + cliff tables. RUNS ANYWHERE.

Two questions, one pass over the stored per-frame rows:

* SENSITIVITY -- how far does each knob move the objective across the range the tuner searches?
  Reported as the spread of the swept mean-F1, together with the loss the shipped value leaves
  on the table (max - baseline) and the damage the worst setting does (baseline - min).
* SHAPE -- how many DISTINCT F1 values does the sweep take? Counted on the 25-frame mean and,
  more diagnostically, per frame: a 95 %-coverage rule makes a single frame's score a step
  function of a few integers, and averaging 25 staircases hides that.

    python scripts/tune_diag_report.py --oat tune_diag_out/oat21.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oat", required=True)
    ap.add_argument("--csv", default="")
    a = ap.parse_args()
    with open(a.oat) as fh:
        d = json.load(fh)
    R, base = d["results"], d["base_f1"]
    print(f"BASE mean_f1 = {base:.4f} over {d['n_frames']} frames "
          f"(prob_thr held at {d['baseline']['prob_thr']:.4f})\n")

    rows = []
    for k, vals in d["grids"].items():
        f1 = np.array([R[f"{k}#{i}"]["mean_f1"] for i in range(len(vals))])
        per = np.array([R[f"{k}#{i}"]["f1"] for i in range(len(vals))])   # (n_vals, n_frames)
        tp = np.array([sum(R[f"{k}#{i}"]["tp"]) for i in range(len(vals))])
        dist_frame = [len(set(np.round(per[:, j], 9))) for j in range(per.shape[1])]
        best = int(np.argmax(f1))
        rows.append(dict(knob=k, lo=vals[0], hi=vals[-1], n=len(vals),
                         shipped=d["baseline"].get(k), f1min=f1.min(), f1max=f1.max(),
                         spread=f1.max() - f1.min(), gain=f1.max() - base,
                         drop=base - f1.min(), argmax=vals[best], tp_at_max=int(tp[best]),
                         tp_base=int(tp[int(np.argmin(np.abs(np.array(vals, float)
                                                             - float(d["baseline"][k]))))]),
                         d_mean=len(set(np.round(f1, 9))),
                         d_frame_med=float(np.median(dist_frame)),
                         d_frame_max=int(max(dist_frame))))
    rows.sort(key=lambda r: -r["spread"])

    hdr = (f"{'knob':>15} {'range swept':>16} {'n':>3} {'shipped':>9} {'F1 min':>7} "
           f"{'F1 max':>7} {'spread':>7} {'gain':>7} {'drop':>7} {'argmax':>9} "
           f"{'dist(mean)':>10} {'dist/frame med':>14} {'max':>4}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        sh = "-" if r["shipped"] is None else f"{r['shipped']:.4g}"
        rng = "%.4g-%.4g" % (r["lo"], r["hi"])
        print(f"{r['knob']:>15} {rng:>16} {r['n']:>3} {sh:>9} "
              f"{r['f1min']:7.4f} {r['f1max']:7.4f} {r['spread']:7.4f} {r['gain']:+7.4f} "
              f"{r['drop']:+7.4f} {r['argmax']:9.4g} {r['d_mean']:>10} "
              f"{r['d_frame_med']:14.1f} {r['d_frame_max']:4d}")

    print("\nfull sweep curves for the three widest-spread knobs:")
    for r in rows[:3]:
        k = r["knob"]
        vals = d["grids"][k]
        print(f"\n  {k}  (shipped {r['shipped']:.4g}, base {base:.4f})")
        for i, v in enumerate(vals):
            mark = "  <- shipped" if abs(float(v) - float(r["shipped"])) < 1e-9 else ""
            print(f"    {float(v):9.4g}  {R[f'{k}#{i}']['mean_f1']:.4f}  "
                  f"tp={sum(R[f'{k}#{i}']['tp']):4d} fp={sum(R[f'{k}#{i}']['fp']):4d}{mark}")

    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("\nwrote", a.csv)


if __name__ == "__main__":
    main()
