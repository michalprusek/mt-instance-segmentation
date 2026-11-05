#!/usr/bin/env python3
"""Is the evidence for the missing filament IN the map, or absent from it?

Everything post-hoc is now closed: local mask repair at crossings (read 37, ceiling +0.013 with a
PERFECT repair), consuming the multiplicity map in the arc graph (+0.013 with a perfect map), the
overlap-conditioned fill (loses to a foreground-matched threshold drop at every setting), the
connectivity-loss family (clDice degenerates to Dice on a thin target, structurally), and the
assembler (on real data it resolves 0.90-0.95 wherever the mask carries the filament through).

That leaves one question, and it decides whether the remaining route is CALIBRATION or EVIDENCE:
along the GT centerline of the stretches the model loses, what probability does it actually
assign? If those pixels sit at p = 0.3-0.9, the signal is present and mis-ranked, and a better
operating point or calibration can reach it -- bounded by the +0.048 a per-frame ORACLE threshold
is worth. If they sit near 0, the evidence is not in the map at all and no post-hoc method can
recover it; only a better model or a better generator can.

Measures, on cached maps, per GT polyline:
  * the per-vertex probability profile along the GT centerline (sampled bilinearly)
  * the LOW STRETCHES: maximal runs where p < thr, their length and their minimum and median p
  * split by whether that GT curve is matched by the instancer at all
so "how much headroom is there below the threshold" is answered with a distribution, not a claim.

    PYTHONPATH=src:scripts python scripts/diag_evidence_headroom.py --data data/real/roi303_eval \
        --pred-dir <dump> --thr 0.97 --out headroom.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import map_coordinates

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from instance.geometry import resample  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402

UP = 1.5
COV_THRS = (0.5, 0.9, 0.97, 0.99)


def runs_below(v, thr):
    """(start, length, min, median) of every maximal run of v below thr."""
    out, i, n = [], 0, len(v)
    while i < n:
        if v[i] >= thr:
            i += 1
            continue
        j = i
        while j < n and v[j] < thr:
            j += 1
        seg = v[i:j]
        out.append((i, j - i, float(seg.min()), float(np.median(seg))))
        i = j
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--thr", type=float, default=0.97)
    ap.add_argument("--split", default="val")
    ap.add_argument("--min-run", type=int, default=5, help="eval px; shorter dips are noise")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prof_all, runs, cov = [], [], []
    n_curves = frames = 0
    for p in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
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
        frames += 1
        for q in gt:
            pts = resample(q, ds=1.0)
            if len(pts) < 10:
                continue
            v = map_coordinates(prob, [pts[:, 1], pts[:, 0]], order=1, mode="nearest")
            n_curves += 1
            prof_all.append(v.astype(np.float16))
            # PER-CURVE coverage: the strict metric needs >= 95 % of ONE curve's length carried by
            # ONE instance, so aggregate pixel coverage says nothing. This is the recall ceiling
            # the MAP allows at this threshold, before any assembly.
            cov.append([float((v >= t).mean()) for t in COV_THRS])
            for st, L, mn, md in runs_below(v, args.thr):
                if L >= args.min_run:
                    runs.append((L, mn, md, len(v)))
        print(f"{stem}: {len(gt)} curves", flush=True)

    C = np.array(cov) if cov else np.zeros((0, len(COV_THRS)))
    v = np.concatenate([x.astype(np.float32) for x in prof_all]) if prof_all else np.zeros(1)
    R = np.array(runs, dtype=float) if runs else np.zeros((0, 4))
    q = [5, 10, 25, 50, 75, 90, 95]
    summ = {
        "frames": frames, "curves": n_curves, "thr": args.thr, "min_run": args.min_run,
        "gt_pixels": int(v.size),
        "share_of_GT_below_thr": round(float((v < args.thr).mean()), 4),
        "p_on_GT_quantiles": {str(k): round(float(np.percentile(v, k)), 4) for k in q},
        "n_low_runs": int(len(R)),
    }
    if len(R):
        summ["run_length_quantiles"] = {str(k): round(float(np.percentile(R[:, 0], k)), 1) for k in q}
        summ["run_min_p_quantiles"] = {str(k): round(float(np.percentile(R[:, 1], k)), 4) for k in q}
        summ["run_median_p_quantiles"] = {str(k): round(float(np.percentile(R[:, 2], k)), 4) for k in q}
        # THE decisive split: of the stretches the model loses, how many are merely mis-ranked?
        for lo in (0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9):
            summ[f"share_of_low_runs_with_min_p_above_{lo}"] = round(float((R[:, 1] > lo).mean()), 4)
        summ["low_run_px_total"] = int(R[:, 0].sum())
        summ["low_run_px_share_of_GT"] = round(float(R[:, 0].sum() / max(v.size, 1)), 4)
    if len(C):
        # the map's own recall ceiling: curves whose length is covered at each level
        summ["map_recall_ceiling"] = {
            f"thr{t:g}": {f"cov>={c:g}": round(float((C[:, i] >= c).mean()), 4)
                          for c in (0.95, 0.90, 0.80, 0.50)}
            for i, t in enumerate(COV_THRS)}
        summ["curve_coverage_quantiles"] = {
            f"thr{t:g}": {str(k): round(float(np.percentile(C[:, i], k)), 4) for k in (5, 25, 50, 75, 95)}
            for i, t in enumerate(COV_THRS)}
    json.dump(summ, open(args.out, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
