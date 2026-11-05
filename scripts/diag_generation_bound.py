#!/usr/bin/env python3
"""The generation bound: could ANY curve-level candidate step complete the curves we lose?

The 13-agent review ranked "curve-level accept on cached maps" as the last route with a non-zero
expected gain and ZERO GPU cost, and set its own first gate: of the addressable GT curves, how many
would acquire a completing candidate at all. Under half and the direction dies for free.

This measures the bound WITHOUT building the candidate generator, by asking the question the
generator would have to answer. For each GT curve:
  * sample the thresholded mask along it -> a per-vertex covered/not profile;
  * treat every uncovered RUN as a gap the candidate step would have to bridge;
  * a gap is BRIDGEABLE if it is at most `--max-gap` px long (a curvature-bounded path can span it;
    the GT itself proves a physically legal path exists, so this is an upper bound by construction);
  * report the coverage the curve would reach if every bridgeable gap were filled, and how many
    curves cross the 0.80 the headline metric needs.

Because gaps are bridged with the GT's own geometry, every number here is an UPPER BOUND on what a
real candidate generator plus a perfect accept step could reach. It cannot be beaten, only missed.

    PYTHONPATH=src:scripts python scripts/diag_generation_bound.py --data data/real/roi303_eval \
        --pred-dir <dump> --thr 0.96 --max-gap 40 --out bound.json
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
for sub in ("src", "scripts"):
    sys.path.insert(0, os.path.join(ROOT, sub))

from instance.geometry import resample  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402

UP = 1.5


def bridged_coverage(cov, max_gap):
    """Coverage after filling every uncovered run no longer than `max_gap`."""
    out = cov.copy()
    i, n = 0, len(cov)
    while i < n:
        if out[i]:
            i += 1
            continue
        j = i
        while j < n and not out[j]:
            j += 1
        # an interior gap is bridgeable; a gap at either END is not (nothing to link to)
        if j - i <= max_gap and i > 0 and j < n:
            out[i:j] = True
        i = j
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--thr", type=float, default=0.96)
    ap.add_argument("--max-gap", type=float, default=40.0, help="eval px a candidate may span")
    ap.add_argument("--need", type=float, default=0.80, help="the headline metric's coverage")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw, brid, lens = [], [], []
    frames = 0
    for p in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        stem = os.path.basename(p)[:-3]
        npz = os.path.join(args.pred_dir, stem + ".npz")
        if not os.path.exists(npz):
            continue
        fr = read_frame_h5(p)
        if str(fr["attrs"].get("split")) != args.split:
            continue
        prob = np.load(npz)["prob"].astype(np.float32)[0]
        frames += 1
        for q in fr["polylines"]:
            pts = resample(np.asarray(q, float) * UP, ds=1.0)
            if len(pts) < 10:
                continue
            v = map_coordinates(prob, [pts[:, 1], pts[:, 0]], order=1, mode="nearest")
            cov = v >= args.thr
            raw.append(float(cov.mean()))
            brid.append(float(bridged_coverage(cov, args.max_gap).mean()))
            lens.append(len(pts))
        print(f"{stem}", flush=True)

    R, B = np.array(raw), np.array(brid)
    summ = {"frames": frames, "curves": len(R), "thr": args.thr, "max_gap": args.max_gap,
            "need": args.need,
            "share_reaching_need_RAW": round(float((R >= args.need).mean()), 4),
            "share_reaching_need_BRIDGED": round(float((B >= args.need).mean()), 4),
            "share_reaching_0.95_RAW": round(float((R >= 0.95).mean()), 4),
            "share_reaching_0.95_BRIDGED": round(float((B >= 0.95).mean()), 4),
            "median_coverage_RAW": round(float(np.median(R)), 4),
            "median_coverage_BRIDGED": round(float(np.median(B)), 4)}
    # the gate the review declared: of curves that FAIL the need today, how many are rescued?
    fail = R < args.need
    summ["curves_failing_today"] = int(fail.sum())
    summ["of_those_rescued_by_bridging"] = round(float((B[fail] >= args.need).mean()), 4) if fail.any() else None
    summ["GATE"] = ("PASS (>= 0.50 rescued -- a curve-level accept step has something to accept)"
                    if fail.any() and (B[fail] >= args.need).mean() >= 0.5 else
                    "FAIL (< 0.50 rescued -- the direction dies here, for free)")
    json.dump(summ, open(args.out, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
