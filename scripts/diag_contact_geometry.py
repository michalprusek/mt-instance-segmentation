#!/usr/bin/env python3
"""How much TRUE close-parallel structure does each block contain? Model-free, GT only.

Screen A (2026-09-13) attributed 50.3 % of in-domain breaks to a "close parallel" and 40.9 % to a
crossing, and that split is load-bearing for the overlap-permitting emission. But
`diag_failure_geometry.py:164-172` defines the parallel class by DISTANCE ALONE -- "this GT within
[3, 9] eval px of ANOTHER GT" -- with no tangent-angle gate. Two filaments converging into a
crossing are within 3-9 px for a stretch on either side of the junction, so a crossing approach
counts as "parallel" wherever `count_crossings` did not flag the junction itself. The parallel class
is therefore an UPPER BOUND, and an adversarial review of a related design re-measured exactly this
confound and found most of a comparable pool to be crossing contact.

This script measures the contact geometry directly, with the angle gate, on ground truth only -- no
model, no threshold, no instancer, so nothing here depends on a checkpoint:

  for every pair of GT curves that come within `--hi` eval px of each other, walk the contact and
  record, per sample, the gap and the ABSOLUTE ANGLE between the two tangents (mod 180 deg, so
  antiparallel counts as parallel -- the filaments have no polarity for this purpose).

Contact is then classified by that angle:
  PARALLEL   < `--par-deg` (default 20 deg)  -- the pair runs side by side; one pixel band, two
                                               filaments; this is the project's bottleneck #1 and
                                               the population an overlap-permitting emission
                                               converts.
  OBLIQUE    in between                      -- ambiguous, reported separately, never merged into
                                               either headline number.
  CROSSING   > `--cross-deg` (default 50 deg) -- the pair is converging or diverging; this is a
                                               junction approach, not a parallel.

Reported per block, in ARC LENGTH (eval px) rather than sample counts, because a long parallel run
and a one-sample grazing contact are not the same population.

    PYTHONPATH=src python scripts/diag_contact_geometry.py --data data/real/roi303_eval \\
        --split val --out contact_roi303.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from mt_bench.cvat_import import read_frame_h5  # noqa: E402

UP = 1.5   # the eval frame, the convention every number in this project is quoted in


def source_of(name: str) -> str:
    if name.startswith("htw_"):
        return "htw"
    if name.startswith("alice_"):
        return "alice"
    if name.startswith("training_img_"):
        return "roi303-derived"
    if name.startswith("roi_img_"):
        return "roi303"
    return "synth"


def densify(poly: np.ndarray, step: float = 1.0) -> np.ndarray:
    d = np.hypot(*np.diff(poly, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 0:
        return poly[:1]
    t = np.arange(0.0, s[-1] + 1e-9, step)
    return np.stack([np.interp(t, s, poly[:, 0]), np.interp(t, s, poly[:, 1])], axis=1)


def tangents(pts: np.ndarray, win: int = 9) -> np.ndarray:
    """Tangent angle in degrees mod 180, smoothed over `win` samples.

    The window matters: a vertex-level tangent on a coarse human polyline is quantisation noise --
    the same reason kappa_max is measured at an 8 px baseline and not at 2 px.
    """
    n = len(pts)
    h = max(1, win // 2)
    lo = np.clip(np.arange(n) - h, 0, n - 1)
    hi = np.clip(np.arange(n) + h, 0, n - 1)
    d = pts[hi] - pts[lo]
    return np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180.0


def ang_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Absolute difference of two mod-180 angles, in [0, 90]."""
    d = np.abs(a - b) % 180.0
    return np.minimum(d, 180.0 - d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--lo", type=float, default=2.0, help="eval px; below this the two curves are "
                    "annotation duplicates, not resolvable filaments (roi303 carries 15 exact "
                    "duplicate polylines and 32 near-coincident pairs)")
    ap.add_argument("--hi", type=float, default=9.0, help="eval px; the parallel window used by "
                    "diag_failure_geometry (3-9 eval px)")
    ap.add_argument("--par-deg", type=float, default=20.0)
    ap.add_argument("--cross-deg", type=float, default=50.0)
    ap.add_argument("--tan-win", type=int, default=9)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    acc: dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        name = os.path.basename(path)
        fr = read_frame_h5(path)
        if args.split != "all" and str(fr["attrs"].get("split", "")) != args.split:
            continue
        src = source_of(name)
        a = acc.setdefault(src, {"frames": 0, "curves": 0, "gt_len": 0.0,
                                 "par": 0.0, "obl": 0.0, "cross": 0.0, "dup": 0.0,
                                 "par_runs": [], "gap_par": []})
        a["frames"] += 1

        curves = [densify(np.asarray(p, float) * UP) for p in fr["polylines"] if len(p) >= 2]
        curves = [c for c in curves if len(c) >= 4]
        a["curves"] += len(curves)
        a["gt_len"] += float(sum(len(c) for c in curves))       # 1 px sampling => length in px
        if len(curves) < 2:
            continue
        tans = [tangents(c, args.tan_win) for c in curves]
        trees = [cKDTree(c) for c in curves]

        for i, ci in enumerate(curves):
            # nearest OTHER curve per sample, and the tangent of that other curve at the hit
            best_d = np.full(len(ci), np.inf)
            best_a = np.zeros(len(ci))
            for j, cj in enumerate(curves):
                if j == i:
                    continue
                d, idx = trees[j].query(ci, k=1)
                m = d < best_d
                best_d[m] = d[m]
                best_a[m] = tans[j][idx[m]]
            dtheta = ang_diff(tans[i], best_a)

            dup = best_d < args.lo
            inband = (best_d >= args.lo) & (best_d <= args.hi)
            a["dup"] += float(dup.sum())
            par = inband & (dtheta < args.par_deg)
            cross = inband & (dtheta > args.cross_deg)
            obl = inband & ~par & ~cross
            a["par"] += float(par.sum())
            a["obl"] += float(obl.sum())
            a["cross"] += float(cross.sum())
            a["gap_par"].extend(best_d[par].tolist())
            # contiguous parallel runs, so a grazing single sample is not a bundle
            if par.any():
                idx = np.flatnonzero(par)
                for run in np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1):
                    if len(run) >= 5:
                        a["par_runs"].append(int(len(run)))

    out = {"data": args.data, "split": args.split, "band_eval_px": [args.lo, args.hi],
           "par_deg": args.par_deg, "cross_deg": args.cross_deg, "blocks": {}}
    for src, a in acc.items():
        tot = a["par"] + a["obl"] + a["cross"]
        runs = np.array(a["par_runs"], float)
        out["blocks"][src] = {
            "frames": a["frames"], "curves": a["curves"],
            "gt_len_eval_px": a["gt_len"],
            "contact_len_eval_px": tot,
            "contact_share_of_gt_len": tot / max(a["gt_len"], 1),
            "of_contact_parallel": a["par"] / max(tot, 1),
            "of_contact_oblique": a["obl"] / max(tot, 1),
            "of_contact_crossing": a["cross"] / max(tot, 1),
            "parallel_len_per_frame": a["par"] / max(a["frames"], 1),
            "parallel_share_of_gt_len": a["par"] / max(a["gt_len"], 1),
            "duplicate_len_below_lo": a["dup"],
            "n_parallel_runs_ge5px": int(runs.size),
            "parallel_runs_per_frame": float(runs.size) / max(a["frames"], 1),
            "parallel_run_len_p50_p90": ([float(np.percentile(runs, 50)),
                                          float(np.percentile(runs, 90))] if runs.size else None),
            "parallel_gap_p50": (float(np.median(a["gap_par"])) if a["gap_par"] else None),
        }

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)

    print(f"contact band {args.lo}-{args.hi} eval px | PARALLEL < {args.par_deg} deg, "
          f"CROSSING > {args.cross_deg} deg (tangent window {args.tan_win} px)")
    print(f"{'block':<16} {'frm':>4} {'curves':>7} {'contact%GT':>11} {'par':>6} {'obl':>6} "
          f"{'cross':>6} {'par px/frm':>11} {'runs>=5px/frm':>14} {'run p50/p90':>12}")
    for src, b in out["blocks"].items():
        rl = b["parallel_run_len_p50_p90"]
        rs = f"{rl[0]:.0f}/{rl[1]:.0f}" if rl else "-"
        print(f"{src:<16} {b['frames']:4d} {b['curves']:7d} "
              f"{100*b['contact_share_of_gt_len']:10.2f}% {b['of_contact_parallel']:6.3f} "
              f"{b['of_contact_oblique']:6.3f} {b['of_contact_crossing']:6.3f} "
              f"{b['parallel_len_per_frame']:11.0f} {b['parallel_runs_per_frame']:14.1f} {rs:>12}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
