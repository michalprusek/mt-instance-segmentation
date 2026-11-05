#!/usr/bin/env python3
"""Repair two measured defects of the ROI-303 block IN PLACE, VAL and TEST alike, and log every change.

Declared 2026-09-04 (docs/TODO.md §0.0-CAMPAIGN) before any arm was scored on the repaired block.

1. **Frames whose image carries no signal.** Five frames have an empty or saturated image
   (`roi_img_8`: black, 42 curves; `roi_img_149`; TEST `roi_img_139` 27 curves, `roi_img_143`,
   `roi_img_148`): annotation on nothing, a guaranteed false negative for every method. Gate: the
   [1, 99] percentile span of the stored image must exceed ``--min-span`` grey levels AND at least
   half the pixels must differ from the frame minimum. Failing frames are MOVED to
   ``<data>/_excluded/`` (not deleted) with the reason in their attrs.

2. **Duplicate polylines.** The scale/gap audit found 15 exact duplicates and a further set of
   near-coincident pairs (>= 40 px of one curve within 1.5 px of another). Rule, purely geometric:
   polyline B is dropped iff at least ``--cov`` (0.90) of B's 1-px samples lie within ``--tol``
   (1.5 px) of polyline A and B is not longer than A. A pair that coincides only along part of
   B is kept -- two filaments drawn over each other for a stretch is a bundle, not a duplicate.

The split attribute is preserved; nothing is re-randomised. The frame list in
``docs/roi303_split.json`` stays the declaration -- excluded frames are simply absent from the
directory afterwards, and the exclusion list is written next to it.

    python scripts/repair_roi_dataset.py --data data/real/roi303_eval [--dry-run]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

import h5py
import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from instance.geometry import arclength, resample  # noqa: E402


def image_ok(img: np.ndarray, min_span: float) -> tuple[bool, str]:
    lo, hi = np.percentile(img, [1, 99])
    frac_var = float(np.mean(img > img.min()))
    if hi - lo < min_span:
        return False, f"p1-p99 span {hi - lo:.1f} < {min_span}"
    if frac_var < 0.5:
        return False, f"only {100 * frac_var:.1f} % of pixels above the frame minimum"
    return True, ""


def dedupe(polys: list[np.ndarray], tol: float, cov: float) -> tuple[list[np.ndarray], list[dict]]:
    res = [resample(np.asarray(q, float), ds=1.0) for q in polys]
    lens = [float(arclength(r)[-1]) if len(r) > 1 else 0.0 for r in res]
    trees = [cKDTree(r) if len(r) > 1 else None for r in res]
    order = np.argsort(-np.asarray(lens))                 # longest first: it is the keeper
    drop, log = set(), []
    for bi in order:
        if trees[bi] is None or bi in drop:
            continue
        for ai in order:
            if ai == bi or ai in drop or trees[ai] is None or lens[ai] < lens[bi]:
                continue
            d, _ = trees[ai].query(res[bi], k=1)
            c = float(np.mean(d <= tol))
            if c >= cov:
                drop.add(int(bi))
                log.append({"dropped": int(bi), "kept": int(ai), "coverage": round(c, 3),
                            "len_dropped": round(lens[bi], 1), "len_kept": round(lens[ai], 1),
                            "exact": bool(len(polys[ai]) == len(polys[bi])
                                          and np.allclose(polys[ai], polys[bi]))})
                break
    return [q for k, q in enumerate(polys) if k not in drop], log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/roi303_eval")
    ap.add_argument("--min-span", type=float, default=2.0)
    ap.add_argument("--tol", type=float, default=1.5)
    ap.add_argument("--cov", type=float, default=0.90)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log", default=None, help="JSON log path (default <data>/_repair_log.json)")
    args = ap.parse_args()
    excl_dir = os.path.join(args.data, "_excluded")
    log = {"excluded_frames": [], "dedupe": {}, "params": vars(args)}
    n_dropped = 0
    for p in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        nm = os.path.basename(p)
        with h5py.File(p, "r") as f:
            img = f["image"][...]
            keys = sorted(f["polylines"].keys())
            polys = [f["polylines"][k][...] for k in keys]
            split = str(f.attrs.get("split"))
        ok, why = image_ok(img, args.min_span)
        if not ok:
            log["excluded_frames"].append({"name": nm, "split": split, "n_polylines": len(polys),
                                           "reason": why})
            print(f"EXCLUDE {nm} [{split}] {len(polys)} curves: {why}")
            if not args.dry_run:
                os.makedirs(excl_dir, exist_ok=True)
                shutil.move(p, os.path.join(excl_dir, nm))
                with h5py.File(os.path.join(excl_dir, nm), "r+") as f:
                    f.attrs["excluded_reason"] = why
            continue
        kept, dl = dedupe(polys, args.tol, args.cov)
        if dl:
            n_dropped += len(dl)
            log["dedupe"][nm] = dl
            print(f"DEDUPE  {nm} [{split}] {len(polys)} -> {len(kept)}: "
                  + ", ".join(f"#{d['dropped']}->#{d['kept']} cov={d['coverage']}{' exact' if d['exact'] else ''}" for d in dl))
            if not args.dry_run:
                with h5py.File(p, "r+") as f:
                    del f["polylines"]
                    g = f.create_group("polylines")
                    for k, q in enumerate(kept):
                        g.create_dataset(f"pl_{k:04d}", data=np.asarray(q, np.float64))
                    f.attrs["n_polylines"] = len(kept)
                    f.attrs["n_manual"] = len(kept)
                    f.attrs["sources"] = ",".join(["manual"] * len(kept))
                    f.attrs["n_dedup_dropped"] = len(dl)
                    f.attrs["dedup_tol"] = args.tol
                    f.attrs["dedup_cov"] = args.cov
    print(f"\n{len(log['excluded_frames'])} frames excluded, {n_dropped} duplicate polylines dropped "
          f"in {len(log['dedupe'])} frames{' (DRY RUN)' if args.dry_run else ''}")
    out = args.log or os.path.join(args.data, "_repair_log.json")
    if not args.dry_run:
        with open(out, "w") as fh:
            json.dump(log, fh, indent=1)
        print("log ->", out)


if __name__ == "__main__":
    main()
