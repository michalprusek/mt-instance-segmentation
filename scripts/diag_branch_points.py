#!/usr/bin/env python3
"""Branch points of the decoded ridge, per frame: the HEAT claim's instrument (docs/TODO.md, HEAT
decode declaration: a Gaussian peak has no saturated plateau, so NMS spurs must not recur).
For each frame of a 2-channel dump: skeleton branch points (8-neighbour degree >= 3) of
  B(0.98)  = skeletonize(existence > 0.98)
  N1(t)    = NMS ridge of the HEAT channel at t (decode_ridge.ridge_nms on channel 1)
  G(t,tx)  = the gated variant's mask as built by decode_ridge.build_masks
and, for reference, the same on a 1-channel dump (SPARSE35): B(0.98) and N(t) on channel 0.
RUNS ON TULEN (CPU, minutes).

    PYTHONPATH=.:src:scripts python scripts/diag_branch_points.py --pred-dir preds/HEAT_roi_ep40_native --out runs/ridge_heat/branch_points.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from skimage.morphology import skeletonize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from decode_ridge import build_masks  # noqa: E402


def branch_points(mask):
    sk = skeletonize(mask).astype(np.uint8)
    if not sk.any():
        return 0, 0
    pad = np.pad(sk, 1)
    nb = sum(pad[1 + dy:pad.shape[0] - 1 + dy, 1 + dx:pad.shape[1] - 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0))
    return int(((sk > 0) & (nb >= 3)).sum()), int(sk.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--exclude", default="roi_img_8,roi_img_149")
    ap.add_argument("--variants", default="B(0.98),N(0.5),G(0.5,0.3),G(0.3,0.3),G(0.7,0.5),GU(0.5,0.3)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    excl = set(args.exclude.split(","))
    variants = [v.strip() for v in args.variants.split(",")]
    fixed, buf = [], ""
    for tok in variants:
        buf = tok if not buf else buf + "," + tok
        if buf.count("(") == buf.count(")"):
            fixed.append(buf); buf = ""
    variants = fixed
    rows = []
    for f in sorted(glob.glob(os.path.join(args.pred_dir, "*.npz"))):
        stem = os.path.basename(f)[:-4]
        if stem in excl:
            continue
        allch = np.load(f)["prob"].astype(np.float32)
        prob = allch[0] if allch.ndim == 3 else allch; heat = allch[1] if (allch.ndim == 3 and allch.shape[0] > 1) else None
        vs = [v for v in variants if not (v.startswith("G") and heat is None)]
        masks = build_masks(prob, vs, heat=heat)
        rec = {"frame": stem}
        for v, m in masks.items():
            bp, L = branch_points(m); rec[v] = {"bp": bp, "skel_px": L, "bp_per_kpx": (1000.0 * bp / L) if L else 0.0}
        rows.append(rec); print(stem, {v: rec[v]["bp"] for v in masks}, flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(rows, open(args.out, "w"))
    for v in variants:
        rr = [r[v] for r in rows if v in r]
        if rr:
            print("%-12s branch points/frame q50 %.0f q90 %.0f  per 1000 skeleton px q50 %.2f  frames %d" % (v, np.median([x["bp"] for x in rr]), np.percentile([x["bp"] for x in rr], 90), np.median([x["bp_per_kpx"] for x in rr]), len(rr)))


if __name__ == "__main__":
    main()
