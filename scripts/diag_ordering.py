#!/usr/bin/env python3
"""What occupies the 0.5-0.97 probability band: filament the model under-scores, or background it over-scores?

Read 38b measured that at threshold 0.50 the map already covers 85 % of GT curves to >= 95 % of
their length, and at 0.97 only 33 % -- so the information for recall 0.85 exists and cannot be
extracted, because decoding at 0.5 floods the frame. The whole remaining gap is ORDERING. This
names which side of the ordering is broken, which decides the intervention:

  * if the band is mostly ON-GT pixels the model under-scores, the fix is evidence/sensitivity
    (a generator that makes faint filaments learnable, or a loss that stops treating them as easy);
  * if the band is mostly OFF-GT pixels the model over-scores, the fix is harder negatives
    (background structures the generator does not produce), and raising sensitivity would hurt.

Reports, per threshold band, the share of pixels within `tol` of any GT centerline, plus the
pixel-level average precision of the map against a GT-proximity label, in domain and on real, so
the two can be compared as a DOMAIN GAP rather than as one number.

    PYTHONPATH=src:scripts python scripts/diag_ordering.py --data data/real/roi303_eval \
        --pred-dir <dump> --out ordering.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from instance.oracle import oracle_instance_masks  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402

UP = 1.5
BANDS = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 0.97), (0.97, 1.01)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--tol", type=float, default=5.0, help="eval px; the metric's own tolerance")
    ap.add_argument("--split", default="val")
    ap.add_argument("--edge-band", type=float, default=30.0,
                    help="eval px from the frame border; off-GT mass there is the known "
                         "edge-band artefact of cropped ROI frames, not a generator failure")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    on = np.zeros(len(BANDS)); off = np.zeros(len(BANDS)); off_edge = np.zeros(len(BANDS))
    ap_num = ap_den = 0.0
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
        gt = [np.asarray(q, float) * UP for q in fr["polylines"]]
        if not gt:
            continue
        core = np.zeros(prob.shape, bool)
        for m in oracle_instance_masks(gt, prob.shape, half_width=0.5, up=1.0):
            core |= m
        near = distance_transform_edt(~core) <= args.tol
        # Off-GT is NOT the same as background. The known anatomy of off-annotation predictions
        # (memory: strict-f1-blind-to-bridging) is ~1/3-2/3 EDGE BANDS on cropped ROI frames,
        # most of the rest UNANNOTATED VISIBLE FILAMENTS, ~1/6 junk rims -- only the last is a
        # model error. The border distance separates the edge-band share, which a FOV rule fixes
        # and no generator change should be asked to.
        h, w = prob.shape
        yy, xx = np.mgrid[0:h, 0:w]
        border = np.minimum(np.minimum(yy, h - 1 - yy), np.minimum(xx, w - 1 - xx))
        edge = border <= args.edge_band
        for i, (lo, hi) in enumerate(BANDS):
            b = (prob >= lo) & (prob < hi)
            on[i] += int((b & near).sum()); off[i] += int((b & ~near).sum())
            off_edge[i] += int((b & ~near & edge).sum())
        # pixel-level AP of the map against GT proximity, subsampling negatives 1:50
        y = near.ravel(); s = prob.ravel()
        neg = np.where(~y)[0]
        keep = np.concatenate([np.where(y)[0], neg[::50]])
        ys, ss = y[keep], s[keep]
        o = np.argsort(-ss)
        ys = ys[o]
        tp = np.cumsum(ys); prec = tp / np.arange(1, len(ys) + 1)
        if ys.sum():
            ap_num += float((prec * ys).sum() / ys.sum()); ap_den += 1
        frames += 1
        print(f"{stem}", flush=True)
        if args.limit and frames >= args.limit:
            break

    tot = on + off
    summ = {"frames": frames, "tol": args.tol,
            "pixel_AP_vs_GT_proximity": round(ap_num / max(ap_den, 1), 4),
            "bands": {f"{lo:g}-{hi:g}": {
                "px": int(tot[i]),
                "share_of_all_px": round(float(tot[i] / max(tot.sum(), 1)), 5),
                "on_GT_share": round(float(on[i] / max(tot[i], 1)), 4)} for i, (lo, hi) in enumerate(BANDS)},
            "on_GT_px_total": int(on.sum()), "off_GT_px_total": int(off.sum())}
    # the decisive summary: of the pixels in the extractable band, how many are filament?
    mid = sum(tot[i] for i, (lo, hi) in enumerate(BANDS) if lo >= 0.5 and hi <= 0.97)
    mid_on = sum(on[i] for i, (lo, hi) in enumerate(BANDS) if lo >= 0.5 and hi <= 0.97)
    mid_edge = sum(off_edge[i] for i, (lo, hi) in enumerate(BANDS) if lo >= 0.5 and hi <= 0.97)
    summ["band_0.5_0.97"] = {"px": int(mid), "on_GT_share": round(float(mid_on / max(mid, 1)), 4),
                             "on_GT_px": int(mid_on), "off_GT_px": int(mid - mid_on),
                             "off_GT_in_edge_band_px": int(mid_edge),
                             "off_GT_edge_share": round(float(mid_edge / max(mid - mid_on, 1)), 4)}
    summ["off_GT_edge_share_by_band"] = {f"{lo:g}-{hi:g}": round(float(off_edge[i] / max(off[i], 1)), 4)
                                         for i, (lo, hi) in enumerate(BANDS)}
    json.dump(summ, open(args.out, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
