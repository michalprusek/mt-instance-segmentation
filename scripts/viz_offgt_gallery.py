#!/usr/bin/env python3
"""Crop gallery of an arm's OFF-ANNOTATION predictions (diag_fp_anatomy class off_other) so the
one model-owned cost can be counted by eye: junk rims (bright blob with a dark ring traced as a
curve) vs straight dark lines that are unannotated filaments. RUNS ON TULEN (CPU).

Takes the anatomy JSONs, keeps the N longest off_other predictions on frames with < --max-gt GT,
re-instances those frames, and draws a crop around each prediction's midpoint (image, GT faint,
the prediction in red) with the frame, length and local SNR in the title.

    PYTHONPATH=.:src:scripts python scripts/viz_offgt_gallery.py --anat "runs/diag_roi/anat/NOISE_roi_ep40_native_*.json" \
        --pred-dir preds/NOISE_roi_ep40_native --n 30 --max-gt 10 --out runs/diag_roi/offgt_gallery_NOISE.png
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import zoom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from diag_fp_anatomy import UP, KAPPA_MAX, K_ORI, resample  # noqa: E402
from instance.instancer_a import instance_a  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/roi303_eval")
    ap.add_argument("--anat", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--thr", type=float, default=0.99)
    ap.add_argument("--params", default="src/instance/params_a_model_v5H.json")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--max-gt", type=int, default=10)
    ap.add_argument("--crop", type=int, default=160, help="eval px")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    params = json.load(open(args.params)); params.pop("kappa_max", None); params["prob_thr"] = args.thr
    cand = []
    for f in glob.glob(args.anat):
        for rec in json.load(open(f))["frames"]:
            if rec["n_gt"] >= args.max_gt:
                continue
            for x in rec["fps"]:
                if x["cls"] == "off_other":
                    cand.append((x["len"], rec["frame"], x["pred"]))
    cand.sort(reverse=True); cand = cand[: args.n]
    print(len(cand), "candidates from", len({c[1] for c in cand}), "frames", flush=True)
    by_frame = {}
    for L, stem, i in cand:
        by_frame.setdefault(stem, []).append((i, L))
    crops = []
    for stem, items in by_frame.items():
        fr = read_frame_h5(os.path.join(args.data, stem + ".h5"))
        if str(fr["attrs"].get("split")) == "test":
            raise SystemExit("refusing to touch TEST")
        allch = np.load(os.path.join(args.pred_dir, stem + ".npz"))["prob"].astype(np.float32)
        prob = allch[0]; chans = allch[:K_ORI]; H, W = prob.shape
        polys, _ = instance_a(prob > args.thr, KAPPA_MAX, params=params, channels=chans, prob=prob)
        img = fr["image"].astype(np.float32); lo, hi = np.percentile(img, [1, 99]); img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
        img = zoom(img, UP, order=1)
        gt = [np.asarray(q, float) * UP for q in fr["polylines"]]
        for i, L in items:
            if i >= len(polys):
                continue
            c = resample(np.asarray(polys[i], float)); mid = c[len(c) // 2]
            crops.append((stem, i, L, img, gt, c, mid))
    n = len(crops); cols = 6; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows))
    axes = np.atleast_2d(axes)
    h = args.crop // 2
    for k, (stem, i, L, img, gt, c, mid) in enumerate(crops):
        ax = axes[k // cols, k % cols]
        x0, y0 = int(mid[0]) - h, int(mid[1]) - h
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        for g in gt:
            ax.plot(g[:, 0], g[:, 1], color="yellow", lw=0.6, alpha=0.5)
        ax.plot(c[:, 0], c[:, 1], color="red", lw=1.0)
        ax.set_xlim(x0, x0 + 2 * h); ax.set_ylim(y0 + 2 * h, y0); ax.axis("off")
        ax.set_title(f"{stem.replace('roi_img_', '#')} p{i} {L}px", fontsize=8)
    for k in range(n, rows * cols):
        axes[k // cols, k % cols].axis("off")
    plt.tight_layout(); plt.savefig(args.out, dpi=100); print("->", args.out)


if __name__ == "__main__":
    main()
