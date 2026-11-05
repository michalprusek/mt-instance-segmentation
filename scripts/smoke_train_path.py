#!/usr/bin/env python3
"""Smoke the ACTUAL training path of a calibration: crops from OnlineDS.__getitem__ exactly as
train_v5.py builds them (render, crop, stretch, flips), for two calibs side by side. RUNS ON TULEN.

Reports, on the off-label part of each crop (further than 5 px from the target's foreground):
lag-1 residual autocorrelation, 2x2 block variance ratio, kurtosis -- the statistics the NOISE arm
is supposed to move -- plus the foreground fraction, and writes a montage (crop | target) so the
frames the arm trains on can be SEEN.

    PYTHONPATH=.:src:scripts:synth:third_party python scripts/smoke_train_path.py \
        --calib-a calib_V6_c070_narrow.json --calib-b calib_V6_noise.json --n 12 \
        --montage runs/diag_roi/smoke_noise_montage.png --out runs/diag_roi/smoke_noise.json
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.stats import kurtosis

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "src", "scripts", "synth", "third_party"):
    sys.path.insert(0, os.path.join(ROOT, d))


def build_ds(calib, bgs, tirf_prob):
    from gen_train import build_cfg
    from mt_generator import generate_frame
    from train_v5 import OnlineDS
    cal = json.load(open(calib))
    cfg = build_cfg(cal["best_params"], mask_hw=1.0)
    if cal.get("gen_overrides"):
        cfg = dataclasses.replace(cfg, **cal["gen_overrides"])
    if tirf_prob > 0:
        cfg = dataclasses.replace(cfg, tirf_label_sparse=True)
    return OnlineDS(cfg, bgs, generate_frame, crop=512, epoch_len=1000, tirf_prob=tirf_prob)


def stats(im, fg):
    H, W = im.shape
    far = distance_transform_edt(~(fg > 0.5)) > 5.0 if fg.any() else np.ones_like(im, bool)
    r = im - gaussian_filter(im, 2.0); r = r - np.median(r[far])
    v = float(np.var(r[far])) + 1e-12
    a = r[:, 1:]; b = r[:, :-1]; m = far[:, 1:] & far[:, :-1]
    acf = float(np.mean(a[m] * b[m]) / v)
    rb = r[:H // 2 * 2, :W // 2 * 2].reshape(H // 2, 2, W // 2, 2).mean((1, 3))
    fb = far[:H // 2 * 2, :W // 2 * 2].reshape(H // 2, 2, W // 2, 2).all((1, 3))
    return {"acf1": acf, "block2": float(np.var(rb[fb]) / v), "kurt": float(kurtosis(r[far])),
            "fg": float((fg > 0.5).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-a", default="calib_V6_c070_narrow.json")
    ap.add_argument("--calib-b", default="calib_V6_noise.json")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--tirf-prob", type=float, default=0.0, help="0 = IRM only (the arm's scope)")
    ap.add_argument("--montage", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from train_v5 import IMA_M, IMA_S, K_ORI
    bgs = []
    for d in ("/home/prusek/mt_enc_exp/irm_backgrounds_v2",
              "/home/prusek/BIOCEV/datasets/microtubules/IRM_backgrounds_v2",
              os.path.join(ROOT, "data/real/irm_backgrounds_v2")):
        bgs = sorted(glob.glob(os.path.join(d, "*.tif")))
        if bgs:
            break
    res, crops = {}, {}
    for tag, calib in (("A", args.calib_a), ("B", args.calib_b)):
        ds = build_ds(calib, bgs, args.tirf_prob)
        rows, ims = [], []
        for i in range(args.n):
            t, gt = ds[i]
            im = (t[0].numpy() * float(IMA_S[0]) + float(IMA_M[0])) if hasattr(IMA_S, "__len__") else (t[0].numpy() * float(IMA_S) + float(IMA_M))
            fg = gt[K_ORI].numpy()
            rows.append(stats(im, fg)); ims.append((im, fg))
        res[tag] = {"calib": calib, "per_crop": rows,
                    **{k: float(np.median([r[k] for r in rows])) for k in ("acf1", "block2", "kurt", "fg")}}
        crops[tag] = ims
        print(f"{tag} {calib}: median acf1 {res[tag]['acf1']:.3f} block2 {res[tag]['block2']:.3f} "
              f"kurt {res[tag]['kurt']:.2f} fg {100 * res[tag]['fg']:.2f}%", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    if args.montage:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = min(6, args.n)
        fig, axes = plt.subplots(n, 4, figsize=(14, 3.5 * n))
        for i in range(n):
            for j, tag in enumerate(("A", "B")):
                im, fg = crops[tag][i]
                axes[i, 2 * j].imshow(im, cmap="gray", vmin=0, vmax=1); axes[i, 2 * j].set_title(f"{tag} crop {i}", fontsize=8)
                axes[i, 2 * j + 1].imshow(fg, cmap="gray"); axes[i, 2 * j + 1].set_title(f"{tag} target", fontsize=8)
            for ax in axes[i]:
                ax.axis("off")
        plt.tight_layout(); plt.savefig(args.montage, dpi=80); print("montage ->", args.montage)


if __name__ == "__main__":
    main()
