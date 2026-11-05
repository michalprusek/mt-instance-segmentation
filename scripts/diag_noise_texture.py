#!/usr/bin/env python3
"""Is the NOISE TEXTURE the domain gap?  RUNS ON TULEN (CPU, a minute).

The break anatomy (2026-09-04 23:30) found real faint stretches look like synthetic ones in width,
roughness and along-length statistics, yet the model bridges the synthetic ones (97 % of label at
>= 0.98) and not the real ones (41-66 % of samples below local SNR 1 fall under 0.98). One
remaining candidate is the noise itself: the generator adds WHITE shot + read noise, while a real
IRM camera frame carries spatially correlated noise (pixel cross-talk, interference speckle,
compression). A line integrator tuned on white noise under-weights evidence in correlated noise.

Per frame, on the [1, 99]-stretched native image, the high-pass residual r = img - gaussian(img, 2)
is measured OFF the filaments (pixels further than 6 px from any GT / label centerline):
  * normalised autocorrelation at lags (0,1), (1,0), (1,1), (0,2), (2,0), (0,3);
  * the ratio of residual variance at 2x2-block scale to pixel scale (white noise: 0.25);
  * the residual's kurtosis (impulsive dirt vs gaussian noise).
Real (roi303 VAL, raw block) vs synthetic renders of the training calibration.

    PYTHONPATH=.:src:scripts:synth:third_party python scripts/diag_noise_texture.py \
        --exclude roi_img_8,roi_img_149 --n-synth 60 --out runs/diag_roi/noise_texture.json
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

from diag_filament_contrast import stretch  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402

LAGS = [(0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (0, 3)]


def texture(img01, polys):
    H, W = img01.shape
    lab = np.zeros((H, W), bool)
    for c in polys:
        c = np.asarray(c, float)
        ok = (c[:, 0] >= 0) & (c[:, 0] < W - 0.5) & (c[:, 1] >= 0) & (c[:, 1] < H - 0.5)
        c = c[ok]
        if len(c):
            lab[np.round(c[:, 1]).astype(int), np.round(c[:, 0]).astype(int)] = True
    far = distance_transform_edt(~lab) > 6.0 if lab.any() else np.ones((H, W), bool)
    r = img01 - gaussian_filter(img01, 2.0)
    r = r - np.median(r[far])
    v = float(np.var(r[far])) + 1e-12
    out = {"var": v}
    for dy, dx in LAGS:
        a = r[dy:, dx:]; b = r[:H - dy, :W - dx]; m = far[dy:, dx:] & far[:H - dy, :W - dx]
        out[f"acf_{dy}_{dx}"] = float(np.mean(a[m] * b[m]) / v)
    # block-average variance ratio (white noise -> 1/4 at 2x2, 1/16 at 4x4)
    for k in (2, 4):
        rb = r[:H // k * k, :W // k * k].reshape(H // k, k, W // k, k).mean((1, 3))
        fb = far[:H // k * k, :W // k * k].reshape(H // k, k, W // k, k).all((1, 3))
        out[f"block{k}_var_ratio"] = float(np.var(rb[fb]) / v) if fb.any() else None
    out["kurtosis"] = float(kurtosis(r[far]))
    return out


def q(v, ps=(10, 25, 50, 75, 90)):
    v = np.asarray([x for x in v if x is not None], float)
    return {str(p): float(np.percentile(v, p)) for p in ps} if len(v) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/roi303_eval")
    ap.add_argument("--split", default="val")
    ap.add_argument("--exclude", default="")
    ap.add_argument("--calib", default="calib_V6_c070_narrow.json")
    ap.add_argument("--n-synth", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.split == "test":
        raise SystemExit("refusing to touch TEST")
    excl = set(args.exclude.split(",")) if args.exclude else set()
    real = []
    for p in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        stem = os.path.basename(p)[:-3]
        if stem in excl:
            continue
        fr = read_frame_h5(p)
        if str(fr["attrs"].get("split")) != args.split:
            continue
        t = texture(stretch(fr["image"]), fr["polylines"]); t["frame"] = stem
        real.append(t)
    synth = []
    if args.n_synth > 0:
        from gen_train import build_cfg
        from mt_generator import generate_frame
        from PIL import Image
        cal = json.load(open(args.calib))
        cfg = build_cfg(cal["best_params"], mask_hw=1.0)
        if cal.get("gen_overrides"):
            cfg = dataclasses.replace(cfg, **cal["gen_overrides"])
        bgs = []
        for d in ("/home/prusek/mt_enc_exp/irm_backgrounds_v2",
                  "/home/prusek/BIOCEV/datasets/microtubules/IRM_backgrounds_v2",
                  os.path.join(ROOT, "data/real/irm_backgrounds_v2")):
            bgs = sorted(glob.glob(os.path.join(d, "*.tif")))
            if bgs:
                break
        rng = np.random.default_rng(args.seed)
        g = 640
        for k in range(args.n_synth):
            b = np.asarray(Image.open(bgs[int(rng.integers(len(bgs)))]), np.float32)
            if b.ndim == 3:
                b = b.mean(2)
            if b.shape[0] < g or b.shape[1] < g:
                b = np.asarray(Image.fromarray(b).resize((max(g, b.shape[1]), max(g, b.shape[0]))), np.float32)
            y0, x0 = int(rng.integers(0, b.shape[0] - g + 1)), int(rng.integers(0, b.shape[1] - g + 1))
            bg = b[y0:y0 + g, x0:x0 + g]
            img, inst, _ = generate_frame(bg, rng, cfg)
            t = texture(stretch(img), [z["centerline"] for z in inst]); t["frame"] = f"synth_{k}"
            # the empty background alone, stretched the same way: what the real texture would be
            # without the generator's added noise
            tb = texture(stretch(bg), []); t["bg_only"] = tb
            synth.append(t)
    keys = [f"acf_{dy}_{dx}" for dy, dx in LAGS] + ["block2_var_ratio", "block4_var_ratio", "kurtosis", "var"]
    S = {"real": {k: q([r[k] for r in real]) for k in keys},
         "synth": {k: q([r[k] for r in synth]) for k in keys},
         "synth_background_only": {k: q([r["bg_only"][k] for r in synth]) for k in keys},
         "n_real": len(real), "n_synth": len(synth)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"summary": S, "real": real, "synth": synth}, open(args.out, "w"), indent=1)
    for k in keys:
        print(f"{k:18s} real q10/50/90 {S['real'][k].get('10', float('nan')):.3f}/{S['real'][k].get('50', float('nan')):.3f}/{S['real'][k].get('90', float('nan')):.3f}"
              f"   synth {S['synth'][k].get('10', float('nan')):.3f}/{S['synth'][k].get('50', float('nan')):.3f}/{S['synth'][k].get('90', float('nan')):.3f}"
              f"   bg-only {S['synth_background_only'][k].get('10', float('nan')):.3f}/{S['synth_background_only'][k].get('50', float('nan')):.3f}/{S['synth_background_only'][k].get('90', float('nan')):.3f}")


if __name__ == "__main__":
    main()
