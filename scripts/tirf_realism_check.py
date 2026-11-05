#!/usr/bin/env python3
"""Does the sparse-labelling renderer actually reproduce real TIRF? Measured, not eyeballed.

The generator now renders TIRF filaments as a chain of discrete fluorophore puncta. Whether that
was worth doing is decided by putting synthetic and real frames through the SAME measurement -- the
duty cycle along each filament, thresholded against each frame's OWN off-filament noise, so an
exposure or gain difference between synthetic and real cannot pass for a difference in labelling.

Real curves come from the annotation; synthetic ones from the generator's own centerlines, so the
two sides differ in the image and nothing else.

    python scripts/tirf_realism_check.py --real <real_eval_alice_tirf/h5> --bg <irm_backgrounds_v2>
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import h5py
import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "synth"))

from mt_generator import GenConfig, generate_frame  # noqa: E402


def densify(p, step=1.0):
    d = np.sqrt(((p[1:] - p[:-1]) ** 2).sum(1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 0:
        return p
    t = np.arange(0.0, s[-1], step)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)


def profile(img, poly, half=7.0):
    q = densify(poly)
    if len(q) < 5:
        return None
    t = np.gradient(q, axis=0)
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    H, W = img.shape

    def take(p):
        c = np.round(p[:, 0]).astype(int)
        r = np.round(p[:, 1]).astype(int)
        ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
        v = np.full(len(p), np.nan)
        v[ok] = img[r[ok], c[ok]]
        return v

    return take(q) - np.nanmean(np.stack([take(q + n * half), take(q - n * half)]), axis=0)


def frame_duty(img, polys, k_noise=3.0, seed=7):
    """Median duty cycle over the frame's filaments, against the frame's own noise floor.

    Identical for real and synthetic, which is the whole point -- the noise floor is measured from
    displaced copies of the same curves in the same image, so brightness and gain divide out."""
    polys = [np.asarray(q, float) for q in polys if len(q) >= 2]
    if not polys:
        return np.nan, np.nan
    anno = cKDTree(np.concatenate([densify(q, 2.0) for q in polys], axis=0))
    H, W = img.shape
    rng = np.random.default_rng(seed)
    nvals = []
    for q in polys:
        for _ in range(40):
            z = q + np.array([rng.uniform(-.4, .4) * W, rng.uniform(-.4, .4) * H])
            if not ((z[:, 0] >= 0).all() and (z[:, 0] < W).all()
                    and (z[:, 1] >= 0).all() and (z[:, 1] < H).all()):
                continue
            if anno.query(densify(z, 2.0), k=1)[0].min() <= 8.0:
                continue
            v = profile(img, z)
            if v is not None:
                nvals.append(v[np.isfinite(v)])
            break
    if not nvals:
        return np.nan, np.nan
    sigma = float(np.std(np.concatenate(nvals)))
    duty = []
    for q in polys:
        v = profile(img, q)
        if v is None:
            continue
        v = v[np.isfinite(v)]
        if len(v) >= 30:
            duty.append(float((v > k_noise * sigma).mean()))
    duty = [d for d in duty if d >= 0.05]      # a curve on nothing says nothing about labelling
    return (float(np.median(duty)) if duty else np.nan), sigma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--bg", required=True, help="directory of real empty-field backgrounds")
    ap.add_argument("--pattern", default="atp_ch4", help="real frames that carry TIRF signal")
    ap.add_argument("--exclude", default="NOatp")
    ap.add_argument("--n-synth", type=int, default=8, help="synthetic frames per density")
    ap.add_argument("--fig", default="tirf_realism.png")
    args = ap.parse_args()

    print("REAL")
    real = []
    for p in sorted(glob.glob(os.path.join(args.real, "*.h5"))):
        nm = os.path.basename(p)
        if args.pattern not in nm or (args.exclude and args.exclude in nm):
            continue
        with h5py.File(p, "r") as f:
            img = f["image"][()].astype(np.float64)
            polys = [f["polylines"][k][()] for k in sorted(f["polylines"])]
        d, s = frame_duty(img, polys)
        real.append(d)
        print(f"  {nm[:50]:50s} duty {d:5.2f}   noise sigma {s:7.1f}")
    real = np.array([d for d in real if np.isfinite(d)])
    print(f"  -> real duty: min {real.min():.2f}  median {np.median(real):.2f}  max {real.max():.2f}")

    bgs = []
    for ext in ("*.tif", "*.npy", "*.png"):
        bgs = sorted(glob.glob(os.path.join(args.bg, ext)))[:200]
        if bgs:
            break
    if not bgs:
        raise SystemExit(f"no backgrounds under {args.bg}")

    def load_bg(p):
        if p.endswith(".npy"):
            return np.load(p).astype(np.float64)
        from PIL import Image
        a = np.asarray(Image.open(p), float)
        while a.ndim > 2:
            a = a[..., 0]
        return a

    print("\nSYNTHETIC (same measurement)")
    rows = []
    for label, cfg in (("continuous (old renderer)", GenConfig(tirf_label_sparse=False)),
                       ("sparse, full prior", GenConfig(tirf_label_sparse=True))):
        ds = []
        for i in range(args.n_synth):
            bg = load_bg(bgs[i % len(bgs)])
            img, ins, _ = generate_frame(bg, np.random.default_rng(500 + i), cfg, modality="tirf")
            polys = [np.asarray(z["centerline"], float) for z in ins]
            d, _ = frame_duty(img, polys, seed=11 + i)
            if np.isfinite(d):
                ds.append(d)
        ds = np.array(ds)
        rows.append((label, ds))
        print(f"  {label:28s} duty: min {ds.min():.2f}  median {np.median(ds):.2f}  max {ds.max():.2f}")

    print("\nVERDICT")
    for label, ds in rows:
        covers = ds.min() <= real.min() + 0.05 and ds.max() >= real.max() - 0.05
        inside = (ds >= real.min() - 0.05) & (ds <= real.max() + 0.05)
        print(f"  {label:28s} brackets the real range: {'YES' if covers else 'no':3s}   "
              f"{100 * inside.mean():5.1f} % of synthetic frames land inside it")
    print("\nBracketing the real range is the target, NOT matching its median: this corpus is one")
    print("lab's labelling ratio, and the generator has to cover the labs it has not seen.")


if __name__ == "__main__":
    main()
