#!/usr/bin/env python3
"""BREAK ANATOMY: what does a real faint stretch look like, and does the generator render it?
RUNS ON TULEN (CPU, minutes).

Declared in docs/TODO.md 0.0-CAMPAIGN (2026-09-04 23:05) after check (i) showed the model bridges
synthetic faint stretches (p 0.999) but not real ones. At every GT sample (2 px arc-length steps,
in-frame) this reads, with one instrument on both sides:

  snr        |c| / frame noise (c = smoothed core minus 4-6 px background, along the normal);
  width      FWHM of the normal profile (px), relative to the FILAMENT's own median FWHM;
  rough      background roughness: std of the profile's outer taps (|d| >= 4) / frame noise;
  iso        no other centerline within 8 px;
  flip       sign(c) differs from the filament's dominant sign;
  p          (real only) the model's probability at the sample, from the cached 1.5x map.

Real: p < 0.98 samples vs p >= 0.98 samples -- which covariates separate them. Real vs synth: the
joint (width, snr) behaviour along a filament -- the fraction of samples wider than 1.5x the
filament median, and the within-filament correlation of width and contrast (defocus conserves
contrast x width, so it shows as a NEGATIVE correlation; a multiplicative fade at constant width
shows none).

    PYTHONPATH=.:src:scripts:synth:third_party python scripts/diag_break_anatomy.py \
        --pred-dir /disk2/prusek/mt_work/preds/SPARSE35_roi_native --thr 0.98 \
        --exclude roi_img_8,roi_img_149 --n-synth 80 --out runs/diag_roi/break_anatomy.json
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "src", "scripts", "synth", "third_party"):
    sys.path.insert(0, os.path.join(ROOT, d))

from diag_filament_contrast import frame_noise, resample, stretch  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402

UP = 1.5
TAPS = np.arange(-6, 6.01, 0.5)
MARGIN, ISO_R = 7.0, 8.0


def profiles(img_s, Q, t):
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    xs = Q[:, 0][:, None] + TAPS[None, :] * n[:, 0][:, None]
    ys = Q[:, 1][:, None] + TAPS[None, :] * n[:, 1][:, None]
    return map_coordinates(img_s, [ys.ravel(), xs.ravel()], order=1, mode="nearest").reshape(xs.shape)


def fwhm(prof, core, bg):
    """Half-maximum width of |profile - bg| around the centre, in px (nan when no half crossing)."""
    a = np.abs(prof - bg[:, None])
    peak = np.abs(core - bg)
    half = 0.5 * peak
    out = np.full(len(prof), np.nan)
    mid = len(TAPS) // 2
    for i in range(len(prof)):
        if peak[i] <= 0:
            continue
        row = a[i]
        l = mid
        while l > 0 and row[l] >= half[i]:
            l -= 1
        r = mid
        while r < len(TAPS) - 1 and row[r] >= half[i]:
            r += 1
        if l == 0 or r == len(TAPS) - 1:
            continue
        out[i] = TAPS[r] - TAPS[l]
    return out


def sample_table(img01, polys, prob=None):
    """One row per in-frame sample: [snr, width_rel, width_px, rough, iso, flip, p, fil_snr_med]."""
    img_s = gaussian_filter(img01, 1.0)
    noise = frame_noise(img01)
    H, W = img01.shape
    rs = [resample(poly) for poly in polys]
    pts = [(k, Q) for k, (Q, t) in enumerate(rs) if Q is not None]
    if not pts:
        return np.zeros((0, 8))
    P = np.concatenate([Q for _, Q in pts]); owner = np.concatenate([np.full(len(Q), k) for k, Q in pts])
    tree = cKDTree(P)
    rows = []
    for k, (Q, t) in enumerate(rs):
        if Q is None:
            continue
        inside = (Q[:, 0] >= MARGIN) & (Q[:, 0] < W - MARGIN) & (Q[:, 1] >= MARGIN) & (Q[:, 1] < H - MARGIN)
        if inside.sum() < 3:
            continue
        Q, t = Q[inside], t[inside]
        prof = profiles(img_s, Q, t)
        core = prof[:, np.abs(TAPS) <= 1].mean(1)
        bg = np.median(prof[:, np.abs(TAPS) >= 4], axis=1)
        c = core - bg
        pol = 1.0 if np.median(c) >= 0 else -1.0
        snr = np.abs(c) / noise
        w = fwhm(prof, core, bg)
        wmed = np.nanmedian(w) if np.isfinite(w).any() else np.nan
        rough = prof[:, np.abs(TAPS) >= 4].std(1) / noise
        nb = tree.query_ball_point(Q, ISO_R)
        iso = np.array([all(owner[j] == k for j in lst) for lst in nb], float)
        flip = (np.sign(c) != pol).astype(float)
        pv = map_coordinates(prob, [Q[:, 1] * UP, Q[:, 0] * UP], order=1, mode="nearest") if prob is not None \
            else np.full(len(Q), np.nan)
        rows.append(np.stack([snr, w / wmed if np.isfinite(wmed) and wmed > 0 else np.full(len(w), np.nan),
                              w, rough, iso, flip, pv, np.full(len(Q), np.median(snr))], 1))
    return np.concatenate(rows, 0) if rows else np.zeros((0, 8))


def within_filament_corr(img01, polys):
    """Per filament: Spearman(width, |contrast|) along its length (defocus < 0; uniform fade ~ 0)."""
    img_s = gaussian_filter(img01, 1.0)
    out = []
    for poly in polys:
        Q, t = resample(poly)
        if Q is None or len(Q) < 15:
            continue
        H, W = img01.shape
        inside = (Q[:, 0] >= MARGIN) & (Q[:, 0] < W - MARGIN) & (Q[:, 1] >= MARGIN) & (Q[:, 1] < H - MARGIN)
        if inside.sum() < 15:
            continue
        prof = profiles(img_s, Q[inside], t[inside])
        core = prof[:, np.abs(TAPS) <= 1].mean(1); bg = np.median(prof[:, np.abs(TAPS) >= 4], axis=1)
        w = fwhm(prof, core, bg); a = np.abs(core - bg)
        m = np.isfinite(w)
        if m.sum() >= 12 and a[m].std() > 0 and w[m].std() > 0:
            out.append(float(spearmanr(w[m], a[m])[0]))
    return out


def q(v, ps=(10, 25, 50, 75, 90)):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return {str(p): float(np.percentile(v, p)) for p in ps} if len(v) else {}


def describe(T, name):
    """Covariate summary of a sample set."""
    if len(T) == 0:
        return {"n": 0}
    return {"n": int(len(T)), "snr": q(T[:, 0]), "width_rel": q(T[:, 1]), "width_px": q(T[:, 2]),
            "rough": q(T[:, 3]), "iso_share": float(np.nanmean(T[:, 4])), "flip_share": float(np.nanmean(T[:, 5])),
            "share_width_rel_gt_1.5": float(np.nanmean(T[:, 1] > 1.5)), "share_snr_lt_1": float(np.mean(T[:, 0] < 1)),
            "share_rough_gt_2": float(np.mean(T[:, 3] > 2))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/roi303_eval")
    ap.add_argument("--split", default="val")
    ap.add_argument("--exclude", default="")
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--channel", type=int, default=0)
    ap.add_argument("--thr", type=float, default=0.98)
    ap.add_argument("--calib", default="calib_V6_c070_narrow.json")
    ap.add_argument("--n-synth", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.split == "test":
        raise SystemExit("refusing to touch TEST")
    excl = set(args.exclude.split(",")) if args.exclude else set()

    real, real_corr = [], []
    for p in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        stem = os.path.basename(p)[:-3]
        if stem in excl:
            continue
        fr = read_frame_h5(p)
        if str(fr["attrs"].get("split")) != args.split:
            continue
        npz = os.path.join(args.pred_dir, stem + ".npz")
        if not os.path.exists(npz):
            continue
        allch = np.load(npz)["prob"].astype(np.float32)
        prob = allch[min(args.channel, allch.shape[0] - 1)]
        img01 = stretch(fr["image"])
        real.append(sample_table(img01, fr["polylines"], prob))
        real_corr += within_filament_corr(img01, fr["polylines"])
    R = np.concatenate(real, 0)
    print(f"real: {len(R)} samples", flush=True)

    synth, synth_corr = [], []
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
            img, inst, _ = generate_frame(b[y0:y0 + g, x0:x0 + g], rng, cfg)
            img01 = stretch(img)
            polys = [z["centerline"] for z in inst]
            synth.append(sample_table(img01, polys, None))
            synth_corr += within_filament_corr(img01, polys)
    Sy = np.concatenate(synth, 0) if synth else np.zeros((0, 8))
    print(f"synth: {len(Sy)} samples", flush=True)

    p = R[:, 6]
    S = {"thr": args.thr,
         "real_all": describe(R, "real"), "real_p_lt_thr": describe(R[p < args.thr], "real weak-p"),
         "real_p_ge_thr": describe(R[p >= args.thr], "real strong-p"),
         "real_share_p_lt_thr": float(np.mean(p < args.thr)),
         "synth_all": describe(Sy, "synth"),
         # like-for-like faint stretches: samples with snr < 1 inside filaments whose median snr >= 1.5
         "real_faint_in_visible": describe(R[(R[:, 0] < 1) & (R[:, 7] >= 1.5)], "real faint"),
         "synth_faint_in_visible": describe(Sy[(Sy[:, 0] < 1) & (Sy[:, 7] >= 1.5)], "synth faint") if len(Sy) else {},
         "real_within_filament_spearman_width_vs_contrast": q(real_corr),
         "synth_within_filament_spearman_width_vs_contrast": q(synth_corr),
         "real_share_corr_lt_-0.3": float(np.mean(np.array(real_corr) < -0.3)) if real_corr else None,
         "synth_share_corr_lt_-0.3": float(np.mean(np.array(synth_corr) < -0.3)) if synth_corr else None}
    # p < thr rate by covariate bins on real (which covariate predicts a break?)
    S["real_p_lt_thr_by"] = {}
    for name, col, edges in (("snr", 0, [0, 0.7, 1, 1.5, 2, 3, 5, 1e9]),
                             ("width_rel", 1, [0, 0.7, 0.9, 1.1, 1.3, 1.6, 2.0, 1e9]),
                             ("rough", 3, [0, 1, 1.5, 2, 3, 5, 1e9])):
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (R[:, col] >= lo) & (R[:, col] < hi)
            if m.sum() > 50:
                rows.append({"lo": lo, "hi": hi, "n": int(m.sum()), "p_lt_thr": float(np.mean(p[m] < args.thr))})
        S["real_p_lt_thr_by"][name] = rows
    for name, col in (("iso", 4), ("flip", 5)):
        S["real_p_lt_thr_by"][name] = {str(v): float(np.mean(p[R[:, col] == v] < args.thr)) for v in (0.0, 1.0)
                                       if (R[:, col] == v).sum() > 50}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(S, open(args.out, "w"), indent=1)
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
