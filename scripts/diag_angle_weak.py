#!/usr/bin/env python3
"""Pre-screen for the orientation-isotropic renderer (generator research 2026-09-16, the one
surviving proposal): does the MODEL's confidence on plain filament stretches depend on the
filament's angle to the pixel grid on REAL data, the way the generator's `_paint` aliasing
(axis-aligned filaments render 20-40 % brighter than oblique ones, symmetric under the rot90/flip
augmentation group) would predict?

Samples: GT polyline points resampled at 1 native px, >= FAR native px from any GT crossing and
>= END px from either end (plain stretches, as in diag_line_integral.py). Network read: max of p
within +-2 eval px of the point (prob_at convention). weak = p < --thr. Angle: local tangent angle
mod 90 deg, binned into 6 bins of 15 deg; AXIS = the two half-bins nearest 0 and 90 (|theta| < 7.5
or > 82.5), OBLIQUE = 22.5..67.5.

Reported PER FRAME then pooled across frames (paired, so nematic frames cannot confound):
  excess = weak share (AXIS) - weak share (OBLIQUE) within the frame, mean over frames with
  >= --min-n samples in both classes, with a frame bootstrap CI; plus the pooled weak share per bin.
Run it on the real dump (roi303 VAL, htw VAL) and on the in-domain synth VAL dump; the kill line
and launch gate are declared in docs/TODO.md before any read.

    PYTHONPATH=src python scripts/diag_angle_weak.py --pred-dir <dump> --data <h5 dir> --split val \\
        --thr 0.98 --out angle.json [--prefix htw_]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from instance.geometry import resample  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.gt_stats import count_crossings  # noqa: E402

UP, K_ORI = 1.5, 6
FAR, END = 40.0, 10.0      # native px
BINS = np.array([0, 15, 30, 45, 60, 75, 90], float)


def normals(pts):
    d = np.gradient(pts, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], axis=1)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9), d


def prob_at(prob, pts_native, win=2.0):
    H, W = prob.shape
    q = pts_native * UP
    nrm, _ = normals(q)
    offs = np.arange(-win, win + 1e-9, 1.0)
    p = q[:, None, :] + nrm[:, None, :] * offs[None, :, None]
    cc = np.clip(np.rint(p[..., 0]).astype(int), 0, W - 1)
    rr = np.clip(np.rint(p[..., 1]).astype(int), 0, H - 1)
    return prob[rr, cc].max(axis=1)


def contrast_along(img, pts, perp=2.0, bg_lo=6.0, bg_hi=12.0):
    H, W = img.shape
    nrm, _ = normals(pts)
    offs_sig = np.arange(-perp, perp + 1e-9, 0.5)
    offs_bg = np.concatenate([np.arange(-bg_hi, -bg_lo + 1e-9, 1.0), np.arange(bg_lo, bg_hi + 1e-9, 1.0)])

    def sample(off):
        p = pts[:, None, :] + nrm[:, None, :] * off[None, :, None]
        cc = np.clip(np.rint(p[..., 0]).astype(int), 0, W - 1)
        rr = np.clip(np.rint(p[..., 1]).astype(int), 0, H - 1)
        return img[rr, cc]

    bg = np.median(sample(offs_bg), axis=1)
    return np.abs(sample(offs_sig) - bg[:, None]).max(axis=1)


def frame_noise(img):
    hp = img - gaussian_filter(img, 3.0)
    return 1.4826 * np.median(np.abs(hp - np.median(hp))) + 1e-9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--thr", type=float, required=True)
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--min-n", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.split == "test":
        raise SystemExit("VAL only")
    rng = np.random.default_rng(0)
    frames = []          # per frame: (weak counts per bin, totals per bin)
    for path in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        name = os.path.basename(path)
        if args.prefix and not name.startswith(args.prefix):
            continue
        fr = read_frame_h5(path)
        if str(fr["attrs"].get("split")) != args.split or not fr["polylines"]:
            continue
        npz = os.path.join(args.pred_dir, name.replace(".h5", ".npz"))
        if not os.path.exists(npz):
            continue
        ch = np.load(npz)["prob"].astype(np.float32)
        prob = ch[:K_ORI].max(axis=0)
        img = np.asarray(fr["image"], float); sigma = frame_noise(img)
        snr_bins = [[] for _ in range(len(BINS) - 1)]
        polys = [np.asarray(q, float) for q in fr["polylines"] if len(q) >= 2]
        cr = count_crossings(polys, tol=2.0, ds=1.0, endpoint_margin=8.0)
        cr_tree = cKDTree(np.array([[c["x"], c["y"]] for c in cr])) if cr else None
        wk = np.zeros(len(BINS) - 1); tot = np.zeros(len(BINS) - 1)
        for q in polys:
            pts = resample(q, ds=1.0)
            if len(pts) < 2 * END + 5:
                continue
            _, d = normals(pts)
            ang = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180.0
            ang = np.where(ang > 90.0, 180.0 - ang, ang)          # fold to [0, 90]
            ok = np.ones(len(pts), bool)
            ok[: int(END)] = False; ok[-int(END):] = False
            if cr_tree is not None:
                ok &= cr_tree.query(pts)[0] >= FAR
            if not ok.any():
                continue
            p = prob_at(prob, pts)
            weak = (p < args.thr)
            snr = contrast_along(img, pts) / sigma
            b = np.clip(np.digitize(ang, BINS) - 1, 0, len(BINS) - 2)
            for i in range(len(BINS) - 1):
                m = ok & (b == i)
                tot[i] += m.sum(); wk[i] += (weak & m).sum()
                if m.any():
                    snr_bins[i].append(snr[m])
        if tot.sum() > 0:
            frames.append((wk, tot, name, [np.concatenate(x) if x else np.zeros(0) for x in snr_bins]))
    if not frames:
        raise SystemExit("no samples")
    WK = np.array([f[0] for f in frames]); TOT = np.array([f[1] for f in frames])
    pooled = WK.sum(0) / np.maximum(TOT.sum(0), 1)
    snr_all = [np.concatenate([f[3][i] for f in frames]) for i in range(len(BINS) - 1)]
    snr_p50 = [float(np.median(x)) if x.size else float("nan") for x in snr_all]
    snr_p25 = [float(np.percentile(x, 25)) if x.size else float("nan") for x in snr_all]
    # axis = bins 0 and 5 (0-15 and 75-90; the half-bins nearest the axes are inside them), oblique = bins 1..4 centre (30-60)
    axis_w, axis_t = WK[:, [0, 5]].sum(1), TOT[:, [0, 5]].sum(1)
    obl_w, obl_t = WK[:, [2, 3]].sum(1), TOT[:, [2, 3]].sum(1)
    keep = (axis_t >= args.min_n) & (obl_t >= args.min_n)
    ex = axis_w[keep] / axis_t[keep] - obl_w[keep] / obl_t[keep]
    boot = [rng.choice(ex, len(ex)).mean() for _ in range(2000)] if ex.size else [np.nan]
    res = {"pred_dir": args.pred_dir, "thr": args.thr, "prefix": args.prefix, "n_frames": len(frames), "n_frames_paired": int(keep.sum()),
           "bins_deg": BINS.tolist(), "pooled_weak_share_per_bin": pooled.tolist(), "n_per_bin": TOT.sum(0).tolist(),
           "overall_weak_share": float(WK.sum() / TOT.sum()),
           "image_snr_p50_per_bin": snr_p50, "image_snr_p25_per_bin": snr_p25,
           "axis_minus_oblique_excess_mean": float(ex.mean()) if ex.size else float("nan"),
           "axis_minus_oblique_excess_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
           "axis_minus_oblique_excess_sd_frames": float(ex.std(ddof=1)) if ex.size > 1 else float("nan")}
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"frames {len(frames)} (paired {int(keep.sum())})  overall weak share {res['overall_weak_share']:.4f}")
    print("  weak share per 15-deg bin (0..90):", " ".join(f"{v:.4f}" for v in pooled), " n:", " ".join(f"{int(v)}" for v in TOT.sum(0)))
    print("  image SNR p50 per bin:            ", " ".join(f"{v:.2f}" for v in snr_p50), "   p25:", " ".join(f"{v:.2f}" for v in snr_p25))
    print(f"  AXIS(0-15,75-90) - OBLIQUE(30-60) excess, mean over frames {res['axis_minus_oblique_excess_mean']:+.4f}  "
          f"CI95 [{res['axis_minus_oblique_excess_ci95'][0]:+.4f}, {res['axis_minus_oblique_excess_ci95'][1]:+.4f}]  frame sd {res['axis_minus_oblique_excess_sd_frames']:.4f}")


if __name__ == "__main__":
    main()
