#!/usr/bin/env python3
"""Show WHERE the foreground drops out along real microtubules, with the image underneath.

Statistics have taken this as far as they can. Measured so far: 15.8 % of a detected
microtubule's length falls below threshold; at 69 % of those pixels the probability is under
0.05; the evidence is not split across orientation bins; the local contrast there is 0.43x
normal; and the training generator contains MORE long faint stretches than reality, not fewer.
So the model is confidently blind at places it was trained to see, and the remaining
explanations -- a different kind of faintness, crossings, or ground truth drawn through empty
image -- are distinguishable by eye in seconds and not by another summary number.

Two panels per frame: the whole frame with each ground-truth centerline coloured GREEN where
the model detects it and RED where it does not, then the N longest gaps zoomed, with the raw
image and nothing but the ground-truth line drawn over it, so the question "is there a filament
here at all?" can actually be answered.

    PYTHONPATH=src python scripts/viz_gaps.py --pred-dir <npz dir> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.ndimage import gaussian_filter, label, zoom  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from instance.geometry import resample  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.gt_stats import count_crossings  # noqa: E402

UP = 1.5
BG_SIGMA = 12.0


def display01(img):
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape[:2]
    core = img[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
    lo, hi = np.percentile(core, [1.0, 99.0])
    return np.clip((img - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/mt34_eval")
    ap.add_argument("--pred-dir", default="/home/prusek/mt_enc_exp/mt34_pred")
    ap.add_argument("--split", default="test")
    ap.add_argument("--thr", type=float, default=0.442)
    ap.add_argument("--frames", nargs="+", default=["training_img_114", "training_img_112",
                                                    "alice_2026_02_06_pll336_100x_atp_ch1_I__061_f00_irm"])
    ap.add_argument("--n-zoom", type=int, default=6)
    ap.add_argument("--zoom-px", type=int, default=90)
    ap.add_argument("--out-dir", default="data/enc_sensitivity_testset/gap_viz")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for name in args.frames:
        path = os.path.join(args.data, f"{name}.h5")
        npz = os.path.join(args.pred_dir, f"{name}.npz")
        if not (os.path.exists(path) and os.path.exists(npz)):
            print(f"  skip {name}", flush=True)
            continue
        fr = read_frame_h5(path)
        prob = np.load(npz)["prob"].astype(np.float32).max(axis=0)
        H, W = prob.shape
        img = zoom(display01(fr["image"]), UP, order=1)[:H, :W]
        raw = zoom(np.asarray(fr["image"], float), UP, order=1)[:H, :W]
        resid = np.abs(raw - gaussian_filter(raw, BG_SIGMA))

        gt = [np.asarray(p, float) * UP for p in fr["polylines"]]
        cross = count_crossings(gt, tol=2.0 * UP, endpoint_margin=8.0 * UP)
        cxy = np.array([[c["x"], c["y"]] for c in cross]) if cross else np.zeros((0, 2))

        # Every gap run, with its length and where it sits.
        segs = []
        for pl in gt:
            q = resample(pl, ds=1.0)
            r = np.clip(np.rint(q[:, 1]).astype(int), 0, H - 1)
            c = np.clip(np.rint(q[:, 0]).astype(int), 0, W - 1)
            v = prob[r, c]
            faint = v < args.thr
            if faint.all():
                continue
            lab, n = label(faint)
            for k in range(1, n + 1):
                idx = np.where(lab == k)[0]
                if len(idx) < 4:
                    continue
                mid = q[idx[len(idx) // 2]]
                dc = float(np.min(np.linalg.norm(cxy - mid, axis=1))) if len(cxy) else np.inf
                segs.append({"len": len(idx), "mid": mid, "pts": q[idx],
                             "contrast": float(np.median(resid[r[idx], c[idx]])),
                             "d_cross": dc})
        segs.sort(key=lambda s: -s["len"])
        top = segs[:args.n_zoom]

        ncol = max(len(top), 1)
        fig = plt.figure(figsize=(4.2 * max(ncol, 3), 11))
        gs = fig.add_gridspec(2, ncol, height_ratios=[2.1, 1.0])

        ax = fig.add_subplot(gs[0, :])
        ax.imshow(img, cmap="gray", interpolation="nearest")
        for pl in gt:
            q = resample(pl, ds=1.0)
            r = np.clip(np.rint(q[:, 1]).astype(int), 0, H - 1)
            c = np.clip(np.rint(q[:, 0]).astype(int), 0, W - 1)
            ok = prob[r, c] >= args.thr
            ax.plot(q[ok, 0], q[ok, 1], ".", ms=0.7, color="#31d843")
            ax.plot(q[~ok, 0], q[~ok, 1], ".", ms=1.4, color="#ff2d2d")
        for i, s in enumerate(top):
            ax.plot(*s["mid"], "o", ms=13, mfc="none", mec="yellow", mew=1.6)
            ax.annotate(str(i + 1), s["mid"], color="yellow", fontsize=11,
                        xytext=(9, 9), textcoords="offset points")
        miss = 100 * float(np.mean([1] * 0 + [s["len"] for s in segs])) if segs else 0
        ax.set_title(f"{name} — GREEN = detected, RED = dropout    "
                     f"({len(segs)} gap runs, longest {top[0]['len'] if top else 0} px)",
                     fontsize=12)
        ax.axis("off")

        for i, s in enumerate(top):
            axz = fig.add_subplot(gs[1, i])
            z = args.zoom_px
            cx, cy = s["mid"]
            r0, c0 = int(np.clip(cy - z / 2, 0, H - z)), int(np.clip(cx - z / 2, 0, W - z))
            axz.imshow(img[r0:r0 + z, c0:c0 + z], cmap="gray", interpolation="nearest")
            p = s["pts"] - [c0, r0]
            axz.plot(p[:, 0], p[:, 1], "-", color="#ff2d2d", lw=1.0, alpha=0.85)
            dc = "inf" if not np.isfinite(s["d_cross"]) else f"{s['d_cross']:.0f}"
            axz.set_title(f"{i + 1}: {s['len']} px, contrast {s['contrast']:.1f}\n"
                          f"crossing {dc} px away", fontsize=9)
            axz.set_xlim(0, z)
            axz.set_ylim(z, 0)
            axz.axis("off")

        fig.tight_layout()
        out = os.path.join(args.out_dir, f"{name}_gaps.png")
        fig.savefig(out, dpi=105, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  wrote {out}  ({len(segs)} gaps, longest {top[0]['len'] if top else 0} px)",
              flush=True)


if __name__ == "__main__":
    main()
