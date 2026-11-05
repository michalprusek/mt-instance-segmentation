#!/usr/bin/env python3
"""Side-by-side overlay of two arms' instances on the same real frames (the eval frame, 1.5x).
RUNS ON TULEN (CPU). For the NOISE read: what are the extra false positives on the near-empty
frames, and what did it gain on the dense ones?

Panels: image + GT (yellow) | arm A instances | arm B instances. A predicted polyline is drawn
GREEN when >= 95 % of its samples lie within 5 px of some GT centerline (it would pass the
precision half of the match) and RED otherwise; GT is drawn faint white under the predictions and
GT filaments with < 95 % of their samples within 5 px of any prediction are drawn MAGENTA (missed).
Counts in the titles are the strict metric on the same masks.

    PYTHONPATH=.:src:scripts python scripts/viz_arm_pair.py --frames roi_img_280,roi_img_44 \
        --pred-a preds/SPARSE35_roi_native --pred-b preds/NOISE_roi_ep40_native \
        --label-a SPARSE35 --label-b NOISE --thr 0.99 --out-dir runs/diag_roi/pairs
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt, zoom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from instance.instancer_a import instance_a  # noqa: E402
from instance.metrics import centerline_f1  # noqa: E402
from instance.oracle import oracle_instance_masks  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402

UP, KAPPA_MAX, TOL = 1.5, 0.25, 5.0


def raster(polys, shape):
    m = np.zeros(shape, bool)
    for c in polys:
        c = np.asarray(c, float)
        if len(c) < 2:
            continue
        for (x0, y0), (x1, y1) in zip(c[:-1], c[1:]):
            n = max(2, int(np.hypot(x1 - x0, y1 - y0)) + 1)
            xs = np.round(np.linspace(x0, x1, n)).astype(int); ys = np.round(np.linspace(y0, y1, n)).astype(int)
            ok = (xs >= 0) & (xs < shape[1]) & (ys >= 0) & (ys < shape[0])
            m[ys[ok], xs[ok]] = True
    return m


def frac_near(poly, dist):
    c = np.asarray(poly, float)
    if len(c) < 2:
        return 0.0
    seg = np.hypot(*(np.diff(c, axis=0).T)); n = max(2, int(seg.sum()) + 1)
    t = np.linspace(0, 1, n); cum = np.concatenate([[0], np.cumsum(seg)]) / max(cum_ := seg.sum(), 1e-9)
    xs = np.interp(t, cum, c[:, 0]); ys = np.interp(t, cum, c[:, 1])
    xi = np.clip(np.round(xs).astype(int), 0, dist.shape[1] - 1); yi = np.clip(np.round(ys).astype(int), 0, dist.shape[0] - 1)
    return float(np.mean(dist[yi, xi] <= TOL))


def run_arm(prob, params, gt, hi):
    m = prob > params["thr"]
    polys, _ = instance_a(m, KAPPA_MAX, {k: v for k, v in params.items() if k != "thr"}, channels=prob[None], prob=prob)
    im = oracle_instance_masks(polys, hi, half_width=1.0, up=1.0)
    r = centerline_f1(im, gt, tol=TOL, length_coverage=0.95, precision_coverage=0.95) if gt else \
        {"tp": 0, "fp": len(polys), "fn": 0, "f1": float("nan")}
    return polys, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/roi303_eval")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--pred-a", required=True); ap.add_argument("--pred-b", required=True)
    ap.add_argument("--label-a", default="A"); ap.add_argument("--label-b", default="B")
    ap.add_argument("--thr", type=float, default=0.99)
    ap.add_argument("--params", default="src/instance/params_a_model_v5H.json")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    params = json.load(open(args.params)); params.pop("kappa_max", None); params.pop("prob_thr", None)
    params["thr"] = args.thr
    os.makedirs(args.out_dir, exist_ok=True)
    for stem in args.frames.split(","):
        fr = read_frame_h5(os.path.join(args.data, stem + ".h5"))
        if str(fr["attrs"].get("split")) == "test":
            raise SystemExit("refusing to touch TEST")
        img = fr["image"].astype(np.float32)
        lo, hi_ = np.percentile(img, [1, 99]); img = np.clip((img - lo) / max(hi_ - lo, 1e-6), 0, 1)
        img = zoom(img, UP, order=1)
        shape = (int(fr["attrs"]["height"]), int(fr["attrs"]["width"]))
        hi = (int(round(shape[0] * UP)), int(round(shape[1] * UP)))
        gt = [np.asarray(q, float) * UP for q in fr["polylines"]]
        gdist = distance_transform_edt(~raster(gt, hi)) if gt else np.full(hi, 1e6)
        fig, axes = plt.subplots(1, 3, figsize=(24, 8.5))
        axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
        for g in gt:
            axes[0].plot(g[:, 0], g[:, 1], color="yellow", lw=0.8)
        axes[0].set_title(f"{stem}: image + GT ({len(gt)})", fontsize=11)
        for ax, pdir, lab in ((axes[1], args.pred_a, args.label_a), (axes[2], args.pred_b, args.label_b)):
            allch = np.load(os.path.join(pdir, stem + ".npz"))["prob"].astype(np.float32)
            prob = allch[0] if allch.ndim == 3 else allch
            polys, r = run_arm(prob, params, gt, hi)
            pdist = distance_transform_edt(~raster(polys, hi)) if polys else np.full(hi, 1e6)
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            n_missed = 0
            for g in gt:
                cov = frac_near(g, pdist)
                if cov < 0.95:
                    n_missed += 1
                ax.plot(g[:, 0], g[:, 1], color="magenta" if cov < 0.95 else "white", lw=0.6, alpha=0.7 if cov < 0.95 else 0.35)
            n_on = 0
            for p in polys:
                on = frac_near(p, gdist) >= 0.95; n_on += int(on)
                ax.plot(p[:, 0], p[:, 1], color="lime" if on else "red", lw=1.0)
            ax.set_title(f"{lab} @{args.thr}: tp {r['tp']} fp {r['fp']} fn {r['fn']} F1 {r['f1']:.3f}   "
                         f"pred {len(polys)} (on-GT {n_on}, off {len(polys) - n_on}); GT missed {n_missed}", fontsize=10)
        for ax in axes:
            ax.set_xlim(0, hi[1]); ax.set_ylim(hi[0], 0); ax.axis("off")
        plt.tight_layout()
        out = os.path.join(args.out_dir, f"pair_{stem}.png"); plt.savefig(out, dpi=110); plt.close(fig)
        print(stem, "->", out, flush=True)


if __name__ == "__main__":
    main()
