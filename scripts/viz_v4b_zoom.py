#!/usr/bin/env python3
"""The shipped system, readable: v4b foreground -> instancer A, with zoom crops. RUNS ON TULEN.

The four-panel full-frame renders show *how much* is found; at 1024 px squeezed into a panel
they cannot show *what the failures look like*. This adds crops centred on the densest
crossing cluster in each frame, GT next to prediction at the same scale, so a break at a
junction or a merged bundle is visible as a break or a merge rather than as a colour smudge.

The crop is chosen by the data, not by eye: the window containing the most GT crossings.

    cd /home/prusek/mt_enc_exp/mt34_work
    PYTHONPATH=src ~/dinov3_env/bin/python scripts/viz_v4b_zoom.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
import numpy as np
from scipy.ndimage import zoom

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from instance.instancer_a import instance_a  # noqa: E402
from instance.metrics import centerline_f1  # noqa: E402
from instance.oracle import oracle_instance_masks  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.fov import fov_mask  # noqa: E402
from mt_bench.gt_stats import count_crossings  # noqa: E402

UP = 1.5
KAPPA_MAX = 0.25

DEFAULT_FRAMES = ["training_img_114", "training_img_112", "training_img_10",
                  "alice_2026_02_06_pll336_100x_atp_ch1_I__061_f00_irm"]


def _norm01(img):
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape[:2]
    core = img[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
    lo, hi = np.percentile(core, [1.0, 99.0])
    return np.clip((img - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


def densest_window(crossings, shape, size):
    """Window (r0, c0) holding the most GT crossings; frame centre if there are none.

    ``count_crossings`` yields dicts with ``x, y`` in the same (x=col, y=row) convention as
    the polylines, already in the 1.5x frame because it was handed scaled GT.
    """
    H, W = shape
    if not crossings:
        return (H - size) // 2, (W - size) // 2
    pts = np.array([[c["x"], c["y"]] for c in crossings], dtype=float)
    best, best_n = ((H - size) // 2, (W - size) // 2), -1
    for p in pts:
        r0 = int(np.clip(p[1] - size / 2, 0, max(0, H - size)))
        c0 = int(np.clip(p[0] - size / 2, 0, max(0, W - size)))
        n = int(np.sum((pts[:, 0] >= c0) & (pts[:, 0] < c0 + size) &
                       (pts[:, 1] >= r0) & (pts[:, 1] < r0 + size)))
        if n > best_n:
            best, best_n = (r0, c0), n
    return best


def draw(ax, img01, polylines, title, box=None, lw=1.1):
    ax.imshow(img01, cmap="gray", interpolation="nearest")
    cmap = plt.get_cmap("hsv")
    for i, p in enumerate(polylines):
        p = np.asarray(p, dtype=float)
        ax.plot(p[:, 0], p[:, 1], "-", lw=lw, color=cmap((i * 0.37) % 1.0), alpha=0.95)
    if box is not None:
        r0, c0, s = box
        ax.plot([c0, c0 + s, c0 + s, c0, c0], [r0, r0, r0 + s, r0 + s, r0],
                "-", color="white", lw=1.6, alpha=0.9)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/mt34_eval")
    ap.add_argument("--pred-dir", default="/home/prusek/mt_enc_exp/mt34_pred")
    ap.add_argument("--params", default="src/instance/params_a_model_v2.json")
    ap.add_argument("--prob-thr", type=float, default=0.35)
    ap.add_argument("--crop", type=int, default=420, help="zoom window side, in 1.5x px")
    ap.add_argument("--frames", nargs="+", default=DEFAULT_FRAMES)
    ap.add_argument("--out-dir", default="data/enc_sensitivity_testset/v4b_zoom")
    args = ap.parse_args()

    params = json.load(open(args.params))
    params.pop("kappa_max", None)
    os.makedirs(args.out_dir, exist_ok=True)

    for name in args.frames:
        path = os.path.join(args.data, f"{name}.h5")
        if not os.path.exists(path):
            print(f"  skip {name}", flush=True)
            continue
        fr = read_frame_h5(path)
        a = fr["attrs"]
        shape = (int(a["height"]), int(a["width"]))
        hi = (int(round(shape[0] * UP)), int(round(shape[1] * UP)))
        gt = [np.asarray(p, float) * UP for p in fr["polylines"]]
        img01 = zoom(_norm01(fr["image"]), UP, order=1)[:hi[0], :hi[1]]

        npz = os.path.join(args.pred_dir, f"{name}.npz")
        if not os.path.exists(npz):
            print(f"  {name}: no prediction", flush=True)
            continue
        chans = np.load(npz)["prob"].astype(np.float32)
        fv = zoom(fov_mask(fr["image"]).astype(np.uint8), UP, order=0).astype(bool)
        fv = np.pad(fv, ((0, max(0, hi[0] - fv.shape[0])),
                         (0, max(0, hi[1] - fv.shape[1]))),
                    constant_values=True)[:hi[0], :hi[1]]
        chans = chans * fv[None, :, :]
        thr = params.get("prob_thr", args.prob_thr)
        pls, _ = instance_a(chans.max(axis=0) > thr, KAPPA_MAX, params,
                            channels=chans, prob=chans.max(axis=0))

        masks = oracle_instance_masks(pls, hi, half_width=1.0, up=1.0)
        f1 = centerline_f1(masks, gt, tol=5.0,
                           length_coverage=0.95, precision_coverage=0.95)["f1"]

        cross = count_crossings(gt, tol=2.0 * UP, endpoint_margin=8.0 * UP)
        r0, c0 = densest_window(cross, hi, args.crop)
        s = args.crop

        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        draw(axes[0, 0], img01, gt,
             f"{name} -- ground truth, {len(gt)} microtubules", box=(r0, c0, s))
        draw(axes[0, 1], img01, pls,
             f"v4b foreground + instancer A -- {len(pls)} inst, F1={f1:.3f}",
             box=(r0, c0, s))
        sub = img01[r0:r0 + s, c0:c0 + s]
        shift = [np.asarray(p, float) - [c0, r0] for p in gt]
        draw(axes[1, 0], sub, shift, f"zoom: ground truth ({len(cross)} crossings in frame)",
             lw=1.8)
        draw(axes[1, 1], sub, [np.asarray(p, float) - [c0, r0] for p in pls],
             "zoom: prediction", lw=1.8)
        for ax in axes[1]:
            ax.set_xlim(0, s)
            ax.set_ylim(s, 0)
        fig.tight_layout()
        out = os.path.join(args.out_dir, f"{name}.png")
        fig.savefig(out, dpi=105, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
