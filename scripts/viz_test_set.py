#!/usr/bin/env python3
"""Render instancer A's TEST-set output next to GT and the image. RUNS ON TULEN.

Four panels per frame, in the 1.5x eval frame everything else is measured in:

  1. the raw image (display stretch from the CENTRAL crop -- several task-586 frames have a
     saturated surround outside an octagonal field stop that would otherwise black out the
     interior);
  2. ground truth, one colour per annotated microtubule;
  3. **A on the ORACLE foreground** -- the ceiling: what the instancer does when the semantic
     step is perfect;
  4. **A on the v4b predicted foreground** -- the real system end to end.

Panels 3 and 4 side by side are the point. They isolate what is left to gain from the
foreground (0.920 vs 0.416 pooled on TEST) from what is left to gain from the grouping, and
one colour per instance makes the failure legible: a microtubule that changes colour mid-
filament was broken, two filaments sharing a colour were merged.

Each panel is titled with its own strict centerline-F1 on that frame, so the picture cannot
drift from the number.

    cd /home/prusek/mt_enc_exp/mt34_work
    PYTHONPATH=src ~/dinov3_env/bin/python scripts/viz_test_set.py
"""
from __future__ import annotations

import argparse
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
from instance.oracle import (oracle_instance_masks, oracle_ori_channels)  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.fov import fov_mask  # noqa: E402

UP = 1.5
KAPPA_MAX = 0.25

DEFAULT_FRAMES = [
    "training_img_10",                                     # 94 GT -- the densest TEST frame
    "training_img_114",                                    # 61 GT -- crossing-dense
    "training_img_112",                                    # 61 GT
    "alice_2026_02_06_pll336_100x_atp_ch1_I__061_f00_irm",  # 38 GT -- densest Alice
    "alice_2026_02_06_pll338_500x_atp_ch4_I__063_f00_irm",  # 22 GT -- typical Alice
    "training_img_101",                                    # 3 GT -- sparse: an over-firing test
]


def _norm01(img: np.ndarray) -> np.ndarray:
    """Display stretch from the central 60%, which is inside the field stop for every frame."""
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape[:2]
    core = img[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
    lo, hi = np.percentile(core, [1.0, 99.0])
    return np.clip((img - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


def draw(ax, img01, polylines, title):
    ax.imshow(img01, cmap="gray", interpolation="nearest")
    cmap = plt.get_cmap("hsv")
    for i, p in enumerate(polylines):
        p = np.asarray(p, dtype=float)
        # 0.37 is an irrational-ish stride through the hue circle: consecutive instances --
        # which are usually spatial neighbours -- get far-apart colours instead of a gradient.
        ax.plot(p[:, 0], p[:, 1], "-", lw=1.1, color=cmap((i * 0.37) % 1.0), alpha=0.95)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/mt34_eval")
    ap.add_argument("--pred-dir", default="/home/prusek/mt_enc_exp/mt34_pred")
    ap.add_argument("--params-a", default="src/instance/params_a_v2.json")
    ap.add_argument("--params-a-model", default="src/instance/params_a_model_v2.json")
    ap.add_argument("--prob-thr", type=float, default=0.35)
    ap.add_argument("--frames", nargs="+", default=DEFAULT_FRAMES)
    ap.add_argument("--out-dir", default="data/enc_sensitivity_testset/test_viz")
    args = ap.parse_args()

    import json
    pa = json.load(open(args.params_a))
    pam = json.load(open(args.params_a_model))
    for p in (pa, pam):
        p.pop("kappa_max", None)          # kappa_max is derived, never read from a params file
    os.makedirs(args.out_dir, exist_ok=True)

    def f1(polys, gt, hi):
        masks = oracle_instance_masks(polys, hi, half_width=1.0, up=1.0)
        return centerline_f1(masks, gt, tol=5.0,
                             length_coverage=0.95, precision_coverage=0.95)["f1"]

    for name in args.frames:
        path = os.path.join(args.data, f"{name}.h5")
        if not os.path.exists(path):
            print(f"  skip {name}: no such frame", flush=True)
            continue
        fr = read_frame_h5(path)
        a = fr["attrs"]
        shape = (int(a["height"]), int(a["width"]))
        hi = (int(round(shape[0] * UP)), int(round(shape[1] * UP)))
        gt = [np.asarray(p, float) * UP for p in fr["polylines"]]
        img01 = zoom(_norm01(fr["image"]), UP, order=1)[:hi[0], :hi[1]]

        # --- panel 3: oracle foreground (the ceiling) ---
        chans_o = oracle_ori_channels(fr["polylines"], shape, K=6, half_width=1.0, up=UP)
        pls_o, _ = instance_a(chans_o.max(axis=0) > 0.5, KAPPA_MAX, pa,
                             channels=chans_o, prob=chans_o.max(axis=0))

        # --- panel 4: the v4b predicted foreground (the real system) ---
        npz = os.path.join(args.pred_dir, f"{name}.npz")
        pls_m = []
        if os.path.exists(npz):
            chans_m = np.load(npz)["prob"].astype(np.float32)
            fv = zoom(fov_mask(fr["image"]).astype(np.uint8), UP, order=0).astype(bool)
            fv = np.pad(fv, ((0, max(0, hi[0] - fv.shape[0])),
                             (0, max(0, hi[1] - fv.shape[1]))),
                        constant_values=True)[:hi[0], :hi[1]]
            chans_m = chans_m * fv[None, :, :]
            thr = pam.get("prob_thr", args.prob_thr)
            pls_m, _ = instance_a(chans_m.max(axis=0) > thr, KAPPA_MAX, pam,
                                  channels=chans_m, prob=chans_m.max(axis=0))
        else:
            print(f"  {name}: no prediction npz, panel 4 will be empty", flush=True)

        fig, axes = plt.subplots(1, 4, figsize=(26, 7))
        draw(axes[0], img01, [], f"{name}\nimage ({a.get('source_task')})")
        draw(axes[1], img01, gt, f"ground truth -- {len(gt)} microtubules")
        draw(axes[2], img01, pls_o,
             f"A on ORACLE foreground -- {len(pls_o)} inst, F1={f1(pls_o, gt, hi):.3f}")
        draw(axes[3], img01, pls_m,
             f"A on v4b foreground -- {len(pls_m)} inst, F1={f1(pls_m, gt, hi):.3f}"
             if pls_m else "A on v4b foreground -- no prediction")
        fig.tight_layout()
        out = os.path.join(args.out_dir, f"{name}.png")
        fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
