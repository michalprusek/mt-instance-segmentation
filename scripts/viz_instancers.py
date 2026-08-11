#!/usr/bin/env python3
"""Side-by-side render of GT vs PySOAX vs A vs B on one MT-34 frame. RUNS ON TULEN.

One colour per instance, so a microtubule broken at a crossing shows up as a colour change
mid-filament -- the failure the whole exercise is about, made visible rather than tabulated.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "third_party"))

from instance.instancer_a import instance_a  # noqa: E402
from instance.instancer_b import instance_b  # noqa: E402
from instance.oracle import oracle_mask, oracle_ori_channels  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402

UP = 1.5
KAPPA_MAX = 0.25
PYSOAX_PARAMS = {
    "min_snake_length": 17, "gaussian_std": 0.8784, "grouping_distance": 17.1466,
    "direction_threshold": 0.87, "orient_weight": 0.0, "embed_weight": 0.0,
    "ridge_threshold": 0.0354, "stretch_factor": 0.998, "alpha": 0.25, "beta": 0.06,
    "gamma": 4.597, "external_factor": 1.918, "max_iterations": 5000,
    "change_threshold": 0.001, "check_period": 100, "point_spacing": 1.0,
}


def draw(ax, polylines, title, r0, c0, sz):
    cmap = plt.get_cmap("hsv")
    for i, p in enumerate(polylines):
        p = np.asarray(p, float)
        ax.plot(p[:, 0] - c0, p[:, 1] - r0, "-", lw=1.4,
                color=cmap((i * 0.41) % 1.0), alpha=0.95)
    ax.set_xlim(0, sz)
    ax.set_ylim(sz, 0)
    ax.set_facecolor("black")
    ax.axis("off")
    ax.set_title(f"{title}  ({len(polylines)} instances)", fontsize=11)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="training_img_114")
    ap.add_argument("--data", default="data/real/mt34_eval")
    ap.add_argument("--crop", type=int, nargs=3, default=[500, 500, 600],
                    metavar=("R0", "C0", "SIZE"), help="in the 1.5x eval frame")
    ap.add_argument("--out", default="data/enc_sensitivity_testset/instancer_compare.png")
    args = ap.parse_args()
    r0, c0, sz = args.crop

    fr = read_frame_h5(os.path.join(args.data, f"{args.frame}.h5"))
    shape = (int(fr["attrs"]["height"]), int(fr["attrs"]["width"]))
    gt = [np.asarray(p, float) * UP for p in fr["polylines"]]
    chans = oracle_ori_channels(fr["polylines"], shape, K=6, half_width=1.0, up=UP)
    mask = oracle_mask(fr["polylines"], shape, half_width=1.0, up=UP)

    import pysoax
    soax = [np.stack([np.asarray(d["centerline"], float)[:, 1],
                      np.asarray(d["centerline"], float)[:, 0]], axis=1)
            for d in pysoax.extract_soax_instances(mask.astype(np.uint8) * 255,
                                                   PYSOAX_PARAMS)
            if len(d["centerline"]) >= 2]
    pa, _ = instance_a(mask, KAPPA_MAX)
    pb, _ = instance_b(chans, KAPPA_MAX)

    def inside(pls):
        out = []
        for p in pls:
            q = np.asarray(p, float)
            sel = ((q[:, 0] >= c0 - 5) & (q[:, 0] <= c0 + sz + 5)
                   & (q[:, 1] >= r0 - 5) & (q[:, 1] <= r0 + sz + 5))
            if sel.sum() >= 2:
                out.append(q[sel])
        return out

    fig, axes = plt.subplots(1, 4, figsize=(26, 7))
    for ax, (pls, name) in zip(axes, [(gt, "human GT"), (soax, "PySOAX"),
                                      (pa, "A: curvature-bounded matching"),
                                      (pb, "B: orientation-lifted tracing")]):
        draw(ax, inside(pls), name, r0, c0, sz)
    fig.suptitle(f"{args.frame} (oracle foreground) -- one colour per instance; "
                 "a colour change mid-filament is a break", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=95, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
