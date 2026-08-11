#!/usr/bin/env python3
"""Semantic (foreground) comparison of segmentation models on MT-34. RUNS ON TULEN.

Reports the project's tolerant centerline-coverage F1 at tol 1/2/5 -- the metric the generator
and v4b were tuned on -- plus the foreground fraction, so a model that buys recall by firing
more is visible rather than flattered. Dice is deliberately absent: it measures how well an
arbitrary GT dilation band is filled and was already found confounded for 2-px filaments.

    ~/dinov3_env/bin/python scripts/semantic_compare.py --split val \\
        v4b=/home/prusek/mt_enc_exp/mt34_pred_fovnorm \\
        nnunet15=/home/prusek/mt_enc_exp/mt34_pred_nnunet_s15
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))
from instance.geometry import resample  # noqa: E402
from instance.oracle import oracle_mask  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402

UP = 1.5


def coverage_f1(pred_mask: np.ndarray, gt_pts: np.ndarray, tol: float) -> tuple[float, float]:
    """(recall, precision) of centerline coverage within ``tol`` px.

    recall    = fraction of GT centerline covered by predicted foreground
    precision = fraction of the predicted SKELETON lying on a GT centerline
    """
    # EVERYTHING here is (row, col), including gt_pts. Mixing (x, y) and (row, col) is the
    # bug this project has a dedicated test for; it silently reports ~0.02 instead of ~0.94.
    ys, xs = np.where(pred_mask)
    if len(ys) == 0 or len(gt_pts) == 0:
        return 0.0, 0.0
    rec = float((cKDTree(np.stack([ys, xs], 1)).query(gt_pts, k=1)[0] <= tol).mean())
    sy, sx = np.where(skeletonize(pred_mask))
    if len(sy) == 0:
        return rec, 0.0
    prec = float((cKDTree(gt_pts).query(np.stack([sy, sx], 1), k=1)[0] <= tol).mean())
    return rec, prec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="name=/path/to/pred_dir")
    ap.add_argument("--data", default="data/real/mt34_eval")
    ap.add_argument("--split", default="val")
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()

    models = dict(m.split("=", 1) for m in args.models)
    acc: dict[tuple[str, str], list] = {}

    for path in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        fr = read_frame_h5(path)
        a = fr["attrs"]
        if args.split != "all" and str(a.get("split")) != args.split:
            continue
        if not fr["polylines"]:
            continue
        task = str(a.get("source_task"))
        shape = (int(a["height"]), int(a["width"]))
        gt_pts = np.concatenate([resample(np.asarray(p, float) * UP, 1.0)
                                 for p in fr["polylines"]])
        gt_pts = np.stack([gt_pts[:, 1], gt_pts[:, 0]], axis=1)      # -> (row, col)
        gt_fg = float(oracle_mask(fr["polylines"], shape, half_width=1.0, up=UP).mean())

        for name, d in models.items():
            f = os.path.join(d, os.path.basename(path).replace(".h5", ".npz"))
            if not os.path.exists(f):
                continue
            prob = np.load(f)["prob"].astype(np.float32)
            m = prob.max(axis=0) > args.thr
            row = {"fg": float(m.mean()), "gt_fg": gt_fg}
            for tol in (1.0, 2.0, 5.0):
                r, p = coverage_f1(m, gt_pts, tol)
                row[f"r{tol:g}"], row[f"p{tol:g}"] = r, p
            acc.setdefault((name, task), []).append(row)

    print(f"\n=== MT-34 {args.split.upper()} semantic (threshold {args.thr}) ===")
    hdr = f"{'model':10s} {'task':6s} {'n':>3s}"
    for tol in (1, 2, 5):
        hdr += f" {'tol' + str(tol) + ' F1':>9s}"
    hdr += f" {'fg%':>6s} {'GTfg%':>6s} {'ratio':>6s}"
    print(hdr)
    for (name, task), rows in sorted(acc.items()):
        label = "Alice" if task == "585" else "new-22"
        line = f"{name:10s} {label:6s} {len(rows):3d}"
        for tol in (1, 2, 5):
            r = np.mean([x[f"r{tol}"] for x in rows])
            p = np.mean([x[f"p{tol}"] for x in rows])
            line += f" {2 * r * p / max(r + p, 1e-9):9.3f}"
        fg = np.mean([x["fg"] for x in rows])
        gf = np.mean([x["gt_fg"] for x in rows])
        line += f" {100 * fg:6.2f} {100 * gf:6.2f} {fg / max(gf, 1e-9):6.2f}"
        print(line)


if __name__ == "__main__":
    main()
