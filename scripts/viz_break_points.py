#!/usr/bin/env python3
"""Montage of MID-FILAMENT break points of residual fragments under the "band + nothing off the
annotation" ceiling: what does the model's near-filament foreground look like where the fixed
assembler breaks a filament it traverses whole on the clean band?

Per panel (crop --crop native px around a fragment end classified 'mid' by diag_break_locus.py's
rule): left = raw frame with GT (green) and the fragment (red, its break end marked); middle =
the map the instancer saw (band + kept model foreground, thresholded) with the break end;
right = the model's ORIGINAL probability map (before painting / deleting), same crop, so the eye
can see the spur that made the break.

    PYTHONPATH=src python scripts/viz_break_points.py --ceil-dir preds/CEIL_hb_all_delfar5 \\
        --orig-dir preds/SPARSE35_hb_ep40_native --data data/real/mt34_htw_eval --split val \\
        --thr 0.995 --prefix htw_ --n 18 --out brk_htw.png
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from instance.geometry import resample  # noqa: E402
from instance.instancer_a import instance_a  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.gt_stats import count_crossings  # noqa: E402

UP, KAPPA_MAX, TOL = 1.5, 0.25, 5.0


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--ceil-dir", required=True)
    ap.add_argument("--orig-dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--thr", type=float, required=True)
    ap.add_argument("--params", default="src/instance/params_a_derived.json")
    ap.add_argument("--cov", type=float, default=0.80)
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--crop", type=int, default=48, help="native px")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.split == "test":
        raise SystemExit("VAL only")
    params = json.load(open(args.params))
    rng = np.random.default_rng(args.seed)
    panels = []
    files = sorted(glob.glob(os.path.join(args.data, "*.h5")))
    rng.shuffle(files)
    for path in files:
        if len(panels) >= args.n * 3:
            break
        name = os.path.basename(path)
        if args.prefix and not name.startswith(args.prefix):
            continue
        fr = read_frame_h5(path)
        if str(fr["attrs"].get("split")) != args.split or not fr["polylines"]:
            continue
        npz = os.path.join(args.ceil_dir, name.replace(".h5", ".npz"))
        onpz = os.path.join(args.orig_dir, name.replace(".h5", ".npz"))
        if not (os.path.exists(npz) and os.path.exists(onpz)):
            continue
        ch = np.load(npz)["prob"].astype(np.float32)
        prob = ch.max(axis=0)
        oprob = np.load(onpz)["prob"].astype(np.float32).max(axis=0)
        img = np.asarray(fr["image"], float)
        gts = [resample(np.asarray(q, float), ds=1.0) * UP for q in fr["polylines"] if len(q) >= 2]
        gts = [g for g in gts if len(g) >= 2]
        if not gts:
            continue
        gt_pts = np.concatenate(gts, 0)
        gt_id = np.concatenate([np.full(len(g), i) for i, g in enumerate(gts)])
        tree = cKDTree(gt_pts)
        end_tree = cKDTree(np.array([g[0] for g in gts] + [g[-1] for g in gts]))
        cr = count_crossings([np.asarray(q, float) for q in fr["polylines"]], tol=2.0, ds=1.0, endpoint_margin=8.0)
        cr_tree = cKDTree(np.array([[c["x"] * UP, c["y"] * UP] for c in cr])) if cr else None
        polys, _ = instance_a(prob > args.thr, KAPPA_MAX, params=params, channels=ch, prob=prob)
        polys = [resample(np.asarray(p, float), ds=1.0) for p in polys if len(p) >= 2]
        for pl in polys:
            if len(pl) < 2:
                continue
            d, idx = tree.query(pl)
            on = d <= TOL
            if on.mean() < 0.95:
                continue
            g = int(np.bincount(gt_id[idx[on]]).argmax())
            gcov = float((cKDTree(pl).query(gts[g])[0] <= TOL).mean())
            if gcov >= args.cov:
                continue
            for e in (pl[0], pl[-1]):
                if end_tree.query(e)[0] <= 10.0:
                    continue
                if cr_tree is not None and cr_tree.query(e)[0] <= 15.0:
                    continue
                panels.append((name, img, prob > args.thr, oprob, gts, polys, pl, e))
                break
    if not panels:
        raise SystemExit("no mid-filament fragment ends found")
    panels = panels[:args.n]
    n = len(panels)
    cols = 3
    fig, axes = plt.subplots(n, cols, figsize=(cols * 3.0, n * 3.0))
    axes = np.atleast_2d(axes)
    h = args.crop // 2
    for i, (name, img, mask, oprob, gts, polys, pl, e) in enumerate(panels):
        H, W = img.shape
        cx, cy = e[0] / UP, e[1] / UP
        x0, y0 = int(np.clip(cx - h, 0, W - args.crop)), int(np.clip(cy - h, 0, H - args.crop))
        crop = img[y0:y0 + args.crop, x0:x0 + args.crop]
        X0, Y0 = int(round(x0 * UP)), int(round(y0 * UP)); CE = int(round(args.crop * UP))
        mcrop = mask[Y0:Y0 + CE, X0:X0 + CE]
        ocrop = oprob[Y0:Y0 + CE, X0:X0 + CE]
        lo, hi = np.percentile(crop, [1, 99])
        ax = axes[i, 0]
        ax.imshow(crop, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        for g in gts:
            q = g / UP - [x0, y0]
            m = (q[:, 0] >= 0) & (q[:, 0] < args.crop) & (q[:, 1] >= 0) & (q[:, 1] < args.crop)
            if m.any():
                ax.plot(q[m, 0], q[m, 1], "-", color="lime", lw=0.8, alpha=0.8)
        for p in polys:
            q = p / UP - [x0, y0]
            m = (q[:, 0] >= 0) & (q[:, 0] < args.crop) & (q[:, 1] >= 0) & (q[:, 1] < args.crop)
            if m.any():
                ax.plot(q[m, 0], q[m, 1], "-", color="orange", lw=0.7, alpha=0.7)
        q = pl / UP - [x0, y0]
        ax.plot(q[:, 0], q[:, 1], "-", color="red", lw=1.0)
        ax.plot([e[0] / UP - x0], [e[1] / UP - y0], "o", mfc="none", mec="red", ms=9)
        ax.set_title(f"{name[:-3]}  raw + GT / preds", fontsize=7)
        ax = axes[i, 1]
        ax.imshow(mcrop, cmap="gray", interpolation="nearest")
        ax.plot([e[0] - X0], [e[1] - Y0], "o", mfc="none", mec="red", ms=9)
        ax.set_title("map the assembler saw (band + kept fg)", fontsize=7)
        ax = axes[i, 2]
        ax.imshow(ocrop, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        ax.plot([e[0] - X0], [e[1] - Y0], "o", mfc="none", mec="cyan", ms=9)
        ax.set_title("model p (original)", fontsize=7)
        for a in axes[i]:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle("mid-filament break ends of residual fragments (band painted, off-annotation deleted)", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out} with {n} panels")


if __name__ == "__main__":
    main()
