#!/usr/bin/env python3
"""Montage of the model's OFF-ANNOTATION predictions on raw frames, so a human can say what they are.

Step 2c (2026-09-16) found that the model's off-annotation foreground costs ~0.17 of the 0.27 gap on
roi303 and that those predictions have the contrast of strong annotated filaments. Whether they are
unannotated microtubules (annotation convention: nothing model-side can fix it) or filament-shaped
debris (a model target) is a question for eyes, not statistics. This renders crops.

For each of --n-frames random VAL frames: run the instancer at --thr, keep predictions whose
centerline lies < 50 % within 5 eval px of any GT (the off_other class), sort by length, take the
--per-frame SHORTEST ones above --min-len (the +705 short blobs the derived constant admitted) and
crop --crop px around each. Panel = raw crop (robust 1/99 % stretch) with GT (green) and the
prediction (red) overlaid; a second copy without overlays so the eye is not led.

    PYTHONPATH=src python scripts/viz_offgt_fp.py --pred-dir <dump> --data data/real/roi303_eval \\
        --split val --thr 0.98 --n-frames 12 --per-frame 3 --out offgt_fp.png
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

UP, KAPPA_MAX = 1.5, 0.25


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--data", default="data/real/roi303_eval")
    ap.add_argument("--split", default="val")
    ap.add_argument("--thr", type=float, default=0.98)
    ap.add_argument("--params", default="src/instance/params_a_derived.json")
    ap.add_argument("--n-frames", type=int, default=12)
    ap.add_argument("--per-frame", type=int, default=3)
    ap.add_argument("--min-len", type=float, default=15.0, help="eval px")
    ap.add_argument("--max-len", type=float, default=90.0, help="eval px: the short-blob class")
    ap.add_argument("--crop", type=int, default=96, help="native px")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    params = json.load(open(args.params))

    files = [p for p in sorted(glob.glob(os.path.join(args.data, "*.h5")))]
    frames = []
    for p in files:
        fr = read_frame_h5(p)
        if str(fr["attrs"].get("split")) == args.split and fr["polylines"] and \
                os.path.exists(os.path.join(args.pred_dir, os.path.basename(p).replace(".h5", ".npz"))):
            frames.append((p, fr))
    pick = rng.choice(len(frames), size=min(args.n_frames, len(frames)), replace=False)

    panels = []   # (name, crop_img, gt_segments_in_crop, pred_pts_in_crop, length, on_gt)
    for idx in pick:
        p, fr = frames[idx]
        name = os.path.basename(p)[:-3]
        ch = np.load(os.path.join(args.pred_dir, name + ".npz"))["prob"].astype(np.float32)
        prob = ch.max(axis=0)
        img = np.asarray(fr["image"], float)
        H, W = img.shape
        gt = [resample(np.asarray(q, float), ds=1.0) for q in fr["polylines"] if len(q) >= 2]
        tree = cKDTree(np.concatenate(gt, 0))
        polys, _ = instance_a(prob > args.thr, KAPPA_MAX, params=params, channels=ch, prob=prob)
        cands = []
        for pl in polys:
            pl = np.asarray(pl, float)
            if len(pl) < 2:
                continue
            nat = resample(pl / UP, ds=1.0)
            if len(nat) < 3:
                continue
            L = float(len(nat) * UP)
            on = float((tree.query(nat)[0] <= 5.0 / UP).mean())
            if on < 0.5 and args.min_len <= L <= args.max_len:
                cands.append((L, on, nat))
        cands.sort(key=lambda t: t[0])
        for L, on, nat in cands[:args.per_frame]:
            cy, cx = nat[:, 1].mean(), nat[:, 0].mean()
            h = args.crop // 2
            y0, x0 = int(np.clip(cy - h, 0, H - args.crop)), int(np.clip(cx - h, 0, W - args.crop))
            crop = img[y0:y0 + args.crop, x0:x0 + args.crop]
            gts = [g - [x0, y0] for g in gt]
            panels.append((name, crop, gts, nat - [x0, y0], L, on))

    if not panels:
        raise SystemExit("no off-annotation predictions in the sampled frames")
    n = len(panels)
    cols = 6
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 2.6, rows * 2 * 2.7))
    axes = np.atleast_2d(axes)
    for i, (name, crop, gts, pred, L, on) in enumerate(panels):
        r, c = divmod(i, cols)
        lo, hi = np.percentile(crop, [1, 99])
        for k, ax in enumerate((axes[2 * r, c], axes[2 * r + 1, c])):
            ax.imshow(crop, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if k == 0:
                for g in gts:
                    m = (g[:, 0] >= 0) & (g[:, 0] < crop.shape[1]) & (g[:, 1] >= 0) & (g[:, 1] < crop.shape[0])
                    if m.any():
                        ax.plot(g[m, 0], g[m, 1], "-", color="lime", lw=0.8, alpha=0.8)
                ax.plot(pred[:, 0], pred[:, 1], "-", color="red", lw=1.0, alpha=0.9)
                ax.set_title(f"{name[-7:]} L={L:.0f} onGT={on:.2f}", fontsize=7)
    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[2 * r, c].axis("off"); axes[2 * r + 1, c].axis("off")
    fig.suptitle(f"off-annotation predictions, {args.min_len:.0f}-{args.max_len:.0f} eval px, thr {args.thr} "
                 f"(top: GT green, prediction red; bottom: raw)", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out} with {n} panels")


if __name__ == "__main__":
    main()
