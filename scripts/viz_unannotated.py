#!/usr/bin/env python3
"""Look at the detections that lie away from every annotation. Filaments, or debris?

The audit found 102 of them on MT-34 TEST, carrying 7.66x the local contrast of the same curve
dropped at a random place -- so they sit on real image structure. Whether that structure is a
microtubule the annotator did not draw, or an edge, a scratch or a speck of dirt, decides
something the statistics cannot: if they are filaments, the measurable F1 ceiling is below 1.0
and optimising precision against this benchmark is partly optimising against its gaps.

Each crop shows the raw image with the detection drawn over it, plus the nearest annotated
microtubule for scale, so "does this look like the things the annotator DID draw?" can be
answered by eye. Crops are ordered by contrast, strongest first: if the top ones are debris,
that is the more damaging finding.

    PYTHONPATH=src python scripts/viz_unannotated.py --pred-dir <npz dir> --out <png>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.ndimage import gaussian_filter, zoom  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from instance.geometry import arclength, resample  # noqa: E402
from instance.instancer_a import instance_a  # noqa: E402
from instance.metrics import centerline_f1  # noqa: E402
from instance.oracle import oracle_instance_masks  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.fov import fov_mask  # noqa: E402

UP, KAPPA_MAX, BG_SIGMA = 1.5, 0.25, 12.0


def display01(img):
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape[:2]
    core = img[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
    lo, hi = np.percentile(core, [1.0, 99.0])
    return np.clip((img - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/mt34_eval")
    ap.add_argument("--split", default="test")
    ap.add_argument("--pred-dir", default="/home/prusek/mt_enc_exp/mt34_pred")
    ap.add_argument("--params", default="src/instance/params_a_model_synthtuned.json")
    ap.add_argument("--near-px", type=float, default=8.0)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--crop", type=int, default=150)
    ap.add_argument("--out", default="data/enc_sensitivity_testset/unannotated.png")
    args = ap.parse_args()

    params = json.load(open(args.params))
    params.pop("kappa_max", None)
    thr = params.get("prob_thr", 0.35)

    found = []
    for path in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        fr = read_frame_h5(path)
        a = fr["attrs"]
        if str(a.get("split")) != args.split or not fr["polylines"]:
            continue
        npz = os.path.join(args.pred_dir, os.path.basename(path).replace(".h5", ".npz"))
        if not os.path.exists(npz):
            continue
        ch = np.load(npz)["prob"].astype(np.float32)
        H, W = ch.shape[1:]
        fv = zoom(fov_mask(fr["image"]).astype(np.uint8), UP, order=0).astype(bool)
        fv = np.pad(fv, ((0, max(0, H - fv.shape[0])), (0, max(0, W - fv.shape[1]))),
                    constant_values=True)[:H, :W]
        ch = ch * fv[None]
        prob = ch.max(axis=0)
        pls, _ = instance_a(prob > thr, KAPPA_MAX, params, channels=ch, prob=prob)
        if not pls:
            continue
        gt = [np.asarray(p, float) * UP for p in fr["polylines"]]
        masks = oracle_instance_masks(pls, (H, W), half_width=1.0, up=1.0)
        res = centerline_f1(masks, gt, tol=5.0, length_coverage=0.95, precision_coverage=0.95)
        matched = {p["pred"] for p in res["pairs"] if p["tp"]}

        raw = zoom(np.asarray(fr["image"], float), UP, order=1)[:H, :W]
        disp = zoom(display01(fr["image"]), UP, order=1)[:H, :W]
        resid = np.abs(raw - gaussian_filter(raw, BG_SIGMA))
        gt_tree = cKDTree(np.concatenate([resample(g, ds=1.0) for g in gt]))

        for i, pl in enumerate(pls):
            if i in matched:
                continue
            q2 = resample(np.asarray(pl, float), ds=2.0)
            if float(np.median(gt_tree.query(q2, k=1)[0])) <= args.near_px:
                continue
            q = resample(np.asarray(pl, float), ds=1.0)
            r = np.clip(np.rint(q[:, 1]).astype(int), 0, H - 1)
            c = np.clip(np.rint(q[:, 0]).astype(int), 0, W - 1)
            found.append({"contrast": float(np.median(resid[r, c])),
                          "len": float(arclength(q)[-1]), "poly": q,
                          "disp": disp, "gt": gt,
                          "name": os.path.basename(path).replace(".h5", "")})
    if not found:
        raise SystemExit("no unannotated detections found")
    found.sort(key=lambda d: -d["contrast"])
    sel = found[:args.n]
    print(f"{len(found)} detections away from any annotation; showing the {len(sel)} strongest",
          flush=True)

    ncol = 6
    nrow = int(np.ceil(len(sel) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.25 * nrow))
    axes = np.atleast_2d(axes)
    z = args.crop
    for k, d in enumerate(sel):
        ax = axes[k // ncol][k % ncol]
        p = d["poly"]
        cx, cy = p[:, 0].mean(), p[:, 1].mean()
        Hd, Wd = d["disp"].shape
        r0 = int(np.clip(cy - z / 2, 0, max(0, Hd - z)))
        c0 = int(np.clip(cx - z / 2, 0, max(0, Wd - z)))
        ax.imshow(d["disp"][r0:r0 + z, c0:c0 + z], cmap="gray", interpolation="nearest")
        # every annotated microtubule crossing this crop, for scale and appearance
        for g in d["gt"]:
            gg = np.asarray(g, float) - [c0, r0]
            m = (gg[:, 0] > -20) & (gg[:, 0] < z + 20) & (gg[:, 1] > -20) & (gg[:, 1] < z + 20)
            if m.sum() > 3:
                ax.plot(gg[m, 0], gg[m, 1], "-", color="#31d843", lw=1.6, alpha=0.85)
        ax.plot(p[:, 0] - c0, p[:, 1] - r0, "-", color="#ff2d2d", lw=1.6)
        ax.set_title(f"contrast {d['contrast']:.1f} · {d['len']:.0f} px", fontsize=8)
        ax.set_xlim(0, z)
        ax.set_ylim(z, 0)
        ax.axis("off")
    for k in range(len(sel), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("RED = detection with no annotation nearby   ·   GREEN = annotated microtubules",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=105, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
