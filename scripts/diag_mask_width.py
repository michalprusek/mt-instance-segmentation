#!/usr/bin/env python3
"""Perpendicular WIDTH of the thresholded foreground across annotated centerlines: is the model's
map near a filament uniformly thicker than the 2-px band it was trained on, or lumpy?

Advisor 2026-09-16: the residual under the "band + nothing off the annotation" ceiling minus the
same-path oracle's is fragments, not junctions (htw), i.e. the model's near-filament foreground
spawns spurs the fixed assembler breaks on. This measures the shape of that foreground directly.

For every GT polyline (resampled 1 native px, taken to eval scale), at each sample the mask is read
along the normal at 0.5-px steps within +-HALF eval px; width = length of the contiguous run of
mask pixels containing offset 0 (0 if the centerline pixel itself is off). Reported per set:
  on-mask fraction, width p10/p50/p90 over on-mask samples, fraction of samples with width >= 6
  (lumps), and LUMPINESS = fraction of on-mask samples whose width differs from the median width
  of its own 20-px window by more than 2 px. The same statistics on the painted oracle band
  (half_width 1.0) give the reference the model was trained toward.

    PYTHONPATH=src python scripts/diag_mask_width.py --pred-dir <dump> --data <h5 dir> --split val \\
        --thr 0.98 --out width.json [--prefix htw_]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from instance.geometry import resample  # noqa: E402
from instance.oracle import oracle_ori_channels  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402

UP = 1.5
HALF = 12.0
STEP = 0.5


def normals(pts):
    d = np.gradient(pts, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], axis=1)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)


def widths(mask, pts_eval):
    H, W = mask.shape
    nrm = normals(pts_eval)
    offs = np.arange(-HALF, HALF + 1e-9, STEP)
    p = pts_eval[:, None, :] + nrm[:, None, :] * offs[None, :, None]
    cc = np.clip(np.rint(p[..., 0]).astype(int), 0, W - 1)
    rr = np.clip(np.rint(p[..., 1]).astype(int), 0, H - 1)
    m = mask[rr, cc]                      # (n, n_offs)
    c = len(offs) // 2
    out = np.zeros(len(pts_eval), float)
    for i in range(len(pts_eval)):
        if not m[i, c]:
            continue
        lo = c
        while lo > 0 and m[i, lo - 1]:
            lo -= 1
        hi = c
        while hi < len(offs) - 1 and m[i, hi + 1]:
            hi += 1
        out[i] = (hi - lo + 1) * STEP
    return out


def summarize(w_all):
    w = np.concatenate(w_all) if w_all else np.zeros(0)
    on = w > 0
    lump = []
    for wi in w_all:
        for k in range(0, len(wi), 20):
            seg = wi[k:k + 20]
            seg = seg[seg > 0]
            if len(seg) >= 5:
                lump.append(np.abs(seg - np.median(seg)) > 2.0)
    lump = np.concatenate(lump) if lump else np.zeros(0, bool)
    return {"n": int(w.size), "on_mask": float(on.mean()) if w.size else float("nan"),
            "p10": float(np.percentile(w[on], 10)) if on.any() else float("nan"),
            "p50": float(np.percentile(w[on], 50)) if on.any() else float("nan"),
            "p90": float(np.percentile(w[on], 90)) if on.any() else float("nan"),
            "frac_ge6": float((w[on] >= 6).mean()) if on.any() else float("nan"),
            "lumpiness": float(lump.mean()) if lump.size else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--thr", type=float, required=True)
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.split == "test":
        raise SystemExit("VAL only")
    w_model, w_band = [], []
    n_frames = 0
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
        prob = np.load(npz)["prob"].astype(np.float32).max(axis=0)
        H, W = prob.shape
        mask = prob > args.thr
        shape_native = (int(fr["attrs"]["height"]), int(fr["attrs"]["width"]))
        band = oracle_ori_channels(fr["polylines"], shape_native, K=6, half_width=1.0, up=UP).max(axis=0) > 0.5
        band = np.pad(band, ((0, max(0, H - band.shape[0])), (0, max(0, W - band.shape[1]))))[:H, :W]
        for q in fr["polylines"]:
            q = np.asarray(q, float)
            if len(q) < 2:
                continue
            pts = resample(q, ds=1.0) * UP
            if len(pts) < 3:
                continue
            w_model.append(widths(mask, pts))
            w_band.append(widths(band, pts))
        n_frames += 1
    res = {"frames": n_frames, "thr": args.thr, "pred_dir": args.pred_dir, "prefix": args.prefix,
           "model": summarize(w_model), "band": summarize(w_band)}
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    for k in ("model", "band"):
        r = res[k]
        print(f"{k:<6} n={r['n']} on_mask={r['on_mask']:.3f} width p10/p50/p90={r['p10']:.1f}/{r['p50']:.1f}/{r['p90']:.1f} "
              f"frac>=6px={r['frac_ge6']:.3f} lumpiness={r['lumpiness']:.3f}")


if __name__ == "__main__":
    main()
