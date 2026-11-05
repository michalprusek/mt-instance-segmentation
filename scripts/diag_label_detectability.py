#!/usr/bin/env python3
"""Is the generator labelling microtubules the network cannot possibly see -- and does that get
WORSE for short filaments?

The only intervention that ever produced a significant gain in this project (+0.085, p=0.001) was
fixing exactly this defect at the population level: synthetic labelled microtubules carried a third
of the contrast-to-noise of real ones, so a third of training positives were unlearnable. That fix
was applied to the MARGINAL distribution. This script asks the stratified question the marginal one
cannot: several of the generator's appearance mechanisms are specified in ABSOLUTE pixels --
``detach_len_range`` is 10-55 px of dimmed filament, and the along-length intensity modulation has a
fixed correlation length -- while a short microtubule is only ~25 px long in total. On a long
filament a 30 px dim stretch is a blemish; on a short one it is the whole object.

If that holds, the frame-level length prior added a population of filaments that are labelled but
invisible, which would explain why it measured as a null.

Output: per-microtubule, the fraction of its own labelled centerline whose local contrast-to-noise
falls below a detectability floor, binned by contour length.

    python scripts/diag_label_detectability.py --calib calib.json --n 60
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import glob

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ("synth", "scripts", "src"):
    sys.path.insert(0, os.path.join(ROOT, p))


def densify(poly, step=1.0):
    seg = np.sqrt(((poly[1:] - poly[:-1]) ** 2).sum(1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return poly
    t = np.arange(0.0, s[-1], step)
    return np.stack([np.interp(t, s, poly[:, 0]), np.interp(t, s, poly[:, 1])], axis=1)


def local_cnr(img, poly, noise, halfwin=7.0):
    """Per-point |filament - flanking background| / noise, along one centerline.

    Local rather than per-object on purpose: a filament whose middle is bright and whose ends are
    invisible has a fine average and two unlearnable ends, and the average hides that.
    """
    q = densify(poly)
    if len(q) < 5:
        return None
    tang = np.gradient(q, axis=0)
    nrm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    H, W = img.shape

    def take(pts):
        c = np.round(pts[:, 0]).astype(int)
        r = np.round(pts[:, 1]).astype(int)
        ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
        out = np.full(len(pts), np.nan)
        out[ok] = img[r[ok], c[ok]]
        return out

    core = take(q)
    bg = np.nanmean(np.stack([take(q + nrm * halfwin), take(q - nrm * halfwin)]), axis=0)
    val = np.abs(core - bg) / max(noise, 1e-9)
    return val[np.isfinite(val)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", nargs="+", required=True,
                    help="one or more calibration json files to compare")
    ap.add_argument("--bg-dir", default="data/real/irm_backgrounds_v2")
    ap.add_argument("--n", type=int, default=60, help="frames per calibration")
    ap.add_argument("--crop", type=int, default=640)
    ap.add_argument("--floor", type=float, default=1.0,
                    help="CNR below which a labelled pixel carries no learnable evidence")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from gen_train import build_cfg
    from mt_generator import generate_frame

    bgs = sorted(glob.glob(os.path.join(args.bg_dir, "*.tif")))
    if not bgs:
        raise SystemExit(f"no backgrounds under {args.bg_dir}")
    from PIL import Image

    BINS = [(0, 30), (30, 60), (60, 120), (120, 250), (250, 500), (500, 1e9)]
    report = {}
    for calib in args.calib:
        cal = json.load(open(calib))
        cfg = build_cfg(cal["best_params"], mask_hw=1.0)
        if cal.get("gen_overrides"):
            cfg = dataclasses.replace(cfg, **cal["gen_overrides"])
        rows = []
        for i in range(args.n):
            rng = np.random.default_rng(90000 + i)
            bg = np.asarray(Image.open(bgs[rng.integers(len(bgs))]), float)
            while bg.ndim > 2:
                bg = bg[..., 0]
            H, W = bg.shape
            c = args.crop
            bg = bg[H // 2 - c // 2:H // 2 + c // 2, W // 2 - c // 2:W // 2 + c // 2]
            img, inst, _meta = generate_frame(bg, rng, cfg)
            img = np.asarray(img, float)
            # noise from the residual against a heavily smoothed copy, so structure does not
            # inflate it the way a plain std would
            from scipy.ndimage import median_filter
            noise = float(np.median(np.abs(img - median_filter(img, 5))) * 1.4826)
            for ins in inst:
                cl = np.asarray(ins["centerline"], float)
                keep = ((cl[:, 0] >= 0) & (cl[:, 0] < c) & (cl[:, 1] >= 0) & (cl[:, 1] < c))
                if keep.sum() < 5:
                    continue
                cl = cl[keep]
                L = float(np.sqrt(((cl[1:] - cl[:-1]) ** 2).sum(1)).sum())
                v = local_cnr(img, cl, noise)
                if v is None or len(v) < 5:
                    continue
                rows.append((L, float((v < args.floor).mean()), float(np.median(v))))
        A = np.array(rows)
        report[os.path.basename(calib)] = dict(
            n=len(A),
            overall_dead_frac=float(A[:, 1].mean()),
            bins={f"{lo}-{hi if hi < 1e8 else 'inf'}": dict(
                n=int(((A[:, 0] >= lo) & (A[:, 0] < hi)).sum()),
                dead_frac=float(A[(A[:, 0] >= lo) & (A[:, 0] < hi), 1].mean())
                if ((A[:, 0] >= lo) & (A[:, 0] < hi)).any() else None,
                median_cnr=float(np.median(A[(A[:, 0] >= lo) & (A[:, 0] < hi), 2]))
                if ((A[:, 0] >= lo) & (A[:, 0] < hi)).any() else None,
            ) for lo, hi in BINS})

        print(f"\n=== {os.path.basename(calib)} ===  {len(A)} microtubules, "
              f"{A[:,1].mean()*100:.1f} % of labelled centerline below CNR {args.floor}")
        print(f'{"length bin (px)":>16s} {"n":>6s} {"% of its OWN label below the floor":>36s} '
              f'{"median CNR":>11s}')
        for lo, hi in BINS:
            m = (A[:, 0] >= lo) & (A[:, 0] < hi)
            if not m.any():
                continue
            lbl = f"{lo}-{'inf' if hi > 1e8 else int(hi)}"
            print(f'{lbl:>16s} {m.sum():6d} {A[m,1].mean()*100:35.1f}% '
                  f'{np.median(A[m,2]):11.2f}')

    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
