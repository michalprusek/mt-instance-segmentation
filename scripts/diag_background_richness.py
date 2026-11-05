#!/usr/bin/env python3
"""Are the generator's backgrounds cleaner than the background AROUND real filaments?

The ordering gap is now localised: in domain the map's 0.5-0.97 band is 97.4 % on-GT, on real
66.5 %, and only 13 % of the real off-GT mass is the known edge-band artefact. So the model
over-scores something real that it never had to reject during training.

The generator composites synthetic filaments onto REAL empty fields, so its negatives are real
images -- but an empty field is not the same as the neighbourhood of a real assay: filament-bearing
frames carry debris, out-of-focus filaments, rims and drift artefacts an empty field may not have.
If so, the model is never taught to reject them and scores them in exactly the extractable band.

Compares three populations of OFF-FILAMENT pixels (further than `--exclude` px from any centerline):
  real     -- roi303 frames, off human-annotated centerlines
  empty    -- the generator's background set (whole frame; there are no filaments)
  synth    -- generator renders, off the exact synthetic centerlines
on statistics that measure STRUCTURE rather than amplitude, so they survive normalisation:
high-pass energy, the lag-1 autocorrelation of the high-pass residual, its kurtosis, and the
share of pixels on an elongated ridge (Hessian-based, the shape the network is tuned to fire on).

    PYTHONPATH=src:scripts python scripts/diag_background_richness.py \
        --real data/real/roi303_eval --empty data/real/irm_backgrounds_v2 \
        --synth data/synth_eval_s35 --out bg.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from instance.oracle import oracle_instance_masks  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402


def norm(img):
    """Percentile normalisation, the same shape of scaling the pipeline uses.

    ⚠ THE SUPPORT IS PART OF THE MEASUREMENT. `np.percentile` over a 2048² real frame and over a
    768² render return different spans, so the SAME pixels normalise to different amplitudes:
    measured on 60 raw 16-bit roi303 frames, the high-pass std of identical real pixels is 0.0228
    at full-frame support and 0.1077 at 768 support — a factor of 3.2 at the median and 6 at q75.
    Comparing a whole real frame against a whole render therefore measures the FRAME SIZE, not the
    imagery. That error reversed the sign of arm 42's premise and cost 12 GPU-hours; use
    `matched_support()` below for any cross-population comparison.
    """
    lo, hi = np.percentile(img, [1, 99])
    return np.clip((img.astype(float) - lo) / (hi - lo + 1e-9), 0, 1)


def matched_support(img, side, mask=None):
    """Centre-crop to `side` BEFORE normalising, so two populations are comparable.

    Returns (normalised crop, cropped mask or None). Frames smaller than `side` are returned whole
    and the caller must drop them rather than pad — padding changes the percentiles too.
    """
    h, w = img.shape
    if min(h, w) < side:
        return None, None
    y, x = (h - side) // 2, (w - side) // 2
    sl = (slice(y, y + side), slice(x, x + side))
    return norm(img[sl]), (None if mask is None else mask[sl])


def ridgeness(x, sigma=1.5):
    """Share of pixels lying on an ELONGATED dark-or-bright ridge (polarity-agnostic).

    The network fires on elongated structure; a background that carries a lot of it is a
    background the model must be taught to reject, and an empty field may simply not have any.
    """
    g = gaussian_filter(x, sigma)
    gy, gx = np.gradient(g)
    gyy, gyx = np.gradient(gy)
    gxy, gxx = np.gradient(gx)
    tr = gxx + gyy
    det = gxx * gyy - gxy * gyx
    disc = np.sqrt(np.maximum(tr * tr / 4 - det, 0))
    l1, l2 = tr / 2 + disc, tr / 2 - disc          # |l1| >= |l2| not guaranteed; fix below
    big = np.where(np.abs(l1) >= np.abs(l2), l1, l2)
    small = np.where(np.abs(l1) >= np.abs(l2), l2, l1)
    aniso = np.abs(big) / (np.abs(small) + 1e-6)
    return (np.abs(big) > 0.02) & (aniso > 3.0)


def stats(x, m):
    """Structure statistics of the masked region: amplitude-free where possible."""
    hp = x - gaussian_filter(x, 3.0)
    v = hp[m]
    if v.size < 1000:
        return None
    a = hp[:, :-1][m[:, :-1]]
    b = hp[:, 1:][m[:, :-1]]
    acf = float(np.corrcoef(a, b)[0, 1]) if a.size > 1000 else float("nan")
    return {"hp_std": float(v.std()),
            "hp_kurtosis": float(((v - v.mean()) ** 4).mean() / (v.var() ** 2 + 1e-12) - 3),
            "hp_lag1_acf": acf,
            "ridge_share": float((ridgeness(x) & m).sum() / max(m.sum(), 1))}


def collect(files, exclude, kind, limit):
    out = []
    for p in sorted(files)[:limit or None]:
        try:
            if kind == "empty":
                import imageio.v3 as iio
                img = np.asarray(iio.imread(p), float) if not p.endswith(".npy") else np.load(p)
                x = norm(img)
                m = np.ones(x.shape, bool)
            else:
                fr = read_frame_h5(p)
                x = norm(np.asarray(fr["image"], float))
                gt = [np.asarray(q, float) for q in fr["polylines"]]
                core = np.zeros(x.shape, bool)
                for mm in oracle_instance_masks(gt, x.shape, half_width=1.0, up=1.0):
                    core |= mm
                m = distance_transform_edt(~core) > exclude
            s = stats(x, m)
            if s:
                out.append(s)
        except Exception as e:                       # a malformed frame must not kill the census
            print(f"  skip {os.path.basename(p)}: {type(e).__name__}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--empty", default="")
    ap.add_argument("--synth", default="")
    ap.add_argument("--exclude", type=float, default=10.0)
    ap.add_argument("--support", type=int, default=0,
                    help="centre-crop every frame to this side BEFORE normalising. 0 keeps the "
                         "old whole-frame behaviour, which is only valid within one population. "
                         "Set it to the render size (768) for any real-vs-synth comparison.")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pops = {"real_off_filament": (sorted(glob.glob(os.path.join(args.real, "*.h5"))), "h5")}
    if args.empty:
        f = sorted(glob.glob(os.path.join(args.empty, "*")))
        pops["generator_backgrounds"] = ([x for x in f if not x.endswith(".json")], "empty")
    if args.synth:
        pops["synth_off_filament"] = (sorted(glob.glob(os.path.join(args.synth, "*.h5"))), "h5")

    summ = {}
    for name, (files, kind) in pops.items():
        print(f"--- {name}: {len(files)} files", flush=True)
        rows = collect(files, args.exclude, kind, args.limit)
        if not rows:
            summ[name] = {"n": 0}
            continue
        summ[name] = {"n": len(rows)}
        for k in rows[0]:
            v = np.array([r[k] for r in rows], float)
            v = v[np.isfinite(v)]
            summ[name][k] = {q: round(float(np.percentile(v, int(q))), 5) for q in ("25", "50", "75")}
    json.dump(summ, open(args.out, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
