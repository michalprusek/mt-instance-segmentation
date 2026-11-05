#!/usr/bin/env python3
"""Does a filament's CONTRAST vary along its own length -- in the image, not in the model?

`diag_model_only.py` (2026-09-11) found that the median per-curve MINIMUM predicted probability is
0.94 in domain at every length and 0.51-0.87 on real data, while the per-curve MEAN is 0.95-0.99
everywhere. That says the real failure is a local dip rather than global faintness. Before any
generator change is declared, the premise has to be checked WITHOUT the model: is there a dip in
the raw image, or is the model inventing it?

Measured per GT curve, model-free:

  * contrast profile along arc length -- at each sample the PERPENDICULAR MAXIMUM of
    |I - background| within +-`perp` px, so a ground truth drawn a pixel or two off the filament
    cannot masquerade as fading. `background` is the median of the raw frame in a perpendicular
    annulus at `bg_lo`..`bg_hi` px, recomputed per sample, so an uneven illumination field does
    not enter.
  * the statistics that matter for the dip hypothesis: min / median of the profile, the fraction
    of arc length below half the curve's own median, and the number of distinct dips.

If synthetic filaments are flat (min/median near 1) and real ones are not, the generator is
missing along-filament contrast heterogeneity and that is a declared, falsifiable defect. If BOTH
are flat, the dip is the model's and the generator is not the place to fix it.

    PYTHONPATH=src:scripts python scripts/diag_along_filament.py \
        --data data/real/mt34_htw_eval --split val --out along_htw.json
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

from mt_bench.cvat_import import read_frame_h5  # noqa: E402

LEN_BINS = ((0, 20), (20, 30), (30, 53), (53, 107), (107, 213), (213, 10 ** 9))  # native px


def source_of(name: str) -> str:
    if name.startswith("htw_"):
        return "htw"
    if name.startswith("alice_"):
        return "alice"
    if name.startswith("training_img_"):
        return "roi303-derived"
    if name.startswith("roi_img_"):
        return "roi303"
    return "synth"


def densify(poly: np.ndarray, step: float = 1.0) -> np.ndarray:
    d = np.hypot(*np.diff(poly, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 0:
        return poly[:1]
    t = np.arange(0.0, s[-1] + 1e-9, step)
    return np.stack([np.interp(t, s, poly[:, 0]), np.interp(t, s, poly[:, 1])], axis=1)


def normals(pts: np.ndarray) -> np.ndarray:
    d = np.gradient(pts, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], axis=1)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)


def profile(img: np.ndarray, poly: np.ndarray, perp: float, bg_lo: float, bg_hi: float):
    """Background-subtracted |contrast| along the curve, perpendicular-max within +-perp px."""
    H, W = img.shape
    pts = densify(poly)
    if len(pts) < 4:
        return None
    nrm = normals(pts)
    offs_sig = np.arange(-perp, perp + 1e-9, 0.5)
    offs_bg = np.concatenate([np.arange(-bg_hi, -bg_lo + 1e-9, 1.0),
                              np.arange(bg_lo, bg_hi + 1e-9, 1.0)])

    def sample(off):
        p = pts[:, None, :] + nrm[:, None, :] * off[None, :, None]
        cc = np.clip(np.rint(p[..., 0]).astype(int), 0, W - 1)
        rr = np.clip(np.rint(p[..., 1]).astype(int), 0, H - 1)
        return img[rr, cc]

    bg = np.median(sample(offs_bg), axis=1)
    sig = sample(offs_sig) - bg[:, None]
    # IRM polarity is per-frame and can even flip within a frame; |contrast| is the honest read.
    return np.abs(sig).max(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--perp", type=float, default=2.0, help="native px; kills annotation drift")
    ap.add_argument("--bg-lo", type=float, default=6.0)
    ap.add_argument("--bg-hi", type=float, default=12.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    recs: list[dict] = []
    n_frames: dict[str, int] = {}
    for path in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        name = os.path.basename(path)
        fr = read_frame_h5(path)
        if args.split != "all" and str(fr["attrs"].get("split", "")) != args.split:
            continue
        img = np.asarray(fr["image"], np.float32)
        # Per-frame robust scale only; no percentile stretch, so a bright rim cannot rescale the
        # filaments' contrast the way the training normalisation does.
        src = source_of(name)
        n_frames[src] = n_frames.get(src, 0) + 1
        for poly in fr["polylines"]:
            poly = np.asarray(poly, float)
            if len(poly) < 2:
                continue
            L = float(np.hypot(*np.diff(poly, axis=0).T).sum())
            if L < 8:
                continue
            c = profile(img, poly, args.perp, args.bg_lo, args.bg_hi)
            if c is None or len(c) < 8:
                continue
            med = float(np.median(c))
            if med <= 0:
                continue
            r = c / med
            below = r < 0.5
            # distinct dips = maximal runs below half the curve's own median
            ndip = int(np.sum(np.diff(below.astype(int)) == 1) + (1 if below[0] else 0))
            recs.append({"src": src, "len": L,
                         "min_over_med": float(r.min()),
                         "p05_over_med": float(np.percentile(r, 5)),
                         "cv": float(c.std() / med),
                         "frac_below_half": float(below.mean()),
                         "n_dips": ndip,
                         "contrast_med": med})
        if args.limit and sum(n_frames.values()) >= args.limit:
            break

    out = {"data": args.data, "split": args.split, "perp": args.perp,
           "n_frames": n_frames, "blocks": {}}
    for src in sorted({r["src"] for r in recs}):
        rs = [r for r in recs if r["src"] == src]
        def med(k):
            return float(np.median([r[k] for r in rs]))
        blk = {"n_curves": len(rs),
               "min_over_med_median": med("min_over_med"),
               "p05_over_med_median": med("p05_over_med"),
               "cv_median": med("cv"),
               "frac_below_half_median": med("frac_below_half"),
               "frac_below_half_mean": float(np.mean([r["frac_below_half"] for r in rs])),
               "n_dips_median": med("n_dips"),
               "curves_with_a_dip": float(np.mean([r["n_dips"] > 0 for r in rs])),
               "by_length": {}}
        for lo, hi in LEN_BINS:
            sel = [r for r in rs if lo <= r["len"] < hi]
            if not sel:
                continue
            blk["by_length"][f"{lo}-{hi if hi < 10**9 else 'inf'}"] = {
                "n": len(sel),
                "min_over_med": float(np.median([r["min_over_med"] for r in sel])),
                "frac_below_half": float(np.mean([r["frac_below_half"] for r in sel])),
                "curves_with_a_dip": float(np.mean([r["n_dips"] > 0 for r in sel])),
            }
        out["blocks"][src] = blk

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)

    for src, b in out["blocks"].items():
        print(f"\n=== {src}  (frames {n_frames.get(src, 0)}, curves {b['n_curves']}) ===")
        print(f"  median per-curve  min/median contrast : {b['min_over_med_median']:.3f}")
        print(f"  median per-curve  p05/median contrast : {b['p05_over_med_median']:.3f}")
        print(f"  median per-curve  CV of contrast      : {b['cv_median']:.3f}")
        print(f"  mean fraction of arc below half median: {b['frac_below_half_mean']:.3f}")
        print(f"  curves carrying at least one dip      : {b['curves_with_a_dip']:.3f}")
        print(f"  {'len bin (native px)':>20} {'n':>6} {'min/med':>8} {'frac<half':>10} "
              f"{'with dip':>9}")
        for k, r in b["by_length"].items():
            print(f"  {k:>20} {r['n']:6d} {r['min_over_med']:8.3f} {r['frac_below_half']:10.3f} "
                  f"{r['curves_with_a_dip']:9.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
