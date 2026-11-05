#!/usr/bin/env python3
"""Why are the crossing-dense block's microtubules BRIGHT when every other block's are dark?

Measured 2026-08-28: 98.1 % of the annotated filaments in the 585 block are darker than their
surroundings, and only 3.6 % of the 586 block's are. Both are interference reflection microscopy
from the same lab. In IRM a microtubule in contact with the coverslip is dark (a pi reflection phase
shift) and becomes bright only once it lifts past h = lambda/(8*n_water) = 56.4 nm, so "bright"
means one of two very different things:

  (a) the images are INVERTED relative to 585, a display or export convention, or
  (b) the microtubules really are elevated -- a whole block of detached filaments.

These are distinguishable without any extra data. Surface debris and dirt in IRM are strongly DARK
whatever the microtubules are doing, because they sit on the glass. So:

  debris dark + filaments bright  ->  the filaments are genuinely elevated
  debris bright + filaments bright ->  the image is inverted

The answer decides whether 586 is a legitimate second imaging regime the model must handle, or a
preprocessing inconsistency that has been silently splitting the training signal.

    python scripts/diag_586_polarity.py
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.fov import fov_mask  # noqa: E402


def source_of(name):
    b = os.path.basename(name)
    return "htw" if b.startswith("htw") else ("585" if b.startswith("alice") else "586")


def densify(p, step=1.0):
    seg = np.sqrt(((p[1:] - p[:-1]) ** 2).sum(1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return p
    t = np.arange(0.0, s[-1], step)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)


def filament_contrast(img, polys):
    """Median signed core-minus-flank contrast over a frame's annotated filaments."""
    out = []
    H, W = img.shape
    for p in polys:
        q = densify(p)
        if len(q) < 5:
            continue
        t = np.gradient(q, axis=0)
        n = np.stack([-t[:, 1], t[:, 0]], 1)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)

        def take(pt):
            c = np.round(pt[:, 0]).astype(int); r = np.round(pt[:, 1]).astype(int)
            ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
            v = np.full(len(pt), np.nan); v[ok] = img[r[ok], c[ok]]; return v

        v = take(q) - np.nanmean(np.stack([take(q + n * 7), take(q - n * 7)]), axis=0)
        if np.isfinite(v).any():
            out.append(float(np.nanmedian(v)))
    return np.array(out)


def debris_contrast(img, polys, fov, min_px=25):
    """Signed contrast of compact, NON-filamentous blobs away from every annotation.

    Debris sits on the glass and is dark in IRM regardless of what the microtubules do, so its sign
    is the frame's absolute reference.
    """
    H, W = img.shape
    near = np.zeros((H, W), bool)
    for p in polys:
        q = densify(p, 0.5)
        c = np.clip(np.round(q[:, 0]).astype(int), 0, W - 1)
        r = np.clip(np.round(q[:, 1]).astype(int), 0, H - 1)
        near[r, c] = True
    near = ndi.binary_dilation(near, iterations=8)

    med = float(np.median(img[fov])) if fov.any() else float(np.median(img))
    mad = float(np.median(np.abs(img - med))) * 1.4826
    out = []
    for sign, name in ((-1, "dark"), (+1, "bright")):
        m = ((img - med) * sign > 4 * mad) & fov & ~near
        m = ndi.binary_opening(m, np.ones((3, 3)))
        lab, n = ndi.label(m, np.ones((3, 3)))
        if n == 0:
            continue
        sizes = ndi.sum(m, lab, range(1, n + 1))
        for i in np.where(sizes >= min_px)[0] + 1:
            ys, xs = np.where(lab == i)
            pts = np.stack([xs - xs.mean(), ys - ys.mean()], 1)
            if len(pts) < 3:
                continue
            ev = np.linalg.eigvalsh(np.cov(pts.T))
            if ev[0] <= 1e-6 or ev[1] / ev[0] > 6.0:
                continue                      # elongated -> a filament, not debris
            sel = lab == i
            ring = ndi.binary_dilation(sel, iterations=6) & ~ndi.binary_dilation(sel, iterations=2)
            if ring.sum() < 12:
                continue
            out.append(float(img[sel].mean() - img[ring].mean()))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/real/mt34_htw_eval")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.data_dir, "*.h5"))):
        fr = read_frame_h5(path)
        if args.split != "all" and str(fr["attrs"].get("split")) != args.split:
            continue
        img = fr["image"].astype(np.float64)
        polys = [np.asarray(p, float) for p in fr["polylines"] if len(p) >= 2]
        if not polys:
            continue
        fov = fov_mask(fr["image"])
        fc = filament_contrast(img, polys)
        dc = debris_contrast(img, polys, fov)
        if not len(fc):
            continue
        rows.append(dict(name=os.path.basename(path), src=source_of(path),
                         fil=float(np.median(fc)), fil_dark=float((fc < 0).mean()),
                         n_deb=len(dc),
                         deb=float(np.median(dc)) if len(dc) else np.nan,
                         deb_dark=float((dc < 0).mean()) if len(dc) else np.nan))

    print("FILAMENT vs DEBRIS polarity, per block  (negative = darker than surroundings)\n")
    print(f'{"block":>6s} {"frames":>7s} {"filament contrast":>18s} {"% frames dark":>14s} | '
          f'{"n debris":>9s} {"debris contrast":>16s} {"% debris dark":>14s}')
    for s in ("585", "586", "htw"):
        R = [r for r in rows if r["src"] == s]
        if not R:
            continue
        fil = np.array([r["fil"] for r in R])
        deb = np.array([r["deb"] for r in R if np.isfinite(r["deb"])])
        dd = np.array([r["deb_dark"] for r in R if np.isfinite(r["deb_dark"])])
        print(f'{s:>6s} {len(R):7d} {np.median(fil):18.2f} {(fil < 0).mean() * 100:13.1f}% | '
              f'{int(np.sum([r["n_deb"] for r in R])):9d} '
              f'{np.median(deb) if len(deb) else float("nan"):16.2f} '
              f'{np.mean(dd) * 100 if len(dd) else float("nan"):13.1f}%')

    print()
    print("READING")
    print("  debris DARK  + filaments BRIGHT -> the microtubules really are elevated (detached)")
    print("  debris BRIGHT+ filaments BRIGHT -> the image is inverted relative to the other blocks")
    print()
    print("Per-frame detail for the 586 block (is it every frame, or a subset?):")
    print(f'  {"frame":42s} {"filament":>9s} {"% dark":>7s} {"debris":>9s}')
    for r in [r for r in rows if r["src"] == "586"]:
        print(f'  {r["name"][:42]:42s} {r["fil"]:9.2f} {r["fil_dark"] * 100:6.0f}% '
              f'{r["deb"]:9.2f}')


if __name__ == "__main__":
    main()
