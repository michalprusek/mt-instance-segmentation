#!/usr/bin/env python3
"""Effect test for the orientation-isotropic renderer (generator research 2026-09-16, guard on the
EFFECT, not the plumbing): peak rendered contrast of a straight filament vs its angle to the pixel
grid, shipped stamp vs `paint_aa`, raw and after the frame PSF. Prints max/min over angles and the
orientation-averaged peak, and the gain that equalises the averaged peaks at the shipped PSF.

    python scripts/paint_aa_effect_test.py [--psf 0.705] [--sigma 1.78]
"""
import argparse
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "synth"))
import mt_generator as G  # noqa: E402

ANGLES = np.arange(0.0, 90.0 + 1e-9, 7.5)


def peak(cfg, theta_deg, sigma, psf, L=400, H=512):
    th = np.deg2rad(theta_deg)
    t = np.arange(0, L, 1.0)
    cl = np.stack([H / 2 + (t - L / 2) * np.cos(th), H / 2 + (t - L / 2) * np.sin(th)], 1)
    field = np.zeros((H, H))
    G._paint(field, cl, 1.0, np.ones(len(cl)), sigma, cfg)
    mass = field.sum() / L
    if psf:
        field = gaussian_filter(field, psf)
    n = np.array([-np.sin(th), np.cos(th)])
    pk = []
    for i in range(len(cl) // 4, 3 * len(cl) // 4, 2):
        offs = np.arange(-4, 4.01, 0.25)
        p = cl[i][None, :] + offs[:, None] * n[None, :]
        pk.append(field[np.clip(np.rint(p[:, 1]).astype(int), 0, H - 1), np.clip(np.rint(p[:, 0]).astype(int), 0, H - 1)].max())
    return float(np.mean(pk)), float(np.std(pk)), float(mass)


def profile(cfg, sigma, psf):
    rows = [peak(cfg, a, sigma, psf) for a in ANGLES]
    pk = np.array([r[0] for r in rows]); rip = np.array([r[1] for r in rows]); mass = np.array([r[2] for r in rows])
    return pk, rip, mass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psf", type=float, default=0.705)
    ap.add_argument("--sigma", type=float, default=1.78)
    ap.add_argument("--gain", type=float, default=None, help="paint_aa_gain to test; default = the equalising gain")
    args = ap.parse_args()
    shipped = G.GenConfig(paint_aa=False)
    aa1 = G.GenConfig(paint_aa=True, paint_aa_gain=1.0)
    for lab, psf in (("raw", None), (f"psf {args.psf}", args.psf)):
        ps, rs, ms = profile(shipped, args.sigma, psf)
        pa, ra, ma = profile(aa1, args.sigma, psf)
        print(f"[{lab}] shipped: peak by angle " + " ".join(f"{v:.2f}" for v in ps) + f"  max/min {ps.max() / ps.min():.3f}  mean {ps.mean():.3f}  ripple sd {rs.mean():.3f}  mass/len {ms.mean():.2f}")
        print(f"[{lab}] aa g=1 : peak by angle " + " ".join(f"{v:.2f}" for v in pa) + f"  max/min {pa.max() / pa.min():.3f}  mean {pa.mean():.3f}  ripple sd {ra.mean():.3f}  mass/len {ma.mean():.2f}")
    ps, _, _ = profile(shipped, args.sigma, args.psf)
    pa, _, _ = profile(aa1, args.sigma, args.psf)
    gain = ps.mean() / pa.mean() if args.gain is None else args.gain
    aag = G.GenConfig(paint_aa=True, paint_aa_gain=gain)
    pg, rg, mg = profile(aag, args.sigma, args.psf)
    print(f"equalising gain at psf {args.psf}: {ps.mean() / pa.mean():.4f}; tested gain {gain:.4f} -> aa peak mean {pg.mean():.3f} vs shipped {ps.mean():.3f} "
          f"(ratio {pg.mean() / ps.mean():.4f}), max/min {pg.max() / pg.min():.3f}, mass/len {mg.mean():.2f} vs shipped {profile(shipped, args.sigma, None)[2].mean():.2f}")


if __name__ == "__main__":
    main()
