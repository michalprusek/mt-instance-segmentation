"""The orientation-isotropic filament stamp (ARM 52, 2026-09-16): guards on the EFFECT.

The shipped `_paint` rounds 0.5-px normal offsets to pixels, so a filament's rendered peak depends on
its angle to the grid (max/min 1.40 raw) and the rot90/flip augmentation preserves it. With
`paint_aa` the peak is isotropic (max/min <= 1.05), the mass per unit length is unchanged, and with
the default gain the orientation-averaged peak after the shipped PSF matches the shipped renderer
within 2 %, so the fix is not a contrast change in disguise. OFF stays the shipped renderer.
"""
import os
import sys

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "synth"))
import mt_generator as G  # noqa: E402

SIGMA, PSF, L, H = 1.78, 0.705, 300, 400
ANGLES = np.arange(0.0, 90.0 + 1e-9, 7.5)


def _peak(cfg, theta_deg, psf):
    th = np.deg2rad(theta_deg)
    t = np.arange(0, L, 1.0)
    cl = np.stack([H / 2 + (t - L / 2) * np.cos(th), H / 2 + (t - L / 2) * np.sin(th)], 1)
    field = np.zeros((H, H))
    G._paint(field, cl, 1.0, np.ones(len(cl)), SIGMA, cfg)
    mass = field.sum() / L
    if psf:
        field = gaussian_filter(field, psf)
    n = np.array([-np.sin(th), np.cos(th)])
    pk = []
    for i in range(len(cl) // 4, 3 * len(cl) // 4, 3):
        offs = np.arange(-4, 4.01, 0.25)
        p = cl[i][None, :] + offs[:, None] * n[None, :]
        pk.append(field[np.clip(np.rint(p[:, 1]).astype(int), 0, H - 1), np.clip(np.rint(p[:, 0]).astype(int), 0, H - 1)].max())
    return float(np.mean(pk)), mass


def _profile(cfg, psf):
    r = [_peak(cfg, a, psf) for a in ANGLES]
    return np.array([x[0] for x in r]), np.array([x[1] for x in r])


def test_the_shipped_stamp_is_anisotropic_and_the_flag_off_keeps_it():
    pk, _ = _profile(G.GenConfig(), None)
    assert pk.max() / pk.min() > 1.3          # the defect this arm is about; OFF must still show it


def test_the_isotropic_stamp_is_isotropic_and_conserves_mass():
    pk, mass = _profile(G.GenConfig(paint_aa=True, paint_aa_gain=1.0), None)
    _, mass0 = _profile(G.GenConfig(), None)
    assert pk.max() / pk.min() <= 1.05
    assert abs(mass.mean() / mass0.mean() - 1.0) < 0.01


def test_the_default_gain_matches_the_shipped_orientation_averaged_peak_after_the_psf():
    pk0, _ = _profile(G.GenConfig(), PSF)
    pk1, _ = _profile(G.GenConfig(paint_aa=True), PSF)
    assert abs(pk1.mean() / pk0.mean() - 1.0) < 0.02
    assert pk1.max() / pk1.min() <= 1.05


def test_modulation_and_sign_follow_the_nearest_sample():
    cfg = G.GenConfig(paint_aa=True, paint_aa_gain=1.0)
    t = np.arange(0, 200, 1.0)
    cl = np.stack([100 + t - 100, np.full_like(t, 150.0)], 1)
    mod = np.where(t < 100, 1.0, -1.0)
    field = np.zeros((300, 300))
    G._paint(field, cl, 1.0, mod, SIGMA, cfg)
    assert field[150, 50] > 0 and field[150, 150] < 0
    assert np.isclose(field[150, 50], -field[150, 150], rtol=0.05)
