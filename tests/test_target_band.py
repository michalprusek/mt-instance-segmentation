"""The training target band (ARM 53, 2026-09-16): the inherited 3x3 SQUARE dilation of the rasterised
centerline puts more target pixels per unit length on oblique curves than on axis-aligned ones,
and the rot90/flip augmentation preserves that; `--target-band euclid` (exact 1.5 px Euclidean band)
is orientation-isotropic. The unflagged path must stay bit-identical to the square dilation.
"""
import os
import sys

import numpy as np
from scipy.ndimage import binary_dilation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import train_v5 as T  # noqa: E402

# 0.5 deg off the multiples of 7.5 so no line runs EXACTLY along the pixel-centre diagonal: at 45.000 deg the
# pixel centres at perpendicular distance sqrt(2) sit on the 1.5 px radius and the count jumps to 3.5 for that
# measure-zero angle; real and synthetic curves never hold it.
ANGLES = np.arange(0.0, 90.0 + 1e-9, 7.5) + 0.5
L, S = 240, 400


def _line(theta_deg):
    th = np.deg2rad(theta_deg)
    t = np.arange(0, L, 1.0)
    return np.stack([S / 2 + (t - L / 2) * np.cos(th), S / 2 + (t - L / 2) * np.sin(th)], 1)


def _px_per_len(mode):
    old = T.TARGET_BAND
    T.TARGET_BAND = mode
    try:
        out = []
        for a in ANGLES:
            gt = T.ori_channels_soft([{"centerline": _line(a)}], (S, S), half_width=1.0)
            out.append(gt[T.K_ORI].sum() / L)
    finally:
        T.TARGET_BAND = old
    return np.array(out)


def test_the_square_band_is_anisotropic_and_the_euclid_band_is_not():
    sq = _px_per_len("square")
    eu = _px_per_len("euclid")
    assert sq.max() / sq.min() > 1.15
    assert eu.max() / eu.min() <= 1.08
    assert abs(eu.mean() - 3.0) < 0.3           # a 1.5 px band is 3 px wide


def test_the_unflagged_path_is_the_square_dilation():
    assert T.TARGET_BAND == "square"
    cl = _line(30.0)
    gt = T.ori_channels_soft([{"centerline": cl}], (S, S), half_width=1.0)
    from instance.geometry import resample
    pts = resample(cl, ds=0.4)
    one = np.zeros((S, S), bool)
    one[np.clip(np.rint(pts[:, 1]).astype(int), 0, S - 1), np.clip(np.rint(pts[:, 0]).astype(int), 0, S - 1)] = True
    ref = binary_dilation(one, structure=np.ones((3, 3), bool))
    assert np.array_equal(gt[T.K_ORI] > 0, ref)


def test_euclid_band_keeps_the_orientation_channels_and_count_layout():
    old = T.TARGET_BAND
    T.TARGET_BAND = "euclid"
    try:
        gt = T.ori_channels_soft([{"centerline": _line(20.0)}, {"centerline": _line(110.0)}], (S, S), half_width=1.0)
    finally:
        T.TARGET_BAND = old
    assert gt.shape[0] == T.K_ORI + 2
    assert (gt[:T.K_ORI].max(axis=0) > 0).sum() == (gt[T.K_ORI] > 0).sum()   # union == any channel
    assert gt[T.K_ORI + 1].max() >= 2                                          # the crossing is counted twice
