"""ARM 54 (2026-09-17): arbitrary-angle rotation of a synthetic frame together with its target stack.
The rotated target must still sit on the rotated filament; the union stays binary; the count map
stays integer; probability 0 leaves the dataset path untouched (no call is made).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import train_v5 as T  # noqa: E402

S = 256


def _frame_with_line(theta_deg):
    th = np.deg2rad(theta_deg)
    t = np.arange(-100, 100, 1.0)
    cl = np.stack([S / 2 + t * np.cos(th), S / 2 + t * np.sin(th)], 1)
    img = np.full((S, S), 100.0)
    for x, y in cl:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < S and 0 <= yi < S:
            img[yi, xi] -= 40.0                     # a dark IRM-like ridge
    gt = T.ori_channels_soft([{"centerline": cl}], (S, S), half_width=1.0)
    return img, gt


def test_rotated_target_still_covers_the_rotated_ridge():
    img, gt = _frame_with_line(20.0)
    im2, gt2 = T.rotate_pair(img, gt, 33.0)
    band = gt2[T.K_ORI] > 0
    inner = band.copy(); inner[:40] = inner[-40:] = False; inner[:, :40] = inner[:, -40:] = False
    assert inner.sum() > 200
    assert im2[inner].mean() < im2[~band].mean() - 10.0     # the band is on the dark ridge, not beside it
    assert set(np.unique(gt2[T.K_ORI])) <= {0.0, 1.0}
    assert np.allclose(gt2[T.K_ORI + 1], np.round(gt2[T.K_ORI + 1]))


def test_shapes_and_dtypes_are_preserved():
    img, gt = _frame_with_line(60.0)
    im2, gt2 = T.rotate_pair(img, gt, 71.5)
    assert im2.shape == img.shape and gt2.shape == gt.shape and gt2.dtype == gt.dtype
