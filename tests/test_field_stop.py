"""Tests for the synthetic octagonal field stop.

The strongest available check is a ROUND TRIP: a field stop synthesised for training must be
found by ``mt_bench.fov.fov_mask``, which is the detector written against the real task-586
aperture. If the detector cannot see the synthetic one, the model is being trained on a
boundary that does not resemble the boundary it will meet.
"""
import numpy as np
import pytest

from mt_bench.fov import fov_mask

from train_v5 import K_ORI, apply_field_stop


def _frame(h=256, w=256, rng=None):
    """A textured 'background' with a couple of bright filaments, in [0, 1]."""
    rng = rng or np.random.default_rng(0)
    im = 0.5 + 0.02 * rng.standard_normal((h, w)).astype(np.float32)
    im[h // 3, :] = 0.9
    im[:, w // 2] = 0.9
    gt = np.zeros((K_ORI + 2, h, w), np.float32)
    gt[0, h // 3, :] = 1.0
    gt[K_ORI, h // 3, :] = 1.0          # union channel
    gt[K_ORI, :, w // 2] = 1.0
    return im, gt


def test_field_stop_saturates_the_surround_and_keeps_the_centre():
    rng = np.random.default_rng(1)
    im, gt = _frame()
    out, _ = apply_field_stop(im, gt, rng)
    h, w = im.shape
    assert out[h // 2, w // 2] == pytest.approx(im[h // 2, w // 2], abs=0.05)
    # Corners are outside any octagon whose apothem is at most 0.52 of the frame; the surround
    # is at least 1.3x the interior median (the measured roi303 floor is 1.45x, MT-34 2.5x).
    interior = float(np.median(im))
    assert out[0, 0] > interior * 1.25
    assert out[-1, -1] > interior * 1.25


def test_the_surround_covers_the_measured_range_of_both_blocks():
    """roi303 stops sit at 1.45-3.5x the interior, MT-34 at 2.5-5x; over many draws the synthetic
    surround must reach below 2.5x (else roi303's stops are never rendered) and above 3.5x."""
    ratios = []
    for seed in range(40):
        rng = np.random.default_rng(100 + seed)
        im, gt = _frame()
        out, _ = apply_field_stop(im, gt, rng)
        ratios.append(out[0, 0] / float(np.median(im)))
    ratios = np.array(ratios)
    assert ratios.min() < 2.0 and ratios.max() > 3.5, ratios


def test_labels_outside_the_aperture_are_dropped():
    """The instrument has no data outside the stop and the annotators drew none there.

    The aperture covers ~91 % of the frame (measured on the 19 MT-34 frames that have one), so
    only the octagon's corner wedges fall outside -- a label has to be put there to test this.
    """
    rng = np.random.default_rng(2)
    im, gt = _frame()
    h, w = im.shape
    for d in range(12):                                  # a stub in the top-left corner wedge
        gt[K_ORI, d, d] = 1.0
    out, gt2 = apply_field_stop(im, gt, rng)
    assert gt2.sum() < gt.sum(), "nothing was dropped -- is the aperture covering the frame?"
    assert gt2[K_ORI, 0, 0] == 0.0
    # ...but the centre of a labelled filament survives.
    assert gt2[K_ORI, h // 3, w // 2] == pytest.approx(gt[K_ORI, h // 3, w // 2])


def test_the_real_detector_finds_the_synthetic_stop():
    """Round trip: mt_bench.fov was written for the real aperture; it must see this one."""
    rng = np.random.default_rng(3)
    im, gt = _frame(384, 384)
    # The MT-34 regime (2.5-5x) is what the detector was written against; the fixture's bright
    # filaments widen its intensity band, so a 1.3x surround is below what it can see HERE even
    # though it sees 1.45x on real roi303 frames (76 detected stops, 2026-09-16).
    out, _ = apply_field_stop(im, gt, rng, sat_range=(2.5, 5.0))
    m = fov_mask(out)
    assert 0.15 < m.mean() < 0.95, f"detector found {m.mean():.2f} of the frame inside"
    h, w = out.shape
    assert m[h // 2, w // 2]
    assert not m[0, 0]


def test_a_frame_without_a_stop_is_returned_fully_valid():
    """The same detector must not invent an aperture where there is none (Alice frames)."""
    im, _ = _frame(384, 384)
    assert fov_mask(im).mean() > 0.95


def test_shape_and_dtype_are_preserved():
    rng = np.random.default_rng(4)
    im, gt = _frame()
    out, gt2 = apply_field_stop(im, gt, rng)
    assert out.shape == im.shape and out.dtype == np.float32
    assert gt2.shape == gt.shape
    assert np.isfinite(out).all()
