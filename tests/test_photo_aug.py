"""Effect tests for the photometric augmentation (synth/photo_aug.py).

The project's rule after the dead `half_width` parameter: a guard must test the EFFECT, not the
plumbing. So these check that (a) strength 0 is a strict no-op that draws nothing, (b) strength 1
actually changes the image while leaving the training TARGET untouched through the real dataset
path, and (c) the output obeys the evaluation's normalisation contract.
"""
from __future__ import annotations

import os
import random
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "synth"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from photo_aug import _stretch, low_frequency_field, photometric_augment  # noqa: E402


def _ridge_image(rng, n=160):
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    im = 0.5 + 0.02 * rng.normal(size=(n, n)).astype(np.float32)
    for c in (40, 80, 120):
        im -= 0.25 * np.exp(-0.5 * ((yy - c) / 1.5) ** 2)
    return np.clip(im, 0, 1)


def test_strength_zero_is_a_noop_and_draws_nothing():
    rng = np.random.default_rng(3)
    im = _ridge_image(rng)
    state = rng.bit_generator.state
    out = photometric_augment(im, rng, 0.0)
    assert out is im
    assert rng.bit_generator.state == state


def test_strength_one_changes_the_image_but_stays_in_contract():
    changed, n = 0, 40
    for seed in range(n):
        rng = np.random.default_rng(seed)
        im = _ridge_image(rng)
        out = photometric_augment(im, rng, 1.0)
        assert out.shape == im.shape and out.dtype == np.float32
        assert out.min() >= 0.0 and out.max() <= 1.0
        # the evaluation's [1, 99] stretch is (up to percentile interpolation) idempotent on
        # its own output, so train and test inputs obey one rule
        assert np.abs(_stretch(out) - out).max() < 0.02
        if np.abs(out - im).mean() > 0.01:
            changed += 1
    assert changed >= 0.9 * n, f"only {changed}/{n} draws changed the image"


def test_every_component_fires_with_a_measurable_effect():
    """Force each branch by driving the probability draws, then check its footprint."""
    rng = np.random.default_rng(0)
    im = _ridge_image(rng)
    n_levels = []
    blurred = 0
    noisy = 0
    for seed in range(60):
        rng = np.random.default_rng(seed)
        out = photometric_augment(im, rng, 1.0)
        n_levels.append(len(np.unique(out)))
        # blur lowers the high-frequency energy, noise raises it
        hf = lambda a: float(np.abs(np.diff(a, axis=1)).mean())  # noqa: E731
        if hf(out) < 0.8 * hf(im):
            blurred += 1
        if hf(out) > 1.5 * hf(im):
            noisy += 1
    assert min(n_levels) <= 256, "quantisation never fired"
    assert max(n_levels) > 1000, "quantisation always fired"
    assert blurred > 0 and noisy > 0


def test_low_frequency_field_contract():
    f = low_frequency_field((97, 131), np.random.default_rng(1))
    assert f.shape == (97, 131)
    assert np.abs(f).max() <= 1.0 + 1e-6
    # smooth: neighbouring pixels differ by far less than the range
    assert np.abs(np.diff(f, axis=0)).max() < 0.2


def test_dataset_path_changes_image_not_target(tmp_path):
    """Through OnlineDS: same seed, photo_aug 0 vs 1 -> identical target, different image."""
    pytest.importorskip("torch")
    from PIL import Image
    from train_v5 import OnlineDS
    rng = np.random.default_rng(0)
    bg = (0.5 + 0.05 * rng.normal(size=(700, 700))).astype(np.float32)
    p = tmp_path / "bg.tif"
    Image.fromarray(bg).save(p)

    def gen_fn(bgc, r, cfg, modality="irm"):
        n = bgc.shape[0]
        t = np.linspace(0, n - 1, 400)
        cl = np.stack([t, 0.5 * n + 30 * np.sin(t / 60)], 1)
        img = bgc.copy()
        rr = np.clip(np.rint(cl[:, 1]).astype(int), 0, n - 1)
        cc = np.clip(np.rint(cl[:, 0]).astype(int), 0, n - 1)
        img[rr, cc] -= 0.2
        return img, [{"centerline": cl}], {}

    outs = []
    for pa in (0.0, 1.0):
        ds = OnlineDS(None, [str(p)], gen_fn, crop=256, gen=640, epoch_len=1, photo_aug=pa)
        random.seed(1234)
        outs.append(ds[0])
    (im0, gt0), (im1, gt1) = outs
    assert gt0.shape == gt1.shape
    assert bool((gt0 == gt1).all()), "the augmentation must not touch the target"
    assert float((im0 - im1).abs().mean()) > 0.01, "photo_aug=1 did not change the image"


def _make_ds(tmp_path, **kw):
    from PIL import Image
    from train_v5 import OnlineDS
    rng = np.random.default_rng(0)
    bg = (0.5 + 0.02 * rng.normal(size=(1000, 1000))).astype(np.float32)
    p = tmp_path / "bg2.tif"
    Image.fromarray(bg).save(p)

    def gen_fn(bgc, r, cfg, modality="irm"):
        # one horizontal Gaussian ridge of sigma 1.7 px through the frame centre
        n = bgc.shape[0]
        yy = np.arange(n)[:, None].astype(np.float32)
        img = bgc - 0.3 * np.exp(-0.5 * ((yy - n / 2) / 1.7) ** 2)
        cl = np.stack([np.linspace(0, n - 1, 300), np.full(300, n / 2.0)], 1)
        return img.astype(np.float32), [{"centerline": cl}], {}

    return OnlineDS(None, [str(p)], gen_fn, crop=512, gen=640, epoch_len=1, rot=False, **kw)


def _ridge_fwhm(gray):
    """FWHM of the (dark) ridge profile averaged along columns, in px."""
    prof = gray.mean(1)
    prof = prof.max() - prof
    pk = prof.argmax()
    half = 0.5 * prof[pk]
    lo = pk
    while lo > 0 and prof[lo] > half:
        lo -= 1
    hi = pk
    while hi < len(prof) - 1 and prof[hi] > half:
        hi += 1
    return hi - lo


def test_scale_aug_scales_the_image_not_the_target_width(tmp_path):
    """s = 0.8 vs s = 1.5: the filament's apparent width scales ~1.9x, the 3 px target does not."""
    pytest.importorskip("torch")
    import train_v5
    widths, tw = {}, {}
    for lo_hi in ((0.8, 0.8001), (1.5, 1.5001)):
        ds = _make_ds(tmp_path, scale_aug=lo_hi, scale_prob=1.0)
        random.seed(5)
        im, gt = ds[0]
        gray = im[0].numpy() * 0.229 + 0.485
        fg = gt[train_v5.K_ORI].numpy() > 0
        # the target's across-ridge run length must be 3 px whatever the scale
        runs = fg.sum(0)[fg.sum(0) > 0]
        tw[lo_hi[0]] = float(np.median(runs))
        widths[lo_hi[0]] = _ridge_fwhm(gray)
        assert im.shape[-1] == 512 and gt.shape[-1] == 512
    assert tw[0.8] == 3.0 and tw[1.5] == 3.0, tw
    ratio = widths[1.5] / widths[0.8]
    assert 1.5 < ratio < 2.4, (widths, ratio)


def test_scale_aug_off_is_bit_identical_to_the_flag_absent(tmp_path):
    pytest.importorskip("torch")
    outs = []
    for kw in ({}, {"scale_aug": None, "lowres_prob": 0.0, "photo_aug": 0.0}):
        ds = _make_ds(tmp_path, **kw)
        random.seed(11)
        outs.append(ds[0])
    assert bool((outs[0][0] == outs[1][0]).all()) and bool((outs[0][1] == outs[1][1]).all())


def test_lowres_sim_blurs_but_keeps_the_target(tmp_path):
    pytest.importorskip("torch")
    ims, gts = [], []
    for lp in (0.0, 1.0):
        ds = _make_ds(tmp_path, lowres_prob=lp)
        random.seed(3)
        im, gt = ds[0]
        ims.append(im[0].numpy()); gts.append(gt.numpy())
    # the extra draws shift the crop/flip sequence, so compare an invariant of the target rather
    # than the array: a full-width horizontal 3 px line has 512 x 3 foreground pixels in ANY crop
    import train_v5
    assert int((gts[0][train_v5.K_ORI] > 0).sum()) == 512 * 3 == int((gts[1][train_v5.K_ORI] > 0).sum())
    hf = lambda a: float(np.abs(np.diff(a, axis=1)).mean())  # noqa: E731
    assert hf(ims[1]) < hf(ims[0])


def test_heat_channel_peaks_on_the_centerline_and_dips_between_parallels():
    pytest.importorskip("torch")
    from train_v5 import K_ORI, heat_channel, ori_channels_soft
    n = 96
    a = np.stack([np.linspace(0, n - 1, 200), np.full(200, 40.0)], 1)      # y = 40
    b = np.stack([np.linspace(0, n - 1, 200), np.full(200, 44.5)], 1)      # y = 44.5 -> gap 4.5
    h = heat_channel([{"centerline": a}, {"centerline": b}], (n, n), sigma=1.2)
    assert h.shape == (n, n) and h.dtype == np.float32
    assert h[40, 50] == 1.0                       # exactly on a centerline
    assert 0.1 < h[42, 50] < 0.3                  # midway between the two: a real dip, not 0
    assert h[20, 50] < 1e-6                       # far away
    # appended LAST and only when asked; existing indices untouched
    g0 = ori_channels_soft([{"centerline": a}], (n, n))
    g1 = ori_channels_soft([{"centerline": a}], (n, n), heat_sigma=1.2)
    assert g0.shape[0] == K_ORI + 2 and g1.shape[0] == K_ORI + 3
    assert np.array_equal(g0, g1[:K_ORI + 2])
    assert g1[K_ORI + 2].max() == 1.0 and g1[K_ORI + 2][40, 50] == 1.0
