"""The second-generation noise model (2026-09-04): defaults bit-identical to the original function,
a correlated share at constant variance, impulses for the kurtosis tail, and a median reference
that decouples the noise level from the background crop's dynamic range."""
import dataclasses
import sys, os
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import kurtosis

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "synth"))
from mt_generator import GenConfig, add_irm_noise, _shot_read_noise  # noqa: E402


def _old_add_irm_noise(img, rng, cfg):
    """The function as it was before 2026-09-04, verbatim."""
    lo, hi = np.percentile(img, [0.5, 99.5]); span = max(hi - lo, 1e-6)
    z = _shot_read_noise(np.clip((img - lo) / span, 0, 1), rng, cfg.poisson_gain_range, cfg.read_noise_range)
    return lo + np.clip(z, 0, 1) * span


def _flat(seed=0, level=1000.0, shape=(256, 256)):
    rng = np.random.default_rng(seed)
    return level * (1.0 + 0.02 * gaussian_filter(rng.normal(0, 1, shape), 6.0))


def _acf1(r):
    r = r - r.mean(); v = (r * r).mean() + 1e-12
    return float((r[:, 1:] * r[:, :-1]).mean() / v)


def _residual(img):
    return img - gaussian_filter(img, 2.0)


def test_defaults_are_bit_identical_to_the_old_function():
    cfg = GenConfig()
    img = _flat()
    a = add_irm_noise(img, np.random.default_rng(7), cfg)
    b = _old_add_irm_noise(img, np.random.default_rng(7), cfg)
    assert np.array_equal(a, b)


def test_correlated_share_keeps_the_variance_and_correlates_the_residual():
    img = _flat()
    base = GenConfig(poisson_gain_range=(150.0, 150.0), read_noise_range=(0.01, 0.01))
    corr = dataclasses.replace(base, noise_corr_frac_range=(0.6, 0.6), noise_corr_sigma_range=(1.2, 1.2))
    r0 = _residual(add_irm_noise(img, np.random.default_rng(3), base) - img)
    r1 = _residual(add_irm_noise(img, np.random.default_rng(3), corr) - img)
    assert abs(r1.std() / r0.std() - 1.0) < 0.25          # same total noise power (high-pass part)
    assert _acf1(r0) < 0.05 and _acf1(r1) > 0.2           # white before, correlated after


def test_impulses_make_the_residual_heavy_tailed():
    img = _flat()
    base = GenConfig(poisson_gain_range=(150.0, 150.0), read_noise_range=(0.01, 0.01))
    imp = dataclasses.replace(base, noise_impulse_frac_range=(1e-3, 1e-3), noise_impulse_amp_range=(10.0, 10.0))
    k0 = kurtosis(_residual(add_irm_noise(img, np.random.default_rng(5), base)).ravel())
    k1 = kurtosis(_residual(add_irm_noise(img, np.random.default_rng(5), imp)).ravel())
    assert k0 < 1.0 and k1 > 3.0


def test_median_reference_decouples_noise_from_the_background_range():
    """A graded background (2x across the crop) got ~10x the noise of a clean one under the span
    reference; under the median reference both get the same noise relative to their median."""
    clean = _flat(level=1000.0)
    rng = np.random.default_rng(1)
    dirty = clean * np.linspace(0.6, 1.6, clean.shape[1])[None, :]
    cfg_span = GenConfig(poisson_gain_range=(150.0, 150.0), read_noise_range=(0.01, 0.01))
    cfg_med = dataclasses.replace(cfg_span, noise_ref="median", noise_span_rel_range=(0.02, 0.02))
    def rel_noise(cfg, im):
        r = _residual(add_irm_noise(im, np.random.default_rng(2), cfg) - im)
        return r.std() / np.median(im)
    ratio_span = rel_noise(cfg_span, dirty) / rel_noise(cfg_span, clean)
    ratio_med = rel_noise(cfg_med, dirty) / rel_noise(cfg_med, clean)
    assert ratio_span > 3.0                                # the defect, reproduced
    assert 0.8 < ratio_med < 1.25                          # the fix


def test_median_reference_keeps_a_dark_filament_dip_at_every_photon_scale():
    """A 6 % dark line on a flat background must keep its depth (within 20 %) after noise, at the
    smallest and the largest photon scale -- the review-caught floor would have flattened it."""
    img = _flat(level=1000.0)
    line = img.copy(); line[:, 120:123] *= 0.94
    for rel in (0.003, 0.02, 0.3):
        cfg = GenConfig(poisson_gain_range=(150.0, 150.0), read_noise_range=(0.01, 0.01),
                        noise_ref="median", noise_span_rel_range=(rel, rel))
        out = np.mean([add_irm_noise(line, np.random.default_rng(s), cfg) for s in range(8)], axis=0)
        depth = 1.0 - out[:, 120:123].mean() / out[:, 60:100].mean()
        assert abs(depth / 0.06 - 1.0) < 0.2, (rel, depth)


def test_median_reference_shot_noise_scales_with_local_intensity():
    img = _flat(level=1000.0)
    half = img.copy(); half[:, :128] *= 0.25
    cfg = GenConfig(poisson_gain_range=(150.0, 150.0), read_noise_range=(0.0, 0.0),
                    noise_ref="median", noise_span_rel_range=(0.05, 0.05))
    r = _residual(add_irm_noise(half, np.random.default_rng(4), cfg) - half)
    ratio = r[:, :120].std() / r[:, 136:].std()
    assert 0.4 < ratio < 0.6                            # sqrt(0.25) = 0.5: photon statistics
