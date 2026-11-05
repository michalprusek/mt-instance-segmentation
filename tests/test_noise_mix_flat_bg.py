"""Noise-regime mixture and flat-fielded backgrounds (2026-10-05): off = bit-identical, on = the declared effect."""
import dataclasses
import sys, os
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "synth"))
from mt_generator import GenConfig, add_irm_noise, flatten_background, generate_frame  # noqa: E402

ALT = {"noise_ref": "median", "noise_span_rel_range": [0.01, 0.3], "noise_corr_frac_range": [0.2, 0.95],
       "noise_corr_sigma_range": [0.8, 2.0], "noise_impulse_frac_range": [0.0, 0.002]}


def _bg(seed=0, shape=(256, 256)):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return 1000.0 * (1.0 + 0.3 * xx / shape[1] + 0.02 * gaussian_filter(rng.normal(0, 1, shape), 1.5))


def _acf1(img):
    r = img - gaussian_filter(img, 2.0); r = r - r.mean()
    return float((r[:, 1:] * r[:, :-1]).mean() / ((r * r).mean() + 1e-12))


def test_mixture_off_is_bit_identical_even_with_an_alt_dict():
    img = _bg()
    a = add_irm_noise(img, np.random.default_rng(3), GenConfig())
    b = add_irm_noise(img, np.random.default_rng(3), dataclasses.replace(GenConfig(), noise_alt=ALT))
    assert np.array_equal(a, b)


def test_mixture_prob_one_equals_the_alt_regime_after_one_draw():
    img = _bg()
    alt_cfg = dataclasses.replace(GenConfig(), **{k: tuple(v) if isinstance(v, list) else v for k, v in ALT.items()})
    r1 = np.random.default_rng(5); r1.random()
    want = add_irm_noise(img, r1, alt_cfg)
    got = add_irm_noise(img, np.random.default_rng(5), dataclasses.replace(GenConfig(), noise_alt_prob=1.0, noise_alt=ALT))
    assert np.array_equal(want, got)


def test_mixture_draws_both_regimes_at_the_declared_rate():
    cfg = dataclasses.replace(GenConfig(), noise_alt_prob=0.5, noise_alt=ALT)
    img = _bg(); flags = []
    for s in range(200):
        m = {}; add_irm_noise(img, np.random.default_rng(s), cfg, meta=m); flags.append(m["noise_alt"])
    assert 0.38 < np.mean(flags) < 0.62


def test_flatten_removes_the_gradient_and_keeps_mean_and_fine_texture():
    bg = _bg(); f = flatten_background(bg, 20.0)
    assert abs(f.mean() / bg.mean() - 1.0) < 0.01
    assert gaussian_filter(f, 25.0).std() < 0.2 * gaussian_filter(bg, 25.0).std()
    hp = lambda z: (z - gaussian_filter(z, 3.0))[40:-40, 40:-40].std()
    assert 0.8 < hp(f) / hp(bg) < 1.25


def test_flatten_off_is_bit_identical_and_on_is_recorded():
    bg = _bg(1, (192, 192))
    a = generate_frame(bg, np.random.default_rng(11), GenConfig())[0]
    b = generate_frame(bg, np.random.default_rng(11), dataclasses.replace(GenConfig(), bg_flatten_sigma_range=(5.0, 9.0)))[0]
    assert np.array_equal(a, b)
    img, _, meta = generate_frame(bg, np.random.default_rng(11), dataclasses.replace(GenConfig(), bg_flatten_prob=1.0))
    assert 3.0 <= meta["bg_flatten_sigma"] <= 60.0 and np.isfinite(img).all()
