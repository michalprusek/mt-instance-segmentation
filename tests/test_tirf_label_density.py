"""Sparse fluorophore labelling in the TIRF renderer.

A real TIRF microtubule at a low labelling ratio is a chain of puncta, not a continuous line, and
the generator used to render only the continuous case. The properties below are the ones the rest
of the pipeline depends on; each was mutation-tested (break the code, confirm the test fails).
"""
import numpy as np
import pytest

from mt_generator import (GenConfig, along_label_density, generate_frame, sample_scene)


def _cfg(**kw):
    return GenConfig(**kw)


def test_default_is_off_so_existing_results_are_untouched():
    """Every number measured so far used the continuous renderer; the new path must be opt-in."""
    assert GenConfig().tirf_label_sparse is False


def test_mean_is_one_so_density_is_not_secretly_a_brightness_knob():
    """If a lower labelling density also dimmed the frame, a sweep over density would be a sweep
    over exposure and the two effects could never be separated afterwards."""
    cfg = _cfg()
    for lam in (0.1, 0.3, 1.0, 5.0):
        m = along_label_density(np.random.default_rng(0), 20000, lam, cfg)
        assert m.mean() == pytest.approx(1.0, abs=0.06), f"lam={lam} mean={m.mean()}"


def test_duty_cycle_rises_with_density():
    cfg = _cfg()
    duty = [(along_label_density(np.random.default_rng(1), 20000, lam, cfg) > 0.2).mean()
            for lam in (0.1, 0.3, 1.0, 5.0)]
    assert all(a < b for a, b in zip(duty, duty[1:])), duty
    assert duty[0] < 0.25, "the sparse end must actually be sparse"
    assert duty[-1] > 0.90, "the dense end must converge to a continuous filament"


def test_dense_limit_is_flat():
    """High density has to reproduce the old continuous appearance, so one prior spans both
    regimes instead of the generator having two disjoint TIRF modes."""
    m = along_label_density(np.random.default_rng(2), 20000, 50.0, _cfg())
    assert m.std() < 0.25


def test_sparse_end_reaches_the_measured_regime():
    """The real 500x-diluted frames sit at a duty cycle of roughly 0.1 to 0.8 (measured by
    scripts/tirf_puncta_stats.py). The prior must BRACKET that, not sit beside it."""
    cfg = _cfg()
    lo, hi = cfg.tirf_label_density_log10
    d_lo = (along_label_density(np.random.default_rng(3), 20000, 10.0 ** lo, cfg) > 0.2).mean()
    d_hi = (along_label_density(np.random.default_rng(3), 20000, 10.0 ** hi, cfg) > 0.2).mean()
    assert d_lo <= 0.11, f"prior's sparse end (duty {d_lo:.2f}) does not reach the real one"
    assert d_hi >= 0.79, f"prior's dense end (duty {d_hi:.2f}) does not reach the real one"


def test_amplitude_jitter_widens_the_spread_but_keeps_the_mean():
    cfg_j, cfg_n = _cfg(tirf_label_amp_cv=0.6), _cfg(tirf_label_amp_cv=0.0)
    a = along_label_density(np.random.default_rng(4), 20000, 1.0, cfg_j)
    b = along_label_density(np.random.default_rng(4), 20000, 1.0, cfg_n)
    assert a.std() > b.std() * 1.15
    assert a.mean() == pytest.approx(b.mean(), abs=0.08)


def test_zero_jitter_gives_integer_counts():
    m = along_label_density(np.random.default_rng(5), 500, 1.0, _cfg(tirf_label_amp_cv=0.0))
    assert np.allclose(m, np.round(m), atol=1e-9)


def test_the_gt_mask_stays_continuous():
    """The whole point: the mask comes from the centerline, so a filament rendered as puncta still
    carries ONE continuous instance. If the mask followed the render, the model would learn that a
    dark gap ends an instance -- the opposite of what it must do."""
    cfg = _cfg(tirf_label_sparse=True)
    bg = np.random.default_rng(9).normal(0.4, 0.05, (256, 256))
    ins_a, _ = sample_scene(bg.shape, np.random.default_rng(11), cfg)
    ins_b, _ = sample_scene(bg.shape, np.random.default_rng(11), _cfg(tirf_label_sparse=False))
    assert len(ins_a) == len(ins_b) and len(ins_a) > 0
    for a, b in zip(ins_a, ins_b):
        assert np.array_equal(a["mask"], b["mask"])


def test_sparse_labelling_changes_the_tirf_render_only():
    """It must reach TIRF and leave IRM alone -- IRM images mass, which is continuous whatever the
    labelling ratio is."""
    bg = np.random.default_rng(9).normal(0.4, 0.05, (192, 192))
    off, on = _cfg(tirf_label_sparse=False), _cfg(tirf_label_sparse=True)
    t0, _, _ = generate_frame(bg, np.random.default_rng(3), off, modality="tirf")
    t1, _, _ = generate_frame(bg, np.random.default_rng(3), on, modality="tirf")
    i0, _, _ = generate_frame(bg, np.random.default_rng(3), off, modality="irm")
    i1, _, _ = generate_frame(bg, np.random.default_rng(3), on, modality="irm")
    assert not np.array_equal(t0, t1), "the switch does nothing to TIRF"
    assert np.array_equal(i0, i1), "the switch leaked into IRM"


def test_nothing_is_marked_unlabelled_when_the_feature_is_off():
    """The marker must never appear on the paths that produced every existing result."""
    bg = np.random.default_rng(9).normal(0.4, 0.05, (256, 256))
    for modality in ("irm", "tirf"):
        _, ins, _ = generate_frame(bg, np.random.default_rng(4), _cfg(tirf_label_sparse=False),
                                   modality=modality)
        assert not any("tirf_unlabelled" in z for z in ins)


def test_irm_never_marks_unlabelled_even_with_the_feature_on():
    """IRM images mass, so a microtubule is there whether or not it carries a fluorophore."""
    bg = np.random.default_rng(9).normal(0.4, 0.05, (256, 256))
    _, ins, _ = generate_frame(bg, np.random.default_rng(4), _cfg(tirf_label_sparse=True),
                               modality="irm")
    assert not any("tirf_unlabelled" in z for z in ins)


def test_an_unlabelled_filament_really_is_absent_from_the_image():
    """The marker has to mean what it says: a marked filament contributed nothing to the render,
    so dropping it from the target removes an unlearnable pair rather than a hard one."""
    bg = np.zeros((256, 256))
    cfg = _cfg(tirf_label_sparse=True, tirf_label_density_log10=(-2.4, -2.4))  # near-zero labelling
    found = False
    for seed in range(40):
        img, ins, _ = generate_frame(bg, np.random.default_rng(seed), cfg, modality="tirf")
        marked = [z for z in ins if z.get("tirf_unlabelled")]
        if not marked:
            continue
        found = True
        # the render must be flat wherever ONLY marked filaments run
        others = np.zeros_like(img, bool)
        for z in ins:
            if not z.get("tirf_unlabelled"):
                others |= z["mask"]
        from scipy.ndimage import binary_dilation
        others = binary_dilation(others, np.ones((15, 15), bool))
        for z in marked:
            only = z["mask"] & ~others
            if only.sum() >= 20:
                assert img[only].std() < img.std() * 1.5, "a marked filament still shows up"
        break
    assert found, "the near-zero density never produced an unlabelled filament -- test is vacuous"
