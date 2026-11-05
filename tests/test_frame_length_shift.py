"""The frame-level length scale must have a measurable EFFECT, not merely be plumbed through.

A previous parameter (`half_width`) was accepted by a function that never read it, and a guard
that asserted the *string* appeared in the source passed while two GPUs compared a configuration
with itself for three hours. These tests therefore compare generated geometry, never signatures.
"""
import numpy as np
import pytest

from synth.mt_generator import GenConfig, _sample_length, sample_scene

SHAPE = (384, 384)


def _lengths(cfg, seeds):
    out = []
    for s in seeds:
        inst, _ = sample_scene(SHAPE, np.random.default_rng(s), cfg)
        for ins in inst:
            cl = ins["centerline"]
            if len(cl) > 1:
                out.append(float(np.sqrt(((cl[1:] - cl[:-1]) ** 2).sum(1)).sum()))
    return np.asarray(out)


def test_default_is_bit_identical_to_the_previous_behaviour():
    """(0.0, 0.0) must leave the sampler untouched, so every earlier run stays reproducible."""
    cfg = GenConfig()
    assert cfg.frame_length_shift_range == (0.0, 0.0)
    a = _lengths(cfg, range(4))
    b = _lengths(GenConfig(frame_length_shift_range=(0.0, 0.0)), range(4))
    assert np.array_equal(a, b)


def test_a_negative_shift_actually_shortens_the_microtubules():
    """The whole point: reach the 15-60 px regime the HTW corpus lives in."""
    base = GenConfig(length_range=(8.0, 950.0))
    short = GenConfig(length_range=(8.0, 950.0), frame_length_shift_range=(-2.5, -2.5))
    lb, ls = _lengths(base, range(6)), _lengths(short, range(6))
    assert len(lb) > 20 and len(ls) > 20
    assert np.median(ls) < 0.25 * np.median(lb), (np.median(ls), np.median(lb))


def test_the_shift_is_per_FRAME_not_per_microtubule():
    """Within one frame every filament shares the scale; across frames the medians spread.

    A per-filament draw would give the same within-frame spread as the between-frame spread;
    a per-frame draw makes the between-frame variance dominate.
    """
    cfg = GenConfig(length_range=(8.0, 950.0), frame_length_shift_range=(-2.5, 0.5))
    per_frame = []
    for s in range(12):
        inst, _ = sample_scene(SHAPE, np.random.default_rng(s), cfg)
        L = [float(np.sqrt(((i["centerline"][1:] - i["centerline"][:-1]) ** 2).sum(1)).sum())
             for i in inst if len(i["centerline"]) > 1]
        if len(L) >= 3:
            per_frame.append(np.log(np.asarray(L)))
    assert len(per_frame) >= 8
    between = np.var([f.mean() for f in per_frame])
    within = np.mean([f.var() for f in per_frame])
    assert between > 0.3 * within, (between, within)


def test_the_scene_reports_the_shift_it_used():
    """Downstream code (and any future calibration) must be able to read the frame's scale."""
    cfg = GenConfig(frame_length_shift_range=(-1.0, -1.0))
    _, cond = sample_scene(SHAPE, np.random.default_rng(0), cfg)
    assert cond["length_shift"] == pytest.approx(-1.0)


def test_the_clamp_no_longer_piles_samples_onto_its_upper_edge():
    """The shipped calibration puts ~35 % of draws exactly on length_range[1] = 950.

    That spike is an artefact, not a length distribution, so a widened range must remove it.
    The clamp acts inside ``_sample_length``; a finished centerline's arc length differs from
    the drawn value because the curve is bent and smoothed afterwards, so measure it here.
    """
    rng = np.random.default_rng(0)
    piled = GenConfig(length_log_mean=6.6, length_log_sigma=0.55, length_range=(80.0, 950.0))
    spread = GenConfig(length_log_mean=6.6, length_log_sigma=0.55, length_range=(10.0, 3000.0),
                       frame_length_shift_range=(-3.0, 0.0))
    lp = np.array([_sample_length(rng, piled) for _ in range(4000)])
    ls = np.array([_sample_length(rng, spread, length_shift=rng.uniform(-3.0, 0.0))
                   for _ in range(4000)])
    assert float(np.mean(np.isclose(lp, 950.0))) > 0.15          # the defect is real
    assert float(np.mean(np.isclose(ls, 3000.0))) < 0.01         # and the fix removes it
    # and the widened prior must actually reach the short regime the HTW corpus lives in
    assert float(np.mean(ls < 60.0)) > 0.10, float(np.mean(ls < 60.0))


def test_count_coupling_keeps_short_frames_from_being_nearly_empty():
    """Without it, a short-filament frame carries a fraction of the polymer of a long one and the
    network would learn that short microtubules are RARE rather than small."""
    short = dict(length_range=(8.0, 950.0), frame_length_shift_range=(-2.5, -2.5))
    off = GenConfig(**short)
    on = GenConfig(**short, frame_length_count_coupling=1.0)
    long_ref = GenConfig(length_range=(8.0, 950.0))

    def total_fg(cfg, seeds):
        out = []
        for s in seeds:
            inst, _ = sample_scene(SHAPE, np.random.default_rng(s), cfg)
            m = np.zeros(SHAPE, bool)
            for i in inst:
                m |= i["mask"]
            out.append(float(m.mean()))
        return float(np.mean(out))

    fg_long, fg_off, fg_on = (total_fg(c, range(6)) for c in (long_ref, off, on))
    assert fg_off < 0.5 * fg_long, (fg_off, fg_long)          # the problem is real
    assert fg_on > 2.0 * fg_off, (fg_on, fg_off)              # and the coupling fixes it


def test_count_coupling_defaults_to_off():
    assert GenConfig().frame_length_count_coupling == 0.0
    cfg = GenConfig(length_range=(8.0, 950.0), frame_length_shift_range=(-2.0, -2.0))
    a = _lengths(cfg, range(3))
    b = _lengths(GenConfig(length_range=(8.0, 950.0), frame_length_shift_range=(-2.0, -2.0),
                           frame_length_count_coupling=0.0), range(3))
    assert np.array_equal(a, b)
