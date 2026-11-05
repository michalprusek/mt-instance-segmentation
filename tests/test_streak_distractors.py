"""Arm 41: elongated non-microtubule distractors in the generator background.

The default must be a no-op. Every calibration file, every render and every published number
predates this feature, so a non-zero default would silently invalidate the whole campaign rather
than fail loudly.
"""
import numpy as np

from mt_generator import GenConfig, add_streaks


def test_default_is_a_no_op():
    cfg = GenConfig()
    assert cfg.streak_rate == 0.0
    f = np.zeros((64, 64), np.float32)
    n = add_streaks(f, np.random.default_rng(0), cfg)
    assert n == 0
    assert not f.any()


def test_streaks_are_drawn_when_enabled():
    cfg = GenConfig(streak_rate=8.0)
    f = np.zeros((256, 256), np.float32)
    n = add_streaks(f, np.random.default_rng(1), cfg)
    assert n > 0
    assert f.any()


def test_streaks_are_elongated_not_blobs():
    """A distractor must look like a line, or it is just another dirt speck (`add_spots`)."""
    cfg = GenConfig(streak_rate=1.0, streak_len_range=(60.0, 60.0),
                    streak_width_range=(1.0, 1.0), streak_amp_range=(1.0, 1.0),
                    streak_curv_range=(0.0, 0.0), streak_kink_frac=0.0)
    best = 0.0
    for seed in range(12):
        f = np.zeros((256, 256), np.float32)
        if add_streaks(f, np.random.default_rng(seed), cfg) == 0:
            continue
        yx = np.argwhere(np.abs(f) > 0.2 * np.abs(f).max())
        if len(yx) < 30:
            continue
        c = yx - yx.mean(0)
        ev = np.linalg.eigvalsh(np.cov(c.T))
        best = max(best, float(ev[1] / max(ev[0], 1e-6)))
    assert best > 20.0, f"most elongated component had axis ratio {best:.1f}"


def test_rate_range_overrides_the_scalar_rate():
    cfg = GenConfig(streak_rate=0.0, streak_rate_range=(20.0, 20.0))
    f = np.zeros((256, 256), np.float32)
    assert add_streaks(f, np.random.default_rng(2), cfg) > 0
