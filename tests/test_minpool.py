"""Regression tests for the WORST-POINT term (train_v5.minpool_term + the curve-id channel).

Declared 2026-09-11 after `scripts/diag_model_only.py` measured, with no instancer in the loop,
that the median per-curve MEAN probability is 0.95-0.99 in every domain while the median per-curve
MINIMUM is 0.94 in domain and 0.51-0.87 on real data. The term exists to make the objective read
the minimum, so these are the properties that must never break:

  * OFF is bit-identical: without `curve_ids` the target array is unchanged, byte for byte.
  * the id channel is appended LAST, so every historical channel index still means what it meant.
  * the term reads the MINIMUM, not the mean: appending confident pixels to a curve must not
    dilute it, which is exactly what the per-pixel BCE does.
  * background (id 0) is ignored, and short crop fragments are skipped.
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))


def _mod():
    path = os.path.join(ROOT, "scripts", "train_v5.py")
    spec = importlib.util.spec_from_file_location("train_v5_for_minpool_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _mod()


def _curve(x0, x1, y):
    return {"centerline": np.stack([np.arange(x0, x1, 0.5), np.full(int((x1 - x0) * 2), y)],
                                   axis=1)}


# ------------------------------------------------------------------ the target channel

def test_off_is_bit_identical():
    """The flag must not perturb the target of any previous run."""
    ins = [_curve(5, 40, 10), _curve(5, 40, 30)]
    a = T.ori_channels_soft(ins, (48, 48))
    b = T.ori_channels_soft(ins, (48, 48), curve_ids=False)
    assert a.shape == b.shape
    assert np.array_equal(a, b)


def test_id_channel_is_appended_last_and_owns_the_foreground():
    ins = [_curve(5, 40, 10), _curve(5, 40, 30)]
    base = T.ori_channels_soft(ins, (48, 48))
    with_ids = T.ori_channels_soft(ins, (48, 48), curve_ids=True)
    assert with_ids.shape[0] == base.shape[0] + 1
    # every historical index is untouched
    assert np.array_equal(with_ids[:base.shape[0]], base)
    cid = with_ids[-1]
    fg = with_ids[T.K_ORI]
    assert set(np.unique(cid)) == {0.0, 1.0, 2.0}
    # the id map covers exactly the union foreground: no owner-less foreground pixel, and no id
    # outside the target the BCE sees.
    assert np.array_equal(cid > 0, fg > 0)
    assert cid[10, 20] == 1.0 and cid[30, 20] == 2.0


def test_id_channel_survives_the_heat_channel():
    ins = [_curve(5, 40, 10)]
    both = T.ori_channels_soft(ins, (48, 48), heat_sigma=2.0, curve_ids=True)
    heat_only = T.ori_channels_soft(ins, (48, 48), heat_sigma=2.0)
    assert both.shape[0] == heat_only.shape[0] + 1
    assert np.array_equal(both[:heat_only.shape[0]], heat_only)   # heat keeps K_ORI+2
    assert set(np.unique(both[-1])) == {0.0, 1.0}


# ------------------------------------------------------------------ the loss term

def _pair(logit_values, ids, shape=(1, 8)):
    lg = torch.tensor(np.asarray(logit_values, np.float32).reshape(1, 1, *shape))
    cd = torch.tensor(np.asarray(ids, np.float32).reshape(1, 1, *shape))
    return lg, cd


def test_a_single_dip_dominates_the_term():
    """One weak pixel in an otherwise confident curve must cost a lot. That is the whole point."""
    flat = [6.0] * 8
    dipped = [6.0] * 7 + [-4.0]
    l_flat = T.minpool_term(*_pair(flat, [1] * 8), temp=1.0, min_px=4)
    l_dip = T.minpool_term(*_pair(dipped, [1] * 8), temp=1.0, min_px=4)
    assert float(l_flat) < 0.01
    assert float(l_dip) > 1.0
    # and the per-pixel BCE, which this term exists to supplement, barely notices
    import torch.nn.functional as F
    b_flat = F.binary_cross_entropy_with_logits(torch.tensor(flat), torch.ones(8))
    b_dip = F.binary_cross_entropy_with_logits(torch.tensor(dipped), torch.ones(8))
    assert float(b_dip - b_flat) < float(l_dip - l_flat)


def test_it_reads_the_minimum_not_the_mean():
    """Padding a curve with confident pixels dilutes a mean; it must not dilute this."""
    short = T.minpool_term(*_pair([-4.0] + [6.0] * 7, [1] * 8), temp=1.0, min_px=4)
    long_ = T.minpool_term(*_pair([-4.0] + [6.0] * 63, [1] * 64, shape=(1, 64)),
                           temp=1.0, min_px=4)
    assert abs(float(short) - float(long_)) < 0.05 * float(short)


def test_background_is_ignored():
    a = T.minpool_term(*_pair([6.0] * 4 + [-9.0] * 4, [1, 1, 1, 1, 0, 0, 0, 0]),
                       temp=1.0, min_px=4)
    b = T.minpool_term(*_pair([6.0] * 4 + [+9.0] * 4, [1, 1, 1, 1, 0, 0, 0, 0]),
                       temp=1.0, min_px=4)
    assert float(a) == pytest.approx(float(b), abs=1e-6)


def test_curves_are_weighted_equally_regardless_of_length():
    """A 4 px curve in trouble must count as much as a 60 px curve that is fine."""
    ids = [1] * 4 + [2] * 60
    logits = [-4.0] * 4 + [6.0] * 60
    got = float(T.minpool_term(*_pair(logits, ids, shape=(1, 64)), temp=1.0, min_px=4))
    lonely = float(T.minpool_term(*_pair([-4.0] * 4, [1] * 4, shape=(1, 4)), temp=1.0, min_px=4))
    assert got == pytest.approx(lonely / 2.0, rel=0.05)


def test_short_fragments_are_skipped():
    ids = [1, 1, 1] + [0] * 5                      # 3 px curve, below min_px
    val = T.minpool_term(*_pair([-9.0] * 3 + [0.0] * 5, ids), temp=1.0, min_px=8)
    assert float(val) == 0.0


def test_empty_batch_is_an_exact_zero_and_still_differentiable():
    lg, cd = _pair([1.0] * 8, [0] * 8)
    lg.requires_grad_(True)
    val = T.minpool_term(lg, cd, temp=1.0, min_px=4)
    assert float(val) == 0.0
    val.backward()
    assert torch.allclose(lg.grad, torch.zeros_like(lg))


def test_gradient_reaches_the_weak_pixel():
    lg, cd = _pair([6.0] * 7 + [-4.0], [1] * 8)
    lg.requires_grad_(True)
    T.minpool_term(lg, cd, temp=1.0, min_px=4).backward()
    g = lg.grad.reshape(-1).abs()
    assert float(g[-1]) > 10 * float(g[:7].max())    # the dip carries the gradient


def test_temperature_limit_is_the_hard_minimum():
    logits = [6.0, 2.0, -4.0, 5.0, 6.0, 6.0, 6.0, 6.0]
    hard = T.minpool_term(*_pair(logits, [1] * 8), temp=1e-3, min_px=4)
    import torch.nn.functional as F
    want = F.binary_cross_entropy_with_logits(torch.tensor([-4.0]), torch.ones(1))
    assert float(hard) == pytest.approx(float(want), abs=1e-3)
