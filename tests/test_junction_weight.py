"""Regression tests for the crossing-region loss weight (train_v5.junction_weight).

Declared as campaign arm 32 (docs/TODO.md 0.0-CAMPAIGN, 2026-09-07). The weight multiplies the
per-pixel BCE inside a radius of crossing pixels; the crossing seed is the generator's own
multiplicity channel. Two properties must never break: OFF must be bit-identical to the old
loss, and close parallels must not be labelled as crossings.
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))


def _jw():
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "scripts", "train_v5.py")
    spec = importlib.util.spec_from_file_location("train_v5_for_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_v5_for_test"] = mod
    try:
        spec.loader.exec_module(mod)
    except SystemExit:          # the module parses argv at import in some paths
        pass
    return mod.junction_weight


def test_weight_is_wj_inside_the_radius_and_one_outside():
    cnt = torch.zeros(1, 1, 40, 40)
    cnt[0, 0, 20, 20] = 2.0
    w = _jw()(cnt, 4, 6.0)
    assert float(w.max()) == 6.0
    assert float(w.min()) == 1.0
    assert int((w > 1).sum()) == 9 * 9          # square dilation of one seed pixel


def test_multiplicity_one_is_never_a_crossing():
    """A close parallel raster has cnt == 1 on each ridge; it must not be weighted."""
    w = _jw()(torch.ones(1, 1, 8, 8), 3, 6.0)
    assert bool((w == 1).all())


def test_unit_weight_reproduces_the_plain_bce_exactly():
    """w_J = 1 must leave the loss untouched -- this is what makes --jw-radius 0 safe."""
    torch.manual_seed(0)
    o = torch.randn(2, 1, 16, 16)
    t = (torch.rand(2, 1, 16, 16) > 0.9).float()
    pw = torch.tensor(8.0)
    cnt = torch.zeros(2, 1, 16, 16)
    cnt[:, :, 8, 8] = 2
    wm = _jw()(cnt, 3, 1.0)
    el = F.binary_cross_entropy_with_logits(o, t, pos_weight=pw, reduction="none")
    weighted = float((wm * el).sum() / (wm.sum() * o.shape[1]))
    plain = float(F.binary_cross_entropy_with_logits(o, t, pos_weight=pw))
    assert weighted == pytest.approx(plain, abs=1e-6)


def test_radius_scales_for_deep_supervision_levels():
    """A deep level halves the radius with the resolution; the region must shrink accordingly."""
    cnt = torch.zeros(1, 1, 64, 64)
    cnt[0, 0, 32, 32] = 2.0
    full = int((_jw()(cnt, 8, 6.0) > 1).sum())
    half = int((_jw()(F.max_pool2d(cnt, 2, 2), 4, 6.0) > 1).sum())
    assert full == 17 * 17 and half == 9 * 9
