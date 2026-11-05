"""The training target's width.

`ori_channels_soft` accepted a `half_width` argument and then IGNORED it: the width was a
hard-coded 3x3 dilation. A sweep of that flag would have compared a model against itself and
recorded a guaranteed null, which is worse than no lever at all. Found 2026-09-01.
"""
import importlib.util
import sys

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("t5", "scripts/train_v5.py")
t5 = importlib.util.module_from_spec(spec)
sys.modules["t5"] = t5
try:
    spec.loader.exec_module(t5)
except SystemExit:
    pass


def _scene(seed=0, n=4):
    rng = np.random.default_rng(seed)
    return [{"centerline": np.stack([np.cumsum(rng.normal(0, 3, 120)) + 128,
                                     np.cumsum(rng.normal(0, 3, 120)) + 128], 1)}
            for _ in range(n)]


def test_half_width_actually_changes_the_target():
    """The property the argument existed to have, and did not."""
    a = t5.ori_channels_soft(_scene(), (256, 256), half_width=1.0)
    b = t5.ori_channels_soft(_scene(), (256, 256), half_width=2.0)
    assert not np.array_equal(a, b)


def test_wider_is_wider():
    fg = [t5.ori_channels_soft(_scene(), (256, 256), half_width=h)[t5.K_ORI].sum()
          for h in (1.0, 2.0, 3.0, 4.0)]
    assert all(x < y for x, y in zip(fg, fg[1:])), fg


def test_default_is_the_inherited_three_pixel_target():
    """Every number this project has measured used a 1 px centerline dilated by a full 3x3 SQUARE.
    skimage's disk(1) is a cross, not a square, so the default must not go through disk()."""
    from scipy.ndimage import binary_dilation
    line = np.zeros((64, 64), bool)
    line[32, 10:54] = True
    square = binary_dilation(line, structure=np.ones((3, 3), bool))
    ins = [{"centerline": np.stack([np.arange(10, 54, 0.5),
                                    np.full(88, 32.0)], 1)}]
    fg = t5.ori_channels_soft(ins, (64, 64), half_width=1.0)[t5.K_ORI] > 0
    assert fg[:, 12:52].sum() == square[:, 12:52].sum(), "the default target width changed"


def test_zero_half_width_is_the_bare_one_pixel_centerline():
    """The THINNEST target the project can ask for, which is the direction the user wants and the
    one the old hard-coded 3x3 made unreachable."""
    a = t5.ori_channels_soft(_scene(), (256, 256), half_width=0.0)[t5.K_ORI]
    b = t5.ori_channels_soft(_scene(), (256, 256), half_width=1.0)[t5.K_ORI]
    assert 0 < a.sum() < b.sum() * 0.6, (a.sum(), b.sum())
    # A diagonal sampled every 0.4 px does fill the odd 2x2 block, so demanding none would be
    # wrong; what must hold is that the undilated target has far fewer of them.
    def blocks(x):
        return float((x[:-1, :-1] * x[1:, :-1] * x[:-1, 1:] * x[1:, 1:]).sum())
    assert blocks(a) < blocks(b) * 0.35, (blocks(a), blocks(b))
