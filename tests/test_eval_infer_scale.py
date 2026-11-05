"""``eval_v5.py --infer-scale``: the network may run at native scale, the metric never moves.

Two guarantees, both mutation-tested by hand when written:
* the historical path (network on the 1.5x image) is an IDENTITY through ``resample_to_eval`` --
  same object out, so no historical number can move;
* a native-scale stack lands on exactly the shape ``scipy.ndimage.zoom(img, 1.5)`` produces, for
  even AND odd sizes, so the resampled probabilities line up with the 1.5x-scaled polylines.
"""
import os
import sys

import numpy as np
import pytest
from scipy.ndimage import zoom

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from eval_v5 import UP, checkpoint_in_channels, eval_frame_shape, resample_to_eval  # noqa: E402


@pytest.mark.parametrize("shape", [(1024, 1024), (1192, 1192), (730, 732), (2048, 2047), (511, 513)])
def test_eval_frame_shape_matches_scipy_zoom(shape):
    img = np.zeros(shape, np.float32)
    assert eval_frame_shape(shape) == zoom(img, UP, order=1).shape


def test_historical_path_is_identity():
    ch = np.random.default_rng(0).random((7, 1536, 1536), np.float32)
    out = resample_to_eval(ch, (1536, 1536))
    assert out is ch                                   # not merely equal: the same object


@pytest.mark.parametrize("shape", [(1024, 1024), (1192, 1192), (730, 732), (511, 513)])
def test_native_stack_lands_on_eval_frame(shape):
    ch = np.full((2,) + shape, 0.25, np.float32)
    ch[1] = np.linspace(0, 1, shape[1], dtype=np.float32)[None, :]
    target = eval_frame_shape(shape)
    out = resample_to_eval(ch, target)
    assert out.shape == (2,) + target
    assert out.dtype == np.float32
    # a constant channel stays constant; a linear ramp stays a ramp with the same endpoints
    assert np.allclose(out[0], 0.25)
    assert out[1, 0, 0] == pytest.approx(0.0, abs=1e-6) and out[1, 0, -1] == pytest.approx(1.0, abs=1e-6)


def test_native_ridge_moves_with_the_frame():
    """A ridge at native column 100 must land at eval column 150 -- the same place the 1.5x-scaled
    polyline puts the ground truth. This is the alignment the whole flag depends on."""
    ch = np.zeros((1, 400, 400), np.float32)
    ch[0, :, 100] = 1.0
    out = resample_to_eval(ch, eval_frame_shape((400, 400)))
    assert int(np.argmax(out[0, 200])) == 150


def test_checkpoint_in_channels_reads_the_stem():
    class T:
        def __init__(self, shape):
            self.shape, self.ndim = shape, len(shape)
    plain = {"stem.w": T((32, 3, 3, 3)), "b1.w": T((64, 32, 3, 3)), "bias": T((32,))}
    prior = {"stem.w": T((32, 38, 3, 3)), "b1.w": T((64, 32, 3, 3)), "head.w": T((1, 32, 1, 1))}
    assert checkpoint_in_channels(plain) == 3
    assert checkpoint_in_channels(prior) == 38
    assert checkpoint_in_channels({}) == 3


def test_resampling_is_bilinear_not_nearest():
    """Nearest-neighbour would leave a 1 px impulse as 0/1 only; bilinear must produce in-between
    values. The probabilities are what the instancer thresholds, so the interpolant matters."""
    ch = np.zeros((1, 64, 64), np.float32)
    ch[0, 32, 32] = 1.0
    out = resample_to_eval(ch, eval_frame_shape((64, 64)))
    mid = out[(out > 0.05) & (out < 0.95)]
    assert mid.size > 0
    assert out.max() <= 1.0 + 1e-6 and out.min() >= 0.0


def test_checkpoint_in_channels_prefers_the_narrowest_external_width():
    """A decoder conv that reads a skip CONCATENATION (64 + 32 = 96) also has an input width no
    single layer produces. The stem is the narrowest such width, never the widest."""
    class T:
        def __init__(self, shape):
            self.shape, self.ndim = shape, len(shape)
    st = {"stem.w": T((32, 38, 3, 3)), "e1.w": T((64, 32, 3, 3)),
          "dec.w": T((32, 96, 3, 3)), "head.w": T((1, 32, 1, 1))}
    assert checkpoint_in_channels(st) == 38
