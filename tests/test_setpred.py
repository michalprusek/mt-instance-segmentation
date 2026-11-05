"""Guards for the two pure functions the set predictor's loss rests on."""
import numpy as np
import pytest
import torch

from setpred_model import curve_l1, resample_curve


def test_resample_is_uniform_in_arclength_not_vertex_index():
    """A polyline with deliberately uneven vertex spacing must still resample uniformly.

    This is the whole point: generator and human polylines cluster vertices wherever the tracer
    happened to click, so index-uniform sampling would weight the loss by annotation habit.
    """
    p = torch.tensor([[0., 0.], [1., 0.], [2., 0.], [3., 0.], [100., 0.]])
    out = resample_curve(p, 5)
    step = (out[1:] - out[:-1]).norm(dim=1)
    assert torch.allclose(step, step[0].expand_as(step), atol=1e-4), step


def test_resample_keeps_endpoints():
    p = torch.tensor([[0., 0.], [10., 5.], [20., 0.]])
    out = resample_curve(p, 7)
    assert torch.allclose(out[0], p[0], atol=1e-5)
    assert torch.allclose(out[-1], p[-1], atol=1e-5)


def test_resample_degenerate_inputs():
    assert resample_curve(torch.tensor([[3., 4.]]), 4).shape == (4, 2)
    same = torch.tensor([[1., 1.], [1., 1.], [1., 1.]])
    assert resample_curve(same, 3).shape == (3, 2)


def test_curve_l1_is_direction_invariant():
    """The same filament stored backwards must cost exactly the same."""
    a = torch.rand(16, 2)
    b = torch.rand(16, 2)
    assert torch.allclose(curve_l1(a, b), curve_l1(a, b.flip(0)), atol=1e-6)


def test_curve_l1_zero_on_identical_and_reversed():
    a = torch.rand(16, 2)
    assert curve_l1(a, a) < 1e-6
    assert curve_l1(a, a.flip(0)) < 1e-6


def test_curve_l1_direction_invariance_actually_matters():
    """Guard the guard: a plain forward L1 would FAIL these, so the min() is load-bearing."""
    a = torch.linspace(0, 1, 16)[:, None].repeat(1, 2)
    rev = a.flip(0)
    plain_forward = (a - rev).abs().mean()
    assert plain_forward > 0.3, "test curve too symmetric to detect a direction bug"
    assert curve_l1(a, rev) < 1e-6


def test_curve_l1_batches():
    p = torch.rand(3, 5, 16, 2)
    t = torch.rand(3, 5, 16, 2)
    assert curve_l1(p, t).shape == (3, 5)
