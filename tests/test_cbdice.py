import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cbdice_loss import soft_cbdice  # noqa: E402


def _line_target(h=64, w=64, row=32, half=1, k=6, chan=0):
    """A single horizontal filament in one orientation channel."""
    t = torch.zeros(1, k, h, w)
    t[0, chan, row - half:row + half + 1, :] = 1.0
    return t


def _logits_from(mask: torch.Tensor, on=6.0, off=-6.0):
    """Confident logits reproducing a given binary mask."""
    return torch.where(mask > 0.5, torch.full_like(mask, on), torch.full_like(mask, off))


def test_a_perfect_prediction_scores_near_zero():
    t = _line_target()
    loss = soft_cbdice(_logits_from(t), t)
    assert 0.0 <= float(loss) < 0.35


def test_a_broken_filament_costs_more_than_a_whole_one():
    """The property the loss is being adopted for.

    The measured failure is a mask with the right coverage that is shattered into pieces: a
    one-pixel break splits a component while costing one pixel of overlap. A loss that does not
    charge more for the break than for the missing pixel cannot fix it.
    """
    t = _line_target()
    whole = t.clone()
    broken = t.clone()
    broken[0, 0, :, 30:33] = 0.0                     # a 3 px gap, ~5 % of the length
    l_whole = float(soft_cbdice(_logits_from(whole), t))
    l_broken = float(soft_cbdice(_logits_from(broken), t))
    assert l_broken > l_whole + 0.02, f"break not penalised: {l_whole:.3f} -> {l_broken:.3f}"


def test_a_break_costs_more_than_the_same_pixels_removed_from_an_end():
    """Losing three pixels from a tip changes no topology; losing three from the middle does.

    This separates cbDice from a plain overlap loss, which cannot tell the two apart.
    """
    t = _line_target(w=64)
    middle = t.clone()
    middle[0, 0, :, 30:33] = 0.0
    tip = t.clone()
    tip[0, 0, :, 61:64] = 0.0
    l_mid = float(soft_cbdice(_logits_from(middle), t))
    l_tip = float(soft_cbdice(_logits_from(tip), t))
    assert l_mid > l_tip, f"a mid-filament break must cost more than a tip trim ({l_mid:.3f} vs {l_tip:.3f})"


def test_gradients_reach_the_logits_when_there_is_something_to_learn():
    """Gradient must flow from an IMPERFECT prediction.

    Asking for a non-zero gradient at a perfect prediction would be asking for the wrong thing:
    there the loss is exactly 0 and a vanishing gradient is correct. cbDice also saturates once
    the thresholded mask is already right, which is why it is added ALONGSIDE a pixel loss
    rather than replacing one -- it shapes topology and has nothing to say about a mask that is
    topologically correct already.
    """
    t = _line_target()
    broken = t.clone()
    broken[0, 0, :, 30:33] = 0.0
    logits = _logits_from(broken).clone().requires_grad_(True)
    loss = soft_cbdice(logits, t)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0.0


def test_a_perfect_prediction_has_no_gradient_left_to_give():
    t = _line_target()
    logits = _logits_from(t).clone().requires_grad_(True)
    loss = soft_cbdice(logits, t)
    loss.backward()
    assert float(loss.detach()) == pytest.approx(0.0, abs=1e-6)
    assert float(logits.grad.abs().sum()) == pytest.approx(0.0, abs=1e-9)


def test_an_empty_prediction_is_penalised_and_finite():
    t = _line_target()
    empty = torch.zeros_like(t)
    loss = soft_cbdice(_logits_from(empty), t)
    assert torch.isfinite(loss)
    assert float(loss) > 0.5


def test_it_reduces_orientation_channels_by_max_like_soft_cldice():
    """A crossing puts two filaments in different channels; the loss must see one foreground."""
    t = torch.zeros(1, 6, 64, 64)
    t[0, 0, 31:34, :] = 1.0                          # horizontal, channel 0
    t[0, 3, :, 31:34] = 1.0                          # vertical, channel 3
    same = torch.zeros(1, 6, 64, 64)
    same[0, 0] = t.amax(1)[0]                        # both filaments in ONE channel
    a = float(soft_cbdice(_logits_from(t), t))
    b = float(soft_cbdice(_logits_from(same), t))
    assert a == pytest.approx(b, abs=1e-5)


def test_batch_dimension_is_handled():
    t = torch.cat([_line_target(row=20), _line_target(row=44)], dim=0)
    loss = soft_cbdice(_logits_from(t), t)
    assert torch.isfinite(loss)
