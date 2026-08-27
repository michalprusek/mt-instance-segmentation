"""The tracking metric must not reward a foreground for missing microtubules.

``tracks_per_object`` averages over every ground-truth object, and an object nobody detected
contributes ZERO tracks -- the best possible value. So a foreground that simply finds fewer
filaments scores as less fragmented, which is an error in the measured model's favour and in
exactly the direction that gets quoted. ``detected_frac`` / ``tracks_per_detected_object`` /
``completeness_detected`` were added to separate the two populations, and these tests pin the
property that makes them worth having.

``score()`` needs no model and no files: the oracle arm rasterises ground-truth polylines and
runs the real instancer and tracker over them, so a hand-built sequence exercises the whole
aggregation in well under a second.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from eval_tracking import score

_PARAMS = json.load(open(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "instance", "params_a_model_synthtuned.json")))
H = W = 200


def _straight(y, x0, x1, n=40):
    """A ground-truth polyline, ``(x=col, y=row)`` -- the transpose of numpy indexing."""
    return np.stack([np.linspace(x0, x1, n), np.full(n, float(y))], axis=1)


def _sequence(polylines_per_frame, ids):
    """One sequence in the shape ``load_sequences`` returns."""
    frames = []
    for k in range(len(polylines_per_frame)):
        fr = {"attrs": {"height": H, "width": W, "seq_id": 0, "frame_idx": k,
                        "inst_ids": list(ids), "speeds": [0.0] * len(ids)},
              "polylines": [np.asarray(p, float) for p in polylines_per_frame[k]]}
        frames.append((k, f"seq000_f{k:02d}.h5", fr))
    return {0: frames}


def _score(polylines_per_frame, ids):
    return score(_sequence(polylines_per_frame, ids), "oracle", "", _PARAMS, 0.5)


LONG_A = _straight(60, 20, 170)
LONG_B = _straight(120, 20, 170)
TINY = _straight(170, 95, 103, n=5)      # far below the instancer's min_length: never detected


def test_two_well_separated_filaments_are_each_followed_by_one_track():
    """The reference point the other tests are read against."""
    r = _score([[LONG_A, LONG_B]] * 3, [1, 2])
    assert r["n_objects"] == 2
    assert r["detected_frac"] == pytest.approx(1.0)
    assert r["tracks_per_object"] == pytest.approx(1.0)
    assert r["completeness"] == pytest.approx(1.0)


def test_when_everything_is_detected_the_new_metrics_equal_the_old_ones():
    """The additive claim, as a property rather than a one-off check: the decomposition may
    only ever differ from the original where something was missed."""
    r = _score([[LONG_A, LONG_B]] * 3, [1, 2])
    assert r["detected_frac"] == pytest.approx(1.0)
    assert r["tracks_per_detected_object"] == pytest.approx(r["tracks_per_object"])
    assert r["completeness_detected"] == pytest.approx(r["completeness"])


def test_an_undetected_microtubule_must_not_improve_the_fragmentation_score():
    """Add one filament too short for the instancer to ever emit, and change nothing else.

    The foreground is now strictly worse -- it misses an object it did not miss before -- so
    ``tracks_per_object`` FALLING is the bug this decomposition exists to expose. The
    conditional metrics must not move, and the detection rate must carry the loss instead.
    """
    both = _score([[LONG_A, LONG_B]] * 3, [1, 2])
    plus_missed = _score([[LONG_A, LONG_B, TINY]] * 3, [1, 2, 3])

    assert plus_missed["n_objects"] == both["n_objects"] + 1
    assert plus_missed["tracks_per_object"] < both["tracks_per_object"], \
        "the flattering behaviour is gone -- this test no longer measures anything"
    assert plus_missed["detected_frac"] < both["detected_frac"]
    assert plus_missed["tracks_per_detected_object"] == pytest.approx(
        both["tracks_per_detected_object"]), \
        "a missed object leaked into the conditional metric"
    assert plus_missed["completeness_detected"] == pytest.approx(both["completeness_detected"])


def test_detected_frac_is_a_per_object_rate_not_a_per_frame_one():
    """``n_objects`` is the denominator, so the ratio must not move with sequence length --
    which is what counting frames instead of objects would do."""
    r2 = _score([[LONG_A, LONG_B, TINY]] * 2, [1, 2, 3])
    r5 = _score([[LONG_A, LONG_B, TINY]] * 5, [1, 2, 3])
    assert r2["detected_frac"] == pytest.approx(2 / 3)
    assert r2["detected_frac"] == pytest.approx(r5["detected_frac"])
    assert r2["n_objects"] == r5["n_objects"] == 3


# --------------------------------------------------------------------------------------------
# The headline claim of this harness is a switch RATE, and nothing pinned it.
# --------------------------------------------------------------------------------------------

def test_a_clean_sequence_has_no_identity_switches():
    r = _score([[LONG_A, LONG_B]] * 3, [1, 2])
    assert r["id_switches"] == 0
    assert r["links"] > 0, "a zero denominator would make switch_rate 0.000 for free"
    assert r["switch_rate"] == pytest.approx(0.0)


def test_two_filaments_that_swap_places_are_counted_as_switches():
    """The metric must be able to report a nonzero rate at all -- `switches += ...` deleted
    outright would otherwise leave every test green while the headline number reads perfect."""
    close_a = _straight(100, 20, 170)
    close_b = _straight(106, 20, 170)          # 6 px apart: inside the association gates
    frames = [[close_a, close_b], [close_b, close_a], [close_a, close_b]]
    r = score(_sequence(frames, [1, 2]), "oracle", "", _PARAMS, 0.5)
    assert r["links"] > 0
    assert r["id_switches"] > 0, "a deliberate identity swap produced no switches"


def test_switch_rate_has_a_real_denominator():
    """`switch_rate = switches / max(links, 1)` reads 0.000 when links is 0, which is how a
    total failure to recover the correspondence would have looked like a perfect score."""
    r = _score([[LONG_A, LONG_B]] * 4, [1, 2])
    assert r["links"] == pytest.approx(6), "2 objects x 3 frame steps"


# --------------------------------------------------------------------------------------------
# The pieces the model arm needs, exercised without a checkpoint.
# --------------------------------------------------------------------------------------------

def test_a_missing_prediction_is_refused_before_any_scoring(tmp_path):
    """One absent .npz used to drop a whole sequence from the model arm while the oracle arm
    kept it, so two rows printed as a comparison were scored on different populations."""
    from eval_tracking import MissingPrediction, missing_predictions

    seqs = _sequence([[LONG_A]] * 2, [1])
    assert missing_predictions(seqs, str(tmp_path)) == ["seq000_f00.h5", "seq000_f01.h5"]

    np.savez(tmp_path / "seq000_f00.npz", prob=np.zeros((1, 10, 10), np.float16))
    assert missing_predictions(seqs, str(tmp_path)) == ["seq000_f01.h5"]

    with pytest.raises(MissingPrediction, match="seq000_f01"):
        score(seqs, "model", str(tmp_path), _PARAMS, None)


def test_a_missing_prediction_does_not_kill_a_caller_that_handles_exceptions():
    """SystemExit derives from BaseException and escapes `except Exception`, so a sweep driver
    looping over thresholds would be terminated rather than told."""
    from eval_tracking import MissingPrediction

    assert issubclass(MissingPrediction, Exception)
    assert not issubclass(MissingPrediction, SystemExit)


def test_the_cli_threshold_beats_the_params_file():
    from eval_tracking import effective_threshold

    assert effective_threshold({"prob_thr": 0.44}, None) == pytest.approx(0.44)
    assert effective_threshold({"prob_thr": 0.44}, 0.9) == pytest.approx(0.9)
    assert effective_threshold({}, None) == pytest.approx(0.35)


# --------------------------------------------------------------------------------------------
# The rule that decides which object a track follows, and the unit that scales its speed.
# --------------------------------------------------------------------------------------------

def test_assign_to_gt_needs_half_the_detection_to_lie_on_one_object():
    """This threshold defines the whole notion of 'detected', so it drives both the switch
    count and detected_frac."""
    from eval_tracking import assign_to_gt

    gt = [_straight(60, 0, 100)]
    mostly_on = np.stack([np.linspace(0, 100, 20), np.full(20, 60.0)], axis=1)
    mostly_off = np.stack([np.linspace(0, 100, 20),
                           np.concatenate([np.full(6, 60.0), np.full(14, 300.0)])], axis=1)
    assert assign_to_gt([mostly_on], gt, [7]) == [7]
    assert assign_to_gt([mostly_off], gt, [7]) == [None]
    assert assign_to_gt([mostly_on], [], []) == [None]


def test_the_stored_speed_is_lifted_into_the_eval_frame():
    """Speeds are stored at native scale while every polyline here lives at 1.5x. Dropping the
    conversion is the exact mismatch that once produced a false velocity underestimate, and it
    still reaches the JSON even though the printed table omits velocity."""
    import eval_tracking

    moving = [[_straight(60, 20 + 6 * k, 170 + 6 * k)] for k in range(4)]
    seqs = _sequence(moving, [1])
    for _, _, fr in seqs[0]:
        fr["attrs"]["speeds"] = [6.0]
    r = score(seqs, "oracle", "", _PARAMS, 0.5)
    assert r["velocity_n"] > 0, "no velocity was measured, so this pins nothing"
    # The ratio is measured against 6.0 * UP; without the scaling it would be inflated by 1.5.
    assert r["velocity_ratio_median"] == pytest.approx(1.0, abs=0.35)
    assert eval_tracking.UP == 1.5
