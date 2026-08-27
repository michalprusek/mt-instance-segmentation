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
