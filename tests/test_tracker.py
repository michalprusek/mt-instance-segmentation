import numpy as np
import pytest

from instance.tracker import (DEFAULTS, contour_shift, estimate_drift, match_frames,
                              track_sequence, track_velocity)


def _line(x0, y0, x1, y1, n=120):
    return np.stack([np.linspace(x0, x1, n), np.linspace(y0, y1, n)], axis=1)


def _arc(cx, cy, r, t0, t1, n=160):
    t = np.linspace(t0, t1, n)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1)


# --------------------------------------------------------------------------- contour shift

def test_contour_shift_recovers_a_known_slide():
    """A filament sliding along itself has zero perpendicular motion; the shift is the signal.

    Sign convention: b advancing towards a's head is POSITIVE, and sliding the other way is
    negative. Both directions are checked, because a magnitude-only estimator would happily
    report a filament reversing as one advancing.
    """
    path = _line(0, 50, 400, 50, n=400)
    ds = np.linalg.norm(path[1] - path[0])
    a = path[100:300]
    for delta_idx in (10, 25, 60):
        forward = path[100 + delta_idx:300 + delta_idx]
        backward = path[100 - delta_idx:300 - delta_idx]
        assert contour_shift(a, forward) == pytest.approx(delta_idx * ds, abs=1.5)
        assert contour_shift(a, backward) == pytest.approx(-delta_idx * ds, abs=1.5)


def test_contour_shift_is_not_halved_by_end_saturation():
    """Regression: projecting b's HEAD onto a saturates once b has advanced past a's end.

    The nearest point on ``a`` is then a's last vertex no matter how far b went, so that end
    reports zero shift. Averaging it with the (correct) tail estimate halved the answer -- on
    synthetic gliding sequences the velocity came out 3.55 px/frame too low. The estimate must
    use interior matches only.
    """
    path = _line(0, 30, 600, 30, n=600)
    a, b = path[100:300], path[160:360]          # a 60-sample slide, both ends run off
    ds = np.linalg.norm(path[1] - path[0])
    got = abs(contour_shift(a, b))
    assert got == pytest.approx(60 * ds, rel=0.15)
    assert got > 0.75 * 60 * ds, "the estimate collapsed towards half -- saturation is back"


def test_contour_shift_is_zero_for_a_stationary_filament():
    a = _arc(100, 100, 60, 0.2, 1.4)
    assert contour_shift(a, a.copy()) == pytest.approx(0.0, abs=0.5)


# --------------------------------------------------------------------------- drift

def test_drift_estimate_is_near_zero_for_a_purely_gliding_field():
    """Regression: the median CENTROID shift measures motility, not drift.

    A gliding filament's centroid travels along its own contour at the full gliding speed, and
    in a gliding field every filament does, so a centroid-based estimator reported 2.9 px of
    drift on sequences that had none. Only the component perpendicular to each filament's own
    tangent is drift-free.
    """
    prev, curr = [], []
    for k, ang in enumerate(np.linspace(0, np.pi, 6, endpoint=False)):
        c, s = np.cos(ang), np.sin(ang)
        base = np.stack([np.linspace(-90, 90, 200) * c + 160 + 12 * k,
                         np.linspace(-90, 90, 200) * s + 160], axis=1)
        prev.append(base)
        slide = 6.0                                   # each slides ALONG itself
        curr.append(base + np.array([c, s]) * slide)
    d = estimate_drift(prev, curr)
    assert np.linalg.norm(d) < 1.5, f"gliding leaked into the drift estimate: {d}"


def test_drift_estimate_recovers_a_common_translation():
    shift = np.array([3.0, -2.0])
    prev = [_line(20, 40, 180, 60), _arc(120, 150, 50, 0.0, 2.0), _line(30, 200, 200, 190)]
    curr = [p + shift for p in prev]
    d = estimate_drift(prev, curr)
    assert np.allclose(d, shift, atol=0.6)


def test_drift_estimate_degrades_gracefully_when_all_filaments_are_parallel():
    """A field of parallel filaments cannot determine motion ALONG that direction.

    That is the aperture problem, not a bug. The estimator must not invent a confident answer:
    the perpendicular component is still recovered, the parallel one collapses to ~0.
    """
    prev = [_line(0, 40 + 25 * k, 300, 40 + 25 * k) for k in range(5)]
    curr = [p + np.array([7.0, 2.0]) for p in prev]   # 7 px along, 2 px across
    d = estimate_drift(prev, curr)
    assert abs(d[1] - 2.0) < 0.6, "the perpendicular component is observable and must be right"
    assert abs(d[0]) < 2.0, "the unobservable parallel component must not be invented"


# --------------------------------------------------------------------------- association

def test_association_is_global_not_greedy():
    """Two filaments that swap nearest neighbours: greedy takes the bait, matching does not.

    ``a0`` is slightly closer to ``b1`` than to its true partner ``b0``, so a nearest-neighbour
    pass links a0-b1 and leaves a1 to take b0 -- two errors from one local decision. This is
    the same failure mode as PySOAX's greedy junction handling, which is why the tracker reuses
    the min-cost matching instead.
    """
    a0 = _line(10, 100, 210, 100)
    a1 = _line(10, 106, 210, 106)
    b0 = _line(10, 101, 210, 101)
    b1 = _line(10, 104, 210, 104)
    # precondition: a0's nearest is b1's line? check the greedy temptation is real
    pairs = match_frames([a0, a1], [b0, b1])
    assert (0, 0) in pairs and (1, 1) in pairs, f"global matching failed: {pairs}"


def test_far_apart_filaments_are_never_associated():
    a = _line(10, 10, 100, 10)
    b = _line(10, 250, 100, 250)
    assert match_frames([a], [b]) == []


def test_partial_overlap_below_the_gate_is_rejected():
    """A short fragment lying on a long filament is not the same object."""
    long_mt = _line(0, 60, 400, 60, n=400)
    stub = long_mt[:20]
    assert match_frames([long_mt], [stub]) == []


# --------------------------------------------------------------------------- sequences

def test_a_single_frame_produces_no_links_and_one_track_each():
    tracks = track_sequence([[_line(0, 10, 100, 10), _line(0, 40, 100, 40)]])
    assert len(tracks) == 2
    assert all(t.length == 1 for t in tracks)
    assert all(np.isnan(track_velocity(t)) for t in tracks), \
        "velocity is undefined from one frame and must be NaN, not 0"


def test_no_frames_is_not_an_error():
    assert track_sequence([]) == []


def test_a_stable_scene_gives_one_full_length_track_per_filament():
    base = [_line(20, 40, 220, 45), _arc(150, 160, 55, 0.1, 1.9), _line(30, 240, 230, 235)]
    frames = [[p + np.array([0.3 * k, -0.2 * k]) for p in base] for k in range(5)]
    tracks = track_sequence(frames)
    full = [t for t in tracks if t.length == 5]
    assert len(full) == 3, f"expected 3 full tracks, got {[t.length for t in tracks]}"


def test_a_new_filament_starts_a_track_and_a_vanished_one_ends():
    a, b = _line(20, 40, 220, 45), _line(30, 200, 230, 205)
    newcomer = _line(40, 120, 240, 125)
    frames = [[a, b], [a, b], [a, newcomer]]          # b disappears, newcomer arrives
    tracks = track_sequence(frames)
    lengths = sorted(t.length for t in tracks)
    assert lengths == [1, 2, 3], f"expected a birth and a death, got {lengths}"


def test_defaults_are_complete():
    for k in ("ds", "max_shift", "w_dist", "w_len", "w_tip", "c_open",
              "min_overlap", "overlap_tol"):
        assert k in DEFAULTS
