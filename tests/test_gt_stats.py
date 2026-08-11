import numpy as np

from mt_bench.gt_stats import (count_crossings, count_parallel_pairs,
                               curvature_quantile)


def test_curvature_quantile_of_straight_lines_is_near_zero():
    pls = [np.array([[0.0, float(i)], [100.0, float(i)]]) for i in range(10)]
    assert curvature_quantile(pls, ds=8.0, q=99.5) < 1e-3


def test_curvature_quantile_recovers_a_known_circle():
    R, t = 25.0, np.linspace(0.0, np.pi, 500)
    circ = np.stack([R * np.cos(t) + 60, R * np.sin(t) + 60], axis=1)
    k = curvature_quantile([circ], ds=4.0, q=50.0)
    assert abs(k - 1.0 / R) < 0.25 / R


def test_count_crossings_finds_one_perpendicular_x():
    a = np.array([[0.0, 50.0], [100.0, 50.0]])   # horizontal
    b = np.array([[50.0, 0.0], [50.0, 100.0]])   # vertical
    cr = count_crossings([a, b], tol=2.0)
    assert len(cr) == 1
    assert abs(cr[0]["angle_deg"] - 90.0) < 5.0


def test_count_crossings_measures_a_shallow_angle():
    a = np.array([[0.0, 50.0], [100.0, 50.0]])
    ang = np.deg2rad(20.0)
    b = np.array([[50.0 - 50 * np.cos(ang), 50.0 - 50 * np.sin(ang)],
                  [50.0 + 50 * np.cos(ang), 50.0 + 50 * np.sin(ang)]])
    cr = count_crossings([a, b], tol=2.0)
    assert len(cr) == 1 and abs(cr[0]["angle_deg"] - 20.0) < 5.0


def test_count_crossings_ignores_disjoint_lines():
    a = np.array([[0.0, 10.0], [100.0, 10.0]])
    b = np.array([[0.0, 80.0], [100.0, 80.0]])
    assert count_crossings([a, b], tol=2.0) == []


def test_count_crossings_ignores_a_shared_endpoint_touch():
    a = np.array([[0.0, 50.0], [50.0, 50.0]])
    b = np.array([[50.0, 50.0], [50.0, 100.0]])
    assert count_crossings([a, b], tol=2.0) == []


def test_count_parallel_pairs_detects_close_bundle():
    a = np.array([[0.0, 50.0], [100.0, 50.0]])
    b = np.array([[0.0, 54.0], [100.0, 54.0]])   # 4 px apart, inside [2, 6]
    far = np.array([[0.0, 200.0], [100.0, 200.0]])
    assert count_parallel_pairs([a, b, far]) == 1


def test_count_parallel_pairs_ignores_a_too_wide_gap():
    a = np.array([[0.0, 50.0], [100.0, 50.0]])
    b = np.array([[0.0, 70.0], [100.0, 70.0]])   # 20 px apart
    assert count_parallel_pairs([a, b]) == 0
