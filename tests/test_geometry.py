import numpy as np

from instance.geometry import (polyline_curvature, resample, turn_penalty,
                               window_tangent)


def test_resample_gives_constant_spacing():
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    out = resample(pts, ds=1.0)
    d = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert np.allclose(d, 1.0, atol=1e-6)
    assert np.allclose(out[0], [0.0, 0.0]) and np.allclose(out[-1], [10.0, 10.0], atol=1.0)


def test_straight_line_has_zero_curvature():
    pts = resample(np.array([[0.0, 0.0], [50.0, 0.0]]), ds=1.0)
    assert np.max(np.abs(polyline_curvature(pts, ds=1.0))) < 1e-6


def test_right_angle_corner_has_large_curvature():
    pts = resample(np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 20.0]]), ds=1.0)
    assert np.max(np.abs(polyline_curvature(pts, ds=1.0))) > 1.0


def test_circle_curvature_matches_inverse_radius():
    R, t = 40.0, np.linspace(0, np.pi / 2, 400)
    pts = np.stack([R * np.cos(t), R * np.sin(t)], axis=1)
    k = polyline_curvature(resample(pts, ds=1.0), ds=1.0)
    assert abs(np.median(np.abs(k)) - 1.0 / R) < 0.15 / R


def test_signed_curvature_flips_when_the_path_is_reversed():
    R, t = 30.0, np.linspace(0, np.pi / 2, 300)
    pts = resample(np.stack([R * np.cos(t), R * np.sin(t)], axis=1), ds=1.0)
    fwd = np.median(polyline_curvature(pts, ds=1.0))
    rev = np.median(polyline_curvature(pts[::-1], ds=1.0))
    assert abs(fwd + rev) < 1e-3 and abs(fwd) > 1e-3


def test_window_tangent_heads_from_the_terminal_vertex_into_the_body():
    # Horizontal line x=0..60. At the "end" vertex (x=60) the body lies toward -x;
    # at the "start" vertex (x=0) it lies toward +x. Two arms of a straight
    # through-path therefore have tangents pi apart -- the matching convention.
    pts = resample(np.array([[0.0, 5.0], [60.0, 5.0]]), ds=1.0)
    th_end, _ = window_tangent(pts, "end", window=12.0)
    th_start, _ = window_tangent(pts, "start", window=12.0)
    assert abs(np.cos(th_end) + 1.0) < 1e-3
    assert abs(np.cos(th_start) - 1.0) < 1e-3
    assert abs(abs(float(np.arctan2(np.sin(th_end - th_start), np.cos(th_end - th_start))))
               - np.pi) < 1e-6


def test_window_tangent_beats_one_pixel_estimate_on_shallow_angle():
    # 20-degree line: a single 8-connected step can only report 0 or 45 degrees,
    # so it cannot tell a 20-degree crossing from a 0- or 45-degree one.
    ang = np.deg2rad(20.0)
    pts = resample(np.array([[0.0, 0.0], [60 * np.cos(ang), 60 * np.sin(ang)]]), ds=1.0)
    pts = np.round(pts)  # pixel quantization, as a real skeleton would be
    th, _ = window_tangent(pts, "start", window=12.0)
    assert abs(np.rad2deg(th) - 20.0) < 4.0


def test_window_tangent_curvature_is_opposite_at_the_two_ends_of_an_arc():
    # Both ends are measured OUTWARD, so the two rays traverse the same arc in
    # opposite directions -> their signed curvatures are negatives of each other.
    R, t = 50.0, np.linspace(0, 1.0, 400)
    pts = resample(np.stack([R * np.cos(t), R * np.sin(t)], axis=1), ds=1.0)
    _, k_start = window_tangent(pts, "start", window=15.0)
    _, k_end = window_tangent(pts, "end", window=15.0)
    assert abs(k_start + k_end) < 0.2 * max(abs(k_start), 1e-9)


def test_turn_penalty_is_zero_for_collinear_through_path():
    assert turn_penalty(0.0, 0.0) < 1e-9
    assert abs(turn_penalty(0.0, np.pi) - np.pi) < 1e-9
    assert abs(turn_penalty(np.deg2rad(350.0), np.deg2rad(10.0)) - np.deg2rad(20.0)) < 1e-9
