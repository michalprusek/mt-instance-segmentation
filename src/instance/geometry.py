"""Polyline geometry: arc-length resampling, signed curvature, window-fitted tangents.

All polylines are ``(N, 2)`` arrays of ``(x=col, y=row)``. Curvature is the signed turn rate
``kappa = dtheta/ds`` in **rad/px**, positive for a left turn. This is the quantity the
instancer bounds: microtubules have millimetre-scale persistence length and a documented
breaking curvature of ~0.43 um^-1, so a corner in a traced centerline is always a tracer
artifact, never biology.
"""
from __future__ import annotations

import numpy as np


def wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    """Wrap an angle (or array of angles) to ``(-pi, pi]``."""
    return np.arctan2(np.sin(a), np.cos(a))


def _drop_duplicates(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return points
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-12
    return points[keep]


def arclength(points: np.ndarray) -> np.ndarray:
    """Cumulative arc length at each vertex, starting at 0."""
    if len(points) < 2:
        return np.zeros(len(points))
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def total_length(points: np.ndarray) -> float:
    return float(arclength(np.asarray(points, dtype=float))[-1]) if len(points) > 1 else 0.0


def resample(points: np.ndarray, ds: float = 1.0) -> np.ndarray:
    """Resample a polyline to constant arc-length spacing ``ds``.

    Returns at least the two endpoints. Degenerate (zero-length) input is returned as is.
    """
    pts = _drop_duplicates(np.asarray(points, dtype=float))
    if len(pts) < 2:
        return pts
    s = arclength(pts)
    if s[-1] <= 0:
        return pts
    n = max(int(np.floor(s[-1] / ds)) + 1, 2)
    target = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(target, s, pts[:, 0]),
                     np.interp(target, s, pts[:, 1])], axis=1)


def segment_angles(points: np.ndarray) -> np.ndarray:
    """Direction of each segment, in radians."""
    d = np.diff(np.asarray(points, dtype=float), axis=0)
    return np.arctan2(d[:, 1], d[:, 0])


def polyline_curvature(points: np.ndarray, ds: float = 1.0) -> np.ndarray:
    """Signed turn rate at each interior vertex, in rad/px.

    ``ds`` is the assumed spacing; pass the same value used for :func:`resample`. For an
    unevenly sampled polyline the local spacing is used instead, which is why the actual
    inter-vertex distances are read from the data rather than assumed.
    """
    pts = _drop_duplicates(np.asarray(points, dtype=float))
    if len(pts) < 3:
        return np.zeros(0)
    th = segment_angles(pts)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    # Turn happens at the vertex between segment i and i+1; attribute it to the mean of
    # the two adjacent segment lengths so uneven sampling does not inflate curvature.
    span = 0.5 * (seg[:-1] + seg[1:])
    span = np.where(span > 1e-9, span, ds)
    return wrap_angle(np.diff(th)) / span


def max_abs_curvature(points: np.ndarray, ds: float = 1.0) -> float:
    k = polyline_curvature(points, ds=ds)
    return float(np.max(np.abs(k))) if len(k) else 0.0


def window_tangent(points: np.ndarray, end: str,
                   window: float = 12.0) -> tuple[float, float]:
    """Tangent direction and signed curvature at one end of a polyline.

    Both are measured on the ray that starts at the terminal vertex and runs INTO the body
    of the polyline -- i.e. **outward** from whatever junction that end sits in. Two arms of
    a smooth through-path therefore have tangents ~pi apart and signed curvatures that are
    negatives of each other, which is what :mod:`instance.matching` exploits.

    The direction comes from a PCA over every vertex within ``window`` px of the end, not
    from a single step. That is the fix for PySOAX's 45-degree-quantised one-pixel estimate,
    which cannot discriminate shallow crossings at all.

    Returns ``(theta, kappa)`` with ``theta`` in radians and ``kappa`` in rad/px.
    """
    pts = _drop_duplicates(np.asarray(points, dtype=float))
    if len(pts) < 2:
        return 0.0, 0.0
    if end == "end":
        ray = pts[::-1]
    elif end == "start":
        ray = pts
    else:
        raise ValueError(f"end must be 'start' or 'end', got {end!r}")

    s = arclength(ray)
    sel = ray[s <= max(window, 1e-6)]
    if len(sel) < 2:
        sel = ray[:2]

    # PCA direction, sign-fixed to point from the terminal vertex into the body.
    centred = sel - sel.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    direction = vt[0]
    inward = sel[-1] - sel[0]
    if float(np.dot(direction, inward)) < 0:
        direction = -direction
    theta = float(np.arctan2(direction[1], direction[0]))

    k = polyline_curvature(sel)
    kappa = float(np.median(k)) if len(k) else 0.0
    return theta, kappa


def turn_penalty(theta_in: float, theta_out: float) -> float:
    """Absolute turn, in radians ``[0, pi]``, from an incoming to an outgoing heading."""
    return float(abs(wrap_angle(theta_out - theta_in)))
