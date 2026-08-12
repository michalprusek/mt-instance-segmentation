"""Cross-frame association: which microtubule in frame t is which in frame t+1.

Structurally this is the junction matcher from :mod:`instance.matching` pointed at a different
problem. There, arms meeting at a crossing are paired by a global min-cost matching under a
curvature bound, with a priced option to leave an arm open. Here, filaments in consecutive
frames are paired by a global min-cost matching under a displacement bound, with a priced
option to leave a filament unmatched -- which is what track birth and death are.

Reusing the shape is not tidiness. A greedy nearest-neighbour association makes exactly the
mistake PySOAX makes at junctions: it commits to the locally cheapest link and cannot revisit
it, so in a dense bundle the first filament claims the wrong partner and every later one
inherits the error.

Geometry only, no learned weights
---------------------------------
At the single-digit-pixel displacements these acquisitions have, consecutive centerlines
overlap heavily and geometry is highly informative. A learned association would be a component
whose proxy has not been validated against the thing it must improve -- the failure that cost
this project a training run and produced no measurable gain (protocol 17p). It is deferred
until geometry is *measured* to fail, not assumed to.

Stage drift
-----------
A microscope drifts, and every filament in the field moves with it. A tracker that folds that
into per-filament velocity reports drift as motility, which is the one error a motility assay
cannot tolerate. :func:`estimate_drift` therefore recovers the common-mode shift first, and
velocities are reported relative to it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from instance.geometry import arclength, resample

#: Association cost weights and gates. Fitted on SYNTHETIC sequences, never on real
#: annotations -- the choice that beat real-annotation fitting by +0.041 [+0.018, +0.065] on
#: the instancer (protocol 17q) and the one that keeps the pipeline annotation-free.
DEFAULTS = {
    "ds": 2.0,              # resampling step for centerline comparison, px
    "max_shift": 25.0,      # hard gate: no association beyond this displacement, px
    "w_dist": 1.0,          # mean curve-to-curve distance
    "w_len": 0.05,          # length change (a filament does not double in one frame)
    "w_tip": 0.10,          # endpoint continuity
    "c_open": 6.0,          # price of leaving a filament unmatched = birth / death
    "min_overlap": 0.35,    # fraction of the shorter curve that must have a partner nearby
    "overlap_tol": 4.0,     # what "nearby" means when measuring that fraction, px
}


@dataclass
class Track:
    """One microtubule followed through a sequence."""
    track_id: int
    frames: List[int] = field(default_factory=list)
    polylines: List[np.ndarray] = field(default_factory=list)
    #: Signed shift along the filament's own contour, per frame step, drift removed. This is
    #: the gliding velocity in px/frame; it is None for the first frame of a track.
    contour_shift: List[Optional[float]] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.frames)

    def tips(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        return [(p[0], p[-1]) for p in self.polylines]


def _prep(polyline: np.ndarray, ds: float) -> np.ndarray:
    p = np.asarray(polyline, dtype=float)
    return resample(p, ds=ds) if len(p) >= 2 else p


def curve_distance(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Symmetric mean nearest-point distance, and the overlapping fraction.

    Returns ``(mean_distance, overlap_fraction)``. Distance alone is not enough: a short
    fragment sitting on top of a long filament has a tiny mean distance in one direction, so
    the fraction of the *shorter* curve that has a partner nearby is what says whether they
    describe the same object.
    """
    if len(a) < 2 or len(b) < 2:
        return float("inf"), 0.0
    ta, tb = cKDTree(a), cKDTree(b)
    da, _ = tb.query(a, k=1)
    db, _ = ta.query(b, k=1)
    return float(0.5 * (da.mean() + db.mean())), float(min(da.mean(), db.mean()))


def _overlap_fraction(a: np.ndarray, b: np.ndarray, tol: float) -> float:
    ta, tb = cKDTree(a), cKDTree(b)
    fa = float((tb.query(a, k=1)[0] <= tol).mean())
    fb = float((ta.query(b, k=1)[0] <= tol).mean())
    return max(fa, fb)


def contour_shift(a: np.ndarray, b: np.ndarray, edge_frac: float = 0.15) -> float:
    """Signed shift of ``b`` along ``a``'s own contour, in px. Positive = towards a's head.

    A gliding filament slides along itself, so its perpendicular displacement is ~zero and a
    distance-based tracker sees no motion at all. What moves is the *material*, and the way to
    see it is that every point of b sits at a constant arclength offset from its counterpart
    on a.

    The subtlety is what happens at the ends. Once b has advanced, its head lies **beyond** a's
    head, so the nearest point on a is a's last vertex and the projection **saturates** — it
    reports zero shift no matter how far the filament went. Including those points halves the
    estimate. So the offset is taken over the interior matches only: points of b whose nearest
    neighbour on a is not within ``edge_frac`` of either end.
    """
    if len(a) < 2 or len(b) < 2:
        return 0.0
    sa = arclength(a)
    sb = arclength(b)
    if sa[-1] <= 0 or sb[-1] <= 0:
        return 0.0
    _, idx = cKDTree(a).query(b, k=1)
    offs = sa[idx] - sb
    # Drop the saturated ends: those whose match landed on the outer edge_frac of a.
    lo, hi = edge_frac * sa[-1], (1.0 - edge_frac) * sa[-1]
    interior = (sa[idx] > lo) & (sa[idx] < hi)
    if interior.sum() < 3:
        interior = np.ones(len(offs), dtype=bool)     # too short to trim; use everything
    return float(np.median(offs[interior]))


def estimate_drift(prev: Sequence[np.ndarray], curr: Sequence[np.ndarray],
                   max_shift: float = 25.0, ds: float = 2.0) -> np.ndarray:
    """Common-mode translation between two frames, as (dx, dy).

    **Not** the median centroid shift. A gliding filament's centroid travels along its own
    contour at the full gliding speed, and in a gliding field *every* filament does, so the
    median centroid shift measures motility and calls it drift. Measured on synthetic
    sequences with drift switched off, that estimator returned 2.9 px of drift that did not
    exist.

    What separates the two is that gliding is motion **along** the filament while drift is
    motion of the whole field. The component of a filament's displacement perpendicular to its
    own tangent therefore contains no gliding at all — this is the aperture problem, and it is
    solved here the way optical flow solves it: collect the perpendicular ("normal flow")
    constraints from filaments at *different orientations* and least-squares the single
    translation that explains them. Two distinct orientations suffice; a field of parallel
    filaments is genuinely ambiguous and the estimate degrades gracefully towards zero.
    """
    if not prev or not curr:
        return np.zeros(2)
    P = [_prep(p, ds) for p in prev]
    C = [_prep(c, ds) for c in curr]
    cp = np.array([p.mean(axis=0) for p in P])
    cc = np.array([c.mean(axis=0) for c in C])
    d, j = cKDTree(cc).query(cp, k=1)

    rows, vals = [], []
    for i, ok in enumerate(d <= max_shift):
        if not ok or len(P[i]) < 3:
            continue
        a, b = P[i], C[j[i]]
        # Per-point displacement to the nearest partner, and the local unit normal of a.
        _, idx = cKDTree(b).query(a, k=1)
        disp = b[idx] - a
        tang = np.gradient(a, axis=0)
        nrm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
        rows.append(nrm)
        vals.append(np.einsum("ij,ij->i", nrm, disp))
    if not rows:
        return np.zeros(2)
    A = np.concatenate(rows, axis=0)
    y = np.concatenate(vals, axis=0)
    # Rank-deficient when every filament shares one orientation: lstsq returns the minimum-norm
    # solution, which is the honest answer (no evidence for motion along that one direction).
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return np.asarray(sol, dtype=float)


def pair_cost(a: np.ndarray, b: np.ndarray, params: dict) -> Tuple[float, bool]:
    """Cost of calling ``a`` (frame t) and ``b`` (frame t+1) the same microtubule.

    Returns ``(cost, allowed)``. ``allowed`` is False when a hard gate rejects the pair, so a
    forbidden association can never be bought by making everything else expensive.
    """
    dist, _ = curve_distance(a, b)
    if not np.isfinite(dist) or dist > params["max_shift"]:
        return float("inf"), False
    frac = _overlap_fraction(a, b, params["overlap_tol"])
    if frac < params["min_overlap"]:
        return float("inf"), False

    la, lb = arclength(a)[-1], arclength(b)[-1]
    d_len = abs(la - lb) / max(la, lb, 1e-6)
    d_tip = 0.5 * (float(np.linalg.norm(a[0] - b[0])) + float(np.linalg.norm(a[-1] - b[-1])))
    cost = (params["w_dist"] * dist
            + params["w_len"] * d_len * max(la, lb)
            + params["w_tip"] * d_tip)
    return float(cost), True


def match_frames(prev: Sequence[np.ndarray], curr: Sequence[np.ndarray],
                 params: Optional[dict] = None,
                 drift: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
    """Globally optimal association between two frames. Returns (i_prev, j_curr) pairs.

    Unmatched filaments on either side are simply absent from the result -- they are a death
    and a birth respectively. The "leave unmatched" price is implemented by padding the cost
    matrix with ``c_open`` so the assignment solver can choose it, which is the same device the
    junction matcher uses for leaving an arm open.
    """
    p = {**DEFAULTS, **(params or {})}
    if not prev or not curr:
        return []
    shift = np.zeros(2) if drift is None else np.asarray(drift, dtype=float)
    P = [_prep(x, p["ds"]) for x in prev]
    C = [_prep(x, p["ds"]) - shift for x in curr]

    n, m = len(P), len(C)
    big = 1e6
    cost = np.full((n + m, m + n), big, dtype=float)
    allowed = np.zeros((n, m), dtype=bool)
    for i in range(n):
        for j in range(m):
            c, ok = pair_cost(P[i], C[j], p)
            if ok:
                cost[i, j] = c
                allowed[i, j] = True
    # Padding: row i may go unmatched at price c_open, and so may column j.
    for i in range(n):
        cost[i, m + i] = p["c_open"]
    for j in range(m):
        cost[n + j, j] = p["c_open"]
    cost[n:, m:] = 0.0

    rows, cols = linear_sum_assignment(cost)
    return sorted((i, j) for i, j in zip(rows, cols)
                  if i < n and j < m and allowed[i, j])


def track_sequence(frames: Sequence[Sequence[np.ndarray]],
                   params: Optional[dict] = None) -> List[Track]:
    """Link instances across a whole sequence into tracks.

    ``frames[k]`` is the list of polylines detected in frame k. A single frame in, no tracks
    out -- that is the honest answer for a still image, not an error.
    """
    p = {**DEFAULTS, **(params or {})}
    tracks: List[Track] = []
    if not frames:
        return tracks

    active: Dict[int, int] = {}                     # index in frame k -> track_id
    for j, poly in enumerate(frames[0]):
        tracks.append(Track(track_id=len(tracks), frames=[0], polylines=[np.asarray(poly)],
                            contour_shift=[None]))
        active[j] = tracks[-1].track_id

    for k in range(1, len(frames)):
        prev, curr = frames[k - 1], frames[k]
        drift = estimate_drift(prev, curr, p["max_shift"], p["ds"])
        pairs = match_frames(prev, curr, p, drift=drift)
        nxt: Dict[int, int] = {}
        for i, j in pairs:
            tid = active.get(i)
            if tid is None:                          # frame k-1 detection had no track: start one
                tracks.append(Track(track_id=len(tracks), frames=[k - 1],
                                    polylines=[np.asarray(prev[i])], contour_shift=[None]))
                tid = tracks[-1].track_id
            a = _prep(prev[i], p["ds"])
            b = _prep(curr[j], p["ds"]) - drift      # velocity is measured drift-free
            tr = tracks[tid]
            tr.frames.append(k)
            tr.polylines.append(np.asarray(curr[j]))
            tr.contour_shift.append(contour_shift(a, b))
            nxt[j] = tid
        for j, poly in enumerate(curr):              # births
            if j not in nxt:
                tracks.append(Track(track_id=len(tracks), frames=[k],
                                    polylines=[np.asarray(poly)], contour_shift=[None]))
                nxt[j] = tracks[-1].track_id
        active = nxt
    return tracks


def track_velocity(track: Track) -> float:
    """Median contour shift in px/frame over a track, drift already removed. NaN if unmeasured."""
    vals = [v for v in track.contour_shift if v is not None and np.isfinite(v)]
    return float(np.median(vals)) if vals else float("nan")
