"""Instancer A: curvature-bounded junction matching on a skeleton arc graph.

Pipeline: binary mask -> skeleton -> junction-contracted arc graph -> window-fitted tangents
at every arm -> per-junction minimum-cost perfect matching under a hard curvature bound ->
chain the matched arcs -> link free ends across foreground holes -> smooth until the bound
holds -> polylines + masks.

What this fixes relative to PySOAX, cause by cause:

* junction pixels are never raced for -- arcs are whole degree-2 chains, and the crossing
  neighbourhood is contracted away rather than traced through (skeleton_graph);
* tangents come from a PCA over ~12 px, not one 45-degree-quantised pixel step (geometry);
* the junction is solved as a unit by a global matching, not by a sequence of greedy,
  order-dependent, unrevisitable steps (matching);
* the output cannot kink, by construction.

Three additions measured into it afterwards (protocol 17k):

* the pairing cost is **displacement-aware** -- it charges the turn through the gap direction,
  which is what stops two parallel microtubules from being joined across the bundle gap;
* **gap linking** across foreground dropouts, gated by curvature AND by weak image evidence
  along the bridge. This matters because the measured bottleneck is foreground fragmentation:
  v4b's mask breaks a real microtubule 1.56 times on average.
* an optional **orientation-channel agreement** term (``w_ori``), importing the amodal evidence
  instancer B exploits. It vanishes when no channels are supplied, so A still runs on a
  channel-less foreground -- which makes that configuration its own ablation.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
from scipy.ndimage import uniform_filter1d

from instance.geometry import max_abs_curvature, resample, total_length, window_tangent
from instance.matching import ArmEnd, bridge_evidence, match_junction
from instance.oracle import oracle_instance_masks
from instance.skeleton_graph import build_arc_graph

#: Baseline (px) at which curvature is measured. Must match the baseline used to derive
#: kappa_max from the ground truth (mt_bench.gt_stats), otherwise the bound means something
#: different from what was measured: on the MT-34 GT the 99.5th percentile is 0.040 rad/px
#: at an 8 px baseline but 0.074 at 2 px.
KAPPA_DS = 8.0

DEFAULTS = {
    "merge_radius": 3.0,
    "bridge_max_len": 18.0,
    "min_arc_len": 3,
    "window": 12.0,
    "w_theta": 1.0,
    "w_kappa": 10.0,
    "w_gap": 0.02,
    "w_ori": 0.0,
    "c_open": 1.2,
    # Below this tip-to-tip distance the gap DIRECTION is quantisation noise (~30 degrees of
    # error at 2 px), so the cost falls back to the direct turn. Kept independent of
    # merge_radius: tying it to the junction size would switch the displacement term off
    # exactly where bundles need it.
    "gap_floor": 4.0,
    "link_max_gap": 0.0,        # 0 disables gap linking
    "c_open_link": 1.2,
    "bridge_thr": 0.0,          # required mean probability along a bridge (0 = no check)
    "min_length": 15.0,
    "ds": 2.0,
    "half_width": 1.0,
    "smooth_size": 5,
}

_PARAM_FILE = os.path.join(os.path.dirname(__file__), "params_a.json")


def default_params() -> dict:
    """Defaults, overridden by ``params_a.json`` if a tuning run has written one.

    ``kappa_max`` is deliberately dropped if present: it is an explicit argument to
    :func:`instance_a`, so honouring it here would create two disagreeing sources of truth --
    a tuned file value that silently does nothing while looking authoritative.
    """
    p = dict(DEFAULTS)
    if os.path.exists(_PARAM_FILE):
        with open(_PARAM_FILE) as fh:
            loaded = json.load(fh)
        loaded.pop("kappa_max", None)
        p.update(loaded)
    return p


def enforce_curvature(poly: np.ndarray, kappa_max: float, ds: float = 2.0,
                      smooth_size: int = 5, max_iter: int = 40) -> np.ndarray:
    """Smooth a polyline until ``max |dtheta/ds| <= kappa_max`` at the KAPPA_DS baseline.

    Endpoints are pinned so the microtubule does not shrink. Repeated box smoothing strictly
    reduces curvature, so this always terminates; because kappa_max sits above the largest
    curvature seen in 957 human-annotated microtubules, genuine hairpins never trigger it and
    only tracer kinks get smoothed.
    """
    p = resample(np.asarray(poly, dtype=float), ds=ds)
    for _ in range(max_iter):
        if len(p) < 5:
            return p
        probe = resample(p, ds=KAPPA_DS)
        if len(probe) < 3 or max_abs_curvature(probe, ds=KAPPA_DS) <= kappa_max:
            return p
        sm = uniform_filter1d(p, size=smooth_size, axis=0, mode="nearest")
        sm[0], sm[-1] = p[0], p[-1]
        p = resample(sm, ds=ds)
    return p


def _ori_profile(channels: np.ndarray | None, poly: np.ndarray, which: str,
                 window: float) -> np.ndarray | None:
    """Mean orientation-channel response over the arm's tangent window."""
    if channels is None or len(poly) < 2:
        return None
    seq = poly[::-1] if which == "end" else poly
    n = max(int(window), 2)
    sel = seq[:n]
    h, w = channels.shape[1:]
    cc = np.clip(np.rint(sel[:, 0]).astype(int), 0, w - 1)
    rr = np.clip(np.rint(sel[:, 1]).astype(int), 0, h - 1)
    return np.asarray(channels[:, rr, cc], dtype=float).mean(axis=1)


def _arm_at(poly: np.ndarray, arc_idx: int, which: str, window: float,
            channels: np.ndarray | None) -> ArmEnd:
    theta, kappa = window_tangent(poly, which, window=window)
    pos = poly[0] if which == "start" else poly[-1]
    return ArmEnd(arc_idx=arc_idx, which=which, theta=theta, kappa=kappa,
                  pos=np.asarray(pos, dtype=float),
                  ori=_ori_profile(channels, poly, which, window),
                  arc_len=total_length(poly))


def _build_arms(arcs, arc_ends, window: float, channels: np.ndarray | None):
    """Group arc ends into per-junction arm lists."""
    arms_by_j = defaultdict(list)
    for ai, (j0, j1) in enumerate(arc_ends):
        if len(arcs[ai]) < 2:
            continue
        for which, jid in (("start", j0), ("end", j1)):
            if jid is not None:
                arms_by_j[jid].append(_arm_at(arcs[ai], ai, which, window, channels))
    return arms_by_j


def _walk(segments, partner, bridge_pts=None) -> list[np.ndarray]:
    """Follow matched ports into chains.

    ``partner`` maps ``(segment, "start"|"end")`` to ``(other_port, bridge_key)``. When
    ``bridge_pts`` is given, ``bridge_key`` indexes a point inserted at the join -- the
    junction centroid, which both chains through a crossing legitimately share.
    """
    chains, visited = [], set()

    def walk(entry_port):
        pieces, cur = [], entry_port
        while True:
            si, which = cur
            if si in visited:
                break
            visited.add(si)
            pieces.append(segments[si] if which == "start" else segments[si][::-1])
            exit_port = (si, "end" if which == "start" else "start")
            nxt = partner.get(exit_port)
            if nxt is None:
                break
            next_port, key = nxt
            if bridge_pts is not None and key is not None:
                pieces.append(np.asarray(bridge_pts[key], float)[None, :])
            cur = next_port
        return np.concatenate(pieces) if pieces else None

    for si in range(len(segments)):
        for which in ("start", "end"):
            if (si, which) not in partner and si not in visited:
                c = walk((si, which))
                if c is not None:
                    chains.append(c)
    for si in range(len(segments)):                     # closed cycles (rings)
        if si not in visited:
            c = walk((si, "start"))
            if c is not None:
                chains.append(c)
    return chains


def _link_free_ends(chains, kappa_max: float, p: dict,
                    channels: np.ndarray | None, prob: np.ndarray | None, linker=None):
    """Join chains whose free ends continue each other across a foreground hole."""
    if p["link_max_gap"] <= 0 or len(chains) < 2:
        return chains

    arms = [_arm_at(ch, ci, which, p["window"], channels)
            for ci, ch in enumerate(chains) for which in ("start", "end")]

    allow = None
    if prob is not None and p["bridge_thr"] > 0:
        def allow(a, b):
            return bridge_evidence(prob, a, b) >= p["bridge_thr"]

    pairs = match_junction(arms, kappa_max=kappa_max, w_theta=p["w_theta"],
                           w_kappa=p["w_kappa"], w_gap=p["w_gap"],
                           c_open=p["c_open_link"], gap_len=None,
                           gap_floor=p["gap_floor"], w_ori=p["w_ori"],
                           max_gap=p["link_max_gap"], allow=allow,
                           # Gap linking keeps the geometric cost by default: its c_open_link /
                           # link_max_gap / bridge_thr were tuned together, and swapping the
                           # scorer under them changes two things at once.
                           scorer=(_make_scorer(linker, prob, p, is_gap_link=True)
                                   if p.get("linker_gap", False) else None),
                           p_open=p.get("p_open_link", p.get("p_open", 0.5)))
    partner = {}
    for i, j in pairs:
        pa = (arms[i].arc_idx, arms[i].which)
        pb = (arms[j].arc_idx, arms[j].which)
        if pa[0] == pb[0]:
            continue                                    # a chain closing on itself
        partner[pa] = (pb, None)
        partner[pb] = (pa, None)
    return _walk(chains, partner)


def _make_scorer(linker, prob, p, is_gap_link: bool):
    """Wrap a :class:`instance.linker.Linker` into ``match_junction``'s scorer signature."""
    if linker is None:
        return None
    from instance.linker import edge_features

    def scorer(arms, i, j, n_arms):
        f = edge_features(arms[i], arms[j], prob=prob, gap_floor=p["gap_floor"],
                          n_arms=n_arms, is_gap_link=is_gap_link)
        return float(linker.score(f[None])[0])
    return scorer


def _arc_overlap_frac(arc: np.ndarray, overlap: np.ndarray, thr: float) -> float:
    """Fraction of an arc's pixels the multiplicity map marks as carrying two filaments."""
    if len(arc) == 0:
        return 0.0
    h, w = overlap.shape
    r = np.clip(np.rint(arc[:, 1]).astype(int), 0, h - 1)
    c = np.clip(np.rint(arc[:, 0]).astype(int), 0, w - 1)
    return float(np.mean(overlap[r, c] >= thr))


def _pass_through_shared(arcs, arc_ends, junctions, arms_by_j, overlap, kappa_max, p):
    """Let two filaments traverse an arc the multiplicity map says is shared.

    `_absorb_crossing_bridges` documents its own limit: at shallow angles two filaments stay
    skeletally fused over ``~4 * half_width / sin(alpha)`` px, and beyond ``bridge_max_len``
    such an arc is kept because it could equally be a real segment. Read 35 measured the cost of
    that irreducible choice -- the crossing is then two 3-arm junctions joined by a shared arc,
    each of which can match only ONE pair, so one filament is always left open, and identity
    falls from 0.944 (a well-formed 4-arm X) to 0.333.

    The multiplicity map is the evidence that decides the ambiguity the docstring names. Where
    an arc with a junction at BOTH ends is marked shared, the outer arms of its two junctions
    are matched ACROSS it -- one `match_junction` over the pooled arms with the arc's own length
    as the span, so the curvature bound is enforced over the true distance -- and each resulting
    pair gets its own traversal of the arc.

    Returns ``(extra_arcs, partner, consumed)``: copies to append, the port wiring they imply,
    and the ports the ordinary per-junction matching must no longer touch.
    """
    thr, frac = p.get("ov_thr", 0.5), p.get("ov_frac", 0.5)
    min_len = p.get("ov_min_len", 0.0)
    extra_arcs, partner, consumed = [], {}, set()
    for k, (j0, j1) in enumerate(arc_ends):
        if j0 is None or j1 is None or j0 == j1 or len(arcs[k]) < 2:
            continue
        L = total_length(arcs[k])
        if L < min_len or _arc_overlap_frac(arcs[k], overlap, thr) < frac:
            continue
        tagged = [(sd, a) for sd, jid in ((0, j0), (1, j1))
                  for a in arms_by_j.get(jid, []) if a.arc_idx != k]
        if len(tagged) < 2:
            continue
        outer = [a for _, a in tagged]
        side = [sd for sd, _ in tagged]
        pairs = [(i, j) for i, j in
                 match_junction(outer, kappa_max=kappa_max, w_theta=p["w_theta"],
                                w_kappa=p["w_kappa"], w_gap=p["w_gap"], c_open=p["c_open"],
                                gap_len=float(L), gap_floor=p["gap_floor"], w_ori=p["w_ori"])
                 if side[i] != side[j]]
        if not pairs:
            continue
        # the original arc carries the first pair; each further pair gets its own copy
        for n, (i, j) in enumerate(pairs):
            a, b = outer[i], outer[j]
            if side[i] == 1:
                a, b = b, a                       # `a` is always the j0 side
            if n == 0:
                ai = k
            else:
                ai = len(arcs) + len(extra_arcs)
                extra_arcs.append(arcs[k].copy())
            pa, pb = (a.arc_idx, a.which), (b.arc_idx, b.which)
            partner[pa] = ((ai, "start"), j0)
            partner[(ai, "start")] = (pa, j0)
            partner[(ai, "end")] = (pb, j1)
            partner[pb] = ((ai, "end"), j1)
            consumed |= {pa, pb, (ai, "start"), (ai, "end")}
    return extra_arcs, partner, consumed


def instance_a(mask: np.ndarray, kappa_max: float, params: dict | None = None,
               channels: np.ndarray | None = None,
               prob: np.ndarray | None = None,
               linker=None,
               overlap: np.ndarray | None = None) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Instance-segment a binary foreground mask. Returns ``(polylines, masks)``.

    ``channels`` (K, H, W) enables the orientation-agreement term; ``prob`` (H, W) enables the
    bridge-evidence check when linking across gaps. Both are optional and A degrades to its
    geometry-only behaviour without them.

    ``overlap`` (H, W) is a predicted multiplicity map; where it marks an arc with a junction
    at both ends as carrying two filaments, both filaments are allowed to traverse it (see
    :func:`_pass_through_shared`). Passing None leaves every port wiring exactly as before.

    ``linker`` (a :class:`instance.linker.Linker`) replaces the hand-crafted pairing cost with
    a learned one, leaving the hard curvature bound and the global matching untouched. Passing
    None keeps the geometric behaviour exactly, so the learned cost is its own ablation.
    """
    p = {**default_params(), **(params or {})}
    mask = np.asarray(mask, dtype=bool)

    g = build_arc_graph(mask, merge_radius=p["merge_radius"],
                        min_arc_len=int(p["min_arc_len"]),
                        bridge_max_len=p["bridge_max_len"])
    if not g.arcs:
        return [], []

    arcs = [resample(a, ds=p["ds"]) for a in g.arcs]
    arms_by_j = _build_arms(arcs, g.arc_ends, p["window"], channels)

    junction_scorer = _make_scorer(linker, prob, p, is_gap_link=False)
    partner: dict = {}
    consumed: set = set()
    arc_ends = list(g.arc_ends)
    if overlap is not None:
        extra, partner, consumed = _pass_through_shared(
            arcs, arc_ends, g.junctions, arms_by_j, overlap, kappa_max, p)
        arcs = arcs + extra
    for jid, arms in arms_by_j.items():
        free = [a for a in arms if (a.arc_idx, a.which) not in consumed]
        if len(free) < 2:
            continue
        for i, j in match_junction(free, kappa_max=kappa_max, w_theta=p["w_theta"],
                                   w_kappa=p["w_kappa"], w_gap=p["w_gap"],
                                   c_open=p["c_open"], gap_len=None,
                                   gap_floor=p["gap_floor"], w_ori=p["w_ori"],
                                   scorer=junction_scorer,
                                   p_open=p.get("p_open", 0.5)):
            pa = (free[i].arc_idx, free[i].which)
            pb = (free[j].arc_idx, free[j].which)
            partner[pa] = (pb, jid)
            partner[pb] = (pa, jid)

    chains = _walk(arcs, partner, bridge_pts=g.junctions)
    chains = [c for c in chains if total_length(c) >= p["min_length"] * 0.5]
    chains = _link_free_ends(chains, kappa_max, p, channels, prob, linker)

    polylines = []
    for chain in chains:
        if total_length(chain) < p["min_length"]:
            continue
        poly = enforce_curvature(chain, kappa_max, ds=p["ds"],
                                 smooth_size=int(p["smooth_size"]))
        if total_length(poly) >= p["min_length"]:
            polylines.append(poly)

    masks = oracle_instance_masks(polylines, mask.shape,
                                  half_width=p["half_width"], up=1.0)
    return polylines, masks
