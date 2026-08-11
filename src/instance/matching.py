"""Curvature-bounded arm pairing at a junction.

This replaces PySOAX's greedy "follow the straightest continuation" rule, which decides one
step at a time from a single 8-connected pixel step (quantised to 45 degrees) and can never
revisit a wrong choice. Here the junction is solved as a UNIT: its arms are paired by a
minimum-cost perfect matching, so the 4 arms of an X necessarily become 2 consistent
through-paths.

Two ingredients PySOAX lacks:

* a **hard curvature bound** -- a pairing whose implied ``|dtheta|/ds`` exceeds ``kappa_max``
  is not merely expensive, it is forbidden, because microtubules cannot kink;
* an explicit **"leave this arm open"** alternative priced at ``c_open``, so a genuine
  T-junction or a microtubule end is not forced into a spurious continuation.

Angle convention (see :func:`instance.geometry.window_tangent`): ``theta`` points from the
junction INTO the arc's body. Travelling through the junction from arm ``i`` into arm ``j``
means arriving on heading ``theta_i + pi`` and leaving on ``theta_j``, so a straight
through-path has ``turn_penalty(theta_i + pi, theta_j) == 0``. Signed curvatures measured
outward likewise SUM to zero across a smooth continuation.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from instance.geometry import turn_penalty


@dataclass
class ArmEnd:
    """One arc end sitting in a junction (or a free chain end, for gap linking)."""
    arc_idx: int
    which: str                       # "start" or "end"
    theta: float                     # outgoing heading, radians (junction -> arc body)
    kappa: float                     # signed curvature at that end, rad/px, outward
    pos: np.ndarray                  # (x, y) of the terminal vertex
    ori: np.ndarray | None = None    # orientation-channel profile over the tangent window


def _ori_mismatch(a: ArmEnd, b: ArmEnd) -> float:
    """1 - cosine similarity between two arms' orientation-channel profiles, or 0.

    Two arms of ONE microtubule respond in the same amodal orientation channel through the
    junction; two arms of DIFFERENT microtubules crossing steeply do not. This is the evidence
    instancer B exploits, imported into A's cost. It returns 0 when profiles are unavailable,
    so A still runs on a channel-less foreground (nnU-Net) and the term doubles as its own
    ablation.
    """
    if a.ori is None or b.ori is None:
        return 0.0
    na, nb = float(np.linalg.norm(a.ori)), float(np.linalg.norm(b.ori))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(1.0 - np.dot(a.ori, b.ori) / (na * nb))


def pair_cost(a: ArmEnd, b: ArmEnd, gap_len: float | None = None,
              w_theta: float = 1.0, w_kappa: float = 10.0,
              w_gap: float = 0.02, w_ori: float = 0.0,
              gap_floor: float = 4.0) -> tuple[float, float]:
    """Cost of joining two arms into one through-path, and the implied curvature.

    The path is ``tip_a -> tip_b``, so the turn is charged in TWO parts: from the heading that
    arrives along arm ``a`` onto the gap direction, and from the gap direction onto arm ``b``'s
    heading. Charging the direct ``|theta_a + pi - theta_b|`` instead is what let two parallel
    microtubules 4 px apart look like a perfect through-path -- both arms are collinear, so the
    direct turn is zero while the real path has to jog sideways and back. Splitting it makes
    that jog cost two large turns, which is the fix for bundle separation and, with the same
    formula, gives gap linking its cost function.

    Below ``gap_floor`` the direction of ``d`` is quantisation noise (arms of one junction are
    almost coincident), so the direct turn is used instead.

    Returns ``(cost, kappa_implied)``; ``kappa_implied`` is the total turn over the span the
    turn has to happen in, and is what the hard curvature bound applies to.
    """
    d = np.asarray(b.pos, dtype=float) - np.asarray(a.pos, dtype=float)
    L = float(np.linalg.norm(d))
    if gap_len is not None:
        span = gap_len
    else:
        span = max(L, gap_floor)

    if L >= gap_floor:
        ang_d = float(np.arctan2(d[1], d[0]))
        total_turn = (turn_penalty(a.theta + np.pi, ang_d)
                      + turn_penalty(ang_d, b.theta))
    else:
        total_turn = turn_penalty(a.theta + np.pi, b.theta)

    kappa_implied = total_turn / max(span, 1e-6)
    cost = (w_theta * total_turn
            + w_kappa * abs(a.kappa + b.kappa)
            + w_gap * span
            + w_ori * _ori_mismatch(a, b))
    return cost, kappa_implied


def match_junction(arms: list[ArmEnd], kappa_max: float,
                   w_theta: float = 1.0, w_kappa: float = 10.0,
                   w_gap: float = 0.02, c_open: float = 1.2,
                   gap_len: float | None = 4.0,
                   gap_floor: float = 4.0, w_ori: float = 0.0,
                   max_gap: float | None = None,
                   allow: "callable | None" = None) -> list[tuple[int, int]]:
    """Pair the arms of one junction. Returns index pairs into ``arms``.

    Unmatched arms are simply absent from the result -- that is the "leave open" outcome.
    A pair is included only when joining beats leaving BOTH of its arms open, i.e. when
    ``cost < 2 * c_open``; that is why the matching weight is ``2 * c_open - cost``.

    ``gap_len=None`` uses the actual distance between the two arm tips (floored at
    ``gap_floor``), which is the physically correct span for the turn to happen over; pass a
    scalar to force one span for every pair. ``max_gap`` drops candidate pairs farther apart
    than that (used when linking free chain ends across foreground holes), and ``allow`` is an
    optional predicate for extra evidence -- e.g. requiring that the image actually shows
    something along the bridge.
    """
    n = len(arms)
    if n < 2:
        return []

    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if max_gap is not None:
                if float(np.linalg.norm(arms[i].pos - arms[j].pos)) > max_gap:
                    continue
            if allow is not None and not allow(arms[i], arms[j]):
                continue
            cost, kappa_implied = pair_cost(arms[i], arms[j], gap_len,
                                            w_theta, w_kappa, w_gap, w_ori, gap_floor)
            if kappa_implied > kappa_max:
                continue                      # physically impossible: forbidden, not costly
            weight = 2.0 * c_open - cost
            if weight > 0:
                g.add_edge(i, j, weight=weight)

    if g.number_of_edges() == 0:
        return []
    matching = nx.max_weight_matching(g, maxcardinality=False)
    return [tuple(sorted(p)) for p in matching]


def bridge_evidence(prob: np.ndarray, a: ArmEnd, b: ArmEnd,
                    n_samples: int = 24) -> float:
    """Mean foreground probability along the straight tip-to-tip bridge.

    A gap in the predicted foreground is worth bridging when the image still shows WEAK
    evidence there -- a microtubule that dropped below the binarisation threshold, not empty
    background. Sampling the probability map along the bridge separates the two, and the
    threshold applied to this should sit BELOW the binarisation threshold: weak-but-present
    evidence plus curvature consistency is exactly the situation a link is justified in.
    (The equivalent feature bought +0.044 Alice F1 in the earlier learned linker, protocol
    §16.) Returns 0.0 for an empty or out-of-frame bridge.
    """
    h, w = prob.shape
    t = np.linspace(0.0, 1.0, max(n_samples, 2))[:, None]
    pts = np.asarray(a.pos, float)[None, :] * (1 - t) + np.asarray(b.pos, float)[None, :] * t
    cc = np.rint(pts[:, 0]).astype(int)
    rr = np.rint(pts[:, 1]).astype(int)
    ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
    if not ok.any():
        return 0.0
    return float(prob[rr[ok], cc[ok]].mean())
