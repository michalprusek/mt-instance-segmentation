#!/usr/bin/env python3
"""S1 (docs/TODO.md 2026-11-04 08:10): L+H with one term added to the link score, a vote of the neighbouring frames.
A neighbour votes +1 when one of its curves covers both arc ends and runs along the bridge, -1 when two different curves cover them, 0 when it cannot say. The five link features and their coefficients are frozen; lam 0 or no neighbours is instance_lh exactly."""
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from instance.instancer_lh import _arc_graph, _chains, _link_features, hysteresis_mask, to_logit, TooManyArcs
from instance.matching import pair_cost
WINDOW = 15   # eval px of an arc behind its end
TOL = 5.0     # the metric's tolerance at the eval scale


class Neighbour:
    """The curves of t-1 or t+1, (x, y) in frame t at the eval scale, sampled about every px."""

    def __init__(self, curves):
        self.curves = [np.asarray(c, float) for c in curves if len(c)]
        self.tree = cKDTree(np.vstack(self.curves)) if self.curves else None
        self.own = np.concatenate([np.full(len(c), i) for i, c in enumerate(self.curves)]) if self.curves else None
        self.one = {}

    def covering(self, seg):
        """The curve within TOL of most of seg, or -1 when under half of seg has any."""
        if self.tree is None: return -1
        d, i = self.tree.query(seg)
        hit = d <= TOL
        if hit.mean() < 0.5: return -1
        ids, n = np.unique(self.own[i[hit]], return_counts=True)
        return int(ids[n.argmax()])

    def along(self, cid, p, q):
        """Does curve cid run along the bridge p-q? Twice TOL, since a chord cuts inside a bend."""
        if cid not in self.one: self.one[cid] = cKDTree(self.curves[cid])
        n = max(2, int(round(float(np.linalg.norm(p - q)))) + 1)
        return (self.one[cid].query(p[None] + (q - p)[None] * np.linspace(0, 1, n)[:, None])[0] <= 2 * TOL).mean() >= 0.8


def _candidates(arcs, arms, logit, kappa_max, params, neighbours):
    """Every link _match_ends scores, in its order, with the score and the sum of the votes."""
    model = params["link_model"]
    mu, sd, coef = (np.asarray(model[k], float) for k in ("mu", "sd", "coef"))
    intercept, reach = float(model["intercept"]), float(params["match_reach"])
    keys = list(arms); tree = cKDTree(np.array([arms[q].pos for q in keys])); out = []
    seg = {q: (arcs[q[0]][:WINDOW] if q[1] == "start" else arcs[q[0]][-WINDOW:]) for q in keys} if neighbours else {}
    label = {q: [nb.covering(seg[q]) for nb in neighbours] for q in seg}
    for i, j in sorted(tree.query_pairs(reach)):
        if keys[i][0] == keys[j][0]: continue
        a, b = arms[keys[i]], arms[keys[j]]
        cost, kappa_implied = pair_cost(a, b, w_theta=params["w_theta"], w_kappa=params["w_kappa"], w_gap=params["w_gap"], gap_floor=params["gap_floor"])
        if kappa_implied > kappa_max: continue
        features = _link_features(a, b, cost, logit)
        if features[2] > reach: continue
        score = float(((np.asarray(features) - mu) / sd) @ coef + intercept); v = 0
        for n, nb in enumerate(neighbours):
            la, lb = label[keys[i]][n], label[keys[j]][n]
            if la < 0 or lb < 0: continue
            v += -1 if la != lb else int(nb.along(la, a.pos, b.pos))
        out.append((keys[i], keys[j], score, v))
    return out


def _solve(arcs, cands, lam, params):
    g = nx.Graph()
    for p, q, score, v in cands:
        s = score + lam * v
        if s > 0: g.add_edge(p, q, weight=s)
    partner = {}
    for a, b in (nx.max_weight_matching(g) if g.number_of_edges() else ()):
        partner[a] = b; partner[b] = a
    polys = []
    for chain in _chains(arcs, partner):
        p = np.concatenate([arcs[k] if entered == "start" else arcs[k][::-1] for k, entered in chain], 0)
        if float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) >= params["min_length"]: polys.append(p)
    return polys


def decode_ladder(prob, cut, kappa_max, params, neighbours, lams):
    """{lam: polylines (x=col, y=row) at the eval scale}. Raises TooManyArcs as instance_lh does."""
    mask = hysteresis_mask(prob, cut, float(params["hyst_t_lo"]), float(params["hyst_radius"]))
    arcs, arms = _arc_graph(mask, params)
    if not arcs: return {lam: [] for lam in lams}
    if len(arcs) > params["max_arcs"]: raise TooManyArcs(f"{len(arcs)} arcs, limit {params['max_arcs']}")
    cands = _candidates(arcs, arms, to_logit(prob), kappa_max, params, neighbours)
    return {lam: _solve(arcs, cands, lam, params) for lam in lams}
