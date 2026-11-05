"""Shared pieces of the LPC screens and of instancer L (docs/TODO.md 2026-10-06): the instancer's own arc graph, GT
ownership of arcs, arm ends, the hand features of a candidate link, the S0 oracle and the link-score decoder."""
import json
import networkx as nx
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from instance.geometry import resample, turn_penalty, window_tangent
from instance.matching import ArmEnd, pair_cost
from instance.metrics import rasterise_polyline
from instance.skeleton_graph import build_arc_graph

UP, KAPPA, TOL, ARC_FRAC = 1.5, 0.25, 5.0, 0.6
P = {k: v for k, v in json.load(open("src/instance/params_a_derived.json")).items() if not k.startswith("_")}


def to_logit(prob):
    pc = np.clip(prob, 1e-6, 1 - 1e-6)
    return np.clip(np.log(pc / (1 - pc)), -12, 12).astype(np.float32)


def graph(mask):
    """-> arcs {k: (n, 2) resampled at 1 px}, arc_ends [(j_start, j_end)], arms {(k, which): ArmEnd}"""
    g = build_arc_graph(mask, merge_radius=P["merge_radius"], min_arc_len=P["min_arc_len"], bridge_max_len=P["bridge_max_len"])
    arcs = {k: resample(np.asarray(a, float), ds=1.0) for k, a in enumerate(g.arcs) if len(a) >= 2}
    arms = {}
    for k, a in arcs.items():
        for w in ("start", "end"):
            th, ka = window_tangent(a, w, P["window"]); arms[(k, w)] = ArmEnd(k, w, th, ka, np.asarray(a[0] if w == "start" else a[-1], float))
    return arcs, g.arc_ends, arms


def owners(arcs, gt):
    trees = [cKDTree(resample(q, ds=1.0)) for q in gt]; owner, proj = {}, {}
    for k, a in arcs.items():
        bj, bf, bi = -1, 0.0, None
        for j, t in enumerate(trees):
            d, i = t.query(a, k=1); f = float((d <= TOL).mean())
            if f > bf: bj, bf, bi = j, f, i
        if bf >= ARC_FRAC: owner[k] = bj; proj[k] = (float(bi[0]), float(bi[-1]))
    return owner, proj


def link_feats(a, b, logit):
    """-> (5 hand features, implied curvature) exactly as scripts/lpc/s1_links.py computed them for training."""
    gap = float(np.linalg.norm(a.pos - b.pos))
    cost, kimp = pair_cost(a, b, w_theta=P["w_theta"], w_kappa=P["w_kappa"], w_gap=P["w_gap"], gap_floor=P["gap_floor"])
    nb = max(2, int(round(gap)) + 1); bridge = a.pos[None] + (b.pos - a.pos)[None] * np.linspace(0, 1, nb)[:, None]
    cl = map_coordinates(logit, [bridge[:, 1], bridge[:, 0]], order=1, mode="nearest")
    ang = float(np.arctan2(*(b.pos - a.pos)[::-1])) if gap > 1e-6 else a.theta + np.pi
    turn = turn_penalty(a.theta + np.pi, ang) + turn_penalty(ang, b.theta)
    return (float(cl.mean()), float(cl.min()), gap, float(turn), float(cost)), kimp


CTX, HALF, SLEN = 16, 16, 96                       # strip geometry of S1 (scripts/lpc/s1_links.py), frozen
OFF = np.arange(-HALF, HALF + 1, dtype=float)


def link_path(arcs, ea, eb, a, b):
    """Path of a candidate link exactly as S1 built it: last CTX px of arc A (ending at the link), the straight bridge,
    first CTX px of arc B. -> path (n <= SLEN, 2), lens (len_A - 1, n_bridge, n)."""
    gap = float(np.linalg.norm(a.pos - b.pos)); nb = max(2, int(round(gap)) + 1)
    bridge = a.pos[None] + (b.pos - a.pos)[None] * np.linspace(0, 1, nb)[:, None]
    A = arcs[ea[0]] if ea[1] == "end" else arcs[ea[0]][::-1]; B = arcs[eb[0]] if eb[1] == "start" else arcs[eb[0]][::-1]
    A = A[-CTX:]; B = B[:CTX]; pth = np.concatenate([A[:-1], bridge, B[1:]], 0)[:SLEN]
    return pth, (len(A) - 1, nb, pth.shape[0])


def strip_logit(path, logit):
    """Map logit resampled in the path's own frame -> (2 * HALF + 1, SLEN) float16, zero-padded along the path."""
    t = np.gradient(path, axis=0); t /= (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
    nrm = np.stack([-t[:, 1], t[:, 0]], 1)
    xs = path[None, :, 0] + OFF[:, None] * nrm[None, :, 0]; ys = path[None, :, 1] + OFF[:, None] * nrm[None, :, 1]
    out = np.zeros((2 * HALF + 1, SLEN), np.float16); out[:, :path.shape[0]] = map_coordinates(logit, [ys, xs], order=1, mode="nearest")
    return out


def strip_x(lg, lens):
    """(n, H, SLEN) logit strips + (n, 3) lens -> the map-only network input (n, 4, H, SLEN): [0, logit / 12, valid, bridge]."""
    lg = np.asarray(lg, np.float32) / 12.0; lens = np.asarray(lens); s = np.arange(lg.shape[2])[None]
    valid = (s < lens[:, 2:3]).astype(np.float32); bridge = ((s >= lens[:, 0:1]) & (s < lens[:, 0:1] + lens[:, 1:2])).astype(np.float32)
    v = np.repeat(valid[:, None, :], lg.shape[1], 1); b = np.repeat(bridge[:, None, :], lg.shape[1], 1)
    return np.stack([np.zeros_like(lg), lg * v, v, b], 1)


def strip_net():
    """S1's verifier (scripts/lpc/s1_train.py), unchanged."""
    import torch.nn as nn, torch
    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())
            s.b1, s.b2, s.b3 = blk(4, 32), blk(32, 64), blk(64, 96); s.pool = nn.MaxPool2d(2); s.fc = nn.Sequential(nn.Linear(192, 64), nn.ReLU(), nn.Linear(64, 1))
        def forward(s, x):
            h = s.b3(s.pool(s.b2(s.pool(s.b1(x)))))
            return s.fc(torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1))[:, 0]
    return Net()


def strip_scores(nets, lg, lens, dev, bs=512):
    """Mean logit of the given networks over (n, H, SLEN) strips."""
    import torch
    if len(lens) == 0: return np.zeros(0, np.float32)
    out = np.zeros(len(lens), np.float32)
    with torch.no_grad():
        for i in range(0, len(lens), bs):
            x = torch.from_numpy(strip_x(lg[i:i + bs], lens[i:i + bs])).to(dev)
            out[i:i + bs] = sum(n(x).float() for n in nets).cpu().numpy() / len(nets)
    return out


def oracle_masks(arcs, arms, gt, br, shape):
    owner, proj = owners(arcs, gt); preds = []
    for j in range(len(gt)):
        mine = sorted((k for k, o in owner.items() if o == j), key=lambda k: (proj[k][0] + proj[k][1]) / 2)
        if not mine: continue
        ori = [proj[k][0] <= proj[k][1] for k in mine]; runs, cur = [], [0]
        for n in range(1, len(mine)):
            a = arms[(mine[n - 1], "end" if ori[n - 1] else "start")]; b = arms[(mine[n], "start" if ori[n] else "end")]
            ok = float(np.linalg.norm(a.pos - b.pos)) <= br and pair_cost(a, b, gap_floor=P["gap_floor"])[1] <= KAPPA
            if ok: cur.append(n)
            else: runs.append(cur); cur = [n]
        runs.append(cur)
        best = max(runs, key=lambda r: sum(abs(proj[mine[n]][1] - proj[mine[n]][0]) + 1 for n in r))
        poly = np.concatenate([arcs[mine[n]] if ori[n] else arcs[mine[n]][::-1] for n in best], 0)
        preds.append(rasterise_polyline(resample(poly, ds=0.5), shape))
    return preds


def link_edges(prob, cut, model, br_max, strips=False):
    """-> arcs, [(end_a, end_b, gap, log-odds)] for every curvature-admissible pair of ends of different arcs within br_max.
    strips=True also returns (feats (n, 5), logit strips (n, H, SLEN) float16, lens (n, 3)) in edge order (pairs sorted,
    A = the lower end index, as in S1)."""
    mask = prob > cut; arcs, _, arms = graph(mask); E = (np.zeros((0, 5), np.float32), np.zeros((0, 2 * HALF + 1, SLEN), np.float16), np.zeros((0, 3), np.int16))
    if not arcs: return (arcs, [], E) if strips else (arcs, [])
    logit = to_logit(prob); keys = list(arms); tree = cKDTree(np.array([arms[q].pos for q in keys]))
    mu, sd, w, b0 = (np.asarray(model[k], float) for k in ("mu", "sd", "coef", "intercept")); edges, F, S, N = [], [], [], []
    for i, j in sorted(tree.query_pairs(br_max)):
        if keys[i][0] == keys[j][0]: continue
        f, kimp = link_feats(arms[keys[i]], arms[keys[j]], logit)
        if kimp > KAPPA: continue
        edges.append((keys[i], keys[j], f[2], float(((np.asarray(f) - mu) / sd) @ w + b0)))
        if strips:
            pth, ln = link_path(arcs, keys[i], keys[j], arms[keys[i]], arms[keys[j]]); F.append(f); S.append(strip_logit(pth, logit)); N.append(ln)
    if not strips: return arcs, edges
    return arcs, edges, ((np.array(F, np.float32), np.stack(S), np.array(N, np.int16)) if edges else E)


def chain(arcs, edges, br, t, shape=None):
    """Max-weight matching of arc ENDS (weight = log-odds - t > 0, gap <= br), then chaining; polylines or masks."""
    g = nx.Graph()
    for a, b, gap, s in edges:
        if gap <= br and s - t > 0: g.add_edge(a, b, weight=s - t)
    partner = {}
    for a, b in (nx.max_weight_matching(g) if g.number_of_edges() else ()): partner[a] = b; partner[b] = a
    used, polys = set(), []
    def walk(k, w_in):
        out = []
        while k not in used:
            used.add(k); out.append(arcs[k] if w_in == "start" else arcs[k][::-1])
            nxt = partner.get((k, "end" if w_in == "start" else "start"))
            if nxt is None: break
            k, w_in = nxt
        return out
    for k in arcs:                                   # open chains first: start from an end with no partner
        for w in ("start", "end"):
            if (k, w) not in partner and k not in used: polys.append(np.concatenate(walk(k, w), 0))
    for k in arcs:                                   # what is left are closed cycles
        if k not in used: polys.append(np.concatenate(walk(k, "start"), 0))
    polys = [p for p in polys if float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) >= P["min_length"]]
    return polys if shape is None else [rasterise_polyline(resample(p, ds=0.5), shape) for p in polys]


def decode_links(prob, cut, model, br, t):
    """Instancer L (docs/TODO.md 2026-10-06 10:10)."""
    arcs, edges = link_edges(prob, cut, model, br)
    return chain(arcs, edges, br, t, prob.shape) if arcs else []
