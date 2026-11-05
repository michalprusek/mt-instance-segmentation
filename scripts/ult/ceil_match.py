#!/usr/bin/env python3
"""(a) + (c) of docs/TODO.md 2026-11-02 07:15. (a) Ceiling of any link score: matching oracle on L's candidate graph with the H mask
(+1 for consecutive, facing ends of arcs owned by one GT, the S1 label; same matching, chaining, min_length; unowned arcs stay).
(c) For each right link not taken because an end went to another GT's arc: right vs winner on score, gap, turn, turn at 30 px,
|d median logit|, |d mask half-width| near the ends. usage: ceil_match.py DATA PRED CUT OUT.json [htw|all] [NPROC] [SPLIT]"""
import glob, json, os, sys
import numpy as np, networkx as nx
from multiprocessing import Pool
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
sys.path[:0] = [".", "src", "scripts", "scripts/lpc"]
from lpc_lib import P, UP, KAPPA, TOL, graph, owners, link_feats, to_logit, resample, rasterise_polyline
from l_dup import match, walk_all, arclen, row, parallel_pairs, MU, SD, WV, B0
from l_hyst2 import hmask
from instance.geometry import turn_penalty, window_tangent
from mt_bench.cvat_import import read_frame_h5
from mt_bench.gt_stats import count_crossings
from step1_read import source_of
data, pred, cut, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
SRC = sys.argv[5] if len(sys.argv) > 5 else "all"; NP = int(sys.argv[6]) if len(sys.argv) > 6 else 40; SPLIT = sys.argv[7] if len(sys.argv) > 7 else "val"
BRS = (60.0, 100.0); NEAR = 16

def truth_edges(arcs, arms, gt, br):
    """{(end_a, end_b)} S1 label over the candidate graph at bridge br."""
    owner, proj = owners(arcs, gt)
    trees = [cKDTree(resample(q, ds=1.0)) for q in gt]; order = {}
    for j in set(owner.values()):
        mine = sorted((k for k, o in owner.items() if o == j), key=lambda k: (proj[k][0] + proj[k][1]) / 2)
        for n, k in enumerate(mine): order[k] = n
    keys = list(arms); tree = cKDTree(np.array([arms[q].pos for q in keys])); T = []
    for i, j in sorted(tree.query_pairs(br)):
        (ka, wa), (kb, wb) = keys[i], keys[j]
        if ka == kb or owner.get(ka) is None or owner.get(ka) != owner.get(kb) or abs(order[ka] - order[kb]) != 1: continue
        a, b = arms[keys[i]], arms[keys[j]]; gap = float(np.linalg.norm(a.pos - b.pos))
        from instance.matching import pair_cost
        if pair_cost(a, b, w_theta=P["w_theta"], w_kappa=P["w_kappa"], w_gap=P["w_gap"], gap_floor=P["gap_floor"])[1] > KAPPA: continue
        fa = proj[ka][1] if wa == "end" else proj[ka][0]; fb = proj[kb][1] if wb == "end" else proj[kb][0]
        ia, ib = (proj[ka][0] + proj[ka][1]) / 2, (proj[kb][0] + proj[kb][1]) / 2
        if not ((ia < ib and ia <= fa <= fb <= ib + 1e-9) or (ib < ia and ib <= fb <= fa <= ia + 1e-9)): continue
        nb = max(2, int(round(gap)) + 1); bridge = a.pos[None] + (b.pos - a.pos)[None] * np.linspace(0, 1, nb)[:, None]
        if float((trees[owner[ka]].query(bridge, k=1)[0] <= TOL).mean()) >= 0.8: T.append((keys[i], keys[j]))
    return T, owner

def chains_to_masks(arcs, partner, shape):
    polys = []
    for ch in walk_all(arcs, partner):
        p = np.concatenate([arcs[k] if w == "start" else arcs[k][::-1] for k, w in ch], 0)
        if arclen(p) >= P["min_length"]: polys.append(p)
    return [rasterise_polyline(resample(p, ds=0.5), shape) for p in polys]

def one(path):
    name = os.path.basename(path); fr = read_frame_h5(path)
    if SPLIT != "all" and str(fr["attrs"].get("split", "")) != SPLIT: return None
    if SRC == "htw" and source_of(name) != "htw": return None
    f = os.path.join(pred, name.replace(".h5", ".npz"))
    if not os.path.exists(f): return None
    prob = np.load(f)["prob"].astype(np.float32)[0]; gt = [np.asarray(q, float) * UP for q in fr["polylines"] if len(q) >= 2]
    cr = count_crossings(gt, tol=2.0 * UP, endpoint_margin=8.0 * UP) if len(gt) > 1 else []; par = parallel_pairs(gt) if len(gt) > 1 else []
    mask = hmask(prob, cut, "bridge", 0.98, 22.5); arcs, _, arms = graph(mask); res = {}; pairs = []
    if not arcs:
        e = row([], gt, cr, par); return name, {"L": e, "o60": e, "o100": e}, []
    logit = to_logit(prob); partner = match(arms, {}, logit); res["L"] = row(chains_to_masks(arcs, partner, prob.shape), gt, cr, par)
    owner = {}
    if gt:
        for br in BRS:
            T, owner = truth_edges(arcs, arms, gt, br); g = nx.Graph(); g.add_edges_from(T)
            po = {}
            for a, b in (nx.max_weight_matching(g, maxcardinality=True) if g.number_of_edges() else ()): po[a] = b; po[b] = a
            res[f"o{br:g}"] = row(chains_to_masks(arcs, po, prob.shape), gt, cr, par)
            if br == 60.0: T60 = T
        dt = distance_transform_edt(mask)
        def near(k, w):
            a = arcs[k]; s = a[-NEAR:] if w == "end" else a[:NEAR]; xi = np.clip(np.round(s[:, 0]).astype(int), 0, prob.shape[1] - 1); yi = np.clip(np.round(s[:, 1]).astype(int), 0, prob.shape[0] - 1)
            return float(np.median(logit[yi, xi])), float(np.median(dt[yi, xi]))
        def feats(x, y):
            a, b = arms[x], arms[y]; fe, _ = link_feats(a, b, logit); s = float(((np.asarray(fe) - MU) / SD) @ WV + B0)
            ta, _k = window_tangent(arcs[x[0]], x[1], 30.0); tb, _k = window_tangent(arcs[y[0]], y[1], 30.0)
            ang = float(np.arctan2(*(b.pos - a.pos)[::-1])) if fe[2] > 1e-6 else a.theta + np.pi
            t30 = float(turn_penalty(ta + np.pi, ang) + turn_penalty(ang, tb)); la, wa = near(*x); lb, wb = near(*y)
            return dict(score=s, gap=fe[2], turn=fe[3], turn30=t30, dlogit=abs(la - lb), dwidth=abs(wa - wb), minlen=min(arclen(arcs[x[0]]), arclen(arcs[y[0]])))
        for x, y in T60:
            if partner.get(x) == y: continue
            for e, o in ((x, y), (y, x)):
                q = partner.get(e)
                if q is not None and owner.get(q[0], -1) not in (-1, owner.get(e[0])):
                    pairs.append((feats(e, o), feats(e, q))); break
    return name, res, pairs

if __name__ == "__main__":
    with Pool(NP) as pl: rows = [r for r in pl.map(one, sorted(glob.glob(os.path.join(data, "*.h5"))), chunksize=1) if r]
    f1 = lambda t: 2 * t[0] / max(2 * t[0] + t[1] + t[2], 1); tot = {}
    for k in ("L", "o60", "o100"):
        t = np.zeros(5)
        for _, r, _p in rows:
            if k in r: q = r[k]; t += (q["tp"], q["fp"], q["fn"], q["jn"], q["jp"])
        tot[k] = t.tolist(); print(f"{os.path.basename(pred.rstrip('/'))} {k}: F1 {f1(t):.4f} tp/fp/fn {int(t[0])}/{int(t[1])}/{int(t[2])} identity {t[4] / max(t[3], 1):.3f}", flush=True)
    json.dump(dict(total=tot, pairs=[p for _, _r, ps in rows for p in ps]), open(out, "w"))
