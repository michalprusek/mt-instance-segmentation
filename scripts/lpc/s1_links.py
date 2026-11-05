#!/usr/bin/env python3
"""LPC screen S1, step 1 (declared docs/TODO.md 2026-10-06 08:25): candidate links of the S0 graph, their labels,
their rectified strips and the hand features of the comparators.

    s1_links.py DATA PRED_DIR OUT.npz SPLIT REAL(0|1) [NPROC]

A link = two ends of different arcs, 4 <= distance <= 60 eval px, implied curvature <= 0.25 (mask = prob > 0.98).
Strip = 16 px of arc A + straight bridge + 16 px of arc B, 1 px steps, lateral +-16 px, in the 1.5x eval frame;
raw image (per-frame robust normalisation, no model) and map logit.
"""
import glob, json, os, sys
import numpy as np
from multiprocessing import Pool
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
sys.path[:0] = [".", "src", "scripts"]
from instance.geometry import resample, turn_penalty, window_tangent
from instance.matching import ArmEnd, pair_cost
from instance.skeleton_graph import build_arc_graph
from mt_bench.cvat_import import read_frame_h5

UP, KAPPA, TOL, ARC_FRAC, CUT = 1.5, 0.25, 5.0, 0.6, 0.98
BR_MIN, BR_MAX, CTX, HALF, SLEN = 4.0, 60.0, 16, 16, 96
P = {k: v for k, v in json.load(open("src/instance/params_a_derived.json")).items() if not k.startswith("_")}
data, pred, out, split, REAL = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], bool(int(sys.argv[5]))
NP = int(sys.argv[6]) if len(sys.argv) > 6 else 12
OFF = np.arange(-HALF, HALF + 1, dtype=float)


def strip_of(path, img, logit):
    """path (n, 2) x,y in the eval frame -> raw strip, logit strip, each (2*HALF+1, n)."""
    t = np.gradient(path, axis=0); t /= (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
    nrm = np.stack([-t[:, 1], t[:, 0]], 1)
    xs = path[None, :, 0] + OFF[:, None] * nrm[None, :, 0]; ys = path[None, :, 1] + OFF[:, None] * nrm[None, :, 1]
    raw = map_coordinates(img, [ys / UP, xs / UP], order=1, mode="nearest")
    lg = map_coordinates(logit, [ys, xs], order=1, mode="nearest")
    return raw, lg


def one(path):
    name = os.path.basename(path); fr = read_frame_h5(path)
    img = np.asarray(fr["image"], float); lo, hi = np.percentile(img, [1, 99]); img = (img - np.median(img)) / (hi - lo + 1e-6)
    prob = np.load(os.path.join(pred, name.replace(".h5", ".npz")))["prob"].astype(np.float32)[0]
    logit = np.clip(np.log(np.clip(prob, 1e-6, 1 - 1e-6) / (1 - np.clip(prob, 1e-6, 1 - 1e-6))), -12, 12).astype(np.float32)
    gt = [np.asarray(q, float) * UP for q in fr["polylines"] if len(q) >= 2]
    g = build_arc_graph(prob > CUT, merge_radius=P["merge_radius"], min_arc_len=P["min_arc_len"], bridge_max_len=P["bridge_max_len"])
    arcs = [resample(np.asarray(a, float), ds=1.0) for a in g.arcs if len(a) >= 2]
    empty = (np.zeros((0, 2 * HALF + 1, SLEN), np.float16),) * 2 + (np.zeros((0, 3), np.int16), np.zeros(0, np.int8), np.zeros((0, 5), np.float32), 0, 0)
    if len(arcs) < 2: return (name,) + empty
    trees = [cKDTree(resample(q, ds=1.0)) for q in gt]; gts = [resample(q, ds=1.0) for q in gt]
    owner, proj = {}, {}
    for k, a in enumerate(arcs):
        bj, bf, bi = -1, 0.0, None
        for j, t in enumerate(trees):
            d, i = t.query(a, k=1); f = float((d <= TOL).mean())
            if f > bf: bj, bf, bi = j, f, i
        if bf >= ARC_FRAC: owner[k] = bj; proj[k] = (float(bi[0]), float(bi[-1]))
    order = {}
    for j in set(owner.values()):
        mine = sorted((k for k, o in owner.items() if o == j), key=lambda k: (proj[k][0] + proj[k][1]) / 2)
        for n, k in enumerate(mine): order[k] = n
    arms = {}
    for k, a in enumerate(arcs):
        for w in ("start", "end"):
            th, ka = window_tangent(a, w, P["window"]); arms[(k, w)] = ArmEnd(k, w, th, ka, np.asarray(a[0] if w == "start" else a[-1], float))
    keys = list(arms); tree = cKDTree(np.array([arms[q].pos for q in keys]))
    RAW, LG, LEN, Y, F = [], [], [], [], []; n_amb = n_unowned = 0
    for i, j in sorted(tree.query_pairs(BR_MAX)):
        (ka_, wa), (kb_, wb) = keys[i], keys[j]
        if ka_ == kb_: continue
        a, b = arms[(ka_, wa)], arms[(kb_, wb)]; gap = float(np.linalg.norm(a.pos - b.pos))
        if gap < BR_MIN: continue
        cost, kimp = pair_cost(a, b, w_theta=P["w_theta"], w_kappa=P["w_kappa"], w_gap=P["w_gap"], gap_floor=P["gap_floor"])
        if kimp > KAPPA: continue
        nb = max(2, int(round(gap)) + 1); bridge = a.pos[None] + (b.pos - a.pos)[None] * np.linspace(0, 1, nb)[:, None]
        oa, ob = owner.get(ka_), owner.get(kb_)
        if oa is None or ob is None:
            if REAL: n_unowned += 1; continue
            y = 0
        elif oa != ob: y = 0
        else:
            d, _ = trees[oa].query(bridge, k=1)
            # consecutive along the curve, and joined by the ends that face each other along it
            fa = proj[ka_][1] if wa == "end" else proj[ka_][0]; fb = proj[kb_][1] if wb == "end" else proj[kb_][0]
            ia = (proj[ka_][0] + proj[ka_][1]) / 2; ib = (proj[kb_][0] + proj[kb_][1]) / 2
            facing = (ia < ib and ia <= fa <= fb <= ib + 1e-9) or (ib < ia and ib <= fb <= fa <= ia + 1e-9)
            if abs(order[ka_] - order[kb_]) == 1 and facing and float((d <= TOL).mean()) >= 0.8: y = 1
            else: n_amb += 1; continue
        A = arcs[ka_] if wa == "end" else arcs[ka_][::-1]; B = arcs[kb_] if wb == "start" else arcs[kb_][::-1]
        A = A[-CTX:]; B = B[:CTX]; pth = np.concatenate([A[:-1], bridge, B[1:]], 0)[:SLEN]
        raw, lg = strip_of(pth, img, logit); n = pth.shape[0]
        r = np.zeros((2 * HALF + 1, SLEN), np.float16); l = np.zeros((2 * HALF + 1, SLEN), np.float16); r[:, :n] = raw; l[:, :n] = lg
        cl = map_coordinates(logit, [bridge[:, 1], bridge[:, 0]], order=1, mode="nearest")
        ang = float(np.arctan2(*(b.pos - a.pos)[::-1])); turn = turn_penalty(a.theta + np.pi, ang) + turn_penalty(ang, b.theta)
        RAW.append(r); LG.append(l); LEN.append((len(A) - 1, nb, n)); Y.append(y)
        F.append((float(cl.mean()), float(cl.min()), gap, float(turn), float(cost)))
    if not Y: return (name,) + empty[:5] + (n_amb, n_unowned)
    return name, np.stack(RAW), np.stack(LG), np.array(LEN, np.int16), np.array(Y, np.int8), np.array(F, np.float32), n_amb, n_unowned


if __name__ == "__main__":
    paths = [p for p in sorted(glob.glob(os.path.join(data, "*.h5")))
             if os.path.exists(os.path.join(pred, os.path.basename(p).replace(".h5", ".npz")))
             and (split == "all" or str(read_frame_h5(p)["attrs"].get("split")) == split)]
    with Pool(NP) as pool: R = pool.map(one, paths, chunksize=1)
    fid = np.concatenate([np.full(len(r[4]), i, np.int32) for i, r in enumerate(R)])
    np.savez(out, raw=np.concatenate([r[1] for r in R]), logit=np.concatenate([r[2] for r in R]), lens=np.concatenate([r[3] for r in R]),
             y=np.concatenate([r[4] for r in R]), feats=np.concatenate([r[5] for r in R]), frame=fid, names=np.array([r[0] for r in R]))
    y = np.concatenate([r[4] for r in R])
    print(len(R), "frames", len(y), "links  pos", int(y.sum()), "neg", int((1 - y).sum()), "| excluded ambiguous", sum(r[6] for r in R), "unowned(real)", sum(r[7] for r in R))
