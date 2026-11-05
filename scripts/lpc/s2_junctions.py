#!/usr/bin/env python3
"""LPC screen S2, step 1 (declared docs/TODO.md 2026-10-06 09:05): junctions of the instancer's arc graph at which two
different GT curves each own exactly two arms on opposite sides (a through-crossing with a connected mask); every
curvature-admissible arm pair with its strip and hand features; the instancer's own matching (M0).

    s2_junctions.py DATA PRED_DIR OUT_PREFIX SPLIT [NPROC]      -> OUT_PREFIX.npz (pairs) + OUT_PREFIX.json (junctions)
"""
import glob, json, os, sys
import numpy as np
from collections import defaultdict
from multiprocessing import Pool
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
sys.path[:0] = [".", "src", "scripts"]
from instance.geometry import resample, turn_penalty, window_tangent
from instance.matching import ArmEnd, match_junction, pair_cost
from instance.skeleton_graph import build_arc_graph
from mt_bench.cvat_import import read_frame_h5

UP, KAPPA, TOL, ARC_FRAC, CUT, CTX, HALF, SLEN = 1.5, 0.25, 5.0, 0.6, 0.98, 16, 16, 96
P = {k: v for k, v in json.load(open("src/instance/params_a_derived.json")).items() if not k.startswith("_")}
data, pred, outp, split = sys.argv[1:5]; NP = int(sys.argv[5]) if len(sys.argv) > 5 else 12
OFF = np.arange(-HALF, HALF + 1, dtype=float)


def strip_of(path, img, logit):
    t = np.gradient(path, axis=0); t /= (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9); nrm = np.stack([-t[:, 1], t[:, 0]], 1)
    xs = path[None, :, 0] + OFF[:, None] * nrm[None, :, 0]; ys = path[None, :, 1] + OFF[:, None] * nrm[None, :, 1]
    return map_coordinates(img, [ys / UP, xs / UP], order=1, mode="nearest"), map_coordinates(logit, [ys, xs], order=1, mode="nearest")


def one(path):
    name = os.path.basename(path); fr = read_frame_h5(path)
    img = np.asarray(fr["image"], float); lo, hi = np.percentile(img, [1, 99]); img = (img - np.median(img)) / (hi - lo + 1e-6)
    prob = np.load(os.path.join(pred, name.replace(".h5", ".npz")))["prob"].astype(np.float32)[0]
    pc = np.clip(prob, 1e-6, 1 - 1e-6); logit = np.clip(np.log(pc / (1 - pc)), -12, 12).astype(np.float32)
    gt = [np.asarray(q, float) * UP for q in fr["polylines"] if len(q) >= 2]
    g = build_arc_graph(prob > CUT, merge_radius=P["merge_radius"], min_arc_len=P["min_arc_len"], bridge_max_len=P["bridge_max_len"])
    arcs = {k: resample(np.asarray(a, float), ds=1.0) for k, a in enumerate(g.arcs) if len(a) >= 2}
    if len(gt) < 2 or not arcs: return name, [], []
    trees = [cKDTree(resample(q, ds=1.0)) for q in gt]; owner, proj = {}, {}
    for k, a in arcs.items():
        bj, bf, bi = -1, 0.0, None
        for j, t in enumerate(trees):
            d, i = t.query(a, k=1); f = float((d <= TOL).mean())
            if f > bf: bj, bf, bi = j, f, i
        if bf >= ARC_FRAC: owner[k] = bj; proj[k] = (float(bi[0]), float(bi[-1]))
    by_j = defaultdict(list)
    for k, (j0, j1) in enumerate(g.arc_ends):
        if k not in arcs: continue
        for w, jid in (("start", j0), ("end", j1)):
            if jid is not None:
                th, ka = window_tangent(arcs[k], w, P["window"]); by_j[jid].append(ArmEnd(k, w, th, ka, np.asarray(arcs[k][0] if w == "start" else arcs[k][-1], float)))
    juncs, pairs = [], []
    for jid, arms in by_j.items():
        if len(arms) < 3: continue
        own = defaultdict(list)
        for n, a in enumerate(arms):
            if a.arc_idx in owner: own[owner[a.arc_idx]].append(n)
        truth = []
        for c, ns in own.items():
            if len(ns) != 2: continue
            a, b = arms[ns[0]], arms[ns[1]]
            if a.arc_idx == b.arc_idx: continue
            fa = proj[a.arc_idx][0] if a.which == "start" else proj[a.arc_idx][1]; fb = proj[b.arc_idx][0] if b.which == "start" else proj[b.arc_idx][1]
            ia = sum(proj[a.arc_idx]) / 2; ib = sum(proj[b.arc_idx]) / 2; fm = (fa + fb) / 2
            if (ia - fm) * (ib - fm) < 0: truth.append((ns[0], ns[1]))
        if len(truth) != 2: continue
        m0 = match_junction(arms, kappa_max=KAPPA, w_theta=P["w_theta"], w_kappa=P["w_kappa"], w_gap=P["w_gap"], c_open=P["c_open"],
                            gap_len=None, gap_floor=P["gap_floor"], w_ori=0.0)
        jrec = dict(frame=name, n_arms=len(arms), truth=[list(t) for t in truth], m0=[list(map(int, t)) for t in m0], pairs=[])
        for i in range(len(arms)):
            for j in range(i + 1, len(arms)):
                a, b = arms[i], arms[j]
                if a.arc_idx == b.arc_idx: continue
                cost, kimp = pair_cost(a, b, w_theta=P["w_theta"], w_kappa=P["w_kappa"], w_gap=P["w_gap"], gap_floor=P["gap_floor"])
                if kimp > KAPPA: continue
                gap = float(np.linalg.norm(a.pos - b.pos)); nb = max(2, int(round(gap)) + 1)
                bridge = a.pos[None] + (b.pos - a.pos)[None] * np.linspace(0, 1, nb)[:, None]
                A = arcs[a.arc_idx] if a.which == "end" else arcs[a.arc_idx][::-1]; B = arcs[b.arc_idx] if b.which == "start" else arcs[b.arc_idx][::-1]
                A = A[-CTX:]; B = B[:CTX]; pth = np.concatenate([A[:-1], bridge, B[1:]], 0)[:SLEN]; n = pth.shape[0]
                raw, lg = strip_of(pth, img, logit)
                r = np.zeros((2 * HALF + 1, SLEN), np.float16); l = np.zeros((2 * HALF + 1, SLEN), np.float16); r[:, :n] = raw; l[:, :n] = lg
                cl = map_coordinates(logit, [bridge[:, 1], bridge[:, 0]], order=1, mode="nearest")
                ang = float(np.arctan2(*(b.pos - a.pos)[::-1])) if gap > 1e-6 else a.theta + np.pi
                turn = turn_penalty(a.theta + np.pi, ang) + turn_penalty(ang, b.theta)
                jrec["pairs"].append((i, j)); pairs.append((r, l, (len(A) - 1, nb, n), int((i, j) in truth or (j, i) in truth),
                                                            (float(cl.mean()), float(cl.min()), gap, float(turn), float(cost))))
        juncs.append(jrec)
    return name, juncs, pairs


if __name__ == "__main__":
    paths = [p for p in sorted(glob.glob(os.path.join(data, "*.h5")))
             if os.path.exists(os.path.join(pred, os.path.basename(p).replace(".h5", ".npz")))
             and (split == "all" or str(read_frame_h5(p)["attrs"].get("split")) == split)]
    with Pool(NP) as pool: R = pool.map(one, paths, chunksize=1)
    J = [j for r in R for j in r[1]]; PR = [p for r in R for p in r[2]]
    off = 0
    for j in J: j["offset"] = off; off += len(j["pairs"])
    Z = (2 * HALF + 1, SLEN)
    np.savez(outp + ".npz", raw=np.stack([p[0] for p in PR]) if PR else np.zeros((0,) + Z, np.float16), logit=np.stack([p[1] for p in PR]) if PR else np.zeros((0,) + Z, np.float16),
             lens=np.array([p[2] for p in PR], np.int16).reshape(-1, 3), y=np.array([p[3] for p in PR], np.int8), feats=np.array([p[4] for p in PR], np.float32).reshape(-1, 5),
             frame=np.zeros(len(PR), np.int32), names=np.array([r[0] for r in R]))
    json.dump(J, open(outp + ".json", "w"))
    ok = sum(1 for j in J if all(tuple(sorted(t)) in {tuple(sorted(m)) for m in j["m0"]} for t in j["truth"]))
    print(len(R), "frames", len(J), "through-crossing junctions", len(PR), "pairs | M0 identity %.4f (%d/%d)" % (ok / max(len(J), 1), ok, len(J)),
          "| 4-arm", sum(1 for j in J if j["n_arms"] == 4), "other", sum(1 for j in J if j["n_arms"] != 4))
