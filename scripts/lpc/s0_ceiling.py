#!/usr/bin/env python3
"""LPC screen S0 (declared docs/TODO.md 2026-10-06 07:55): the CANDIDATE CEILING.

Candidates = arcs of the skeleton arc graph of a (permissive) mask + every link between two arc ends that is
closer than BR and passes the hard curvature bound (kappa_implied <= 0.25, the instancer's own test). The ORACLE
chooses with the ground truth: an arc belongs to the GT curve that covers most of it (>= ARC_FRAC of its points
within TOL), each arc is used once, and per GT curve the longest run of consecutively ADMISSIBLE links along the
curve is emitted as one path. Scored with the declared metric. The same mask through instance_a is the reference.

    s0_ceiling.py DATA SRC OUT.json SPLIT CUTS BRS [NPROC]
      SRC = a dump directory (npz with 'prob') or the word BAND (perfect band painted from GT, half-widths 2.0 / 3.3)
      CUTS, BRS = comma lists (CUTS ignored for BAND)
"""
import glob, json, os, sys
import numpy as np
from multiprocessing import Pool
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
sys.path[:0] = [".", "src", "scripts"]
from instance.geometry import resample, window_tangent
from instance.instancer_a import instance_a
from instance.matching import ArmEnd, pair_cost
from instance.metrics import centerline_f1, rasterise_polyline
from instance.skeleton_graph import build_arc_graph
from mt_bench.cvat_import import read_frame_h5

UP, KAPPA, TOL, ARC_FRAC = 1.5, 0.25, 5.0, 0.6
P = {k: v for k, v in json.load(open("src/instance/params_a_derived.json")).items() if not k.startswith("_")}
data, src, out, split = sys.argv[1:5]
CUTS = [float(x) for x in sys.argv[5].split(",")]; BRS = [float(x) for x in sys.argv[6].split(",")]
NP = int(sys.argv[7]) if len(sys.argv) > 7 else 16
LE = [0, 12, 20, 35, 60, 120, 1e9]
KEEP_UNOWNED = bool(int(os.environ.get("S0_KEEP_UNOWNED", "0")))


def arm(arc, k, which):
    th, ka = window_tangent(arc, which, P["window"])
    return ArmEnd(k, which, th, ka, np.asarray(arc[0] if which == "start" else arc[-1], float))


def admissible(a, b, br):
    if float(np.linalg.norm(a.pos - b.pos)) > br:
        return False
    return pair_cost(a, b, gap_floor=P["gap_floor"])[1] <= KAPPA


def oracle(mask, gt, brs):
    """-> {br: (pred masks, n_links)}, n_arcs"""
    g = build_arc_graph(mask, merge_radius=P["merge_radius"], min_arc_len=P["min_arc_len"], bridge_max_len=P["bridge_max_len"])
    arcs = [np.asarray(a, float) for a in g.arcs if len(a) >= 2]
    if not arcs:
        return {br: ([], 0) for br in brs}, 0
    arms = {(k, w): arm(a, k, w) for k, a in enumerate(arcs) for w in ("start", "end")}
    gts = [resample(q, ds=1.0) for q in gt]; trees = [cKDTree(q) for q in gts]
    owner, proj = {}, {}
    for k, a in enumerate(arcs):
        pts = resample(a, ds=1.0); best, bf, bi = -1, 0.0, None
        for j, t in enumerate(trees):
            d, i = t.query(pts, k=1)
            f = float((d <= TOL).mean())
            if f > bf: best, bf, bi = j, f, i
        if bf >= ARC_FRAC:
            owner[k] = best; proj[k] = (float(bi[0]), float(bi[-1]))   # GT sample index at the arc's start / end
    # graph size without GT: admissible links between ends of DIFFERENT arcs
    keys = list(arms); pos = np.array([arms[q].pos for q in keys]); tree = cKDTree(pos); res = {}
    for br in brs:
        n_links = sum(1 for i, j in tree.query_pairs(br) if keys[i][0] != keys[j][0] and admissible(arms[keys[i]], arms[keys[j]], br))
        preds = []
        for j in range(len(gt)):
            mine = sorted((k for k, o in owner.items() if o == j), key=lambda k: (proj[k][0] + proj[k][1]) / 2)
            if not mine: continue
            ori = [proj[k][0] <= proj[k][1] for k in mine]          # True: the arc runs along the GT direction
            runs, cur = [], [0]
            for n in range(1, len(mine)):
                a = arms[(mine[n - 1], "end" if ori[n - 1] else "start")]; b = arms[(mine[n], "start" if ori[n] else "end")]
                if admissible(a, b, br): cur.append(n)
                else: runs.append(cur); cur = [n]
            runs.append(cur)
            best = max(runs, key=lambda r: sum(abs(proj[mine[n]][1] - proj[mine[n]][0]) + 1 for n in r))
            poly = np.concatenate([arcs[mine[n]] if ori[n] else arcs[mine[n]][::-1] for n in best], 0)
            preds.append(rasterise_polyline(resample(poly, ds=0.5), mask.shape))
        if KEEP_UNOWNED:      # exploratory decomposition: the oracle loses the EXISTENCE privilege -- every arc no GT curve
            for k, a in enumerate(arcs):   # owns is emitted as its own prediction if it passes the instancer's length floor
                if k not in owner and float(np.linalg.norm(np.diff(a, axis=0), axis=1).sum()) >= P["min_length"]:
                    preds.append(rasterise_polyline(resample(a, ds=0.5), mask.shape))
        res[br] = (preds, n_links)
    return res, len(arcs)


def score(masks, gt):
    r = centerline_f1(masks, gt, tol=TOL, length_coverage=0.8, precision_coverage=0.8, end_tol=0.0)
    return r["tp"], r["fp"], r["fn"], {p["gt"] for p in r.get("pairs", []) if p["tp"]}


def one(path):
    name = os.path.basename(path); fr = read_frame_h5(path); H, W = np.asarray(fr["image"]).shape
    h, w = int(round(H * UP)), int(round(W * UP)); gt = [np.asarray(q, float) * UP for q in fr["polylines"] if len(q) >= 2]
    L = [float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum()) / UP for q in gt]
    variants = {}
    if src == "BAND":
        line = np.zeros((h, w), bool)
        for q in gt:
            p = resample(q, ds=0.25); ok = (p[:, 0] >= 0) & (p[:, 0] <= w - 1) & (p[:, 1] >= 0) & (p[:, 1] <= h - 1)
            line[np.round(p[ok, 1]).astype(int), np.round(p[ok, 0]).astype(int)] = True
        d = distance_transform_edt(~line)
        for hw in (2.0, 3.3): variants["hw%g" % hw] = (d <= hw, (d <= hw).astype(np.float32))
    else:
        prob = np.load(os.path.join(src, name.replace(".h5", ".npz")))["prob"].astype(np.float32)[0]
        for c in CUTS: variants["c%g" % c] = (prob > c, prob)
    res = {}
    for vk, (mask, prob) in variants.items():
        _, m = instance_a(mask, KAPPA, params=P, channels=prob[None], prob=prob)
        tp, fp, fn, _ = score(m, gt); row = dict(inst=(tp, fp, fn))
        if gt:
            orc, n_arcs = oracle(mask, gt, BRS); row["n_arcs"] = n_arcs
            for br, (preds, n_links) in orc.items():
                tp, fp, fn, hit = score(preds, gt)
                row["br%g" % br] = dict(s=(tp, fp, fn), n_links=n_links, miss_len=[L[k] for k in range(len(gt)) if k not in hit])
        else:
            for br in BRS: row["br%g" % br] = dict(s=(0, 0, 0), n_links=0, miss_len=[])
            row["n_arcs"] = 0
        res[vk] = row
    return name, L, res


if __name__ == "__main__":
    paths = [p for p in sorted(glob.glob(os.path.join(data, "*.h5")))
             if (src == "BAND" or os.path.exists(os.path.join(src, os.path.basename(p).replace(".h5", ".npz"))))
             and (split == "all" or str(read_frame_h5(p)["attrs"].get("split")) == split)]
    with Pool(NP) as pool: R = pool.map(one, paths, chunksize=1)
    json.dump(dict(data=data, src=src, split=split, rows=R), open(out, "w"))
    f1 = lambda t: 2 * t[0] / max(2 * t[0] + t[1] + t[2], 1)
    al = np.concatenate([r[1] for r in R]) if R else np.zeros(0)
    print(len(R), "frames", len(al), "GT curves")
    for vk in R[0][2]:
        inst = np.sum([r[2][vk]["inst"] for r in R], 0)
        print("== %s  instancer F1 %.4f (tp/fp/fn %d/%d/%d)  arcs/frame %.0f" % (vk, f1(inst), *inst, np.mean([r[2][vk]["n_arcs"] for r in R])))
        for br in BRS:
            k = "br%g" % br; s = np.sum([r[2][vk][k]["s"] for r in R], 0); ml = np.concatenate([r[2][vk][k]["miss_len"] for r in R] + [np.zeros(0)])
            print("   BR %-4g oracle F1 %.4f P %.3f R %.3f (%d/%d/%d) [emit-only-matched %.4f] links/frame %.0f | miss by len: %s" % (
                br, f1(s), s[0] / max(s[0] + s[1], 1), s[0] / max(s[0] + s[2], 1), *s, f1((s[0], 0, s[2])), np.mean([r[2][vk][k]["n_links"] for r in R]),
                " ".join("%g-%g:%.2f" % (a, b, ((ml >= a) & (ml < b)).sum() / max(((al >= a) & (al < b)).sum(), 1)) for a, b in zip(LE[:-1], LE[1:]))))
