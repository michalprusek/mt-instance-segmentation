#!/usr/bin/env python3
"""Instancer L with SHARED ARCS at crossings ("L-dup", docs/TODO.md 2026-10-31). At a shallow crossing two filaments merge
into one skeleton stretch between two junctions; instancer L can give that stretch to one chain only. L-dup gives every arc
whose BOTH ends sit on (different) junctions and whose length is <= LDUP a second copy with its own two ends, scored and
matched exactly like any other arc end (same candidate links, frozen L-hand link score, max-weight matching, bridge 60,
t 0). Pass 1 matches with all copies; a copy is ACCEPTED only if both copies of the arc are traversed through (linked at
both ends, i.e. two chains run through the stretch). Pass 2 re-matches with the accepted copies only, so wherever no
sharing is accepted the decode is instancer L's. A copy left at a chain end in pass 2 is trimmed. LDUP = 0 is L exactly.
    l_dup.py DATA PRED_DIR CUT OUT.json LDUPS [NPROC] [SPLIT] [htw|all]"""
import dataclasses, glob, json, os, sys
import numpy as np
import networkx as nx
from multiprocessing import Pool
from scipy.spatial import cKDTree
sys.path[:0] = [".", "src", "scripts", "scripts/lpc"]
from lpc_lib import P, UP, KAPPA, graph, link_feats, to_logit, resample, rasterise_polyline
from instance.metrics import centerline_f1, junction_identity, bundle_recovery
from mt_bench.cvat_import import read_frame_h5
from mt_bench.gt_stats import count_crossings
from step1_read import source_of
MODEL = json.load(open("src/instance/params_l_hand.json")); BR, T = 60.0, 0.0
MU, SD, WV, B0 = (np.asarray(MODEL[k], float) for k in ("mu", "sd", "coef", "intercept"))

def parallel_pairs(polylines, gap_lo=2.0 * UP, gap_hi=6.0 * UP, min_len=20.0 * UP, ds=2.0):
    """= scripts/lpc/a_grid.py (copied: a_grid parses sys.argv at import)."""
    res = [resample(np.asarray(p, float), ds=ds) for p in polylines]; trees = [cKDTree(p) if len(p) > 1 else None for p in res]; pairs = []
    for i in range(len(res)):
        if trees[i] is None: continue
        for j in range(i + 1, len(res)):
            if trees[j] is None: continue
            d, _ = trees[j].query(res[i], k=1)
            if float(np.sum((d >= gap_lo) & (d <= gap_hi)) * ds) >= min_len: pairs.append((i, j))
    return pairs

def row(masks, gt, cr, par):
    """= scripts/lpc/a_grid.py row()."""
    r = centerline_f1(masks, gt, tol=5.0, length_coverage=0.8, precision_coverage=0.8, end_tol=0.0)
    ji = junction_identity(masks, gt, cr, tol=5.0, arm_offset=15.0 * UP) if cr else {"n_crossings": 0, "n_preserved": 0}
    br = bundle_recovery(masks, gt, par, tol=1.5 * UP) if par else float("nan")
    return dict(tp=r["tp"], fp=r["fp"], fn=r["fn"], jn=ji["n_crossings"], jp=ji["n_preserved"], npar=len(par), bund=(None if br != br else br * len(par)), n_pred=len(masks))

def arclen(a): return float(np.linalg.norm(np.diff(a, axis=0), axis=1).sum())

def match(arms, twin, logit):
    """= lpc_lib.link_edges + chain's matching, over the given arm ends; no link between the two copies of an arc."""
    keys = list(arms); tree = cKDTree(np.array([arms[q].pos for q in keys])); g = nx.Graph()
    for i, j in sorted(tree.query_pairs(BR)):
        a, b = keys[i][0], keys[j][0]
        if a == b or twin.get(a) == b: continue
        f, kimp = link_feats(arms[keys[i]], arms[keys[j]], logit)
        if kimp > KAPPA or f[2] > BR: continue
        s = float(((np.asarray(f) - MU) / SD) @ WV + B0)
        if s - T > 0: g.add_edge(keys[i], keys[j], weight=s - T)
    partner = {}
    for a, b in (nx.max_weight_matching(g) if g.number_of_edges() else ()): partner[a] = b; partner[b] = a
    return partner

def walk_all(arcs, partner):
    """= lpc_lib.chain's walk, returning each chain as [(arc, entered-at)]."""
    used, chains = set(), []
    def walk(k, w_in):
        out = []
        while k not in used:
            used.add(k); out.append((k, w_in))
            nx_ = partner.get((k, "end" if w_in == "start" else "start"))
            if nx_ is None: break
            k, w_in = nx_
        return out
    for k in arcs:
        for w in ("start", "end"):
            if (k, w) not in partner and k not in used: chains.append(walk(k, w))
    for k in arcs:
        if k not in used: chains.append(walk(k, "start"))
    return chains

def decode(prob, cut, ldup, stats=None):
    arcs, arc_ends, arms = graph(prob > cut)
    if not arcs: return []
    logit = to_logit(prob); cand = {}
    if ldup > 0:
        nxt = max(arcs) + 1
        for k in list(arcs):
            js, je = arc_ends[k]
            if js is None or je is None or js == je or arclen(arcs[k]) > ldup: continue
            cand[k] = nxt; nxt += 1
    def with_copies(sel):
        A, M, tw = dict(arcs), dict(arms), {}
        for k in sel:
            c = cand[k]; A[c] = arcs[k].copy(); tw[c] = k; tw[k] = c
            for w in ("start", "end"): M[(c, w)] = dataclasses.replace(arms[(k, w)], arc_idx=c)
        return A, M, tw
    acc = []
    if cand:
        A, M, tw = with_copies(cand); p1 = match(M, tw, logit)
        through = lambda k: (k, "start") in p1 and (k, "end") in p1
        acc = [k for k in cand if through(k) and through(cand[k])]
    A, M, tw = with_copies(acc); partner = match(M, tw, logit)
    if stats is not None: stats.update(cand=len(cand), acc=len(acc))
    polys = []
    for ch in walk_all(A, partner):
        while ch and ch[0][0] in tw and ch[0][0] > tw[ch[0][0]]: ch = ch[1:]      # a copy at a chain end is not shared
        while ch and ch[-1][0] in tw and ch[-1][0] > tw[ch[-1][0]]: ch = ch[:-1]
        if not ch: continue
        p = np.concatenate([A[k] if w == "start" else A[k][::-1] for k, w in ch], 0)
        if arclen(p) >= P["min_length"]: polys.append(p)
    return [rasterise_polyline(resample(p, ds=0.5), prob.shape) for p in polys]

def one(args):
    path, pred, cut, ldups, src = args
    name = os.path.basename(path)
    if src == "htw" and source_of(name) != "htw": return None
    fr = read_frame_h5(path); prob = np.load(os.path.join(pred, name.replace(".h5", ".npz")))["prob"].astype(np.float32)[0]
    gt = [np.asarray(q, float) * UP for q in fr["polylines"]]
    cr = count_crossings(gt, tol=2.0 * UP, endpoint_margin=8.0 * UP) if len(gt) > 1 else []; par = parallel_pairs(gt) if len(gt) > 1 else []
    res = {}
    for ld in ldups:
        st = {}; res[f"dup{ld:g}"] = dict(row(decode(prob, cut, ld, st), gt, cr, par), **st)
    return name, len(gt), res

if __name__ == "__main__":
    data, pred, cut, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
    ldups = [float(x) for x in sys.argv[5].split(",")]; NP = int(sys.argv[6]) if len(sys.argv) > 6 else 16
    SPLIT = sys.argv[7] if len(sys.argv) > 7 else "val"; SRC = sys.argv[8] if len(sys.argv) > 8 else "all"
    paths = [p for p in sorted(glob.glob(os.path.join(data, "*.h5")))
             if os.path.exists(os.path.join(pred, os.path.basename(p).replace(".h5", ".npz"))) and (SPLIT == "all" or str(read_frame_h5(p)["attrs"].get("split")) == SPLIT)]
    with Pool(NP) as pool: rows = [r for r in pool.map(one, [(p, pred, cut, ldups, SRC) for p in paths], chunksize=1) if r]
    json.dump(dict(data=data, pred=pred, cut=cut, rows=rows), open(out, "w"))
    f1 = lambda tp, fp, fn: 2 * tp / max(2 * tp + fp + fn, 1)
    for ld in ldups:
        k = f"dup{ld:g}"; s = {q: sum(r[2][k][q] for r in rows) for q in ("tp", "fp", "fn", "jn", "jp", "cand", "acc")}
        print(f"{os.path.basename(pred.rstrip('/'))} {k}: frames {len(rows)} F1 {f1(s['tp'], s['fp'], s['fn']):.4f} tp/fp/fn {s['tp']}/{s['fp']}/{s['fn']} identity {s['jp'] / max(s['jn'], 1):.4f} ({s['jn']}) shared cand {s['cand']} accepted {s['acc']}", flush=True)
