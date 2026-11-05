#!/usr/bin/env python3
"""Crossing hysteresis v2 (docs/TODO.md 2026-11-01 07:55), two variants on top of the own-cut mask, then instancer L
unchanged. disc = l_hyst.py (t_lo pixels within R of junctions of skeleton(p > t_lo) whose component touches the own-cut
mask); bridge = only the t_lo pieces within R (R = inf: anywhere) that touch >= 2 DIFFERENT own-cut components (gap fills).
    l_hyst2.py DATA PRED_DIR CUT OUT.json GRID [NPROC] [SPLIT] [htw|all]
    GRID = "L,disc:0.95:22.5,bridge:0.9:inf"   (variant:t_lo:R)"""
import glob, json, os, sys
import numpy as np
from multiprocessing import Pool
from scipy import ndimage
from skimage.morphology import skeletonize
sys.path[:0] = [".", "src", "scripts", "scripts/lpc"]
from lpc_lib import P, UP, graph, to_logit, resample, rasterise_polyline
from l_dup import match, walk_all, row, parallel_pairs, arclen
from mt_bench.cvat_import import read_frame_h5
from mt_bench.gt_stats import count_crossings
from step1_read import source_of
K8, S8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]), np.ones((3, 3), int)

def region_of(lo, R):
    if np.isinf(R): return np.ones_like(lo)
    sk = skeletonize(lo); J = sk & (ndimage.convolve(sk.astype(np.uint8), K8, mode="constant") >= 3)
    return ndimage.distance_transform_edt(~J) <= R if J.any() else np.zeros_like(lo)

def hmask(prob, cut, var, t_lo, R):
    hi = prob > cut
    if var == "L": return hi
    lo = prob > t_lo; reg = region_of(lo, R)
    if var == "disc":
        lab, _ = ndimage.label(hi | (lo & reg), structure=S8); keep = np.unique(lab[hi]); keep = keep[keep > 0]
        return np.isin(lab, keep)
    add = lo & reg & ~hi                                             # bridge: pieces touching >= 2 own-cut components
    la, na = ndimage.label(add, structure=S8)
    if na == 0: return hi
    lh, _ = ndimage.label(hi, structure=S8); H, W = hi.shape
    pa, ph = np.pad(la, 1), np.pad(lh, 1); pairs = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == dx == 0: continue
            sh = ph[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]; m = (la > 0) & (sh > 0)
            pairs.append(np.stack([la[m], sh[m]], 1))
    pr = np.unique(np.concatenate(pairs, 0), axis=0) if pairs else np.zeros((0, 2), int)
    if not len(pr): return hi
    ids, cnt = np.unique(pr[:, 0], return_counts=True)
    return hi | np.isin(la, ids[cnt >= 2])

def decode(prob, cut, var, t_lo, R):
    mask = hmask(prob, cut, var, t_lo, R); arcs, _, arms = graph(mask)
    if not arcs: return [], int((mask & ~(prob > cut)).sum())
    partner = match(arms, {}, to_logit(prob)); polys = []
    for ch in walk_all(arcs, partner):
        p = np.concatenate([arcs[k] if w == "start" else arcs[k][::-1] for k, w in ch], 0)
        if arclen(p) >= P["min_length"]: polys.append(p)
    return [rasterise_polyline(resample(p, ds=0.5), prob.shape) for p in polys], int((mask & ~(prob > cut)).sum())

def parse(g):
    if g == "L": return ("L", 0.0, 0.0)
    v, t, r = g.split(":"); return (v, float(t), float("inf") if r == "inf" else float(r))

def key(s): return "L" if s[0] == "L" else f"{s[0]}{s[1]:g}_{s[2]:g}"

def one(args):
    path, pred, cut, grid, src = args
    name = os.path.basename(path)
    if src == "htw" and source_of(name) != "htw": return None
    fr = read_frame_h5(path); prob = np.load(os.path.join(pred, name.replace(".h5", ".npz")))["prob"].astype(np.float32)[0]
    gt = [np.asarray(q, float) * UP for q in fr["polylines"]]
    cr = count_crossings(gt, tol=2.0 * UP, endpoint_margin=8.0 * UP) if len(gt) > 1 else []; par = parallel_pairs(gt) if len(gt) > 1 else []
    res = {}
    for s in grid:
        masks, added = decode(prob, cut, *s); res[key(s)] = dict(row(masks, gt, cr, par), added=added)
    return name, len(gt), res

if __name__ == "__main__":
    data, pred, cut, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
    grid = [parse(g) for g in sys.argv[5].split(",")]
    NP = int(sys.argv[6]) if len(sys.argv) > 6 else 16; SPLIT = sys.argv[7] if len(sys.argv) > 7 else "val"; SRC = sys.argv[8] if len(sys.argv) > 8 else "all"
    paths = [p for p in sorted(glob.glob(os.path.join(data, "*.h5")))
             if os.path.exists(os.path.join(pred, os.path.basename(p).replace(".h5", ".npz"))) and (SPLIT == "all" or str(read_frame_h5(p)["attrs"].get("split")) == SPLIT)]
    with Pool(NP) as pool: rows = [r for r in pool.map(one, [(p, pred, cut, grid, SRC) for p in paths], chunksize=1) if r]
    json.dump(dict(data=data, pred=pred, cut=cut, grid=[key(s) for s in grid], rows=rows), open(out, "w"))
    f1 = lambda tp, fp, fn: 2 * tp / max(2 * tp + fp + fn, 1)
    for s in grid:
        k = key(s); t = {q: sum(r[2][k].get(q, 0) or 0 for r in rows) for q in ("tp", "fp", "fn", "jn", "jp", "added")}
        print(f"{os.path.basename(pred.rstrip('/'))} {k}: F1 {f1(t['tp'], t['fp'], t['fn']):.4f} tp/fp/fn {t['tp']}/{t['fp']}/{t['fn']} identity {t['jp'] / max(t['jn'], 1):.4f} added {t['added']}", flush=True)
