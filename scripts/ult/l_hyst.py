#!/usr/bin/env python3
"""Crossing-local hysteresis before instancer L (docs/TODO.md 2026-11-01 01:45). Mask = (prob > cut) plus, inside a disc
of radius R around every junction of skeleton(prob > t_lo) — crossing candidates found without GT — the prob > t_lo pixels
whose 8-connected component touches the own-cut mask. Then instancer L unchanged (arc graph of that mask, frozen L-hand
link score on the map's logits, max-weight matching, bridge 60, t 0, min_length). R = 0 is instancer L exactly.
    l_hyst.py DATA PRED_DIR CUT OUT.json GRID [NPROC] [SPLIT] [htw|all]     GRID = "0:0,0.98:15,0.95:30" (t_lo:R pairs)"""
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
K8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

def hyst_mask(prob, cut, t_lo, R, stats=None):
    hi = prob > cut
    if R <= 0: return hi
    lo = prob > t_lo; sk = skeletonize(lo)
    J = sk & (ndimage.convolve(sk.astype(np.uint8), K8, mode="constant") >= 3)
    if not J.any(): return hi
    region = ndimage.distance_transform_edt(~J) <= R
    cand = hi | (lo & region)
    lab, _ = ndimage.label(cand, structure=np.ones((3, 3), int))
    keep = np.unique(lab[hi]); keep = keep[keep > 0]
    out = np.isin(lab, keep)
    if stats is not None: stats.update(added=int((out & ~hi).sum()), junctions=int(ndimage.label(J, structure=np.ones((3, 3), int))[1]))
    return out

def decode(prob, cut, t_lo, R, stats=None):
    mask = hyst_mask(prob, cut, t_lo, R, stats)
    arcs, _, arms = graph(mask)
    if not arcs: return []
    partner = match(arms, {}, to_logit(prob))
    polys = []
    for ch in walk_all(arcs, partner):
        p = np.concatenate([arcs[k] if w == "start" else arcs[k][::-1] for k, w in ch], 0)
        if arclen(p) >= P["min_length"]: polys.append(p)
    return [rasterise_polyline(resample(p, ds=0.5), prob.shape) for p in polys]

def one(args):
    path, pred, cut, grid, src = args
    name = os.path.basename(path)
    if src == "htw" and source_of(name) != "htw": return None
    fr = read_frame_h5(path); prob = np.load(os.path.join(pred, name.replace(".h5", ".npz")))["prob"].astype(np.float32)[0]
    gt = [np.asarray(q, float) * UP for q in fr["polylines"]]
    cr = count_crossings(gt, tol=2.0 * UP, endpoint_margin=8.0 * UP) if len(gt) > 1 else []; par = parallel_pairs(gt) if len(gt) > 1 else []
    res = {}
    for t_lo, R in grid:
        st = {}; res[f"h{t_lo:g}_{R:g}"] = dict(row(decode(prob, cut, t_lo, R, st), gt, cr, par), **st)
    return name, len(gt), res

if __name__ == "__main__":
    data, pred, cut, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
    grid = [tuple(float(v) for v in g.split(":")) for g in sys.argv[5].split(",")]
    NP = int(sys.argv[6]) if len(sys.argv) > 6 else 16; SPLIT = sys.argv[7] if len(sys.argv) > 7 else "val"; SRC = sys.argv[8] if len(sys.argv) > 8 else "all"
    paths = [p for p in sorted(glob.glob(os.path.join(data, "*.h5")))
             if os.path.exists(os.path.join(pred, os.path.basename(p).replace(".h5", ".npz"))) and (SPLIT == "all" or str(read_frame_h5(p)["attrs"].get("split")) == SPLIT)]
    with Pool(NP) as pool: rows = [r for r in pool.map(one, [(p, pred, cut, grid, SRC) for p in paths], chunksize=1) if r]
    json.dump(dict(data=data, pred=pred, cut=cut, rows=rows), open(out, "w"))
    f1 = lambda tp, fp, fn: 2 * tp / max(2 * tp + fp + fn, 1)
    for t_lo, R in grid:
        k = f"h{t_lo:g}_{R:g}"; s = {q: sum(r[2][k].get(q, 0) or 0 for r in rows) for q in ("tp", "fp", "fn", "jn", "jp", "npar", "bund", "added")}
        print(f"{os.path.basename(pred.rstrip('/'))} {k}: frames {len(rows)} F1 {f1(s['tp'], s['fp'], s['fn']):.4f} tp/fp/fn {s['tp']}/{s['fp']}/{s['fn']} "
              f"identity {s['jp'] / max(s['jn'], 1):.4f} ({s['jn']}) bundle {s['bund'] / max(s['npar'], 1):.4f} ({s['npar']}) added px {s['added']}", flush=True)
