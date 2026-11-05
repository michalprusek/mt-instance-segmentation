#!/usr/bin/env python3
"""DIAGNOSTIC, report only (docs/TODO.md 2026-10-31 19:45): failure anatomy of the current recipe (V14 + L-hand).
Decoding = decomp_aa.py / scripts/lpc/a_grid.py (L-hand, bridge 60, t 0); classes = scripts/diag_fp_anatomy.py, coverage
0.80 (declared metric). Per GT also: native length, contact share (fraction of its pixels within 5 eval px of another GT),
map support (fraction of its pixels with prob >= cut within 2 px).
    python anat_l.py DATA PRED_DIR CUT OUT.json [htw|all] [NPROC]"""
import glob, json, os, sys
import numpy as np
from multiprocessing import Pool
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree
sys.path[:0] = [".", "src", "scripts", "scripts/lpc"]
import lpc_lib as LL
from lpc_lib import UP, chain, link_edges
from instance.metrics import _polyline_pixels, centerline_f1
from mt_bench.cvat_import import read_frame_h5
from step1_read import source_of
from diag_fp_anatomy import resample, FP_CLASSES, FN_CLASSES
data, pred, cut, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
SRC = sys.argv[5] if len(sys.argv) > 5 else "all"; NP = int(sys.argv[6]) if len(sys.argv) > 6 else 16
MODEL = json.load(open("src/instance/params_l_hand.json"))
TOL, COV, BORDER = 5.0, 0.80, 8.0

def glen(q):
    q = np.asarray(q, float); return float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum()) if len(q) > 1 else 0.0

def one(path):
    name = os.path.basename(path); fr = read_frame_h5(path)
    if str(fr["attrs"].get("split", "")) != "val": return None
    if SRC == "htw" and source_of(name) != "htw": return None
    f = os.path.join(pred, name.replace(".h5", ".npz"))
    if not os.path.exists(f): return None
    prob = np.load(f)["prob"].astype(np.float32)[0]; H, W = prob.shape
    gt = [np.asarray(q, float) * UP for q in fr["polylines"]]
    arcs, edges = link_edges(prob, cut, MODEL, 60.0)
    polys = chain(arcs, edges, 60.0, 0.0, None) if arcs else []
    masks = [LL.rasterise_polyline(LL.resample(p, ds=0.5), prob.shape) for p in polys]
    r = centerline_f1(masks, gt, tol=TOL, length_coverage=COV, precision_coverage=COV, end_tol=0.0)
    mp = {q["pred"] for q in r["pairs"] if q["tp"]}; mg = {q["gt"] for q in r["pairs"] if q["tp"]}
    gt_px = [_polyline_pixels(g, (H, W)) if len(g) >= 2 else np.zeros((0, 2), int) for g in gt]
    gt_trees = [cKDTree(g) if len(g) else None for g in gt_px]
    pr_yx = [resample(np.asarray(p, float))[:, ::-1] for p in polys]
    allp = [c for c in pr_yx if len(c)]; pr_tree = cKDTree(np.concatenate(allp, 0)) if allp else None
    fp_cls = {c: 0 for c in FP_CLASSES}; fps = []
    for i, c in enumerate(pr_yx):                      # unmatched predictions, diag_fp_anatomy rules
        if i in mp or len(c) < 2: continue
        n = len(c); near = np.zeros(n, bool); foot = {}
        for j, t in enumerate(gt_trees):
            if t is None: continue
            d = t.query(c)[0] <= TOL
            if d.sum() >= max(10, 0.15 * n):
                idx = np.where(d)[0]; cov_j = float((cKDTree(c).query(gt_px[j])[0] <= TOL).mean())
                foot[j] = (int(idx.min()), int(idx.max()), float(d.mean()), cov_j)
            near |= d
        on_gt = float(near.mean())
        edge = np.minimum(np.minimum(c[:, 0], H - 1 - c[:, 0]), np.minimum(c[:, 1], W - 1 - c[:, 1]))
        if on_gt < 0.5: cls = "off_border" if float((edge <= BORDER).mean()) >= 0.7 else "off_other"
        elif len(foot) <= 1:
            cv = max([v[3] for v in foot.values()], default=0.0)
            cls = "overshoot" if cv >= COV else ("fragment" if on_gt >= COV else "frag_overshoot")
        else:
            it = sorted(foot.items(), key=lambda kv: kv[1][0])
            disjoint = all(it[k][1][1] < it[k + 1][1][0] + 3 for k in range(len(it) - 1))
            cls = "gap_bridge" if (disjoint and sum(v[3] >= 0.5 for _, v in it) >= 2) else "junction"
        fp_cls[cls] += 1; fps.append((cls, n / UP))
    hi = maximum_filter(prob, size=5) >= cut           # map support within 2 px
    gts = []
    for j, g in enumerate(gt_px):
        if not len(g): continue
        others = [gt_px[k] for k in range(len(gt_px)) if k != j and len(gt_px[k])]
        contact = float((cKDTree(np.concatenate(others, 0)).query(g)[0] <= TOL).mean()) if others else 0.0
        gi = g.astype(int); support = float(hi[gi[:, 0], gi[:, 1]].mean())   # pixels are (y, x), float32
        if j in mg: cls = "tp"
        else:
            cv = float((pr_tree.query(g)[0] <= TOL).mean()) if pr_tree is not None else 0.0
            cls = "covered_split" if cv >= COV else ("partial" if cv >= 0.5 else "missed")
        gts.append((cls, glen(gt[j]) / UP, contact, support))
    return name, {"tp": r["tp"], "fp": r["fp"], "fn": r["fn"], "fp_classes": fp_cls, "fps": fps, "gts": gts}

if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(data, "*.h5")))
    with Pool(NP) as p: rows = [x for x in p.map(one, files, chunksize=1) if x]
    t = {k: sum(r[k] for _, r in rows) for k in ("tp", "fp", "fn")}
    print(os.path.basename(pred.rstrip("/")), SRC, "cut", cut, "frames", len(rows), t, flush=True)
    json.dump({"data": data, "pred": pred, "cut": cut, "src": SRC, "rows": rows, "total": t}, open(out, "w"))
