#!/usr/bin/env python3
"""ERROR ANALYSIS, report only (docs/TODO.md 2026-11-02 07:00). Classes = anat_l.py under L or L+H; plus why covered_split GT
stay split. usage: anat_lh.py DATA PRED_DIR CUT OUT.json L|H [htw|all] [NPROC]"""
import glob, json, os, sys
import numpy as np
from multiprocessing import Pool
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree
sys.path[:0] = [".", "src", "scripts", "scripts/lpc"]
from lpc_lib import P, UP, KAPPA, graph, owners, link_feats, to_logit, resample, rasterise_polyline
from l_dup import match, walk_all, arclen, MU, SD, WV, B0
from l_hyst2 import hmask
from instance.metrics import _polyline_pixels, centerline_f1
from mt_bench.cvat_import import read_frame_h5
from mt_bench.gt_stats import count_crossings
from step1_read import source_of
from diag_fp_anatomy import resample as rs1, FP_CLASSES, FN_CLASSES
data, pred, cut, out, DEC = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], sys.argv[5]
SRC = sys.argv[6] if len(sys.argv) > 6 else "all"; NP = int(sys.argv[7]) if len(sys.argv) > 7 else 16
TOL, COV, BORDER, XR = 5.0, 0.80, 8.0, 30.0

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
    mask = hmask(prob, cut, "bridge", 0.98, 22.5) if DEC == "H" else prob > cut
    arcs, _, arms = graph(mask); logit = to_logit(prob); polys, cid_of, kept = [], {}, set()
    partner = match(arms, {}, logit) if arcs else {}
    for cid, ch in enumerate(walk_all(arcs, partner) if arcs else []):
        for k, _w in ch: cid_of[k] = cid
        p = np.concatenate([arcs[k] if w == "start" else arcs[k][::-1] for k, w in ch], 0)
        if arclen(p) >= P["min_length"]: polys.append(p); kept.add(cid)
    masks = [rasterise_polyline(resample(p, ds=0.5), prob.shape) for p in polys]
    r = centerline_f1(masks, gt, tol=TOL, length_coverage=COV, precision_coverage=COV, end_tol=0.0)
    mp = {q["pred"] for q in r["pairs"] if q["tp"]}; mg = {q["gt"] for q in r["pairs"] if q["tp"]}
    gt_px = [_polyline_pixels(g, (H, W)) if len(g) >= 2 else np.zeros((0, 2), int) for g in gt]
    gt_trees = [cKDTree(g) if len(g) else None for g in gt_px]
    pr_yx = [rs1(np.asarray(p, float))[:, ::-1] for p in polys]
    allp = [c for c in pr_yx if len(c)]; pr_tree = cKDTree(np.concatenate(allp, 0)) if allp else None
    fp_cls = {c: 0 for c in FP_CLASSES}; fps = []
    for i, c in enumerate(pr_yx):
        if i in mp or len(c) < 2: continue
        n = len(c); near = np.zeros(n, bool); foot = {}
        for j, t in enumerate(gt_trees):
            if t is None: continue
            d = t.query(c)[0] <= TOL
            if d.sum() >= max(10, 0.15 * n):
                idx = np.where(d)[0]; foot[j] = (int(idx.min()), int(idx.max()), float(d.mean()), float((cKDTree(c).query(gt_px[j])[0] <= TOL).mean()))
            near |= d
        on_gt = float(near.mean()); edge = np.minimum(np.minimum(c[:, 0], H - 1 - c[:, 0]), np.minimum(c[:, 1], W - 1 - c[:, 1]))
        if on_gt < 0.5: cls = "off_border" if float((edge <= BORDER).mean()) >= 0.7 else "off_other"
        elif len(foot) <= 1:
            cv = max([v[3] for v in foot.values()], default=0.0); cls = "overshoot" if cv >= COV else ("fragment" if on_gt >= COV else "frag_overshoot")
        else:
            it = sorted(foot.items(), key=lambda kv: kv[1][0])
            cls = "gap_bridge" if (all(it[k][1][1] < it[k + 1][1][0] + 3 for k in range(len(it) - 1)) and sum(v[3] >= 0.5 for _, v in it) >= 2) else "junction"
        fp_cls[cls] += 1; fps.append((cls, n / UP))
    hi = maximum_filter(mask, size=5); gts = []; breaks = []; gtwhy = []
    owner, proj = owners(arcs, [g for g in gt]) if arcs and gt else ({}, {})
    cr = count_crossings(gt, tol=2.0 * UP, endpoint_margin=8.0 * UP) if len(gt) > 1 else []
    xy = np.array([[c["x"], c["y"]] for c in cr if c["i"] != c["j"]], float).reshape(-1, 2); xt = cKDTree(xy) if len(xy) else None
    for j, g in enumerate(gt_px):
        if not len(g): continue
        others = [gt_px[k] for k in range(len(gt_px)) if k != j and len(gt_px[k])]
        contact = float((cKDTree(np.concatenate(others, 0)).query(g)[0] <= TOL).mean()) if others else 0.0
        gi = g.astype(int); support = float(hi[gi[:, 0], gi[:, 1]].mean())
        if j in mg: cls = "tp"
        else:
            cv = float((pr_tree.query(g)[0] <= TOL).mean()) if pr_tree is not None else 0.0
            cls = "covered_split" if cv >= COV else ("partial" if cv >= 0.5 else "missed")
        gts.append((cls, glen(gt[j]) / UP, contact, support))
        if cls != "covered_split": continue
        mine = sorted((k for k, o in owner.items() if o == j), key=lambda k: (proj[k][0] + proj[k][1]) / 2)
        if not mine: gtwhy.append("no_arc"); continue
        cs = {cid_of[k] for k in mine}
        if len(cs) == 1:
            ch_arcs = [k for k, c in cid_of.items() if c in cs]; foreign = [k for k in ch_arcs if owner.get(k, j) != j]; off = [k for k in ch_arcs if k not in owner]
            gtwhy.append("one_chain_chimera" if foreign else ("one_chain_off" if off else "one_chain_other")); continue
        gtwhy.append("split"); ori = {k: proj[k][0] <= proj[k][1] for k in mine}
        for a0, b0 in zip(mine[:-1], mine[1:]):
            if cid_of[a0] == cid_of[b0]: continue
            ka, kb = (a0, "end" if ori[a0] else "start"), (b0, "start" if ori[b0] else "end"); a, b = arms[ka], arms[kb]
            fe, kimp = link_feats(a, b, logit); gap = float(fe[2]); s = float(((np.asarray(fe) - MU) / SD) @ WV + B0)
            at = bool(xt is not None and xt.query(0.5 * (a.pos + b.pos))[0] <= XR)
            if cid_of[a0] not in kept or cid_of[b0] not in kept: why = "short_dropped"
            elif gap > 60: why = "far"
            elif kimp > KAPPA: why = "curvature"
            elif s <= 0: why = "low_score"
            else:
                pa, pb = partner.get(ka), partner.get(kb); ps = [q for q in (pa, pb) if q is not None]
                if not ps: why = "other"
                elif any(owner.get(q[0], -1) not in (-1, j) for q in ps): why = "lost_to_other_gt"
                elif any(q[0] not in owner for q in ps): why = "lost_to_unowned"
                else: why = "lost_same_gt"
            breaks.append((why, at, gap, s))
    return name, {"tp": r["tp"], "fp": r["fp"], "fn": r["fn"], "fp_classes": fp_cls, "fps": fps, "gts": gts, "breaks": breaks, "gtwhy": gtwhy}

if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(data, "*.h5")))
    with Pool(NP) as p: rows = [x for x in p.map(one, files, chunksize=1) if x]
    t = {k: sum(r[k] for _, r in rows) for k in ("tp", "fp", "fn")}
    print(os.path.basename(pred.rstrip("/")), DEC, SRC, "cut", cut, "frames", len(rows), t, flush=True)
    json.dump({"data": data, "pred": pred, "cut": cut, "dec": DEC, "src": SRC, "rows": rows, "total": t}, open(out, "w"))
