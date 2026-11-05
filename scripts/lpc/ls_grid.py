#!/usr/bin/env python3
"""Instancer L with several link scorers on the SAME arc graph and the SAME candidate edges (declared docs/TODO.md
2026-10-06 14:10). Phase 1 (CPU pool): arcs, candidate edges, hand features and map-logit strips per frame. Main process:
strip CNN on the GPU, hand0 control. Phase 2 (CPU pool): matching + chaining + the declared metric per (scorer, bridge, t).

    ls_grid.py DATA PRED_DIR CUT OUT.json WEIGHT_DIR SPEC [NPROC] [SPLIT]
    SPEC = "H:60:0;S:24,36,60:-2,-1,0,1,2,3;Z:60:0"   (H = L-hand, S = L-strip, Z = hand refitted from gap 0; bridges; t values)
Rows carry keys like "S_b60_t0"; env LPC_WITH_A=1 adds instancer A ("A").
"""
import glob, json, os, sys
import numpy as np
from multiprocessing import Pool
sys.path[:0] = [".", "src", "scripts", "scripts/lpc"]
from lpc_lib import P, UP, KAPPA, link_edges, chain, strip_net, strip_scores
from instance.geometry import resample
from instance.instancer_a import instance_a
from instance.metrics import centerline_f1, junction_identity, bundle_recovery
from mt_bench.cvat_import import read_frame_h5
from mt_bench.gt_stats import count_crossings
data, pred, cut, out, wdir = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], sys.argv[5]
SPEC = {a: ([float(x) for x in b.split(",")], [float(x) for x in t.split(",")]) for a, b, t in (q.split(":") for q in sys.argv[6].split(";"))}
NP = int(sys.argv[7]) if len(sys.argv) > 7 else 12; SPLIT = sys.argv[8] if len(sys.argv) > 8 else "val"
WITH_A = bool(int(os.environ.get("LPC_WITH_A", "0"))); BRMAX = max(max(b) for b, _ in SPEC.values())
HAND = json.load(open("src/instance/params_l_hand.json"))


def parallel_pairs(polylines, gap_lo=2.0 * UP, gap_hi=6.0 * UP, min_len=20.0 * UP, ds=2.0):
    from scipy.spatial import cKDTree
    res = [resample(np.asarray(p, float), ds=ds) for p in polylines]; trees = [cKDTree(p) if len(p) > 1 else None for p in res]; pairs = []
    for i in range(len(res)):
        if trees[i] is None: continue
        for j in range(i + 1, len(res)):
            if trees[j] is None: continue
            d, _ = trees[j].query(res[i], k=1)
            if float(np.sum((d >= gap_lo) & (d <= gap_hi)) * ds) >= min_len: pairs.append((i, j))
    return pairs


def row(masks, gt, cr, par):
    r = centerline_f1(masks, gt, tol=5.0, length_coverage=0.8, precision_coverage=0.8, end_tol=0.0)
    ji = junction_identity(masks, gt, cr, tol=5.0, arm_offset=15.0 * UP) if cr else {"n_crossings": 0, "n_preserved": 0}
    br = bundle_recovery(masks, gt, par, tol=1.5 * UP) if par else float("nan")
    return dict(tp=r["tp"], fp=r["fp"], fn=r["fn"], jn=ji["n_crossings"], jp=ji["n_preserved"], npar=len(par), bund=(None if br != br else br * len(par)), n_pred=len(masks))


def load(path):
    name = os.path.basename(path)
    return name, read_frame_h5(path), np.load(os.path.join(pred, name.replace(".h5", ".npz")))["prob"].astype(np.float32)[0]


def phase1(path):
    name, _, prob = load(path); arcs, edges, (F, S, N) = link_edges(prob, cut, HAND, BRMAX, strips=True)
    return path, arcs, edges, F, S, N


def phase2(job):
    path, arcs, edges, scores = job; name, fr, prob = load(path); shape = prob.shape
    gt = [np.asarray(q, float) * UP for q in fr["polylines"]]
    cr = count_crossings(gt, tol=2.0 * UP, endpoint_margin=8.0 * UP) if len(gt) > 1 else []; par = parallel_pairs(gt) if len(gt) > 1 else []
    res = {}
    if WITH_A:
        _, mA = instance_a(prob > cut, KAPPA, params=P, channels=prob[None], prob=prob); res["A"] = row(mA, gt, cr, par)
    for tag, (brs, ts) in SPEC.items():
        ed = [(a, b, g, float(s)) for (a, b, g, _), s in zip(edges, scores[tag])]
        for br in brs:
            for t in ts: res["%s_b%g_t%g" % (tag, br, t)] = row(chain(arcs, ed, br, t, shape) if arcs else [], gt, cr, par)
    return name, len(gt), res


if __name__ == "__main__":
    paths = [p for p in sorted(glob.glob(os.path.join(data, "*.h5")))
             if os.path.exists(os.path.join(pred, os.path.basename(p).replace(".h5", ".npz"))) and (SPLIT == "all" or str(read_frame_h5(p)["attrs"].get("split")) == SPLIT)]
    with Pool(NP) as pool:                                   # forked before CUDA is touched
        nets, dev, z = None, None, None
        if "S" in SPEC:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"; nets = []
            for s in (0, 1):
                n = strip_net(); n.load_state_dict(torch.load(os.path.join(wdir, f"l_strip_s{s}.pt"), map_location="cpu")); nets.append(n.to(dev).eval())
        if "Z" in SPEC: z = json.load(open(os.path.join(wdir, "params_l_hand0.json")))
        jobs, n_e = [], 0
        for path, arcs, edges, F, S, N in pool.imap(phase1, paths, chunksize=1):
            sc = {"H": np.array([e[3] for e in edges], np.float32)}
            if nets: sc["S"] = strip_scores(nets, S, N, dev)
            if z: sc["Z"] = (((F.astype(float) - np.asarray(z["mu"])) / np.asarray(z["sd"])) @ np.asarray(z["coef"]) + z["intercept"]) if len(F) else np.zeros(0)
            jobs.append((path, arcs, edges, {k: v for k, v in sc.items() if k in SPEC})); n_e += len(edges)
        rows = pool.map(phase2, jobs, chunksize=1)
    json.dump(dict(data=data, pred=pred, cut=cut, spec=sys.argv[6], rows=rows), open(out, "w")); print(len(rows), "frames", n_e, "edges", flush=True)
