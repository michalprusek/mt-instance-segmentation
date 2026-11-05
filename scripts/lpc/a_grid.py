#!/usr/bin/env python3
"""Instancer L on a cached dump over a (bridge, t) grid, scored like relink_grid.py (declared metric, junction identity,
bundle recovery). Instancer A on the same mask is computed in the same run as the reference.

    a_grid.py DATA PRED_DIR CUT OUT.json MODEL.json BRS TS [NPROC] [SPLIT]
"""
import glob, json, os, sys
import numpy as np
from multiprocessing import Pool
sys.path[:0] = [".", "src", "scripts", "scripts/lpc"]
from lpc_lib import P, UP, KAPPA, link_edges, chain
from instance.geometry import resample
from instance.instancer_a import instance_a
from instance.metrics import centerline_f1, junction_identity, bundle_recovery
from mt_bench.cvat_import import read_frame_h5
from mt_bench.gt_stats import count_crossings
data, pred, cut, out, model = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], json.load(open(sys.argv[5]))
BRS = [float(x) for x in sys.argv[6].split(",")]; TS = [float(x) for x in sys.argv[7].split(",")]
SKIP_A = bool(int(os.environ.get("LPC_SKIP_A", "0")))
NP = int(sys.argv[8]) if len(sys.argv) > 8 else 12; SPLIT = sys.argv[9] if len(sys.argv) > 9 else "val"


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


def one(path):
    name = os.path.basename(path); fr = read_frame_h5(path)
    prob = np.load(os.path.join(pred, name.replace(".h5", ".npz")))["prob"].astype(np.float32)[0]; mask = prob > cut
    gt = [np.asarray(q, float) * UP for q in fr["polylines"]]
    cr = count_crossings(gt, tol=2.0 * UP, endpoint_margin=8.0 * UP) if len(gt) > 1 else []; par = parallel_pairs(gt) if len(gt) > 1 else []
    res = {}
    if not SKIP_A:
        _, mA = instance_a(mask, KAPPA, params=P, channels=prob[None], prob=prob); res["A"] = row(mA, gt, cr, par)
    arcs, edges = link_edges(prob, cut, model, max(BRS))
    for br in BRS:
        for t in TS:
            res["b%g_t%g" % (br, t)] = row(chain(arcs, edges, br, t, mask.shape) if arcs else [], gt, cr, par)
    return name, len(gt), res


if __name__ == "__main__":
    paths = [p for p in sorted(glob.glob(os.path.join(data, "*.h5")))
             if os.path.exists(os.path.join(pred, os.path.basename(p).replace(".h5", ".npz"))) and (SPLIT == "all" or str(read_frame_h5(p)["attrs"].get("split")) == SPLIT)]
    with Pool(NP) as pool: rows = pool.map(one, paths, chunksize=1)
    json.dump(dict(data=data, pred=pred, cut=cut, rows=rows), open(out, "w")); print(len(rows), "frames", flush=True)
