#!/usr/bin/env python3
"""D1 (declared 2026-10-05 in docs/TODO.md): the w_theta multiplier grid on a cached native dump. The mask is
prob > thr in every variant (bit-identical map); only params['w_theta'] differs. Works on synth (exact GT) and real VAL.

    python relink_grid.py DATA PRED_DIR THR OUT.json [NPROC] [SPLIT] [GRID]
"""
import glob, json, os, sys, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from instance.instancer_a import instance_a
from instance.metrics import centerline_f1, junction_identity, bundle_recovery
from instance.geometry import resample
from mt_bench.cvat_import import read_frame_h5
from mt_bench.gt_stats import count_crossings
UP, KAPPA = 1.5, 0.25
P = {k: v for k, v in json.load(open("src/instance/params_a_derived.json")).items() if not k.startswith("_")}
data, pred, thr, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
NP = int(sys.argv[5]) if len(sys.argv) > 5 else 16
SPLIT = sys.argv[6] if len(sys.argv) > 6 else "val"
GRID = [float(x) for x in (sys.argv[7] if len(sys.argv) > 7 else "1,1.25,1.5,1.75,2,2.25,2.5,3").split(",")]


def parallel_pairs(polylines, gap_lo=2.0 * UP, gap_hi=6.0 * UP, min_len=20.0 * UP, ds=2.0):
    from scipy.spatial import cKDTree
    res = [resample(np.asarray(p, float), ds=ds) for p in polylines]
    trees = [cKDTree(p) if len(p) > 1 else None for p in res]; pairs = []
    for i in range(len(res)):
        if trees[i] is None: continue
        for j in range(i + 1, len(res)):
            if trees[j] is None: continue
            d, _ = trees[j].query(res[i], k=1)
            if float(np.sum((d >= gap_lo) & (d <= gap_hi)) * ds) >= min_len: pairs.append((i, j))
    return pairs


def one(path):
    name = os.path.basename(path); fr = read_frame_h5(path)
    prob = np.load(os.path.join(pred, name.replace(".h5", ".npz")))["prob"].astype(np.float32)[0]
    mask = prob > thr
    gt = [np.asarray(q, float) * UP for q in fr["polylines"]]
    crossings = count_crossings(gt, tol=2.0 * UP, endpoint_margin=8.0 * UP) if len(gt) > 1 else []
    par = parallel_pairs(gt) if len(gt) > 1 else []
    res = {}
    for m in GRID:
        _, masks = instance_a(mask, KAPPA, params={**P, "w_theta": P["w_theta"] * m}, channels=prob[None], prob=prob)
        r = centerline_f1(masks, gt, tol=5.0, length_coverage=0.8, precision_coverage=0.8, end_tol=0.0)
        ji = junction_identity(masks, gt, crossings, tol=5.0, arm_offset=15.0 * UP) if crossings else {"n_crossings": 0, "n_preserved": 0}
        br = bundle_recovery(masks, gt, par, tol=1.5 * UP) if par else float("nan")
        res["x%g" % m] = dict(tp=r["tp"], fp=r["fp"], fn=r["fn"], jn=ji["n_crossings"], jp=ji["n_preserved"],
                              npar=len(par), bund=(None if br != br else br * len(par)), n_pred=len(masks))
    return name, len(gt), str(fr["attrs"].get("source_task", "")), res


if __name__ == "__main__":
    paths = [p for p in sorted(glob.glob(os.path.join(data, "*.h5")))
             if os.path.exists(os.path.join(pred, os.path.basename(p).replace(".h5", ".npz")))
             and str(read_frame_h5(p)["attrs"].get("split")) == SPLIT]
    t0 = time.time()
    with Pool(NP) as pool:
        rows = pool.map(one, paths, chunksize=1)
    json.dump(dict(data=data, pred=pred, thr=thr, split=SPLIT, rows=rows), open(out, "w"))
    print(len(rows), "frames", round(time.time() - t0), "s", flush=True)
