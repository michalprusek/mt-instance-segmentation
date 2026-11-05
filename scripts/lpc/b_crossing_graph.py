#!/usr/bin/env python3
"""Diagnostic (b), declared docs/TODO.md 2026-10-06 10:10: what does the arc graph look like at each GT crossing, and
does identity survive under instancer A and under the S0 oracle (bridge 60)?

    b_crossing_graph.py DATA SRC OUT.json [NPROC]     SRC = dump dir (cut 0.98) or BAND (perfect band hw 2.0)
"""
import glob, json, os, sys
import numpy as np
from collections import Counter, defaultdict
from multiprocessing import Pool
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
sys.path[:0] = [".", "src", "scripts", "scripts/lpc"]
from lpc_lib import P, UP, KAPPA, graph, owners, oracle_masks
from instance.geometry import resample
from instance.instancer_a import instance_a
from instance.metrics import junction_identity
from mt_bench.cvat_import import read_frame_h5
from mt_bench.gt_stats import count_crossings
data, src, out = sys.argv[1:4]; NP = int(sys.argv[4]) if len(sys.argv) > 4 else 12
ARM, NEAR = 15.0 * UP, 6.0


def one(path):
    name = os.path.basename(path); fr = read_frame_h5(path); H, W = np.asarray(fr["image"]).shape
    h, w = int(round(H * UP)), int(round(W * UP)); gt = [np.asarray(q, float) * UP for q in fr["polylines"] if len(q) >= 2]
    if len(gt) < 2: return name, []
    if src == "BAND":
        line = np.zeros((h, w), bool)
        for q in gt:
            p = resample(q, ds=0.25); ok = (p[:, 0] >= 0) & (p[:, 0] <= w - 1) & (p[:, 1] >= 0) & (p[:, 1] <= h - 1)
            line[np.round(p[ok, 1]).astype(int), np.round(p[ok, 0]).astype(int)] = True
        mask = distance_transform_edt(~line) <= 2.0; prob = mask.astype(np.float32)
    else:
        prob = np.load(os.path.join(src, name.replace(".h5", ".npz")))["prob"].astype(np.float32)[0]; mask = prob > 0.98
    cr = count_crossings(gt, tol=2.0 * UP, endpoint_margin=8.0 * UP)
    if not cr: return name, []
    arcs, arc_ends, arms = graph(mask); owner, _ = owners(arcs, gt)
    _, mA = instance_a(mask, KAPPA, params=P, channels=prob[None], prob=prob)
    mO = oracle_masks(arcs, arms, gt, 60.0, mask.shape) if arcs else []
    dmask = distance_transform_edt(~mask); res = [resample(q, ds=1.0) for q in gt]
    atree = {k: cKDTree(a) for k, a in arcs.items()}; rows = []
    for c in cr:
        if c["i"] == c["j"]: continue
        idA = junction_identity(mA, gt, [c], tol=5.0, arm_offset=ARM)
        if idA["n_crossings"] == 0: continue
        idO = junction_identity(mO, gt, [c], tol=5.0, arm_offset=ARM); xy = np.array([c["x"], c["y"]], float)
        broken = False; arm_arc = {}
        for ci, s in ((c["i"], c["s_i"]), (c["j"], c["s_j"])):
            pts = res[ci]
            for side in (-1, 1):
                lo, hi = sorted((int(round(s + side * NEAR)), int(round(s + side * ARM))))
                st = pts[max(lo, 0):min(hi, len(pts) - 1) + 1]
                yi = np.clip(np.round(st[:, 1]).astype(int), 0, h - 1); xi = np.clip(np.round(st[:, 0]).astype(int), 0, w - 1)
                if float((dmask[yi, xi] <= 2.0).mean()) < 0.8: broken = True
                best, bn = None, 0
                for k, o in owner.items():
                    if o != ci: continue
                    n = int((atree[k].query(st, k=1)[0] <= 3.0).sum())
                    if n > bn: best, bn = k, n
                arm_arc[(ci, side)] = best
        if broken: cls = "BROKEN"
        elif any(v is None for v in arm_arc.values()): cls = "UNOWNED"
        elif arm_arc[(c["i"], -1)] == arm_arc[(c["i"], 1)] or arm_arc[(c["j"], -1)] == arm_arc[(c["j"], 1)]: cls = "THROUGH"
        else:
            jids = []
            for k in arm_arc.values():
                e = 0 if np.linalg.norm(arcs[k][0] - xy) <= np.linalg.norm(arcs[k][-1] - xy) else 1
                jids.append(arc_ends[k][e])
            cls = "FREE" if any(j is None for j in jids) else ("SPLIT" if len(set(jids)) > 1 else "CLEAN")
        rows.append((cls, int(idA["n_preserved"]), int(idO["n_preserved"]), float(c.get("angle_deg", 90.0))))
    return name, rows


if __name__ == "__main__":
    paths = sorted(glob.glob(os.path.join(data, "*.h5")))
    if src != "BAND": paths = [p for p in paths if os.path.exists(os.path.join(src, os.path.basename(p).replace(".h5", ".npz")))]
    with Pool(NP) as pool: R = pool.map(one, paths, chunksize=1)
    json.dump(R, open(out, "w")); rows = [r for _, rs in R for r in rs]; n = len(rows)
    failA = sum(1 for r in rows if not r[1])
    print("%d crossings | identity A %.4f | oracle %.4f | A failures %d" % (n, np.mean([r[1] for r in rows]), np.mean([r[2] for r in rows]), failA))
    print("class      n   share  idA    idOracle  share of A-failures  A-fail&oracle-ok  mean angle")
    for cls in ("BROKEN", "UNOWNED", "THROUGH", "FREE", "SPLIT", "CLEAN"):
        rs = [r for r in rows if r[0] == cls]
        if not rs: print("%-9s 0" % cls); continue
        fa = [r for r in rs if not r[1]]
        print("%-9s %4d  %.3f  %.3f  %.3f     %.3f                %4d              %.0f" % (cls, len(rs), len(rs) / n, np.mean([r[1] for r in rs]), np.mean([r[2] for r in rs]),
              len(fa) / max(failA, 1), sum(1 for r in fa if r[2]), np.mean([r[3] for r in rs])))
    print("A failures recoverable by the oracle: %d of %d (%.3f)" % (sum(1 for r in rows if not r[1] and r[2]), failA, sum(1 for r in rows if not r[1] and r[2]) / max(failA, 1)))
