#!/usr/bin/env python3
"""Step (2a), docs/TODO.md 2026-10-31 20:00: scripts/lpc/b_crossing_graph.py with instancer L-hand at the model's own cut
in place of instancer A / the S0 oracle. Classes unchanged (BROKEN / UNOWNED / THROUGH / FREE / SPLIT / CLEAN, same order
and constants); identity = instance.metrics.junction_identity on L's masks (bridge 60, t 0).
    b_crossing_l.py DATA PRED_DIR CUT OUT.json [htw|all] [NPROC]"""
import glob, json, os, sys
import numpy as np
from multiprocessing import Pool
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
sys.path[:0] = [".", "src", "scripts", "scripts/lpc"]
from lpc_lib import UP, graph, owners, link_edges, chain
from instance.geometry import resample
from instance.metrics import junction_identity
from mt_bench.cvat_import import read_frame_h5
from mt_bench.gt_stats import count_crossings
from step1_read import source_of
data, src, cut, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
SRC = sys.argv[5] if len(sys.argv) > 5 else "all"; NP = int(sys.argv[6]) if len(sys.argv) > 6 else 16
ARM, NEAR = 15.0 * UP, 6.0
MODEL = json.load(open("src/instance/params_l_hand.json"))

def one(path):
    name = os.path.basename(path)
    if SRC == "htw" and source_of(name) != "htw": return name, []
    fr = read_frame_h5(path)
    if str(fr["attrs"].get("split", "")) != "val": return name, []
    gt = [np.asarray(q, float) * UP for q in fr["polylines"] if len(q) >= 2]
    prob = np.load(os.path.join(src, name.replace(".h5", ".npz")))["prob"].astype(np.float32)[0]; mask = prob > cut
    h, w = mask.shape
    cr = count_crossings(gt, tol=2.0 * UP, endpoint_margin=8.0 * UP)
    if not cr: return name, []
    arcs, arc_ends, arms = graph(mask); owner, _ = owners(arcs, gt)
    la, le = link_edges(prob, cut, MODEL, 60.0); mL = chain(la, le, 60.0, 0.0, mask.shape) if la else []
    dmask = distance_transform_edt(~mask); res = [resample(q, ds=1.0) for q in gt]
    atree = {k: cKDTree(a) for k, a in arcs.items()}; rows = []
    for c in cr:
        if c["i"] == c["j"]: continue
        idL = junction_identity(mL, gt, [c], tol=5.0, arm_offset=ARM)
        if idL["n_crossings"] == 0: continue
        xy = np.array([c["x"], c["y"]], float); broken = False; arm_arc = {}
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
        rows.append((cls, int(idL["n_preserved"]), float(c.get("angle_deg", 90.0))))
    return name, rows

if __name__ == "__main__":
    paths = [p for p in sorted(glob.glob(os.path.join(data, "*.h5"))) if os.path.exists(os.path.join(src, os.path.basename(p).replace(".h5", ".npz")))]
    with Pool(NP) as pool: R = pool.map(one, paths, chunksize=1)
    json.dump(R, open(out, "w")); rows = [r for _, rs in R for r in rs]; n = len(rows); fail = sum(1 for r in rows if not r[1])
    print("%s %s cut %.3f | %d crossings | identity L %.4f | L failures %d" % (os.path.basename(src.rstrip("/")), SRC, cut, n, np.mean([r[1] for r in rows]) if n else float("nan"), fail))
    for cls in ("BROKEN", "UNOWNED", "THROUGH", "FREE", "SPLIT", "CLEAN"):
        rs = [r for r in rows if r[0] == cls]
        if rs: print("  %-8s n %4d share %.3f idL %.3f share-of-L-failures %.3f angle %.0f" % (cls, len(rs), len(rs) / n, np.mean([r[1] for r in rs]), sum(1 for r in rs if not r[1]) / max(fail, 1), np.mean([r[2] for r in rs])))
