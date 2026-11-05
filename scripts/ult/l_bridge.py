#!/usr/bin/env python3
"""(b') of docs/TODO.md 2026-11-02 07:15: L+H at bridge lengths other than 60 (L-hand, H, t 0 unchanged). usage: l_bridge.py DATA PRED CUT OUT.json BRS [NPROC] [SPLIT] [htw|all]"""
import glob, json, os, sys
import numpy as np
from multiprocessing import Pool
sys.path[:0] = [".", "src", "scripts", "scripts/lpc"]
import l_dup
from l_dup import match, walk_all, row, parallel_pairs, arclen
from l_hyst2 import hmask
from lpc_lib import P, UP, graph, to_logit, resample, rasterise_polyline
from mt_bench.cvat_import import read_frame_h5
from mt_bench.gt_stats import count_crossings
from step1_read import source_of
def one(args):
    path, pred, cut, brs, src = args; name = os.path.basename(path)
    if src == "htw" and source_of(name) != "htw": return None
    fr = read_frame_h5(path); prob = np.load(os.path.join(pred, name.replace(".h5", ".npz")))["prob"].astype(np.float32)[0]
    gt = [np.asarray(q, float) * UP for q in fr["polylines"]]
    cr = count_crossings(gt, tol=2.0 * UP, endpoint_margin=8.0 * UP) if len(gt) > 1 else []; par = parallel_pairs(gt) if len(gt) > 1 else []
    arcs, _, arms = graph(hmask(prob, cut, "bridge", 0.98, 22.5)); logit = to_logit(prob); res = {}
    for br in brs:
        l_dup.BR = br; polys = []
        for ch in (walk_all(arcs, match(arms, {}, logit)) if arcs else []):
            p = np.concatenate([arcs[k] if w == "start" else arcs[k][::-1] for k, w in ch], 0)
            if arclen(p) >= P["min_length"]: polys.append(p)
        res[f"b{br:g}"] = row([rasterise_polyline(resample(p, ds=0.5), prob.shape) for p in polys], gt, cr, par)
    return name, len(gt), res
if __name__ == "__main__":
    data, pred, cut, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]; brs = [float(x) for x in sys.argv[5].split(",")]
    NP = int(sys.argv[6]) if len(sys.argv) > 6 else 24; SPLIT = sys.argv[7] if len(sys.argv) > 7 else "val"; SRC = sys.argv[8] if len(sys.argv) > 8 else "all"
    paths = [p for p in sorted(glob.glob(os.path.join(data, "*.h5"))) if os.path.exists(os.path.join(pred, os.path.basename(p).replace(".h5", ".npz"))) and (SPLIT == "all" or str(read_frame_h5(p)["attrs"].get("split")) == SPLIT)]
    with Pool(NP) as pl: rows = [r for r in pl.map(one, [(p, pred, cut, brs, SRC) for p in paths], chunksize=1) if r]
    json.dump(dict(data=data, pred=pred, cut=cut, rows=rows), open(out, "w"))
    f1 = lambda t: 2 * t[0] / max(2 * t[0] + t[1] + t[2], 1)
    for br in brs:
        k = f"b{br:g}"; t = {q: sum(r[2][k].get(q) or 0 for r in rows) for q in ("tp", "fp", "fn", "jn", "jp", "npar", "bund")}
        print(f"{os.path.basename(pred.rstrip('/'))} {k}: F1 {f1((t['tp'], t['fp'], t['fn'])):.4f} tp/fp/fn {t['tp']}/{t['fp']}/{t['fn']} identity {t['jp'] / max(t['jn'], 1):.4f} bundle {t['bund'] / max(t['npar'], 1):.4f}", flush=True)
