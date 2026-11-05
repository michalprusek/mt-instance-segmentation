#!/usr/bin/env python3
"""Deployment gate (docs/TODO.md 2026-11-03 08:05): the research decoder's polylines per VAL frame, as l_bridge.py decodes them (L+H, b80), before rasterising. usage: research_polys.py DATA PRED CUT OUT.json [NPROC]"""
import glob, json, os, sys
import numpy as np
from multiprocessing import Pool
sys.path[:0] = [".", "src", "scripts", "scripts/lpc"]
import l_dup
from l_dup import match, walk_all, arclen
from l_hyst2 import hmask
from lpc_lib import P, graph, to_logit
from mt_bench.cvat_import import read_frame_h5
l_dup.BR = 80.0
def one(a):
    path, pred, cut = a; name = os.path.basename(path)
    prob = np.load(os.path.join(pred, name.replace(".h5", ".npz")))["prob"].astype(np.float32)[0]
    arcs, _, arms = graph(hmask(prob, cut, "bridge", 0.98, 22.5)); polys = []
    for ch in (walk_all(arcs, match(arms, {}, to_logit(prob))) if arcs else []):
        p = np.concatenate([arcs[k] if w == "start" else arcs[k][::-1] for k, w in ch], 0)
        if arclen(p) >= P["min_length"]: polys.append(p.tolist())
    return name, polys
if __name__ == "__main__":
    data, pred, cut, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]; NP = int(sys.argv[5]) if len(sys.argv) > 5 else 16
    paths = [p for p in sorted(glob.glob(os.path.join(data, "*.h5"))) if os.path.exists(os.path.join(pred, os.path.basename(p).replace(".h5", ".npz"))) and str(read_frame_h5(p)["attrs"].get("split")) == "val"]
    with Pool(NP) as pl: rows = pl.map(one, [(p, pred, cut) for p in paths], chunksize=1)
    json.dump(dict(rows), open(out, "w")); print(len(rows), "frames", sum(len(r[1]) for r in rows), "polylines ->", out)
