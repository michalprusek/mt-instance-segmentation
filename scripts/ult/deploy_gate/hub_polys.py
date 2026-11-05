#!/usr/bin/env python3
"""Deployment gate: the HUB decoder on the research dumps, in whatever environment this runs in (research env on kajman, the ML image on cvat2). Only numpy, scipy, skimage, networkx and the hub package; no research code. usage: hub_polys.py HUB_PKG DUMP_DIR NAMES.json CUT OUT.json [NPROC]"""
import json, os, sys, time
import numpy as np
from multiprocessing import Pool
PKG = sys.argv[1]; sys.path[:0] = [PKG, os.path.join(PKG, "vendor")]
from instance.instancer_lh import instance_lh
PAR = {k: v for k, v in json.load(open(os.path.join(PKG, "params_v16.json"))).items() if not k.startswith("_")}
def one(a):
    dump, name, cut = a; t = time.perf_counter()
    prob = np.load(os.path.join(dump, name.replace(".h5", ".npz")))["prob"].astype(np.float32)[0]
    return name, [p.tolist() for p in instance_lh(prob, cut, 0.25, PAR)], time.perf_counter() - t, prob.shape[0] * prob.shape[1]
if __name__ == "__main__":
    dump, names, cut, out = sys.argv[2], json.load(open(sys.argv[3])), float(sys.argv[4]), sys.argv[5]; NP = int(sys.argv[6]) if len(sys.argv) > 6 else 16
    with Pool(NP) as pl: rows = pl.map(one, [(dump, n, cut) for n in names], chunksize=1)
    json.dump({r[0]: r[1] for r in rows}, open(out, "w")); ts = np.array([r[2] for r in rows])
    import networkx, scipy, skimage
    print(len(rows), "frames", sum(len(r[1]) for r in rows), "polylines; decode s median", round(float(np.median(ts)), 2), "p95", round(float(np.percentile(ts, 95)), 2), "max", round(float(ts.max()), 2), "| np", np.__version__, "scipy", scipy.__version__, "skimage", skimage.__version__, "nx", networkx.__version__)
