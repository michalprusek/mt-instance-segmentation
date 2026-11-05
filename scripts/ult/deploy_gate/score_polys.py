#!/usr/bin/env python3
"""Deployment gate: research metric on a polyline json vs the declared counts of the same frames. usage: score_polys.py DATA POLYS.json REF_BRIDGE.json [htw|all]"""
import json, os, sys
import numpy as np
from multiprocessing import Pool
sys.path[:0] = [".", "src", "scripts", "scripts/lpc"]
from lpc_lib import UP, resample, rasterise_polyline
from instance.metrics import centerline_f1
from mt_bench.cvat_import import read_frame_h5
from step1_read import source_of
DATA, PJ, REF = sys.argv[1:4]; P = json.load(open(PJ)); R = {r[0]: r[2]["b80"] for r in json.load(open(REF))["rows"]}
def one(n):
    fr = read_frame_h5(os.path.join(DATA, n)); gt = [np.asarray(q, float) * UP for q in fr["polylines"]]; H, W = (int(round(s * UP)) for s in np.asarray(fr["image"]).shape[:2])
    masks = [rasterise_polyline(resample(np.asarray(p, float), ds=0.5), (H, W)) for p in P[n]]; r = centerline_f1(masks, gt, tol=5.0, length_coverage=0.8, precision_coverage=0.8, end_tol=0.0)
    return n, (r["tp"], r["fp"], r["fn"])
if __name__ == "__main__":
    with Pool(16) as pl: rows = pl.map(one, sorted(P), chunksize=1)
    f1 = lambda s: 2 * s[0] / max(2 * s[0] + s[1] + s[2], 1); groups = {}
    for n, s in rows:
        g = source_of(n) if "hb" in os.path.basename(REF) or "hball" in os.path.basename(REF) else "roi303"; g = "htw" if g == "htw" else ("mt34" if g != "roi303" else g); a = groups.setdefault(g, [np.zeros(3), np.zeros(3), 0]); a[0] += s; q = R[n]; a[1] += (q["tp"], q["fp"], q["fn"]); a[2] += tuple(s) != (q["tp"], q["fp"], q["fn"])
    for g, (a, b, d) in groups.items(): print(f"{g}: this {a.astype(int).tolist()} F1 {f1(a):.4f} | declared {b.astype(int).tolist()} F1 {f1(b):.4f} | frames differing {d}")
