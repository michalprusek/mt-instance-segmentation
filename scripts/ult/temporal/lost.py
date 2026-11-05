#!/usr/bin/env python3
"""Temporal stability, where the lost length goes (docs/TODO.md 2026-11-03 14:20). V16 only.
usage: lost.py PKG WEIGHTS FRAMES_DIR LIST_JSON V16_JSONS(comma) GEOM_DIR OUT_JSON k/K [search]
search: translation from griddrift.best_shift instead of estimate_drift (2026-11-03 14:45)."""
import json, os, sys
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
PKG, W, ROOT, LST, POL, GEOM, OUT = sys.argv[1:8]; k, K = (int(x) for x in sys.argv[8].split("/"))
sys.path[:0] = [os.path.dirname(PKG), PKG, os.path.join(PKG, "vendor"), GEOM]
from wrapper import MicrotubuleModel, UP
from mt_geometry_cost import estimate_drift, resample
from griddrift import best_shift
SEARCH = len(sys.argv) > 9 and sys.argv[9] == "search"
TOL = 5 / 1.5; CUT = 0.995; LADDER = [6.0, 10.0, 20.0, 50.0]
m = MicrotubuleModel().load_weights(W)
P = {}
for p in POL.split(","): P.update(json.load(open(p)))
yy, xx = np.mgrid[-5:6, -5:6]; FOOT = yy ** 2 + xx ** 2 <= 25   # the 3.33 native px at the eval scale
def pts(curves):
    a = [resample(c, 1.0) for c in curves]
    return np.vstack(a) if a else np.zeros((0, 2))
def peak_map(path):
    a = np.asarray(Image.open(path)).astype(np.float32)
    return ndi.maximum_filter(m.infer_maps(a)["prob_eval"].astype(np.float32), footprint=FOOT)
def lost(A, B, peak):
    if len(A) == 0 or len(B) == 0: return None
    d = cKDTree(B).query(A)[0]; miss = d > TOL
    if not miss.any(): return dict(total=len(A), lost=0, near={str(t): 0 for t in LADDER}, hi=0, mid=0, low=0)
    q = np.rint(A[miss] * UP).astype(int); q[:, 0] = q[:, 0].clip(0, peak.shape[0] - 1); q[:, 1] = q[:, 1].clip(0, peak.shape[1] - 1)
    v = peak[q[:, 0], q[:, 1]]
    return dict(total=len(A), lost=int(miss.sum()), near={str(t): int((d[miss] <= t).sum()) for t in LADDER}, hi=int((v >= CUT).sum()), mid=int(((v >= 0.5) & (v < CUT)).sum()), low=int((v < 0.5).sum()))
R = {}
for r in json.load(open(LST))[k::K]:
    vid = r["vid"]; F = r["frames"]; rows = []
    C = [[np.asarray(c, dtype=float) for c in P[vid]["frames"][f["frame"]]["polys"] if len(c) > 1] for f in F]
    pk = peak_map(os.path.join(ROOT, F[0]["path"]))
    for i in range(len(F) - 1):
        nxt = peak_map(os.path.join(ROOT, F[i + 1]["path"]))
        if F[i]["md5"] != F[i + 1]["md5"]:
            A, B = pts(C[i]), pts(C[i + 1])
            d = best_shift(A, B) if SEARCH else (estimate_drift(C[i], C[i + 1]) if C[i] and C[i + 1] else np.zeros(2))
            for x in (lost(A + d, B, nxt), lost(B - d, A, pk)):
                if x: rows.append(x)
        pk = nxt
    agg = {q: sum(x[q] for x in rows) for q in ("total", "lost", "hi", "mid", "low")}; agg["near"] = {str(t): sum(x["near"][str(t)] for x in rows) for t in LADDER}
    R[vid] = agg; print(vid[:8], agg, flush=True); json.dump(R, open(OUT, "w"))
print("LOST_DONE", k, flush=True)
