#!/usr/bin/env python3
"""Diagnostic (docs/TODO.md 2026-11-04 17:12): R2's partition metric on the synthetic clips, TMP temporal against DEB, frames 1-9, lag 3, six seeds.
usage: syn_partition.py GEOM_DIR R2_DIR"""
import gzip, json, os, sys
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree
GEOM, R2 = sys.argv[1:3]; sys.path.insert(0, GEOM)
from mt_geometry_cost import resample
from griddrift import best_shift
TOL = 5 / 1.5; MINCOV = 20; UP = 1.5; FR = list(range(1, 10))
def partners(A, B, shift):
    if not A or not B: return [(0, len(a)) for a in A]
    own = np.concatenate([np.full(len(b), i) for i, b in enumerate(B)]); T = cKDTree(np.vstack(B)); out = []
    for a in A:
        near = T.query_ball_point(a + shift, TOL); cov = defaultdict(int)
        for hits in near:
            for c in {own[h] for h in hits}: cov[c] += 1
        out.append((sum(v >= MINCOV for v in cov.values()), len(a)))
    return out
def p1l(per, lag):
    R1 = [[resample(np.asarray(c, float)[:, ::-1] / UP, 1.0) for c in per[i] if len(c) > 1] for i in FR]; rows = []
    for i in range(len(R1) - lag):
        if not R1[i] or not R1[i + lag]: continue
        d = best_shift(np.vstack(R1[i]), np.vstack(R1[i + lag])); rows += partners(R1[i], R1[i + lag], d) + partners(R1[i + lag], R1[i], -d)
    n = np.array([r[0] for r in rows]); L = np.array([r[1] for r in rows], float); return float(L[n == 1].sum() / L.sum())
def one(job):
    name, s, t, d = job; return name, s, p1l(t, 3), p1l(d, 3), p1l(t, 1), p1l(d, 1)
if __name__ == "__main__":
    jobs = []
    for s in range(1, 7):
        T = json.load(gzip.open(f"{R2}/TMP_S{s}_temporal_synth.json.gz", "rt"))["clips"]; D = json.load(gzip.open(f"{R2}/DEB_S{s}_single_synth.json.gz", "rt"))["clips"]
        jobs += [(n, s, [x["curves"] for x in T[n]], [x["curves"] for x in D[n]]) for n in sorted(T)]
    with Pool(14) as pl: res = pl.map(one, jobs, chunksize=1)
    for reg in ("static", "gliding", "dynamic", ""):
        r = np.array([x[2:] for x in res if x[0].startswith(reg)]); d3 = r[:, 0] - r[:, 1]; d1 = r[:, 2] - r[:, 3]
        print(f"{reg or 'pooled':8s} P1L lag 3 DEB {r[:, 1].mean():.3f} TMP {r[:, 0].mean():.3f} delta {d3.mean():+.3f} (clips positive {(d3 > 0).mean():.2f}) | lag 1 DEB {r[:, 3].mean():.3f} TMP {r[:, 2].mean():.3f} delta {d1.mean():+.3f}")
    print("SYN_PARTITION_DONE")
