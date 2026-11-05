#!/usr/bin/env python3
"""Temporal stability, the drift arms (docs/TODO.md 2026-11-03 14:45): S1 with no shift, the estimator and the search; the production tracker with each.
usage: measure2.py HUB_SEGMENTATION_DIR RES_DIR LIST_JSON OUT_JSON"""
import json, os, sys
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree
HUB, RES, LST, OUT = sys.argv[1:5]; sys.path[:0] = [HUB, os.path.dirname(os.path.abspath(__file__))]
import api.tracker_kymograph as tk
from api.mt_geometry_cost import estimate_drift, resample, curve_distance
from griddrift import best_shift
TOL = 5 / 1.5; COVER = 0.8
def seglen(p): return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
def pts(cs):
    a = [resample(np.asarray(c, float), 1.0) for c in cs]
    return np.vstack(a) if a else np.zeros((0, 2))
def S(a, b):
    if len(a) == 0 or len(b) == 0: return float("nan")
    return float(((cKDTree(b).query(a)[0] <= TOL).mean() + (cKDTree(a).query(b)[0] <= TOL).mean()) / 2)
ARMS = {"estimator": estimate_drift, "zero": lambda p, c, *a, **k: np.zeros(2), "search": lambda p, c, *a, **k: best_shift(pts(p), pts(c))}
def tracked(curves, arm):
    tk.estimate_drift = ARMS[arm]
    req = tk.TrackRequest(frames=[{"frame": i, "polylines": [{"id": f"{i}::{j}", "points_rc": c.tolist()} for j, c in enumerate(cs)]} for i, cs in enumerate(curves)], cost_threshold=0.6)
    res = tk._track_sync(req); tracks = defaultdict(list)
    for key, tid in res.assignments.items():
        i, j = key.split("::"); tracks[tid].append((int(i), int(j)))
    n = np.mean([len(c) for c in curves]); dur = {t: len({i for i, _ in m}) for t, m in tracks.items()}
    tot = sum(seglen(c) for cs in curves for c in cs); lng = sum(seglen(curves[i][j]) for t, m in tracks.items() if dur[t] >= COVER * len(curves) for i, j in m)
    ratio, cd = [], []
    for m in tracks.values():
        m = sorted(m)
        for (i, j), (i2, j2) in zip(m, m[1:]):
            if i2 == i + 1:
                a, b = curves[i][j], curves[i2][j2]; la, lb = seglen(a), seglen(b); ratio.append(min(la, lb) / max(la, lb, 1e-9)); cd.append(float(curve_distance(a, b)))
    return dict(tracks=len(tracks), T1=len(tracks) / n, T2=lng / tot, T2abs=lng / len(curves), T3=float(np.median(list(dur.values()))), E1_ratio=float(np.median(ratio)), E1_below07=float(np.mean(np.asarray(ratio) < 0.7)), E1_cd=float(np.median(cd)), E1_cd_p90=float(np.percentile(cd, 90)))
def one(job):
    model, vid = job; D = {}
    for k in (0, 1): D.update(json.load(open(f"{RES}/{model}_{k}.json")))
    v = D[vid]; fr = sorted(v["frames"]); curves = [[np.asarray(c, float) for c in v["frames"][f]["polys"] if len(c) > 1] for f in fr]
    S0, Se, Ss, err, est, srch = [], [], [], [], [], []
    for i in range(len(fr) - 1):
        A, B = pts(curves[i]), pts(curves[i + 1])
        e = estimate_drift(curves[i], curves[i + 1]); g = best_shift(A, B)
        S0.append(S(A, B)); Se.append(S(A + e, B)); Ss.append(S(A + g, B)); err.append(float(np.linalg.norm(e - g))); est.append(float(np.linalg.norm(e))); srch.append(float(np.linalg.norm(g)))
    C = v["control"]; cS = []
    for f in sorted(C):
        a = [np.asarray(c, float) for c in C[f]["a"] if len(c) > 1]; b = [np.asarray(c, float) for c in C[f]["b"] if len(c) > 1]
        cS.append(S(pts(a), pts(b)))
    return model, vid, dict(pairs=len(S0), S0=float(np.nanmean(S0)), Se=float(np.nanmean(Se)), Ss=float(np.nanmean(Ss)), Ss_med=float(np.nanmedian(Ss)), C=float(np.nanmean(cS)), err=err, est=est, search=srch, S0_series=S0, Se_series=Se, Ss_series=Ss, arms={a: tracked(curves, a) for a in ARMS})
if __name__ == "__main__":
    L = json.load(open(LST)); jobs = [(m, o["vid"]) for m in ("sparse35", "v16") for o in L]
    R = defaultdict(dict)
    with Pool(12) as pl:
        for model, vid, r in pl.imap_unordered(one, jobs):
            R[model][vid] = r; print(model, vid[:8], f"S0 {r['S0']:.3f} Se {r['Se']:.3f} Ss {r['Ss']:.3f} C {r['C']:.3f} err med {np.median(r['err']):.1f} max {max(r['err']):.1f} | T2abs {r['arms']['estimator']['T2abs']:.0f} {r['arms']['zero']['T2abs']:.0f} {r['arms']['search']['T2abs']:.0f}", flush=True)
    json.dump(R, open(OUT, "w")); print("MEASURE2_DONE")
