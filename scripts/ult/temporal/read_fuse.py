#!/usr/bin/env python3
"""S0 read (docs/TODO.md 2026-11-03 21:10): the partition at lag 1 and at lag 3, the tracker on every third frame, per arm.
usage: read_fuse.py HUB_SEGMENTATION_DIR V16_JSONS(comma) FUSE_JSONS(comma) FUSE2_JSONS(comma) LIST_JSON OUT_JSON"""
import json, os, sys
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree
HUB, POL, FUS, FU2, LST, OUT = sys.argv[1:7]; sys.path[:0] = [HUB, os.path.dirname(os.path.abspath(__file__))]
import api.tracker_kymograph as tk
from api.mt_geometry_cost import resample
from griddrift import best_shift
TOL = 5 / 1.5; MINCOV = 20; COVER = 0.8; ARMS = ("per_frame", "meanimg", "meanmap", "medianmap", "meanmap_q"); MOVER = "9caf72f4"
def seglen(p): return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
def load(spec):
    R = {}
    for p in spec.split(","): R.update(json.load(open(p)))
    return R
def partners(A, B, shift):
    if not A or not B: return [(0, len(a)) for a in A]
    own = np.concatenate([np.full(len(b), i) for i, b in enumerate(B)]); T = cKDTree(np.vstack(B)); out = []
    for a in A:
        near = T.query_ball_point(a + shift, TOL); cov = defaultdict(int)
        for hits in near:
            for c in {own[h] for h in hits}: cov[c] += 1
        out.append((sum(v >= MINCOV for v in cov.values()), len(a)))
    return out
def S(a, b):
    if len(a) == 0 or len(b) == 0: return float("nan")
    return float(((cKDTree(b).query(a)[0] <= TOL).mean() + (cKDTree(a).query(b)[0] <= TOL).mean()) / 2)
def lagread(R1, md5, lag):
    """R1: per frame, curves resampled at 1 px. Every pair (t, t + lag), mean of the pair shares, each direction counted."""
    rows, s1 = [], []
    for i in range(len(R1) - lag):
        if md5[i] == md5[i + lag] or not R1[i] or not R1[i + lag]: continue
        A, B = np.vstack(R1[i]), np.vstack(R1[i + lag]); d = best_shift(A, B)
        rows += partners(R1[i], R1[i + lag], d) + partners(R1[i + lag], R1[i], -d); s1.append(S(A + d, B))
    n = np.array([r[0] for r in rows]); L = np.array([r[1] for r in rows], float)
    return dict(P0L=float(L[n == 0].sum() / L.sum()), P1L=float(L[n == 1].sum() / L.sum()), P2L=float(L[n >= 2].sum() / L.sum()), S1=float(np.nanmean(s1)), pairs=len(s1))
def tracked(curves):
    req = tk.TrackRequest(frames=[{"frame": i, "polylines": [{"id": f"{i}::{j}", "points_rc": c.tolist()} for j, c in enumerate(cs)]} for i, cs in enumerate(curves)], cost_threshold=0.6)
    tracks = defaultdict(list)
    for key, tid in tk._track_sync(req).assignments.items():
        i, j = key.split("::"); tracks[tid].append((int(i), int(j)))
    dur = {t: len({i for i, _ in m}) for t, m in tracks.items()}; n = max(np.mean([len(c) for c in curves]), 1e-9)
    lng = sum(seglen(curves[i][j]) for t, m in tracks.items() if dur[t] >= COVER * len(curves) for i, j in m)
    return dict(T1=len(tracks) / n, T2abs=lng / len(curves), T3=float(np.median(list(dur.values()))))
def one(job):
    vid, fr, md5, polys = job; res = {}
    for arm, per in polys.items():
        cur = [[np.asarray(c, float) for c in p if len(c) > 1] for p in per]; R1 = [[resample(c, 1.0) for c in cs] for cs in cur]
        tr = [tracked(cur[o::3]) for o in range(3)]
        res[arm] = dict(lag1=lagread(R1, md5, 1), lag3=lagread(R1, md5, 3), len_per_frame=float(np.mean([sum(seglen(c) for c in cs) for cs in cur])), n_per_frame=float(np.mean([len(c) for c in cur])), **{k: float(np.mean([x[k] for x in tr])) for k in tr[0]})
    return vid, res
if __name__ == "__main__":
    L = json.load(open(LST)); A = load(POL); F = load(FUS); G = load(FU2); jobs = []
    for o in L:
        v = o["vid"]
        if v not in F: raise SystemExit(f"missing {v}")
        fr = [f["frame"] for f in o["frames"]][1:-1]; md5 = [f["md5"] for f in o["frames"]][1:-1]   # all arms on the same frames
        jobs.append((v, fr, md5, dict(per_frame=[A[v]["frames"][f]["polys"] for f in fr], meanimg=[F[v]["frames"][f]["meanimg"] for f in fr], meanmap=[F[v]["frames"][f]["meanmap"] for f in fr], medianmap=[G[v]["frames"][f]["medianmap"] for f in fr], meanmap_q=[G[v]["frames"][f]["meanmap_q"] for f in fr])))
    with Pool(12) as pl: R = dict(pl.map(one, jobs))
    for v in R: R[v]["attr"] = F[v]["attr"]
    json.dump(R, open(OUT, "w"), indent=1)
    vids = [o["vid"] for o in L]; proj = [o["project"] for o in L]; P = sorted(set(proj)); IDX = {q: [i for i, p in enumerate(proj) if p == q] for q in P}; rng = np.random.default_rng(20261103)
    def ci(d):
        m = [d[np.concatenate([IDX[q] for q in rng.choice(P, len(P))])].mean() for _ in range(10000)]
        return np.percentile(m, [2.5, 97.5])
    get = lambda arm, f: np.array([f(R[v][arm]) for v in vids], float)
    rows = [("P1L lag 3 (PRIMARY)", lambda r: r["lag3"]["P1L"]), ("P1L lag 1 (inflated)", lambda r: r["lag1"]["P1L"]), ("P2L lag 3", lambda r: r["lag3"]["P2L"]), ("P0L lag 3", lambda r: r["lag3"]["P0L"]), ("S1 lag 3", lambda r: r["lag3"]["S1"]), ("S1 lag 1 (inflated)", lambda r: r["lag1"]["S1"]), ("T1, every 3rd", lambda r: r["T1"]), ("T2abs, every 3rd", lambda r: r["T2abs"]), ("T3, every 3rd", lambda r: r["T3"]), ("length per frame", lambda r: r["len_per_frame"]), ("curves per frame", lambda r: r["n_per_frame"])]
    for arm in ARMS[1:]:
        print(f"\n{arm} against per frame\n{'':22s} {'A':>9s} {'arm':>9s} {'arm-A':>9s}  95% CI")
        for name, f in rows:
            x, y = get("per_frame", f), get(arm, f); lo, hi = ci(y - x)
            print(f"{name:22s} {x.mean():9.3f} {y.mean():9.3f} {(y - x).mean():+9.3f}  [{lo:+9.3f},{hi:+9.3f}]")
    print("\nguards: length vs A", {arm: round(float((get(arm, lambda r: r["len_per_frame"]) / get("per_frame", lambda r: r["len_per_frame"])).mean()), 3) for arm in ARMS[1:]})
    mv = next(v for v in vids if v.startswith(MOVER))
    print("mover", MOVER, {arm: {"P0L lag 3": round(R[mv][arm]["lag3"]["P0L"], 3), "P1L lag 3": round(R[mv][arm]["lag3"]["P1L"], 3), "len": round(R[mv][arm]["len_per_frame"])} for arm in ARMS})
    tot = defaultdict(int)
    for v in vids:
        for k, x in R[v]["attr"].items(): tot[k] += x
    for r in ("r3", "r5"):
        n = tot["bundle"] + tot["map_" + r] + tot["decoder_" + r]
        print(f"attribution {r}: one curve becomes two, {n} places: bundle {tot['bundle'] / n:.3f} map {tot['map_' + r] / n:.3f} decoder {tot['decoder_' + r] / n:.3f}")
    print("\nper container (P1L lag 3; attribution r3)")
    for v in vids:
        x = R[v]["attr"]; n = max(x.get("bundle", 0) + x.get("map_r3", 0) + x.get("decoder_r3", 0), 1)
        print(v[:8], " ".join(f"{arm} {R[v][arm]['lag3']['P1L']:.3f}" for arm in ARMS), f"| pairs {R[v]['per_frame']['lag3']['pairs']} bundle {x.get('bundle', 0) / n:.2f} map {x.get('map_r3', 0) / n:.2f} dec {x.get('decoder_r3', 0) / n:.2f}")
    print("READ_FUSE_DONE")
