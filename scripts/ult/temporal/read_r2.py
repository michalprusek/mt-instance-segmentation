#!/usr/bin/env python3
"""ARM 65 R2 (docs/TODO.md 2026-11-04 08:10): TMP_Si on raw (t-1, t, t+1) against DEB_Si per frame on the 12 real windows, six pairs, edge frames out; mean over seeds per container, then bootstrap over projects. Secondary: S1 with lam 1 on both.
usage: read_r2.py HUB_SEGMENTATION_DIR R2_DIR V16_JSONS(comma) LIST_JSON OUT_JSON"""
import gzip, json, os, sys
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree
HUB, R2, POL, LST, OUT = sys.argv[1:6]; sys.path[:0] = [HUB, os.path.dirname(os.path.abspath(__file__))]
import api.tracker_kymograph as tk
from api.mt_geometry_cost import resample
from griddrift import best_shift
TOL = 5 / 1.5; MINCOV = 20; COVER = 0.8; MOVER = "9caf72f4"; SEEDS = range(1, 7)
def seglen(p): return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
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
    rows, s1 = [], []
    for i in range(len(R1) - lag):
        if md5[i] == md5[i + lag] or not R1[i] or not R1[i + lag]: continue
        A, B = np.vstack(R1[i]), np.vstack(R1[i + lag]); d = best_shift(A, B)
        rows += partners(R1[i], R1[i + lag], d) + partners(R1[i + lag], R1[i], -d); s1.append(S(A + d, B))
    n = np.array([r[0] for r in rows]); L = np.array([r[1] for r in rows], float)
    return dict(P0L=float(L[n == 0].sum() / L.sum()), P1L=float(L[n == 1].sum() / L.sum()), P2L=float(L[n >= 2].sum() / L.sum()), S1=float(np.nanmean(s1)))
def tracked(curves):
    req = tk.TrackRequest(frames=[{"frame": i, "polylines": [{"id": f"{i}::{j}", "points_rc": c.tolist()} for j, c in enumerate(cs)]} for i, cs in enumerate(curves)], cost_threshold=0.6)
    tracks = defaultdict(list)
    for key, tid in tk._track_sync(req).assignments.items():
        i, j = key.split("::"); tracks[tid].append((int(i), int(j)))
    dur = {t: len({i for i, _ in m}) for t, m in tracks.items()}; n = max(np.mean([len(c) for c in curves]), 1e-9)
    lng = sum(seglen(curves[i][j]) for t, m in tracks.items() if dur[t] >= COVER * len(curves) for i, j in m)
    return dict(T1=len(tracks) / n, T2abs=lng / len(curves), T3=float(np.median(list(dur.values()))))
def metrics(per, md5):
    cur = [[np.asarray(c, float) for c in p if len(c) > 1] for p in per]; R1 = [[resample(c, 1.0) for c in cs] for cs in cur]
    tr = [tracked(cur[o::3]) for o in range(3)]; al = tracked(cur)
    return dict(lag1=lagread(R1, md5, 1), lag3=lagread(R1, md5, 3), len=float(np.mean([sum(seglen(c) for c in cs) for cs in cur])), n=float(np.mean([len(c) for c in cur])),
                **{k: float(np.mean([x[k] for x in tr])) for k in tr[0]}, **{k + "_all": v for k, v in al.items()})
def one(job):
    v, s, md5, arms = job; return v, s, {a: metrics(p, md5) for a, p in arms.items()}
if __name__ == "__main__":
    L = json.load(open(LST)); V = {}
    for p in POL.split(","): V.update(json.load(open(p)))
    D = {}
    for s in SEEDS:
        for name, f in ((f"tmp{s}", f"TMP_S{s}_temporal_real"), (f"deb{s}", f"DEB_S{s}_single_real")):
            D[name] = json.load(gzip.open(f"{R2}/{f}.json.gz", "rt"))["frames"]
    jobs, same, tot, share = [], 0, 0, []
    for o in L:
        v = o["vid"]; fr = [f["frame"] for f in o["frames"]][1:-1]; md5 = [f["md5"] for f in o["frames"]][1:-1]
        for f in [x["frame"] for x in o["frames"]]:   # DEB_S4 through this path must be the deployed V16 output (rounded to 0.01 px by the compaction)
            a, b = D["deb4"][v][f]["polys"], V[v]["frames"][f]["polys"]; tot += 1
            same += len(a) == len(b) and all(np.allclose(np.asarray(x), np.asarray(y), atol=0.006) for x, y in zip(a, b) if len(x) == len(y)) and all(len(x) == len(y) for x, y in zip(a, b))
        for s in SEEDS:
            t, d = D[f"tmp{s}"][v], D[f"deb{s}"][v]
            share += [t[f]["share"] / max(t[f]["share_single"], 1e-12) for f in fr]
            jobs.append((v, s, md5, dict(tmp=[t[f]["polys"] for f in fr], deb=[d[f]["polys"] for f in fr], tmp_s1=[t[f]["s1"] for f in fr], deb_s1=[d[f]["s1"] for f in fr])))
    print(f"DEB_S4 through tmp_infer == deployed V16 output: {same}/{tot} frames", flush=True)
    print(f"TMP foreground share at the cut, temporal / its own (t,t,t), same frames: median {np.median(share):.3f}, p10 {np.percentile(share, 10):.3f}, p90 {np.percentile(share, 90):.3f}; declared trigger: median off 1 by more than 0.10", flush=True)
    with Pool(12) as pl: res = pl.map(one, jobs, chunksize=1)
    R = defaultdict(dict)
    for v, s, r in res: R[v][s] = r
    json.dump(R, open(OUT, "w"))
    vids = [o["vid"] for o in L]; proj = [o["project"] for o in L]; P = sorted(set(proj)); IDX = {q: [i for i, p in enumerate(proj) if p == q] for q in P}; rng = np.random.default_rng(20261104)
    ci = lambda d: np.percentile([d[np.concatenate([IDX[q] for q in rng.choice(P, len(P))])].mean() for _ in range(10000)], [2.5, 97.5])
    def per(arm, f): return np.array([[f(R[v][s][arm]) for s in SEEDS] for v in vids])   # containers x seeds
    rows = [("P1L lag 3 (PRIMARY)", lambda r: r["lag3"]["P1L"]), ("P1L lag 1 (inflated)", lambda r: r["lag1"]["P1L"]), ("P2L lag 3", lambda r: r["lag3"]["P2L"]), ("P0L lag 3", lambda r: r["lag3"]["P0L"]),
            ("S1 lag 3", lambda r: r["lag3"]["S1"]), ("T1, every 3rd", lambda r: r["T1"]), ("T2abs, every 3rd", lambda r: r["T2abs"]), ("T3, every 3rd", lambda r: r["T3"]),
            ("T1, all frames (rep.)", lambda r: r["T1_all"]), ("T3, all frames (rep.)", lambda r: r["T3_all"]), ("length per frame", lambda r: r["len"]), ("curves per frame", lambda r: r["n"])]
    for arm, ref in (("tmp", "deb"), ("tmp_s1", "deb"), ("deb_s1", "deb"), ("tmp_s1", "deb_s1")):
        print(f"\n{arm} against {ref}, six seed pairs, mean over seeds per container\n{'':24s} {ref:>9s} {arm:>9s} {'delta':>9s}  95% CI            seeds positive")
        for name, f in rows:
            x, y = per(ref, f), per(arm, f); d = (y - x).mean(1); lo, hi = ci(d); sp = ((y - x).mean(0) > 0).sum()
            print(f"{name:24s} {x.mean():9.3f} {y.mean():9.3f} {d.mean():+9.3f}  [{lo:+8.3f},{hi:+8.3f}]  {sp}/6")
    lr = (per("tmp", lambda r: r["len"]) / per("deb", lambda r: r["len"])).mean(); mi = vids.index(next(v for v in vids if v.startswith(MOVER)))
    ml = per("tmp", lambda r: r["len"])[mi].mean() / per("deb", lambda r: r["len"])[mi].mean(); mp = per("tmp", lambda r: r["lag3"]["P0L"])[mi].mean() - per("deb", lambda r: r["lag3"]["P0L"])[mi].mean()
    print(f"\nguards: length tmp/deb {lr:.3f} (within 0.95-1.05); mover length {ml:.3f} (within 0.90-1.10), P0L lag 3 {mp:+.3f} (not above +0.02)")
    d = (per("tmp", lambda r: r["lag3"]["P1L"]) - per("deb", lambda r: r["lag3"]["P1L"])).mean(1); lo, hi = ci(d)
    print("RULE R2:", "PASS" if lo > 0 and d.mean() >= 0.03 and 0.95 <= lr <= 1.05 and 0.90 <= ml <= 1.10 and mp <= 0.02 else "FAIL", f"(P1L lag 3 {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}], line: lower > 0 and point >= +0.03, guards above)")
    print("\nper container P1L lag 3, mean over seeds: deb tmp tmp_s1")
    for i, v in enumerate(vids): print(v[:8], f"{per('deb', lambda r: r['lag3']['P1L'])[i].mean():.3f} {per('tmp', lambda r: r['lag3']['P1L'])[i].mean():.3f} {per('tmp_s1', lambda r: r['lag3']['P1L'])[i].mean():.3f}")
    print("READ_R2_DONE")
