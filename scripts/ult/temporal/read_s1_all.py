#!/usr/bin/env python3
"""S1, REPORT ONLY, added after the declared read (docs/TODO.md 2026-11-04 09:05): the production tracker over all frames, which is what the user runs. Consecutive windows share frames here, so this cannot pass or fail the rule.
usage: read_s1_all.py HUB_SEGMENTATION_DIR V16_JSONS(comma) S1_JSONS(comma) LIST_JSON OUT_JSON"""
import json, os, sys
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
HUB, POL, S1, LST, OUT = sys.argv[1:6]; sys.path.insert(0, HUB)
import api.tracker_kymograph as tk
COVER = 0.8; SHORT = 3
def seglen(p): return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
def track(per):
    cur = [[np.asarray(c, float) for c in p if len(c) > 1] for p in per]
    req = tk.TrackRequest(frames=[{"frame": i, "polylines": [{"id": f"{i}::{j}", "points_rc": c.tolist()} for j, c in enumerate(cs)]} for i, cs in enumerate(cur)], cost_threshold=0.6)
    tracks = defaultdict(list)
    for key, tid in tk._track_sync(req).assignments.items():
        i, j = key.split("::"); tracks[tid].append((int(i), int(j)))
    dur = {t: len({i for i, _ in m}) for t, m in tracks.items()}; n = max(np.mean([len(c) for c in cur]), 1e-9); tot = sum(seglen(c) for cs in cur for c in cs)
    lng = sum(seglen(cur[i][j]) for t, m in tracks.items() if dur[t] >= COVER * len(cur) for i, j in m)
    return dict(T1=len(tracks) / n, T2=lng / tot, T2abs=lng / len(cur), T3=float(np.median(list(dur.values()))), short=sum(d <= SHORT for d in dur.values()), long=sum(d >= COVER * len(cur) for d in dur.values()))
def one(job):
    v, a, b = job; return v, dict(per_frame=track(a), lam1=track(b))
if __name__ == "__main__":
    def load(s, key=None):
        R = {}
        for p in s.split(","):
            j = json.load(open(p)); R.update(j.get("frames", j) if key else j)
        return R
    L = json.load(open(LST)); A = load(POL); B = load(S1, True); jobs = []
    for o in L:
        fr = [f["frame"] for f in o["frames"]][1:-1]; v = o["vid"]
        jobs.append((v, [A[v]["frames"][f]["polys"] for f in fr], [B[v][f]["lam1"] for f in fr]))
    with Pool(12) as pl: R = dict(pl.map(one, jobs))
    json.dump(R, open(OUT, "w"), indent=1)
    vids = [o["vid"] for o in L]; proj = [o["project"] for o in L]; P = sorted(set(proj)); IDX = {q: [i for i, p in enumerate(proj) if p == q] for q in P}; rng = np.random.default_rng(20261104)
    ci = lambda d: np.percentile([d[np.concatenate([IDX[q] for q in rng.choice(P, len(P))])].mean() for _ in range(10000)], [2.5, 97.5])
    print(f"{'all frames, report only':24s} {'per frame':>10s} {'lam 1':>10s} {'delta':>9s}  95% CI")
    for k in ("T1", "T2", "T2abs", "T3", "short", "long"):
        a = np.array([R[v]["per_frame"][k] for v in vids]); b = np.array([R[v]["lam1"][k] for v in vids]); lo, hi = ci(b - a)
        print(f"{k:24s} {a.mean():10.3f} {b.mean():10.3f} {(b - a).mean():+9.3f}  [{lo:+.3f},{hi:+.3f}]  better in {int(((b - a) * (-1 if k in ('T1', 'short') else 1) > 0).sum())}/12")
    print("READ_S1_ALL_DONE")
