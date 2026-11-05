#!/usr/bin/env python3
"""Temporal stability, partition flicker (docs/TODO.md 2026-11-03 14:55): does the same material get cut differently into curves between frames?
usage: partition.py HUB_SEGMENTATION_DIR RES_DIR LIST_JSON OUT_JSON"""
import json, os, sys
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree
HUB, RES, LST, OUT = sys.argv[1:5]; sys.path[:0] = [HUB, os.path.dirname(os.path.abspath(__file__))]
import api.tracker_kymograph as tk
from api.mt_geometry_cost import resample
from griddrift import best_shift
TOL = 5 / 1.5; MINCOV = 20
def seglen(p): return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
def partners(A, B, shift):
    """A, B: lists of curves resampled at 1 px. For each curve of A, how many curves of B cover at least MINCOV px of it."""
    if not A or not B: return [(0, len(a)) for a in A]
    own = np.concatenate([np.full(len(b), i) for i, b in enumerate(B)]); T = cKDTree(np.vstack(B)); out = []
    for a in A:
        near = T.query_ball_point(a + shift, TOL); cov = defaultdict(int)
        for hits in near:
            for c in {own[h] for h in hits}: cov[c] += 1
        out.append((sum(v >= MINCOV for v in cov.values()), len(a)))
    return out
def tally(rows):
    n = np.array([r[0] for r in rows]); L = np.array([r[1] for r in rows], float)
    if len(n) == 0: return dict(P0=np.nan, P1=np.nan, P2=np.nan, P0L=np.nan, P1L=np.nan, P2L=np.nan)
    return dict(P0=float((n == 0).mean()), P1=float((n == 1).mean()), P2=float((n >= 2).mean()), P0L=float(L[n == 0].sum() / L.sum()), P1L=float(L[n == 1].sum() / L.sum()), P2L=float(L[n >= 2].sum() / L.sum()))
def one(job):
    model, vid = job; D = {}
    for k in (0, 1): D.update(json.load(open(f"{RES}/{model}_{k}.json")))
    v = D[vid]; fr = sorted(v["frames"]); C = [[resample(np.asarray(c, float), 1.0) for c in v["frames"][f]["polys"] if len(c) > 1] for f in fr]; raw = [[np.asarray(c, float) for c in v["frames"][f]["polys"] if len(c) > 1] for f in fr]
    rows = []
    for i in range(len(C) - 1):
        d = best_shift(np.vstack(C[i]) if C[i] else np.zeros((0, 2)), np.vstack(C[i + 1]) if C[i + 1] else np.zeros((0, 2)))
        rows += partners(C[i], C[i + 1], d) + partners(C[i + 1], C[i], -d)
    ctl = []
    for f in sorted(v["control"]):
        a = [resample(np.asarray(c, float), 1.0) for c in v["control"][f]["a"] if len(c) > 1]; b = [resample(np.asarray(c, float), 1.0) for c in v["control"][f]["b"] if len(c) > 1]
        ctl += partners(a, b, np.zeros(2)) + partners(b, a, np.zeros(2))
    req = tk.TrackRequest(frames=[{"frame": i, "polylines": [{"id": f"{i}::{j}", "points_rc": c.tolist()} for j, c in enumerate(cs)]} for i, cs in enumerate(raw)], cost_threshold=0.6)
    tracks = defaultdict(list)
    for key, tid in tk._track_sync(req).assignments.items():
        i, j = key.split("::"); tracks[tid].append((int(i), int(j)))
    long_d, short_d, long_l, short_l = [], [], [], []
    for m in tracks.values():
        d = len({i for i, _ in m}); l = float(np.mean([seglen(raw[i][j]) for i, j in m]))
        (long_d if d >= 0.8 * len(raw) else short_d).append(d); (long_l if d >= 0.8 * len(raw) else short_l).append(l)
    return model, vid, dict(real=tally(rows), control=tally(ctl), tracks_long=len(long_d), tracks_short=len(short_d), len_long=float(np.median(long_l)) if long_l else None, len_short=float(np.median(short_l)) if short_l else None, dur_short=float(np.median(short_d)) if short_d else None, single_frame_share=float(np.mean(np.array(short_d) == 1)) if short_d else None)
if __name__ == "__main__":
    L = json.load(open(LST)); R = defaultdict(dict)
    with Pool(12) as pl:
        for model, vid, r in pl.imap_unordered(one, [(m, o["vid"]) for m in ("sparse35", "v16") for o in L]):
            R[model][vid] = r
    json.dump(R, open(OUT, "w")); print("PARTITION_DONE")
    rng = np.random.default_rng(20261103); vids = sorted(R["v16"]); proj = [next(o["project"] for o in L if o["vid"] == v) for v in vids]; P = sorted(set(proj)); IDX = {q: [i for i, p in enumerate(proj) if p == q] for q in P}
    def ci(d):
        m = [d[np.concatenate([IDX[q] for q in rng.choice(P, len(P))])].mean() for _ in range(10000)]
        return np.percentile(m, [2.5, 97.5])
    print(f"{'':16s} SPARSE35    V16   V16-S35  95% CI | control S35  V16")
    for k in ("P0", "P1", "P2", "P0L", "P1L", "P2L"):
        a = np.array([R["sparse35"][v]["real"][k] for v in vids]); b = np.array([R["v16"][v]["real"][k] for v in vids]); ca = np.array([R["sparse35"][v]["control"][k] for v in vids]); cb = np.array([R["v16"][v]["control"][k] for v in vids]); lo, hi = ci(b - a)
        print(f"{k:16s} {a.mean():8.3f} {b.mean():6.3f} {(b - a).mean():+8.3f} [{lo:+.3f},{hi:+.3f}] | {ca.mean():.3f} {cb.mean():.3f}")
    for k in ("tracks_long", "tracks_short", "len_long", "len_short", "dur_short", "single_frame_share"):
        a = np.array([R["sparse35"][v][k] for v in vids], float); b = np.array([R["v16"][v][k] for v in vids], float)
        print(f"{k:18s} S35 {np.nanmean(a):8.2f}  V16 {np.nanmean(b):8.2f}")
    for v in vids:
        r = R["v16"][v]; print(v[:8], f"V16 P1 {r['real']['P1']:.2f} P1L {r['real']['P1L']:.2f} P2L {r['real']['P2L']:.2f} P0L {r['real']['P0L']:.2f} | ctl P1L {r['control']['P1L']:.2f} P2L {r['control']['P2L']:.2f} | long {r['tracks_long']} short {r['tracks_short']} len {r['len_long'] or 0:.0f}/{r['len_short'] or 0:.0f} dur_short {r['dur_short']}")
