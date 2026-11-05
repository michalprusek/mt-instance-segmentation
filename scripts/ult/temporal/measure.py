#!/usr/bin/env python3
"""Temporal stability (docs/TODO.md 2026-11-03 14:10): production tracker and label-free metrics over the curves of infer.py.
usage: measure.py HUB_SEGMENTATION_DIR OUT_JSON sparse35=a.json,b.json v16=c.json,d.json"""
import json, sys, time
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
HUB, OUT = sys.argv[1:3]; sys.path.insert(0, HUB)
from api.mt_geometry_cost import estimate_drift, resample, curve_distance
from api.tracker_kymograph import TrackRequest, _track_sync
TOL = 5 / 1.5   # the benchmark's 5 px at 1.5x, in native px
COVER = 0.8     # share of the window: 16 of 20, 12 of the one 15-frame container
def load(spec):
    R = {}
    for p in spec.split(","): R.update(json.load(open(p)))
    return R
def seglen(p): return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0
def pts(curves):
    a = [resample(np.asarray(c, dtype=float), 1.0) for c in curves if len(c) > 1]
    return np.vstack(a) if a else np.zeros((0, 2))
def persist(A, B):
    """Symmetric share of centerline length within TOL of any centerline of the other frame."""
    if len(A) == 0 or len(B) == 0: return float("nan")
    a = (cKDTree(B).query(A)[0] <= TOL).mean(); b = (cKDTree(A).query(B)[0] <= TOL).mean()
    return float((a + b) / 2)
def jitter(x, y): return abs(x - y) / max((x + y) / 2, 1e-9)
def one_video(v):
    fr = sorted(v["frames"]); F = [v["frames"][f] for f in fr]
    curves = [[np.asarray(c, dtype=float) for c in f["polys"]] for f in F]
    S1, S2, drifts, skipped = [], [], [], 0
    for i in range(len(F) - 1):
        if F[i]["md5"] == F[i + 1]["md5"]: skipped += 1; continue
        a, b = [c for c in curves[i] if len(c) > 1], [c for c in curves[i + 1] if len(c) > 1]
        d = estimate_drift(a, b) if a and b else np.zeros(2); drifts.append(float(np.linalg.norm(d)))
        S1.append(persist(pts(a) + d, pts(b))); S2.append(jitter(len(a), len(b)))
    req = TrackRequest(frames=[{"frame": i, "polylines": [{"id": f"{i}::{j}", "points_rc": np.asarray(c).tolist()} for j, c in enumerate(curves[i]) if len(c) > 1]} for i in range(len(F))], cost_threshold=0.6)
    t0 = time.perf_counter(); res = _track_sync(req); ttrack = time.perf_counter() - t0
    tracks = defaultdict(list)
    for key, tid in res.assignments.items():
        i, j = key.split("::"); tracks[tid].append((int(i), int(j)))
    n = [sum(len(c) > 1 for c in cs) for cs in curves]; length = [sum(seglen(c) for c in cs) for cs in curves]
    dur = {t: len({i for i, _ in m}) for t, m in tracks.items()}
    long_len = sum(seglen(curves[i][j]) for t, m in tracks.items() if dur[t] >= COVER * len(F) for i, j in m)
    ratio, cd = [], []
    for t, m in tracks.items():
        m = sorted(m)
        for (i, j), (i2, j2) in zip(m, m[1:]):
            if i2 == i + 1:
                a, b = curves[i][j], curves[i2][j2]
                la, lb = seglen(a), seglen(b); ratio.append(min(la, lb) / max(la, lb, 1e-9)); cd.append(float(curve_distance(a, b)))
    C = v["control"]; cS1, cS2 = [], []
    for f in sorted(C):
        a, b = [np.asarray(c, dtype=float) for c in C[f]["a"]], [np.asarray(c, dtype=float) for c in C[f]["b"]]
        a, b = [c for c in a if len(c) > 1], [c for c in b if len(c) > 1]
        cS1.append(persist(pts(a), pts(b))); cS2.append(jitter(len(a), len(b)))
    tot = sum(length)
    return dict(frames=len(F), pairs=len(S1), identical_pairs=skipped, n_per_frame=float(np.mean(n)), len_per_frame=float(np.mean(length)),
                decoders=sorted({f["decoder"] for f in F}), drift_px=float(np.median(drifts)) if drifts else None,
                S1=float(np.nanmean(S1)) if S1 else None, S2=float(np.mean(S2)) if S2 else None,
                T1=len(tracks) / max(np.mean(n), 1e-9), tracks=len(tracks), T2=long_len / max(tot, 1e-9), T2abs=long_len / len(F), T3=float(np.median(list(dur.values()))) if dur else 0.0,
                E1_ratio=float(np.median(ratio)) if ratio else None, E1_below07=float(np.mean(np.asarray(ratio) < 0.7)) if ratio else None,
                E1_cd=float(np.median(cd)) if cd else None, E1_cd_p90=float(np.percentile(cd, 90)) if cd else None,
                C_S1=float(np.nanmean(cS1)) if cS1 else None, C_S2=float(np.mean(cS2)) if cS2 else None, t_track=ttrack)
res = {}
for spec in sys.argv[3:]:
    name, files = spec.split("=", 1); D = load(files); res[name] = {}
    for vid, v in D.items():
        res[name][vid] = one_video(v); r = res[name][vid]
        print(name, vid[:8], f"pairs {r['pairs']} n {r['n_per_frame']:.0f} len {r['len_per_frame']:.0f} S1 {r['S1']:.3f} S2 {r['S2']:.3f} T1 {r['T1']:.2f} T2 {r['T2']:.3f} T2abs {r['T2abs']:.0f} T3 {r['T3']:.0f} E1 {r['E1_ratio']:.2f}/{r['E1_below07']:.2f} cd {r['E1_cd']:.1f}/{r['E1_cd_p90']:.1f} ctrl S1 {r['C_S1']:.3f} S2 {r['C_S2']:.3f} track {r['t_track']:.0f}s", flush=True)
json.dump(res, open(OUT, "w"), indent=1); print("MEASURE_DONE")
