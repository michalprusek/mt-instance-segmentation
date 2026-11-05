#!/usr/bin/env python3
"""ARM 65 R3 (docs/TODO.md 2026-11-04 08:10): accuracy with neighbours active, on the held-out clips. Every frame with both neighbours; TMP temporal, TMP (t, t, t), DEB, and S1 on TMP temporal; six seeds pooled; tracks per detected object against the generator's identities.
usage: read_r3.py CLIPS_DIR R2_DIR HUB_SEGMENTATION_DIR OUT_JSON  (cwd: the repo, for lpc_lib and instance.metrics)"""
import gzip, json, os, sys
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree
CLIPS, R2, HUB, OUT = sys.argv[1:5]
sys.path[:0] = [".", "src", "scripts", "scripts/lpc", HUB]
from lpc_lib import UP, resample, rasterise_polyline
from instance.metrics import centerline_f1
from api.tracker_kymograph import TrackRequest, _track_sync
C = 1024; SHAPE = (int(C * UP), int(C * UP)); FR = range(1, 10); REGS = ("static", "gliding", "dynamic"); ARMS = ("tmp_t", "tmp_s", "deb", "tmp_t_s1")
def score(polys, gt):
    masks = [rasterise_polyline(resample(np.asarray(p, float), ds=0.5), SHAPE) for p in polys]
    r = centerline_f1(masks, gt, tol=5.0, length_coverage=0.8, precision_coverage=0.8, end_tol=0.0)
    return [r["tp"], r["fp"], r["fn"]]
def tpo(frames, gtf):
    req = TrackRequest(frames=[{"frame": i, "polylines": [{"id": f"{i}::{j}", "points_rc": (np.asarray(c)[:, ::-1] / UP).tolist()} for j, c in enumerate(cs)]} for i, cs in enumerate(frames)], cost_threshold=0.6)
    seen = defaultdict(set)
    for key, tid in _track_sync(req).assignments.items():
        i, j = (int(x) for x in key.split("::")); p = resample(np.asarray(frames[i][j], float), ds=1.0); best, frac = None, 0.0
        for g, q in gtf[i].items():
            f = (cKDTree(q).query(p)[0] <= 5.0).mean()
            if f > frac: best, frac = g, f
        if best is not None and frac >= 0.5: seen[best].add(tid)
    n = [len(x) for x in seen.values()]; return [float(sum(n)), len(n)]
def one(job):
    name, s, arms = job; meta = json.load(open(f"{CLIPS}/{name}.json"))
    gtl = [[np.asarray(p, float) * UP for p in f.values()] for f in meta["gt"]]; gtf = [{int(i): resample(np.asarray(p, float) * UP, ds=1.0) for i, p in f.items()} for f in meta["gt"]]
    out = {}
    for a, per in arms.items():
        out[a] = dict(f1=np.sum([score(per[i], gtl[i]) for i in FR], axis=0).tolist(), tracks=tpo([per[i] for i in FR], [gtf[i] for i in FR]))
    out["deb_edge"] = np.sum([score(arms["deb"][i], gtl[i]) for i in (0, 10)], axis=0).tolist()
    out["oracle"] = tpo([[np.asarray(p, float) for p in gtl[i]] for i in FR], [gtf[i] for i in FR])
    return name, s, out
if __name__ == "__main__":
    jobs = []
    for s in range(1, 7):
        T = json.load(gzip.open(f"{R2}/TMP_S{s}_temporal_synth.json.gz", "rt"))["clips"]; Ts = json.load(gzip.open(f"{R2}/TMP_S{s}_single_synth.json.gz", "rt"))["clips"]; D = json.load(gzip.open(f"{R2}/DEB_S{s}_single_synth.json.gz", "rt"))["clips"]
        for name in sorted(T):
            jobs.append((name, s, dict(tmp_t=[x["curves"] for x in T[name]], tmp_s=[x["curves"] for x in Ts[name]], deb=[x["curves"] for x in D[name]], tmp_t_s1=[(x["s1"] if x["s1"] is not None else x["curves"]) for x in T[name]])))
    with Pool(16) as pl: res = pl.map(one, jobs, chunksize=1)
    R = defaultdict(dict)
    for name, s, r in res: R[name][s] = r
    json.dump(R, open(OUT, "w"))
    f1 = lambda t: 2 * t[0] / max(2 * t[0] + t[1] + t[2], 1)
    def tot(arm, reg=None, seed=None, key="f1"):
        return np.sum([R[n][s][arm][key] if key == "f1" else R[n][s][arm] for n in R for s in R[n] if (reg is None or n.startswith(reg)) and (seed is None or s == seed)], axis=0)
    print(f"{'F1, frames 1-9':16s} " + " ".join(f"{a:>9s}" for a in ARMS) + "   tmp_t-deb  per seed")
    for reg in REGS + (None,):
        v = {a: f1(tot(a, reg)) for a in ARMS}; ps = [f1(tot("tmp_t", reg, s)) - f1(tot("deb", reg, s)) for s in range(1, 7)]
        print(f"{(reg or 'pooled'):16s} " + " ".join(f"{v[a]:9.4f}" for a in ARMS) + f"   {v['tmp_t'] - v['deb']:+.4f}   " + " ".join(f"{d:+.3f}" for d in ps))
    g = {reg: f1(tot("tmp_t", reg)) - f1(tot("deb", reg)) for reg in REGS}
    print("GUARD R3, TMP temporal not below DEB by more than 0.01 in any regime:", "PASS" if min(g.values()) >= -0.01 else "FAIL", {k: round(v, 4) for k, v in g.items()})
    print(f"{'tracks/det. obj':16s} " + " ".join(f"{a:>9s}" for a in ARMS) + "    oracle")
    for reg in REGS + (None,):
        row = []
        for a in ARMS:
            t = np.sum([R[n][s][a]["tracks"] for n in R for s in R[n] if reg is None or n.startswith(reg)], axis=0); row.append(t[0] / max(t[1], 1))
        o = np.sum([R[n][1]["oracle"] for n in R if reg is None or n.startswith(reg)], axis=0)
        print(f"{(reg or 'pooled'):16s} " + " ".join(f"{x:9.3f}" for x in row) + f"   {o[0] / max(o[1], 1):7.3f}")
    print("DEB F1 on frames 1-9 (neighbours)", round(f1(tot("deb")), 4), "on frames 0 and 10", round(f1(np.sum([R[n][s]["deb_edge"] for n in R for s in R[n]], axis=0)), 4))
    print("READ_R3_DONE")
