#!/usr/bin/env python3
"""S1 on the held-out clips (docs/TODO.md 2026-11-04 08:10): the declared metric per frame with both neighbours, by regime, per frame against each lam; lam chosen by pooled F1, ties to the smaller; the production tracker's tracks per object against the generator's identities. Runs in mt34_work's environment for the metric; the tracker from the hub.
usage: read_s1_syn.py CLIPS_DIR S1_SYN_JSONS(comma) HUB_SEGMENTATION_DIR OUT_JSON"""
import json, os, sys
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree
CLIPS, RES, HUB, OUT = sys.argv[1:5]
sys.path[:0] = [".", "src", "scripts", "scripts/lpc"]
from lpc_lib import UP, resample, rasterise_polyline
from instance.metrics import centerline_f1
LAMS = ("lam0.5", "lam1", "lam2", "lam4"); C = 1024; SHAPE = (int(C * UP), int(C * UP)); NB = 5
sys.path.insert(0, HUB)
from api.tracker_kymograph import TrackRequest, _track_sync
def score(polys, gt):
    masks = [rasterise_polyline(resample(np.asarray(p, float), ds=0.5), SHAPE) for p in polys]
    r = centerline_f1(masks, gt, tol=5.0, length_coverage=0.8, precision_coverage=0.8, end_tol=0.0)
    return r["tp"], r["fp"], r["fn"]
def tracks_per_object(frames, gtf):
    """frames: per frame polylines (x, y) at the eval scale; gtf: per frame {id: polyline}. Tracks whose curve lies within tol of an object for at least half its length count for it."""
    req = TrackRequest(frames=[{"frame": i, "polylines": [{"id": f"{i}::{j}", "points_rc": (np.asarray(c)[:, ::-1] / UP).tolist()} for j, c in enumerate(cs)]} for i, cs in enumerate(frames)], cost_threshold=0.6)
    res = _track_sync(req).assignments; seen = defaultdict(set)
    for key, tid in res.items():
        i, j = (int(x) for x in key.split("::")); p = resample(np.asarray(frames[i][j], float), ds=1.0); best, frac = None, 0.0
        for g, q in gtf[i].items():
            f = (cKDTree(q).query(p)[0] <= 5.0).mean()
            if f > frac: best, frac = g, f
        if best is not None and frac >= 0.5: seen[best].add(tid)
    objs = set().union(*[set(x) for x in gtf]); n = [len(seen[o]) for o in objs if o in seen]
    return dict(objects=len(objs), detected=len(n), tracks=float(np.sum(n)))
def one(job):
    name, cl = job; meta = json.load(open(f"{CLIPS}/{name}.json")); T = len(meta["gt"]); gtf = [{int(i): resample(np.asarray(p, float) * UP, ds=1.0) for i, p in f.items()} for f in meta["gt"]]; gtl = [[np.asarray(p, float) * UP for p in f.values()] for f in meta["gt"]]
    idx = sorted(int(i) for i in cl["lam"]); arms = {"per_frame": [cl["per_frame"][i] for i in idx]}
    for lam in LAMS: arms[lam] = [cl["lam"][str(i)][lam] for i in idx]
    out = {a: np.array([score(ps, gtl[i]) for ps, i in zip(arms[a], idx)]).sum(0).tolist() for a in arms}
    tr = {a: tracks_per_object(arms[a], [gtf[i] for i in idx]) for a in arms}
    oracle = tracks_per_object([[np.asarray(p, float) for p in gtl[i]] for i in idx], [gtf[i] for i in idx])
    edge = np.array([score(cl["per_frame"][i], gtl[i]) for i in idx if i != NB]).sum(0).tolist(); centre = list(score(cl["per_frame"][NB], gtl[NB]))
    return name, dict(f1=out, tracks=tr, oracle=oracle, per_frame_neighbours=edge, per_frame_centre=centre)
if __name__ == "__main__":
    D = {}
    for p in RES.split(","): D.update(json.load(open(p))["clips"])
    with Pool(16) as pl: R = dict(pl.map(one, sorted(D.items()), chunksize=1))
    json.dump(R, open(OUT, "w"), indent=1)
    f1 = lambda s: 2 * s[0] / max(2 * s[0] + s[1] + s[2], 1)
    regs = ("static", "gliding", "dynamic"); arms = ("per_frame",) + LAMS
    tot = lambda reg, a: np.sum([R[n]["f1"][a] for n in R if reg is None or n.startswith(reg)], axis=0)
    print(f"{'F1':12s} " + " ".join(f"{a:>9s}" for a in arms))
    for reg in regs + (None,): print(f"{(reg or 'pooled'):12s} " + " ".join(f"{f1(tot(reg, a)):9.4f}" for a in arms))
    best = max(LAMS, key=lambda a: (round(f1(tot(None, a)), 6), -float(a[3:])))
    print("chosen by pooled F1, ties to the smaller:", best)
    print("guard, every regime not below per frame by more than 0.01:", {reg: round(f1(tot(reg, best)) - f1(tot(reg, "per_frame")), 4) for reg in regs})
    print(f"{'tracks/obj':12s} " + " ".join(f"{a:>9s}" for a in arms) + "    oracle")
    def tpo(reg, a):
        ns = [n for n in R if reg is None or n.startswith(reg)]; x = sum(R[n]["tracks"][a]["tracks"] for n in ns); y = sum(R[n]["tracks"][a]["detected"] for n in ns)
        return x / max(y, 1)
    def oo(reg):
        ns = [n for n in R if reg is None or n.startswith(reg)]; return sum(R[n]["oracle"]["tracks"] for n in ns) / max(sum(R[n]["oracle"]["detected"] for n in ns), 1)
    for reg in regs + (None,): print(f"{(reg or 'pooled'):12s} " + " ".join(f"{tpo(reg, a):9.3f}" for a in arms) + f" {oo(reg):9.3f}")
    print("per frame F1 on frame t", round(f1(np.sum([R[n]["per_frame_centre"] for n in R], axis=0)), 4), "on the neighbour frames", round(f1(np.sum([R[n]["per_frame_neighbours"] for n in R], axis=0)), 4))
    print("READ_S1_SYN_DONE")
