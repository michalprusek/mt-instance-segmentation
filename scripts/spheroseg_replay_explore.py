"""Correction replay, EXPLORATORY (written after the declared read of 2026-10-13, so nothing here is a test).

Two questions the primary count cannot answer:
  1. What did L do to each kept instance it "broke" and to each correction it missed -- cut it up,
     run on past it, join it with another annotated curve, or never have the foreground for it?
  2. At what gap does a person join two pieces, and at what gap does L join pieces a person left apart?

Also prints the gate of the declared read: does the control A on the captured map give back the
proposal stored in the block.

    python scripts/spheroseg_replay_explore.py data/real/spheroseg_verified
"""
from __future__ import annotations

import json
import sys
from collections import Counter

import h5py
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "src")
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.spheroseg_verified import SAME_FRAC, SAME_TOL, densify  # noqa: E402

root = sys.argv[1]
replay = json.load(open(f"{root}/replay/replay.json"))
prov = {r["project_id"][:8]: r for r in json.load(open(f"{root}/provenance.json"))}
TOL = 5.0 / 1.5          # the metric's 5 px of the 1.5x frame, in native px
HIT = 0.8


def cov(a, tree, tol=TOL):
    return float((tree.query(densify(a), distance_upper_bound=tol)[0] <= tol).mean())


def endgap(p, q):
    return min(float(np.linalg.norm(a - b)) for a in (p[0], p[-1]) for b in (q[0], q[-1]))


def order_along(curve, pieces):
    tr = cKDTree(densify(curve))
    return list(np.argsort([float(np.median(tr.query(densify(p))[1])) for p in pieces]))


broke, missed, human, extra = Counter(), Counter(), [], []
print("GATE  stored proposals with a same-curve match (1 px) among the control A's instances on the captured map")
for fr in replay["rows"]:
    p = prov[fr["project"]]
    stored, frame = p["stored"], read_frame_h5(p["h5"])
    G = frame["polylines"]
    with h5py.File(p["h5"], "r") as h:
        P = [h["proposals"][k][:] for k in sorted(h["proposals"].keys())]
    group = "SPARSE35" if fr["tag"] == "sparse35" else "v5H"
    A = [np.asarray(q, float) for q in fr["decoders"]["hubA"]["polylines_native"] if len(q) >= 2]
    a_tr = [cKDTree(densify(q)) for q in A]
    same = 0
    for q in P:
        qt = cKDTree(densify(q))
        same += any(cov(q, t, SAME_TOL) >= SAME_FRAC and cov(a, qt, SAME_TOL) >= SAME_FRAC for a, t in zip(A, a_tr))
    print(f"   {fr['project']} {group:8s} {same:3d} / {len(P):3d} = {100 * same / len(P):5.1f} %")

    L = [np.asarray(q, float) for q in fr["decoders"]["L"]["polylines_native"] if len(q) >= 2]
    l_tr = [cKDTree(densify(q)) for q in L]
    union = cKDTree(np.concatenate([densify(q) for q in L]))
    is_t = np.array([len(s["swallowed"]) >= 2 or s["fate"] == "split" for s in stored])
    is_k = np.array([s["fate"] == "kept" for s in stored]) & ~is_t
    a = np.array(fr["decoders"]["hubA"]["hit"])
    ell = np.array(fr["decoders"]["L"]["hit"])

    for i in np.nonzero((is_k & a & ~ell) | (is_t & ~a & ~ell))[0]:
        covs = [cov(G[i], t) for t in l_tr]
        j = int(np.argmax(covs))
        if covs[j] >= HIT:
            partners = [k for k in range(len(G)) if k != i and cov(G[k], l_tr[j]) >= 0.5]
            kind = "joined with another annotated curve" if partners else "runs on past the annotated curve"
        elif cov(G[i], union) >= HIT:
            kind = "cut into several instances"
        else:
            kind = "no foreground for part of it"
        (missed if is_t[i] else broke)[(group, kind)] += 1

    def owner(c):
        f = [cov(c, t, 3.0) for t in l_tr]
        k = int(np.argmax(f))
        return k if f[k] >= 0.5 else -1

    for i, s in enumerate(stored):                       # joins a person made
        if len(s["swallowed"]) >= 2:
            idx = [s["swallowed"][k] for k in order_along(G[i], [P[k] for k in s["swallowed"]])]
            for u, v in zip(idx[:-1], idx[1:]):
                ou, ov = owner(P[u]), owner(P[v])
                human.append((group, endgap(P[u], P[v]), ou >= 0 and ou == ov))
    by: dict = {}                                        # joins only L made, between two kept curves
    for i in np.nonzero(is_k)[0]:
        k = owner(G[i])
        if k >= 0:
            by.setdefault(k, []).append(int(i))
    for k, ii in by.items():
        if len(ii) >= 2:
            ii = [ii[m] for m in order_along(L[k], [G[i] for i in ii])]
            extra += [(group, endgap(G[u], G[v])) for u, v in zip(ii[:-1], ii[1:])]

print("\n1. kept instances L broke:")
for k, v in sorted(broke.items()):
    print(f"   {k[0]:8s} {v:3d}  {k[1]}")
print("   corrections L missed:")
for k, v in sorted(missed.items()):
    print(f"   {k[0]:8s} {v:3d}  {k[1]}")

print("\n2. gap between the two pieces, native px (A links up to 23.8, L bridges up to 40)")
for g in ("SPARSE35", "v5H"):
    h = np.array([x[1] for x in human if x[0] == g])
    made = np.array([x[2] for x in human if x[0] == g], bool)
    e = np.array([x[1] for x in extra if x[0] == g])
    print(f"   {g}: human joins {len(h)} (median gap {np.median(h):.1f}), L made {int(made.sum())}; "
          f"L-only joins of two kept curves {len(e)}" + (f" (median gap {np.median(e):.1f})" if len(e) else ""))
    for lo, hi in ((0, 12), (12, 24), (24, 40), (40, 1e9)):
        m = (h >= lo) & (h < hi)
        print(f"      gap {lo:>2.0f}-{('%.0f' % hi) if hi < 1e8 else 'up':<3s}: human joins {int(m.sum()):2d}, L made {int((made & m).sum()):2d}"
              f"   | L-only joins {int(((e >= lo) & (e < hi)).sum()):2d}")
