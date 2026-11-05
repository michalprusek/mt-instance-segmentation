"""Correction replay, step 3: the declared read (docs/TODO.md 2026-10-13 21:20).

Joins the decoder outputs of ``spheroseg_replay_decode.py`` with the fates of the 10-13 diff
(``provenance.json``). Every set is fixed by that diff, none by this run:

  T   human topology corrections: stored curves that swallowed >= 2 proposals, or split pieces
  K   curves kept as proposed
  T-  T not hit by the control A        K+  K hit by the control A
  PRIMARY  r = |T- hit by L|  against  b = |K+ not hit by L|

    python scripts/spheroseg_replay_read.py data/real/spheroseg_verified [replay.json] [overlay_dir]
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

import h5py
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import binomtest

sys.path.insert(0, "src")
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.spheroseg_verified import REPLACE_FRAC, REPLACE_TOL, densify  # noqa: E402

root = sys.argv[1]
replay = json.load(open(sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "replay", "replay.json")))
overlay_dir = sys.argv[3] if len(sys.argv) > 3 else ""
prov = {r["project_id"][:8]: r for r in json.load(open(os.path.join(root, "provenance.json")))}
CONTROL = "hubA"


def chamber(title: str) -> str:
    t = title.replace(" ", "")
    if "CH9_DO3" in t:
        return "0521 CH9"
    if "CH10_DO4" in t:
        return "0521 CH10"
    if "kif1a" in t:
        return "0327 ch2[I]"
    return "0410 ch1[I]" if "ch1[I]" in t else "0410 ch2[II]"


def owner(curve: np.ndarray, trees) -> int:
    """Index of the decoder instance carrying >= half of ``curve`` (3 px), or -1."""
    pts = densify(curve)
    best, frac = -1, 0.0
    for i, tr in enumerate(trees):
        f = float((tr.query(pts, distance_upper_bound=REPLACE_TOL)[0] <= REPLACE_TOL).mean())
        if f > frac:
            best, frac = i, f
    return best if frac >= REPLACE_FRAC else -1


def order_along(curve: np.ndarray, pieces) -> list[int]:
    """Indices of ``pieces`` sorted by where they sit along ``curve``."""
    tr = cKDTree(densify(curve))
    return list(np.argsort([float(np.median(tr.query(densify(p))[1])) for p in pieces]))


rows, tot = [], defaultdict(Counter)
mech = defaultdict(Counter)
for fr in replay["rows"]:
    p = prov[fr["project"]]
    stored, group, ch = p["stored"], ("SPARSE35" if fr["tag"] == "sparse35" else "v5H"), chamber(p["title"])
    assert fr["n_gt"] == len(stored), (fr["project"], fr["n_gt"], len(stored))
    is_t = np.array([len(s["swallowed"]) >= 2 or s["fate"] == "split" for s in stored])
    is_k = np.array([s["fate"] == "kept" for s in stored]) & ~is_t
    hit = {k: np.array(v["hit"], bool) for k, v in fr["decoders"].items()}
    a, ell = hit[CONTROL], hit["L"]
    r = int((is_t & ~a & ell).sum())
    b = int((is_k & a & ~ell).sum())
    row = {"project": fr["project"], "title": p["title"], "group": group, "chamber": ch,
           "T": int(is_t.sum()), "T_hitA": int((is_t & a).sum()), "T-": int((is_t & ~a).sum()), "r": r,
           "T_hitL": int((is_t & ell).sum()), "T_lostL": int((is_t & a & ~ell).sum()),
           "K": int(is_k.sum()), "K+": int((is_k & a).sum()), "b": b, "K_gainL": int((is_k & ~a & ell).sum()),
           "other": int((~is_t & ~is_k).sum()), "other_hitA": int((~is_t & ~is_k & a).sum()),
           "other_hitL": int((~is_t & ~is_k & ell).sum()),
           "f1": {k: (v["tp"], v["fp"], v["fn"]) for k, v in fr["decoders"].items()}}
    rows.append(row)
    for key in ("ALL", group, ch):
        for q in ("T", "T_hitA", "T-", "r", "T_hitL", "T_lostL", "K", "K+", "b", "K_gainL", "other", "other_hitA", "other_hitL"):
            tot[key][q] += row[q]
        for k, (tp, fp, fn) in row["f1"].items():
            tot[key][k + "_tp"] += tp
            tot[key][k + "_fp"] += fp
            tot[key][k + "_fn"] += fn

    # mechanism: the human's pairwise decisions, read off each decoder's instances
    frame = read_frame_h5(p["h5"])
    with h5py.File(p["h5"], "r") as h:
        props = [h["proposals"][k][:] for k in sorted(h["proposals"].keys())]
    for dname, d in fr["decoders"].items():
        trees = [cKDTree(densify(np.asarray(q, float))) for q in d["polylines_native"] if len(q) >= 2]
        own_p = {}
        for i, s in enumerate(stored):
            if len(s["swallowed"]) >= 2:                       # a join: consecutive proposals inside one curve
                idx = [s["swallowed"][j] for j in order_along(frame["polylines"][i], [props[j] for j in s["swallowed"]])]
                for u, v in zip(idx[:-1], idx[1:]):
                    for w in (u, v):
                        if w not in own_p:
                            own_p[w] = owner(props[w], trees)
                    mech[(group, dname)]["join_pairs"] += 1
                    mech[(group, dname)]["join_made"] += int(own_p[u] >= 0 and own_p[u] == own_p[v])
        kids = defaultdict(list)
        for i, s in enumerate(stored):
            if s["fate"] == "split":
                kids[s["proposal"]].append(i)
        for par, ii in kids.items():                            # a split: two pieces of one proposal
            if len(ii) < 2:
                continue
            ii = [ii[j] for j in order_along(props[par], [frame["polylines"][i] for i in ii])]
            for u, v in zip(ii[:-1], ii[1:]):
                ou, ov = owner(frame["polylines"][u], trees), owner(frame["polylines"][v], trees)
                mech[(group, dname)]["split_pairs"] += 1
                mech[(group, dname)]["split_made"] += int(ou >= 0 and ov >= 0 and ou != ov)
                mech[(group, dname)]["split_piece_missing"] += int(ou < 0 or ov < 0)

    if overlay_dir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from mt_bench.overlay import _norm01
        os.makedirs(overlay_dir, exist_ok=True)
        hh, ww = frame["image"].shape
        fig, ax = plt.subplots(figsize=(ww / 100.0, hh / 100.0), dpi=200)
        ax.imshow(_norm01(frame["image"]), cmap="gray", interpolation="nearest")
        for q in fr["decoders"]["L"]["polylines_native"]:
            q = np.asarray(q, float)
            ax.plot(q[:, 0], q[:, 1] + 4.0, "-", lw=0.4, color="#35c9ff", alpha=0.9)   # L, drawn 4 px below
        for i, g in enumerate(frame["polylines"]):
            if is_t[i]:
                col = "#3ddc3d" if (ell[i] and not a[i]) else ("#ffd400" if ell[i] else "#ff2a2a")
                ax.plot(g[:, 0], g[:, 1], "-", lw=1.0, color=col, alpha=0.95)
            elif is_k[i] and a[i] and not ell[i]:
                ax.plot(g[:, 0], g[:, 1], "-", lw=1.0, color="#ff8c00", alpha=0.95)
        lab = [("#3ddc3d", "human correction, L makes it (A did not)"), ("#ffd400", "human correction, both decoders hit it"),
               ("#ff2a2a", "human correction, L misses it"), ("#ff8c00", "kept by the human, L loses it"),
               ("#35c9ff", "L instances (4 px below)")]
        ax.legend([Line2D([0], [0], color=c, lw=1.5) for c, _ in lab], [t for _, t in lab], loc="lower right", fontsize=5, framealpha=0.8)
        ax.set_xlim(0, ww)
        ax.set_ylim(hh, 0)
        ax.axis("off")
        fig.subplots_adjust(0, 0, 1, 1)
        fig.savefig(os.path.join(overlay_dir, f"{p['slug']}.jpg"), dpi=200, pil_kwargs={"quality": 90})
        plt.close(fig)


def f1(t, k):
    tp, fp, fn = t[k + "_tp"], t[k + "_fp"], t[k + "_fn"]
    return 2 * tp / max(2 * tp + fp + fn, 1)


print("GATE  control A vs this tree's A on the same map (TP / FP / FN against the annotation):")
for r in rows:
    print(f"   {r['project']} {r['group']:8s} hubA {r['f1']['hubA']}  A {r['f1']['A']}")

print("\nPRIMARY  per frame          T  hitA   T-    r | hitL lostL |    K   K+    b gainL | other hitA hitL")
for r in rows:
    print(f"   {r['project']} {r['group']:8s} {r['T']:4d} {r['T_hitA']:5d} {r['T-']:4d} {r['r']:4d} | {r['T_hitL']:4d} {r['T_lostL']:5d} | "
          f"{r['K']:4d} {r['K+']:4d} {r['b']:4d} {r['K_gainL']:5d} | {r['other']:5d} {r['other_hitA']:4d} {r['other_hitL']:4d}  {r['chamber']}")
print()
for key in ["ALL", "SPARSE35", "v5H"] + sorted({r["chamber"] for r in rows}):
    t = tot[key]
    n = t["r"] + t["b"]
    pv = binomtest(t["r"], n, 0.5).pvalue if n else float("nan")
    print(f"   {key:12s} T {t['T']:3d}  T- {t['T-']:3d}  r {t['r']:3d} ({100 * t['r'] / max(t['T-'], 1):5.1f} % of T-) | "
          f"K+ {t['K+']:3d}  b {t['b']:3d} ({100 * t['b'] / max(t['K+'], 1):4.1f} % of K+) | net {t['r'] - t['b']:+d}  sign test p {pv:.3f}")
a_, s_ = tot["ALL"], tot["SPARSE35"]
if a_["T-"] < 10:
    verdict = "VOID (|T-| < 10)"
elif a_["r"] > a_["b"] and s_["r"] >= s_["b"]:
    verdict = "SUPPORTED"
elif a_["r"] <= a_["b"]:
    verdict = "NOT SUPPORTED"
else:
    verdict = "MIXED"
print(f"\nVERDICT by the declared rule: {verdict}")

print("\nSECONDARY (i) mechanism: the human's pairwise decisions, read off each decoder")
for (group, dname), c in sorted(mech.items()):
    print(f"   {group:8s} {dname:5s} joins made {c['join_made']:2d} / {c['join_pairs']:2d}   "
          f"splits made {c['split_made']:2d} / {c['split_pairs']:2d} (a piece absent: {c['split_piece_missing']})")
print("\nSECONDARY (ii)-(iv) declared-metric F1 against the whole annotation (biased towards the seed decoder)")
for key in ("ALL", "SPARSE35", "v5H"):
    t = tot[key]
    ks = sorted({k[:-3] for k in t if k.endswith("_tp")})
    print(f"   {key:9s} " + "   ".join(f"{k} {f1(t, k):.3f} ({t[k + '_tp']}/{t[k + '_fp']}/{t[k + '_fn']})" for k in ks))
json.dump({"rows": rows, "totals": {k: dict(v) for k, v in tot.items()},
           "mechanism": {f"{g}|{d}": dict(c) for (g, d), c in mech.items()}, "verdict": verdict},
          open(os.path.join(root, "replay", "replay_read.json"), "w"), indent=1)
