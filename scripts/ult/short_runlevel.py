#!/usr/bin/env python3
"""Report only (docs/TODO.md 2026-11-02 23:05): per-run recall of GT < 20 native px at the own L cut, THIN05 vs all six DEB and all six AA (Welch); FP count per run; roi303 FP classes if the anat file has them. Moves no verdict."""
import json, collections, numpy as np
from scipy import stats
U = "/data2/prusek/mt_work/runs/ult/anat/"
AR = {"THIN05": [f"THIN05_S{k}" for k in (3, 4, 5, 6)], "DEB": ["DEB_S1", "DEB_S2"] + [f"DEB_S{k}" for k in (3, 4, 5, 6)], "AA": ["AA", "AA_S2"] + [f"AA_S{k}" for k in (3, 4, 5, 6)]}
def load(m, b): return json.load(open(f"{U}{b}_{m}.json"))
def sr(d, lo=0, hi=20):
    x = [g[0] == "tp" for _, r in d["rows"] for g in r["gts"] if lo <= g[1] < hi]; return float(np.mean(x))
for b in ("hb", "roiv2"):
    D = {n: {m: load(m, b) for m in ms} for n, ms in AR.items()}
    print(f"== {b}")
    for lab, lo, hi in (("< 20", 0, 20), ("20-35", 20, 35), (">= 35", 35, 1e9)):
        V = {n: [sr(d, lo, hi) for d in D[n].values()] for n in AR}
        print(f"   recall {lab}: " + " | ".join(f"{n} " + " ".join(f"{v:.3f}" for v in V[n]) + f" mean {np.mean(V[n]):.3f}" for n in AR))
        for c in ("DEB", "AA"):
            t = stats.ttest_ind(V["THIN05"], V[c], equal_var=False); print(f"      THIN05 - {c}: {np.mean(V['THIN05']) - np.mean(V[c]):+.3f}  Welch p {t.pvalue:.4f}")
    F = {n: [d["total"]["fp"] for d in D[n].values()] for n in AR}; print("   FP: " + " | ".join(f"{n} " + " ".join(map(str, F[n])) + f" mean {np.mean(F[n]):.0f}" for n in AR))
    d0 = D["THIN05"]["THIN05_S3"]; print("   anat keys:", list(d0.keys()), "row keys:", list(d0["rows"][0][1].keys()), "total keys:", list(d0["total"].keys()))
    for n in ("AA", "DEB", "THIN05"):
        ms = [m for m in AR[n] if "_S" in m and m[-1] in "3456"]; C = collections.Counter()
        for m in ms:
            for k, v in D[n][m]["total"].items():
                if k.startswith("fp_") or k in ("off_other", "off_border", "overshoot", "fragment", "frag_overshoot", "gap_bridge", "junction"): C[k] += v
        print(f"   FP classes {n} (mean of S3..S6): " + "  ".join(f"{k} {v / 4:.0f}" for k, v in C.most_common()))
print("RUNLEVEL_DONE")
