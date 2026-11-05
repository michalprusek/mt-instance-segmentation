#!/usr/bin/env python3
"""D2(e) + D4(d): frame and field-cluster bootstrap of paired arm deltas on roi303 VAL (own thresholds), and the
CI half-widths of the already-reported SPARSE35 numbers. Cached per-frame rows only."""
import sys, json, glob, os, numpy as np
sys.path.insert(0, "scripts")
from step1_read import load, rows_by_thr, choose_thr
R = "/disk2/prusek/mt_work/runs/"; G = json.load(open(R + "bench_v2/groups.json")); gid = G["gid"]; EXC = set(G["excluded"])
def rows(dirn, tag, blk="roiVAL", src=None):
    s = load(f"{R}{dirn}/sweep_{blk}_{tag}_der.json")
    if s is None: return None, None
    thr, _ = choose_thr(s, src, 10)
    return thr, {f["name"]: (f["tp"], f["fp"], f["fn"]) for f in rows_by_thr(s, src)[thr]}
def f1(s): return 2 * s[0] / max(2 * s[0] + s[1] + s[2], 1)
def boot(A, B, names, cluster, nb=5000, seed=0):
    rng = np.random.default_rng(seed)
    keys = sorted({(gid.get(n, n) if cluster else n) for n in names})
    idx = {k: i for i, k in enumerate(keys)}
    SA = np.zeros((len(keys), 3)); SB = np.zeros((len(keys), 3))
    for n in names:
        k = idx[gid.get(n, n) if cluster else n]; SA[k] += A[n]
        if B is not None: SB[k] += B[n]
    out = []
    for _ in range(nb):
        s = rng.integers(0, len(keys), len(keys)); a = SA[s].sum(0)
        out.append(f1(a) - f1(SB[s].sum(0)) if B is not None else f1(a))
    lo, hi = np.percentile(out, [2.5, 97.5]); return lo, hi, len(keys)
bt, base = rows("step1", "SPARSE35")
print("== roi303 VAL, paired vs SPARSE35 (own thresholds); 95 % CI of the pooled-micro delta: frame bootstrap | field-cluster bootstrap")
for dirn, tag in [("step1", "SEED2"), ("step1", "SEED3"), ("fov50read", "FOV50"), ("AMODALread", "AMODAL"), ("PAINTAAread", "PAINTAA"),
                  ("EUCLIDread", "EUCLID"), ("ROTAUGread", "ROTAUG"), ("PW2Sread", "PW2S"), ("PW4Sread", "PW4S"), ("V5Hread", "V5H")] + \
                 [("oct05/sw", a) for a in sys.argv[1:]]:
    t, r = rows(dirn, tag)
    if r is None: print(f"{tag:10s} missing"); continue
    names = [n for n in base if n in r and n not in EXC]
    d = f1(np.sum([r[n] for n in names], 0)) - f1(np.sum([base[n] for n in names], 0))
    fl, fh, _ = boot(r, base, names, False); cl, ch, k = boot(r, base, names, True)
    print(f"{tag:10s} thr {t:<6} n {len(names)} delta {d:+.4f}  frame [{fl:+.4f},{fh:+.4f}]  cluster({k}) [{cl:+.4f},{ch:+.4f}]  {'EXCLUDES 0' if cl > 0 or ch < 0 else 'includes 0'}")
print("== CI of reported SPARSE35-recipe numbers (no new read; cached per-frame rows at the VAL-chosen threshold)")
for blk, src, tags in (("roi", None, ("SPARSE35", "SEED2", "SEED3")), ("hb", "htw", ("SPARSE35", "SEED2", "SEED3"))):
    for tag in tags:
        sv = load(f"{R}step1/sweep_{blk}VAL_{tag}_der.json"); st = load(f"{R}step1/sweep_{blk}TEST_{tag}_der.json")
        if sv is None or st is None: print(blk, tag, "missing"); continue
        thr, _ = choose_thr(sv, src, 10)
        for nm, s in (("VAL", sv), ("TEST", st)):
            rb = rows_by_thr(s, src)
            if thr not in rb: print(blk, tag, nm, "thr", thr, "not in grid"); continue
            r = {f["name"]: (f["tp"], f["fp"], f["fn"]) for f in rb[thr]}; names = list(r)
            m = f1(np.sum([r[n] for n in names], 0)); fl, fh, _ = boot(r, None, names, False); cl, ch, k = boot(r, None, names, True)
            print(f"{blk:3s} {tag:8s} {nm:4s} thr {thr} n {len(names)} micro {m:.4f}  frame +-{(fh-fl)/2:.4f}  cluster({k}) +-{(ch-cl)/2:.4f}")
