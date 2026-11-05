#!/usr/bin/env python3
"""Declared docs/TODO.md 2026-10-06 14:10: L-strip (S_b60_t1.5) vs L-hand (H_b60_t0) on identical graphs, each at its OWN
VAL cut (pooled micro, ties to the lower node), paired cluster bootstrap (roi303 v2: fields; htw: frames); plus identity."""
import sys, json, glob, numpy as np
sys.path.insert(0, "scripts")
from step1_read import source_of
R = "/disk2/prusek/mt_work/runs/"; gid = json.load(open(R + "bench_v2/groups.json"))["gid"]
f1 = lambda s: 2 * s[0] / max(2 * s[0] + s[1] + s[2], 1); KEY = {"H": "H_b60_t0", "S": "S_b60_t1.5"}
def best(blk, m, tag):
    out = None; grid = []
    for f in sorted(glob.glob(f"{R}lpc/ssw/LS_{blk}_{m}_t*.json"), key=lambda f: json.load(open(f))["cut"]):
        d = json.load(open(f)); r = {}
        for name, ngt, res in d["rows"]:
            if blk == "hb" and source_of(name) != "htw": continue
            q = res[KEY[tag]]; r[name] = np.array((q["tp"], q["fp"], q["fn"], q["jn"], q["jp"]), float)
        v = f1(sum(r.values())); grid.append((d["cut"], v))
        if out is None or v > out[2] + 1e-12: out = (d["cut"], r, v)
    return out, grid
def boot(X, Y, nb=5000):
    names = sorted(set(X) & set(Y)); keys = sorted({gid.get(n, n) for n in names}); ix = {k: i for i, k in enumerate(keys)}
    SX = np.zeros((len(keys), 5)); SY = np.zeros((len(keys), 5))
    for n in names: SX[ix[gid.get(n, n)]] += X[n]; SY[ix[gid.get(n, n)]] += Y[n]
    rng = np.random.default_rng(0); o = []
    for _ in range(nb):
        s = rng.integers(0, len(keys), len(keys)); o.append(f1(SX[s].sum(0)) - f1(SY[s].sum(0)))
    return np.percentile(o, [2.5, 97.5]), SX.sum(0), SY.sum(0)
for blk in ("hb", "roiv2"):
    print("==", "htw VAL" if blk == "hb" else "roi303 v2 VAL (gauss)"); TS = np.zeros(5); TH = np.zeros(5)
    for m in ("SPARSE35", "S35_SEED2", "S35_SEED3", "NMIX", "NMIX_S2", "NMIX_S3"):
        (cs, rs, vs), gs = best(blk, m, "S"); (ch, rh, vh), gh = best(blk, m, "H"); (lo, hi), s, h = boot(rs, rh)
        if not m.startswith("NMIX"): TS += s; TH += h
        print(f"  {m:10s} strip {vs:.4f}@{cs} hand {vh:.4f}@{ch}  delta {vs-vh:+.4f} [{lo:+.4f},{hi:+.4f}] {'EXCL 0' if lo > 0 or hi < 0 else 'incl 0'}"
              f"  ident strip {s[4]/max(s[3],1):.4f} hand {h[4]/max(h[3],1):.4f} | strip grid " + " ".join(f"{c}:{v:.4f}" for c, v in gs))
    print(f"  pooled baseline seeds: ident strip {TS[4]/max(TS[3],1):.4f} hand {TH[4]/max(TH[3],1):.4f} ({100*(TS[4]/max(TS[3],1)/max(TH[4]/max(TH[3],1),1e-9)-1):+.1f} % rel)")
