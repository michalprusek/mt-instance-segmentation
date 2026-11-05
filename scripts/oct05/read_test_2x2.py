#!/usr/bin/env python3
"""The ONE TEST read (ii), declared docs/TODO.md 2026-10-06 17:20: 2x2 {SPARSE35 recipe, NMIX} x {A, L-hand}, each cell at its
own VAL cut. Pooled micro per seed, paired cluster bootstrap (roi303 v2: fields; htw: frames), junction identity where known."""
import sys, json, glob, numpy as np
sys.path.insert(0, "scripts")
from step1_read import rows_by_thr, source_of
R = "/disk2/prusek/mt_work/runs/"; T = R + "oct05/test2x2/"; gid = json.load(open(R + "bench_v2/groups.json"))["gid"]
f1 = lambda s: 2 * s[0] / max(2 * s[0] + s[1] + s[2], 1)
def ev(path):                      # eval_v5 json, one threshold
    rb = rows_by_thr(json.load(open(path)), None); assert len(rb) == 1
    return {f["name"]: np.array((f["tp"], f["fp"], f["fn"], 0, 0), float) for f in list(rb.values())[0]}
def gr(path, key, htw=False):      # a_grid json
    return {n: np.array((r[key]["tp"], r[key]["fp"], r[key]["fn"], r[key]["jn"], r[key]["jp"]), float)
            for n, _, r in json.load(open(path))["rows"] if not htw or source_of(n) == "htw"}
BASE = ("SPARSE35", "S35_SEED2", "S35_SEED3"); NM = ("NMIX", "NMIX_S2", "NMIX_S3"); HC = dict(zip(BASE + NM, ("0.995", "0.99", "0.99", "0.995", "0.995", "0.995")))
C = {"roi303 v2 TEST": {}, "htw TEST": {}}
for m in BASE: C["roi303 v2 TEST"][m, "A"] = ev(R + f"oct05/testv2/TEST_roiv2_GAUSS_{m}_der_t0.99.json")
for m in NM[:2]: C["roi303 v2 TEST"][m, "A"] = ev(T + f"A_roiv2TEST_{m}_t0.99.json")
C["roi303 v2 TEST"]["NMIX_S3", "A"] = gr(T + "AL_roiv2TEST_NMIX_S3_t0.995.json", "A")
for m in BASE + NM[:2]: C["roi303 v2 TEST"][m, "L"] = gr(T + f"L_roiv2TEST_{m}_t0.995.json", "b60_t0")
C["roi303 v2 TEST"]["NMIX_S3", "L"] = gr(T + "AL_roiv2TEST_NMIX_S3_t0.995.json", "b60_t0")
for m in BASE + NM:
    f = T + f"AL_hbTEST_{m}_t{HC[m]}.json"; C["htw TEST"][m, "A"] = gr(f, "A", True); C["htw TEST"][m, "L"] = gr(f, "b60_t0", True)
def agg(X, names): return sum(X[n] for n in names)
def boot(Xs, Ys, nb=5000):         # lists of per-seed dicts; statistic = mean over seeds of (F1x - F1y); clusters resampled jointly
    names = sorted(set.intersection(*[set(d) for d in Xs + Ys])); keys = sorted({gid.get(n, n) for n in names}); ix = {k: i for i, k in enumerate(keys)}
    def S(d):
        a = np.zeros((len(keys), 5))
        for n in names: a[ix[gid.get(n, n)]] += d[n]
        return a
    SX = [S(d) for d in Xs]; SY = [S(d) for d in Ys]; rng = np.random.default_rng(0); o = []
    st = lambda s: float(np.mean([f1(x[s].sum(0)) - f1(y[s].sum(0)) for x, y in zip(SX, SY)]))
    al = np.arange(len(keys)); d = st(al)
    for _ in range(nb): o.append(st(rng.integers(0, len(keys), len(keys))))
    lo, hi = np.percentile(o, [2.5, 97.5]); return d, lo, hi, len(names), len(keys)
for blk, c in C.items():
    print("==", blk)
    for grp, tag in ((BASE, "base"), (NM, "NMIX")):
        for i in "AL":
            v = []
            for m in grp:
                t = agg(c[m, i], c[m, i]); v.append(f1(t))
                print(f"  {tag:4s} {i} {m:10s} F1 {f1(t):.4f} tp/fp/fn {int(t[0])}/{int(t[1])}/{int(t[2])} P {t[0]/max(t[0]+t[1],1):.3f} R {t[0]/max(t[0]+t[2],1):.3f}" + (f" ident {t[4]/max(t[3],1):.3f} ({int(t[3])})" if t[3] else ""))
            print(f"  {tag:4s} {i} MEAN {np.mean(v):.4f}  sd {np.std(v, ddof=1):.4f}")
    def show(name, Xs, Ys):
        d, lo, hi, n, k = boot(Xs, Ys); print(f"  {name:34s} {d:+.4f} [{lo:+.4f},{hi:+.4f}] n {n} clusters {k} {'EXCL 0' if lo > 0 or hi < 0 else 'incl 0'}")
    for j, (b, n) in enumerate(zip(BASE, NM)):
        show(f"L-A base        seed {j+1}", [c[b, "L"]], [c[b, "A"]]); show(f"L-A NMIX        seed {j+1}", [c[n, "L"]], [c[n, "A"]])
        show(f"NMIX-base | A   seed {j+1}", [c[n, "A"]], [c[b, "A"]]); show(f"NMIX-base | L   seed {j+1}", [c[n, "L"]], [c[b, "L"]])
        show(f"NMIX+L - base+A seed {j+1}", [c[n, "L"]], [c[b, "A"]])
    show("L-A base        3-seed mean", [c[b, "L"] for b in BASE], [c[b, "A"] for b in BASE]); show("L-A NMIX        3-seed mean", [c[n, "L"] for n in NM], [c[n, "A"] for n in NM])
    show("NMIX-base | A   3-seed mean", [c[n, "A"] for n in NM], [c[b, "A"] for b in BASE]); show("NMIX-base | L   3-seed mean", [c[n, "L"] for n in NM], [c[b, "L"] for b in BASE])
    show("NMIX+L - base+A 3-seed mean", [c[n, "L"] for n in NM], [c[b, "A"] for b in BASE])
