#!/usr/bin/env python3
"""Declared read of a retrain arm at the adopted decoder L+H b80 (docs/TODO.md 2026-11-02 09:15, user 09:07). usage: read_arm80.py ARM=CTRL[,ARM=CTRL...] [REF_RUNS]. Same rule as
read_arm.py: ADOPT iff, pooled over the pairs, block mean (htw + roi303 v2) / 2 >= +0.010 AND neither real block < -0.005 AND synth 96b >= -0.010. Arm, control and REF all from
runs/ult/bridge/ key b80, each model at its own VAL L cut. Gate, else VOID: for every model, b60 there == bridge0.98_22.5 of the fold (fold/ or hyst2/), frame by frame, same cut."""
import json, sys, numpy as np
from scipy import stats
sys.path.insert(0, "scripts")
from step1_read import source_of
W = "/data2/prusek/mt_work/"; U = W + "runs/ult/"; KH = "bridge0.98_22.5"; K80 = "b80"
gid = json.load(open(W + "tulen_ckpt/bench_v2/groups.json"))["gid"]
f1 = lambda s: 2 * s[0] / max(2 * s[0] + s[1] + s[2], 1)
PAIRS = [tuple(x.split("=")) for x in sys.argv[1].split(",")]; REF = sys.argv[2].split(",") if len(sys.argv) > 2 else []
FILE = {"roiv2": "roiv2_", "hb": "hball_", "mt34": "hball_", "syn96b": "syn96b_"}
def comparators(m):
    if m.startswith("AA_S") and m != "AA_S2": return [("roiv2_", "hyst2/roiv2_"), ("hball_", "hyst2/hb_"), ("hball_", "hyst2/mt34_"), ("syn96b_", "hyst2/syn96b_")]
    return [("roiv2_", "fold/roiv2_"), ("hball_", "fold/hball_"), ("syn96b_", "fold/syn96b_")]
def gate(m):
    ok = True
    for f, g in comparators(m):
        a = json.load(open(f"{U}{g}{m}.json")); b = json.load(open(f"{U}bridge/{f}{m}.json"))
        ra = {n: r[KH] for n, _, r in a["rows"]}; rb = {n: r["b60"] for n, _, r in b["rows"]}
        same = a["cut"] == b["cut"] and set(ra) <= set(rb) and all(ra[n][q] == rb[n][q] for n in ra for q in ("tp", "fp", "fn"))
        ok &= same; print(f"   {m} {g}: cut {a['cut']} {b['cut']} frames {len(ra)} {'PASS' if same else 'FAIL'}")
    return ok
def load(m, blk):
    d = json.load(open(f"{U}bridge/{FILE[blk]}{m}.json")); r = {}; ji = np.zeros(2)
    for n, _, res in d["rows"]:
        if blk == "hb" and source_of(n) != "htw": continue
        if blk == "mt34" and source_of(n) == "htw": continue
        q = res[K80]; r[n] = np.array((q["tp"], q["fp"], q["fn"]), float); ji += (q["jn"], q["jp"])
    return r, ji
def boot(pairs, blk, nb=5000):
    names = sorted(set.intersection(*[set(x) & set(y) for x, y in pairs]))
    key = (lambda n: gid.get(n, n)) if blk == "roiv2" else (lambda n: n)
    keys = sorted({key(n) for n in names}); ix = {k: i for i, k in enumerate(keys)}
    SX = np.zeros((len(keys), 3)); SY = np.zeros((len(keys), 3))
    for x, y in pairs:
        for n in names: SX[ix[key(n)]] += x[n]; SY[ix[key(n)]] += y[n]
    rng = np.random.default_rng(0); o = []
    for _ in range(nb):
        s = rng.integers(0, len(keys), len(keys)); o.append(f1(SX[s].sum(0)) - f1(SY[s].sum(0)))
    lo, hi = np.percentile(o, [2.5, 97.5]); return f1(SX.sum(0)) - f1(SY.sum(0)), lo, hi
print("== gate: l_bridge b60 == fold L+H")
ms = sorted({m for p in PAIRS for m in p} | set(REF))
if not all([gate(m) for m in ms]): print("GATE FAILED: read void"); sys.exit(1)
BLK = ("hb", "roiv2", "mt34", "syn96b"); D = {(b, m): load(m, b) for b in BLK for m in ms}; F = {}
print("== L+H b80 (primary): arm vs control per pair, F1")
for a, c in PAIRS:
    print(f"   {a} vs {c}: " + "  ".join(f"{b} {f1(sum(D[b, a][0].values())):.4f} vs {f1(sum(D[b, c][0].values())):.4f}" for b in BLK))
for b in BLK:
    F[b] = boot([(D[b, a][0], D[b, c][0]) for a, c in PAIRS], b); ds = [boot([(D[b, a][0], D[b, c][0])], b)[0] for a, c in PAIRS]
    ta, tc = sum(sum(D[b, a][0].values()) for a, _ in PAIRS), sum(sum(D[b, c][0].values()) for _, c in PAIRS); ja, jc = sum(D[b, a][1] for a, _ in PAIRS), sum(D[b, c][1] for _, c in PAIRS)
    print(f"   {b:6s} {F[b][0]:+.4f} [{F[b][1]:+.4f},{F[b][2]:+.4f}]  per seed " + " ".join(f"{d:+.4f}" for d in ds) + f"  positive {sum(d > 0 for d in ds)}/{len(ds)} | P {ta[0] / (ta[0] + ta[1]):.3f} vs {tc[0] / (tc[0] + tc[1]):.3f}  R {ta[0] / (ta[0] + ta[2]):.3f} vs {tc[0] / (tc[0] + tc[2]):.3f}  id {ja[1] / max(ja[0], 1):.3f} vs {jc[1] / max(jc[0], 1):.3f}")
bm = (F["hb"][0] + F["roiv2"][0]) / 2; ad = bm >= 0.010 and F["hb"][0] >= -0.005 and F["roiv2"][0] >= -0.005 and F["syn96b"][0] >= -0.010
print("   ADOPT" if ad else "   NOT ADOPTED", f"(block mean {bm:+.4f}, line +0.010; htw {F['hb'][0]:+.4f}, roi303 {F['roiv2'][0]:+.4f}, floor -0.005; synth {F['syn96b'][0]:+.4f}, floor -0.010)")
if REF:
    print("== run level, L+H b80: arm runs vs REF runs, Welch")
    for b in ("hb", "roiv2"):
        x = [f1(sum(D[b, a][0].values())) for a, _ in PAIRS]; y = [f1(sum(D[b, m][0].values())) for m in REF]
        t = stats.ttest_ind(x, y, equal_var=False); print(f"   {b:6s} arm {np.mean(x):.4f} +- {np.std(x, ddof=1) if len(x) > 1 else 0:.4f}  ref ({len(y)} runs) {np.mean(y):.4f} +- {np.std(y, ddof=1):.4f}  diff {np.mean(x) - np.mean(y):+.4f}  Welch p {t.pvalue:.3f}")
json.dump(dict(block_mean=bm, adopt=bool(ad), **{b: F[b] for b in BLK}), open(f"{U}read80_{PAIRS[0][0]}_{PAIRS[0][1]}.json", "w"), indent=1, default=float)
