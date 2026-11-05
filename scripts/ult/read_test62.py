#!/usr/bin/env python3
"""ARM 62 + L+H TEST read (docs/TODO.md 2026-11-02; third adaptive use of TEST, reported, never used to choose). roi303 v2
TEST (128 fr, field-cluster bootstrap) + htw TEST (33 fr, frame bootstrap), each model at its own VAL L cut. Four cells on
the same frames: V14 + L (AA_S3..S6, already read 10-29), V14 + L+H, V16 + L (DEB_S3..S6), V16 + L+H. Gate: V14 + L here equals
the 10-29 jsons. Primary: candidate vs V14 + L pooled over the four pairs. Run level: candidate runs vs the six AA runs of the
10-29 read (Welch). Confirmed iff block mean >= +0.010."""
import json, sys, numpy as np
from scipy import stats
sys.path.insert(0, "scripts")
from step1_read import source_of
T = "/data2/prusek/mt_work/runs/ult/test/"; N = "/data2/prusek/mt_work/runs/ult/test62/"
CAND = sys.argv[1] if len(sys.argv) > 1 else "H"     # H = V16 + L+H is the candidate, L = V16 + L
gid = json.load(open("/data2/prusek/mt_work/tulen_ckpt/bench_v2/groups.json"))["gid"]
f1 = lambda s: 2 * s[0] / max(2 * s[0] + s[1] + s[2], 1); KH = "bridge0.98_22.5"; BLK = ("roiv2", "hb")
def load(m, blk, k):
    d = json.load(open(f"{N}{blk}_{m}.json")); r = {}; ji = np.zeros(2)
    for name, ngt, res in d["rows"]:
        q = res[k]; r[name] = np.array((q["tp"], q["fp"], q["fn"]), float); ji += (q["jn"], q["jp"])
    return r, ji
def boot(pairs, blk, nb=5000):
    names = sorted(set.intersection(*[set(x) & set(y) for x, y in pairs]))
    if blk == "roiv2": assert all(n in gid for n in names), "frame without field group"
    key = (lambda n: gid[n]) if blk == "roiv2" else (lambda n: n)
    keys = sorted({key(n) for n in names}); ix = {k: i for i, k in enumerate(keys)}
    SX = np.zeros((len(keys), 3)); SY = np.zeros((len(keys), 3))
    for x, y in pairs:
        for n in names: SX[ix[key(n)]] += x[n]; SY[ix[key(n)]] += y[n]
    rng = np.random.default_rng(0); o = []
    for _ in range(nb):
        s = rng.integers(0, len(keys), len(keys)); o.append(f1(SX[s].sum(0)) - f1(SY[s].sum(0)))
    lo, hi = np.percentile(o, [2.5, 97.5]); return f1(SX.sum(0)) - f1(SY.sum(0)), lo, hi, len(names), len(keys)
K = (3, 4, 5, 6); A = [f"AA_S{k}" for k in K]; D_ = [f"DEB_S{k}" for k in K]
cells = {"V14+L": (A, "L"), "V14+L+H": (A, KH), "V16+L": (D_, "L"), "V16+L+H": (D_, KH)}
R = {(c, b): [load(m, b, k) for m in ms] for c, (ms, k) in cells.items() for b in BLK}
print("== gate: V14 + L here vs the 10-29 TEST jsons")
cuts = {l.split()[0]: (l.split()[2], l.split()[3]) for l in open("/data2/prusek/mt_work/runs/ult/test_cuts.txt")}; ok = True
for i, m in enumerate(A):
    for b, c in (("hb", cuts[m][0]), ("roiv2", cuts[m][1])):
        d = json.load(open(f"{T}L_{b}_{m}_t{c}.json")); s = np.zeros(3)
        for name, ngt, res in d["rows"]:
            if b == "hb" and source_of(name) != "htw": continue
            q = res["b60_t0"]; s += (q["tp"], q["fp"], q["fn"])
        here = sum(R["V14+L", b][i][0].values()); g = bool((s == here).all()); ok &= g
        print(f"   {m} {b} cut {c}: 10-29 {s.astype(int).tolist()} here {here.astype(int).tolist()} {'PASS' if g else 'FAIL'}")
if not ok: print("GATE FAILED: read void"); sys.exit(1)
out = {}
print("== per cell: TEST F1 per model | seed mean | pooled P / R | junction identity")
for c in cells:
    for b in BLK:
        fs = [f1(sum(r[0].values())) for r in R[c, b]]; t = sum(sum(r[0].values()) for r in R[c, b]); ji = sum(r[1] for r in R[c, b])
        out[f"{c}_{b}"] = dict(per=fs, mean=float(np.mean(fs)), P=float(t[0] / (t[0] + t[1])), Rc=float(t[0] / (t[0] + t[2])), identity=float(ji[1] / max(ji[0], 1)))
        print(f"   {c:8s} {b:5s} " + " ".join(f"{x:.4f}" for x in fs) + f" | {np.mean(fs):.4f} | P {t[0] / (t[0] + t[1]):.3f} R {t[0] / (t[0] + t[2]):.3f} | id {ji[1] / max(ji[0], 1):.3f} ({ji[0] / 4:.0f})")
print("   frames: roi303", len(R["V14+L", "roiv2"][0][0]), " htw", len(R["V14+L", "hb"][0][0]))
cand = "V16+L+H" if CAND == "H" else "V16+L"
print(f"== PRIMARY: {cand} - V14+L, pooled over the four pairs")
F = {}
for b in BLK:
    p = boot([(x[0], y[0]) for x, y in zip(R[cand, b], R["V14+L", b])], b); F[b] = p; out[f"primary_{b}"] = p[:3]
    ds = [boot([(x[0], y[0])], b)[0] for x, y in zip(R[cand, b], R["V14+L", b])]
    print(f"   {b:5s} {p[0]:+.4f} [{p[1]:+.4f},{p[2]:+.4f}]  frames {p[3]} clusters {p[4]}  per seed " + " ".join(f"{d:+.4f}" for d in ds) + f"  positive {sum(d > 0 for d in ds)}/4")
bm = (F["hb"][0] + F["roiv2"][0]) / 2; out["block_mean"] = bm
print(f"   block mean {bm:+.4f}: " + ("CONFIRMED (>= +0.010)" if bm >= 0.010 else "NOT CONFIRMED on TEST (< +0.010); decision goes to the user"))
print("== decomposition, pooled over the four pairs")
for lab, x, y in (("V16+L - V14+L (generator)", "V16+L", "V14+L"), ("V14+L+H - V14+L (decoder on old maps)", "V14+L+H", "V14+L"), ("V16+L+H - V16+L (decoder on new maps)", "V16+L+H", "V16+L")):
    print("   " + lab + ": " + "  ".join(f"{b} {(q := boot([(u[0], v[0]) for u, v in zip(R[x, b], R[y, b])], b))[0]:+.4f} [{q[1]:+.4f},{q[2]:+.4f}]" for b in BLK))
print("== run level: candidate four runs vs the six AA runs of the 10-29 read (Welch, report only)")
AA6 = {"roiv2": [0.6913, 0.6996, 0.6930, 0.6879, 0.7015, 0.6908], "hb": [0.6861, 0.6741, 0.6927, 0.6736, 0.6859, 0.6703]}
for b in BLK:
    fs = [f1(sum(r[0].values())) for r in R[cand, b]]; t = stats.ttest_ind(fs, AA6[b], equal_var=False); out[f"runlevel_{b}"] = (float(np.mean(fs) - np.mean(AA6[b])), float(t.pvalue))
    print(f"   {b:5s} {np.mean(fs):.4f} vs AA six {np.mean(AA6[b]):.4f}  diff {np.mean(fs) - np.mean(AA6[b]):+.4f}  Welch p {t.pvalue:.3f}")
print("== reference, other host, not comparable to the digit: deployed SPARSE35 + A (3 seeds) roi303 0.637 / htw 0.601")
json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in out.items()}, open("/data2/prusek/mt_work/runs/ult/read_test62.json", "w"), indent=1, default=float)
