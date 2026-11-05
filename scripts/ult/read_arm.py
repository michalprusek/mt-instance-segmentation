#!/usr/bin/env python3
"""Declared read of a retrain arm (docs/TODO.md 2026-11-02 07:15). usage: read_arm.py ARM=CTRL[,ARM=CTRL...] [REF_RUNS]. Decoder L+H (bridge
0.98, R 22.5) primary, L for attribution; each model at its own VAL L cut. ADOPT iff, pooled over the pairs: block mean (htw + roi303 v2) / 2
>= +0.010 AND neither real block < -0.005 AND synth 96b >= -0.010. Reported: per-seed deltas, MT-34 VAL, P / R, identity, run-level Welch of the
arm's runs vs the REF_RUNS (comma list), htw recall by length and FP, roi303 fg at the own cut vs the controls."""
import glob, json, sys, numpy as np
from scipy import stats
sys.path.insert(0, "scripts")
from step1_read import source_of
W = "/data2/prusek/mt_work/"; U = W + "runs/ult/"; KH = "bridge0.98_22.5"
gid = json.load(open(W + "tulen_ckpt/bench_v2/groups.json"))["gid"]
f1 = lambda s: 2 * s[0] / max(2 * s[0] + s[1] + s[2], 1)
PAIRS = [tuple(x.split("=")) for x in sys.argv[1].split(",")]; REF = sys.argv[2].split(",") if len(sys.argv) > 2 else []
def fp(m, blk):
    if m.startswith("AA_S") and m != "AA_S2": return U + {"roiv2": "hyst2/roiv2_", "hb": "hyst2/hb_", "mt34": "hyst2/mt34_", "syn96b": "hyst2/syn96b_"}[blk] + m + ".json"
    return U + "fold/" + {"roiv2": "roiv2_", "hb": "hball_", "mt34": "hball_", "syn96b": "syn96b_"}[blk] + m + ".json"
def load(m, blk, k):
    d = json.load(open(fp(m, blk))); r = {}; ji = np.zeros(2)
    for n, _, res in d["rows"]:
        if blk == "hb" and source_of(n) != "htw": continue
        if blk == "mt34" and source_of(n) == "htw": continue
        q = res[k]; r[n] = np.array((q["tp"], q["fp"], q["fn"]), float); ji += (q["jn"], q["jp"])
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
BLK = ("hb", "roiv2", "mt34", "syn96b"); out = {}
for k, lab in ((KH, "L+H (primary)"), ("L", "L (attribution)")):
    D = {(b, m): load(m, b, k) for b in BLK for p in PAIRS for m in p}
    print(f"== {lab}: arm vs control per pair, F1")
    for a, c in PAIRS:
        print(f"   {a} vs {c}: " + "  ".join(f"{b} {f1(sum(D[b, a][0].values())):.4f} vs {f1(sum(D[b, c][0].values())):.4f}" for b in BLK))
    F = {}
    for b in BLK:
        F[b] = boot([(D[b, a][0], D[b, c][0]) for a, c in PAIRS], b); ds = [boot([(D[b, a][0], D[b, c][0])], b)[0] for a, c in PAIRS]
        ta, tc = sum(sum(D[b, a][0].values()) for a, _ in PAIRS), sum(sum(D[b, c][0].values()) for _, c in PAIRS); ja, jc = sum(D[b, a][1] for a, _ in PAIRS), sum(D[b, c][1] for _, c in PAIRS)
        print(f"   {b:6s} {F[b][0]:+.4f} [{F[b][1]:+.4f},{F[b][2]:+.4f}]  per seed " + " ".join(f"{d:+.4f}" for d in ds) + f"  positive {sum(d > 0 for d in ds)}/{len(ds)} | P {ta[0] / (ta[0] + ta[1]):.3f} vs {tc[0] / (tc[0] + tc[1]):.3f}  R {ta[0] / (ta[0] + ta[2]):.3f} vs {tc[0] / (tc[0] + tc[2]):.3f}  id {ja[1] / max(ja[0], 1):.3f} vs {jc[1] / max(jc[0], 1):.3f}")
    bm = (F["hb"][0] + F["roiv2"][0]) / 2; out[k] = dict(block_mean=bm, **{b: F[b][:3] for b in BLK})
    if k == KH:
        ad = bm >= 0.010 and F["hb"][0] >= -0.005 and F["roiv2"][0] >= -0.005 and F["syn96b"][0] >= -0.010
        print("   ADOPT" if ad else "   NOT ADOPTED", f"(block mean {bm:+.4f}, line +0.010; htw {F['hb'][0]:+.4f}, roi303 {F['roiv2'][0]:+.4f}, floor -0.005; synth {F['syn96b'][0]:+.4f}, floor -0.010)"); out["adopt"] = bool(ad)
        if REF:
            print("== run level, L+H: arm runs vs REF runs, Welch")
            for b in ("hb", "roiv2"):
                x = [f1(sum(load(a, b, KH)[0].values())) for a, _ in PAIRS]; y = [f1(sum(load(m, b, KH)[0].values())) for m in REF]
                t = stats.ttest_ind(x, y, equal_var=False); print(f"   {b:6s} arm {np.mean(x):.4f} +- {np.std(x, ddof=1) if len(x) > 1 else 0:.4f}  ref ({len(y)} runs) {np.mean(y):.4f} +- {np.std(y, ddof=1):.4f}  diff {np.mean(x) - np.mean(y):+.4f}  Welch p {t.pvalue:.3f}")
print("== htw recall by GT length (decomp_aa.py at the own L cut; L decoder) and FP")
def by(m):
    d = json.load(open(f"{U}decomp/hb_{m}.json")); g = [x for _, r in d["rows"] for x in r["gt"]]; t = d["total"]
    return [np.mean([x[1] for x in g if lo <= x[0] < hi]) for lo, hi in ((0, 12), (12, 20), (20, 35), (35, 1e9))], t
for a, c in PAIRS:
    try:
        ra, ta = by(a); rc, tc = by(c)
        print(f"   {a} vs {c}: <12 {ra[0]:.3f} vs {rc[0]:.3f}  12-20 {ra[1]:.3f} vs {rc[1]:.3f}  20-35 {ra[2]:.3f} vs {rc[2]:.3f}  >=35 {ra[3]:.3f} vs {rc[3]:.3f} | TP {ta['tp']} vs {tc['tp']}  FP {ta['fp']} vs {tc['fp']}")
    except FileNotFoundError as e: print(f"   {a}: MISSING {e}")
json.dump(out, open(U + "read_" + PAIRS[0][0].split("_")[0] + ".json", "w"), indent=1, default=float)
