#!/usr/bin/env python3
"""Declared read of ARM 62 (docs/TODO.md 2026-11-01 07:55): compact debris / particles in the generator (V16), seeds
3-6 (DEB_S3..S6) vs the same-host same-seed AA_S3..S6. ADOPT iff, pooled over the four pairs, block mean (htw VAL +
roi303 v2 VAL) / 2 >= +0.010 AND neither real block < -0.005 AND synth 96b >= -0.010. Reported, no line: per-seed deltas,
CIs; run-level Welch vs ALL six AA runs (the 2026-10-31 19:15 proposal); (i) FP at the own cut on the 9 htw frames that
carried >= 3 D tiles in the 2026-10-31 eye pass (amended 07:58, the CBG near-null is the noise reference);
(ii) debris probe (debris-only synthetic frames, fg excess on the footprint);
(iii) ARM 41 guard: htw recall of GT < 20 native px per pair; roi303 fg fraction at the own cut vs the control spread."""
import glob, json, sys, numpy as np
from scipy import stats
sys.path.insert(0, "scripts")
from step1_read import source_of
W = "/data2/prusek/mt_work/"; R = W + "runs/ult/lsw/"; PD = W + "preds/"; U = W + "runs/ult/"
gid = json.load(open(W + "tulen_ckpt/bench_v2/groups.json"))["gid"]
RING_FRAMES = ["htw_006", "htw_024", "htw_036", "htw_038", "htw_046"]   # frames with >= 3 D tiles (23 of the 50)
f1 = lambda s: 2 * s[0] / max(2 * s[0] + s[1] + s[2], 1)
def rows(path, blk):
    d = json.load(open(path)); r = {}
    for name, ngt, res in d["rows"]:
        if blk == "hb" and source_of(name) != "htw": continue
        q = res["b60_t0"]; r[name] = np.array((q["tp"], q["fp"], q["fn"]), float)
    return d["cut"], r
def own(blk, m):
    best = None
    for f in glob.glob(f"{R}L_{blk}_{m}_t*.json"):
        c, r = rows(f, blk); v = f1(sum(r.values()))
        if best is None or v > best[2] + 1e-12 or (abs(v - best[2]) <= 1e-12 and c > best[0]): best = (c, r, v)
    return best
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
PAIRS = [(f"DEB_S{k}", f"AA_S{k}") for k in (3, 4, 5, 6)]; BLK = ("hb", "roiv2", "syn96b")
B = {(blk, m): own(blk, m) for blk in BLK for p in PAIRS for m in p}
miss = [k for k, v in B.items() if v is None]
if miss: print("MISSING", miss); sys.exit(1)
out = {}
print("== own cuts and pooled micro F1 (DEB vs same-seed AA control, kajman)")
for a, c in PAIRS:
    print("   " + "  ".join(f"{blk} {B[blk, a][2]:.4f}@{B[blk, a][0]} vs {B[blk, c][2]:.4f}@{B[blk, c][0]}" for blk in BLK))
print("== per-seed deltas DEB - control")
for blk in BLK:
    ds = [boot([(B[blk, a][1], B[blk, c][1])], blk)[0] for a, c in PAIRS]; out[f"seed_{blk}"] = ds
    print(f"   {blk:6s} " + " ".join(f"{d:+.4f}" for d in ds) + f"   positive {sum(d > 0 for d in ds)}/4")
print("== pool of the four pairs (the declared line)")
F = {}
for blk in BLK:
    F[blk] = boot([(B[blk, a][1], B[blk, c][1]) for a, c in PAIRS], blk); out[f"pool_{blk}"] = F[blk]
    print(f"   {blk:6s} {F[blk][0]:+.4f} [{F[blk][1]:+.4f},{F[blk][2]:+.4f}]")
bm = (F["hb"][0] + F["roiv2"][0]) / 2
adopt = bm >= 0.010 and F["hb"][0] >= -0.005 and F["roiv2"][0] >= -0.005 and F["syn96b"][0] >= -0.010
print("   ADOPT" if adopt else "   NOT ADOPTED", f"(block mean {bm:+.4f}, line +0.010; htw {F['hb'][0]:+.4f}, roi303 {F['roiv2'][0]:+.4f}, floor -0.005; synth {F['syn96b'][0]:+.4f}, floor -0.010)")
out["block_mean"] = bm; out["adopt"] = bool(adopt)
print("== run level (reported, no line): DEB four runs vs ALL six AA runs (AA, AA_S2..S6), Welch")
AA6 = ["AA", "AA_S2", "AA_S3", "AA_S4", "AA_S5", "AA_S6"]
for blk in ("hb", "roiv2"):
    r6 = [own(blk, m)[2] for m in AA6]; r4 = [B[blk, a][2] for a, _ in PAIRS]
    t = stats.ttest_ind(r4, r6, equal_var=False)
    out[f"runlevel_{blk}"] = dict(deb=r4, aa6=r6, diff=float(np.mean(r4) - np.mean(r6)), p=float(t.pvalue))
    print(f"   {blk:6s} DEB  {np.mean(r4):.4f} +- {np.std(r4, ddof=1):.4f}  AA6 {np.mean(r6):.4f} +- {np.std(r6, ddof=1):.4f}  diff {np.mean(r4) - np.mean(r6):+.4f}  Welch p {t.pvalue:.3f}")
print("== (i) FP at the own htw cut on the 5 htw frames with >= 3 D tiles; share of htw FP (CBG near-null range printed below)")
for a, c in PAIRS:
    fa = sum(B["hb", a][1][n + ".h5"][1] for n in RING_FRAMES); fc = sum(B["hb", c][1][n + ".h5"][1] for n in RING_FRAMES)
    oa = sum(v[1] for n, v in B["hb", a][1].items() if n[:-3] not in RING_FRAMES); oc = sum(v[1] for n, v in B["hb", c][1].items() if n[:-3] not in RING_FRAMES)
    print(f"   {a}: debris frames FP {fa:.0f} vs {fc:.0f} ({fa - fc:+.0f})   other htw frames FP {oa:.0f} vs {oc:.0f} ({oa - oc:+.0f})   share {fa / max(fa + oa, 1):.3f} vs {fc / max(fc + oc, 1):.3f} ({fa / max(fa + oa, 1) - fc / max(fc + oc, 1):+.3f})")
    out[f"debfp_{a}"] = [fa, fc, oa, oc]
NULL = []
for k in (3, 4, 5, 6):
    x, y = own("hb", f"CBG_S{k}")[1], own("hb", f"AA_S{k}")[1]
    fx = sum(x[n + ".h5"][1] for n in RING_FRAMES); fy = sum(y[n + ".h5"][1] for n in RING_FRAMES)
    ox = sum(v[1] for n, v in x.items() if n[:-3] not in RING_FRAMES); oy = sum(v[1] for n, v in y.items() if n[:-3] not in RING_FRAMES)
    NULL.append(fx / max(fx + ox, 1) - fy / max(fy + oy, 1))
print("   CBG near-null share deltas " + " ".join(f"{d:+.3f}" for d in NULL) + f"   -> mechanism only if the DEB share falls below {min(NULL):+.3f} in >= 3 of 4 pairs")
print("== (ii) debris probe: fg at 0.995 on debris footprints of debris-only synthetic frames (excess over the bare background)")
for m in [a for a, _ in PAIRS] + [c for _, c in PAIRS]:
    try: d = json.load(open(f"{U}debrisprobe_{m}.json")); print(f"   {m}: fg on debris {d['fg_ring']:.4f}  bare {d['fg_bare']:.4f}  excess {d['fg_excess']:+.4f}"); out[f"probe_{m}"] = d
    except FileNotFoundError: print(f"   {m}: not run")
print("== (iii) ARM 41 guard: htw recall of GT < 20 native px (decomp_aa.py at the own htw cut) and FP at the own cut")
def short(m):
    d = json.load(open(f"{U}decomp/hb_{m}.json")); g = [x for _, r in d["rows"] for x in r["gt"] if x[0] < 20]
    return np.mean([x[1] for x in g])
for a, c in PAIRS:
    try: sa, sc_ = short(a), short(c); sh = f"short recall {sa:.3f} vs {sc_:.3f} ({sa - sc_:+.3f})"
    except FileNotFoundError as e: sh = f"short recall MISSING {e}"
    print(f"   {a}: {sh}   FP htw {sum(v[1] for v in B['hb', a][1].values()):.0f} vs {sum(v[1] for v in B['hb', c][1].values()):.0f}   FP roi303 {sum(v[1] for v in B['roiv2', a][1].values()):.0f} vs {sum(v[1] for v in B['roiv2', c][1].values()):.0f}")
print("== over-firing: roi303 v2 VAL fg fraction at each model's own cut")
def fg(m):
    c = B["roiv2", m][0]; v = []
    for f in sorted(glob.glob(f"{PD}GAUSS_{m}_roiv2_ep40_native/*.npz")):
        pr = np.load(f)["prob"]; pr = pr[0] if pr.ndim == 3 else pr; v.append(float((pr > c).mean()))
    return float(np.mean(v))
G = {m: fg(m) for p in PAIRS for m in p}; out["fg"] = G; lo, hi = min(G[c] for _, c in PAIRS), max(G[c] for _, c in PAIRS)
for a, c in PAIRS:
    print(f"   {a} {G[a]:.5f} vs {c} {G[c]:.5f}", "OVER-FIRING FLAG" if G[a] > hi else "ok", f"(control range {lo:.5f}-{hi:.5f})")
json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in out.items()}, open(U + "read_62.json", "w"), indent=1, default=float)
