#!/usr/bin/env python3
"""Selection on SEL / confirmation on CONF for crossing hysteresis v2 (docs/TODO.md 2026-11-01 07:55). Split: roi303 v2
VAL field clusters sorted by frame count (desc, then id), each to the half with fewer frames so far (ties -> SEL).
Selection pooled over AA_S3..S6: admissible = identity and bundle recovery >= 0.98 x L; highest F1; ties within 0.002 ->
bridge before disc, higher t_lo, smaller R; must beat L. Confirmation (CONF): pooled >= +0.010, all four per-model deltas
> 0, identity up; htw VAL and synth 96b pooled >= -0.005 (read from the htw / syn96b files if present).
    sel_hyst2.py RUNDIR"""
import json, sys, collections, numpy as np
D = sys.argv[1]; MS = ["AA_S3", "AA_S4", "AA_S5", "AA_S6"]
gid = json.load(open("/data2/prusek/mt_work/tulen_ckpt/bench_v2/groups.json"))["gid"]
f1 = lambda t: 2 * t[0] / max(2 * t[0] + t[1] + t[2], 1)
R = {m: json.load(open(f"{D}/roiv2_{m}.json")) for m in MS}
names = sorted(n for n, _, _ in R[MS[0]]["rows"]); grid = R[MS[0]]["grid"]
cnt = collections.Counter(gid.get(n, n) for n in names)
SEL, CONF, nS, nC = set(), set(), 0, 0
for g, c in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])):
    if nS <= nC: SEL.add(g); nS += c
    else: CONF.add(g); nC += c
print(f"split: SEL {len(SEL)} fields / {nS} frames, CONF {len(CONF)} fields / {nC} frames")
def tot(m, k, half):
    t = np.zeros(7)
    for n, _, res in R[m]["rows"]:
        if (gid.get(n, n) in half):
            q = res[k]; t += (q["tp"], q["fp"], q["fn"], q["jn"], q["jp"], q["npar"], q["bund"] or 0)
    return t
def pooled(k, half, ms=MS): return sum(tot(m, k, half) for m in ms)
def summ(t): return f1(t[:3]), t[4] / max(t[3], 1), t[6] / max(t[5], 1)
L = summ(pooled("L", SEL)); print(f"SEL  L: F1 {L[0]:.4f} identity {L[1]:.4f} bundle {L[2]:.4f}")
cand = []
for k in grid:
    if k == "L": continue
    F, I, B = summ(pooled(k, SEL)); adm = I >= 0.98 * L[1] and B >= 0.98 * L[2]
    print(f"SEL  {k:16s} F1 {F:.4f} ({F - L[0]:+.4f}) identity {I:.4f} bundle {B:.4f} {'admissible' if adm else 'NOT admissible'}")
    if adm: cand.append((k, F))
def order(k):
    v, rest = (k[:6], k[6:]) if k.startswith("bridge") else (k[:4], k[4:]); t, r = rest.split("_")
    return (0 if v == "bridge" else 1, -float(t), float(r))
best = max(c[1] for c in cand) if cand else None
if best is None or best <= L[0]: print("SELECTION: nothing admissible beats L on SEL -> STOP"); sys.exit(0)
tied = sorted([c for c in cand if c[1] >= best - 0.002], key=lambda c: order(c[0])); pick = tied[0][0]
print(f"SELECTION: {pick} (best SEL F1 {best:.4f}; ties within 0.002 resolved by the declared order)")
dl = [f1(tot(m, pick, CONF)[:3]) - f1(tot(m, "L", CONF)[:3]) for m in MS]
Lc, Pc = summ(pooled("L", CONF)), summ(pooled(pick, CONF))
print(f"CONF L F1 {Lc[0]:.4f} identity {Lc[1]:.4f} | {pick} F1 {Pc[0]:.4f} ({Pc[0] - Lc[0]:+.4f}) identity {Pc[1]:.4f} | per model " + " ".join(f"{d:+.4f}" for d in dl))
ok = Pc[0] - Lc[0] >= 0.010 and all(d > 0 for d in dl) and Pc[1] > Lc[1]
for blk in ("hb", "syn96b"):
    try:
        T = {m: json.load(open(f"{D}/{blk}_{m}.json")) for m in MS}
        s = lambda k: sum(np.array([(r[2][k]["tp"], r[2][k]["fp"], r[2][k]["fn"]) for r in T[m]["rows"]]).sum(0) for m in MS)
        d = f1(s(pick)) - f1(s("L")); print(f"{blk}: pooled {f1(s('L')):.4f} -> {f1(s(pick)):.4f} ({d:+.4f})"); ok = ok and d >= -0.005
    except (FileNotFoundError, KeyError) as e: print(f"{blk}: MISSING {e}"); ok = False
print("ADOPT" if ok else "NOT ADOPTED")
