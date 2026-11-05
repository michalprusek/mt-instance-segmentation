#!/usr/bin/env python3
"""Fold-in read (docs/TODO.md 2026-11-02 06:45): V16 + L+H vs V16 + L, V14 + L, V14 + L+H on VAL + synth, four models each,
same-host, each at its own L cut. Promotion rule: vs V16 + L, pooled: block mean >= 0, neither real block < -0.005, synth >=
-0.005, roi303 identity up. Gate: L totals = the L jsons."""
import json, sys, collections, numpy as np
sys.path.insert(0, "scripts")
from step1_read import source_of
U = "/data2/prusek/mt_work/runs/ult/"; KH = "bridge0.98_22.5"; K = (3, 4, 5, 6)
gid = json.load(open("/data2/prusek/mt_work/tulen_ckpt/bench_v2/groups.json"))["gid"]
f1 = lambda t: 2 * t[0] / max(2 * t[0] + t[1] + t[2], 1)
def load(path, keep):
    d = json.load(open(path)); out = {}
    for k in ("L", KH):
        out[k] = {n: np.array((r[k]["tp"], r[k]["fp"], r[k]["fn"], r[k]["jn"], r[k]["jp"]), float) for n, _, r in d["rows"] if keep(n)}
    return out
htw = lambda n: source_of(n) == "htw"; mt = lambda n: source_of(n) != "htw"; al = lambda n: True
ARM = {"V14": [f"AA_S{k}" for k in K], "V16": [f"DEB_S{k}" for k in K]}
P = {"V14": U + "hyst2/{}_{}.json", "V16": U + "fold/{}_{}.json"}
def files(arm, blk, m):
    if arm == "V16" and blk in ("hb", "mt34"): return U + f"fold/hball_{m}.json", (htw if blk == "hb" else mt)
    if arm == "V14" and blk == "mt34": return U + f"hyst2/mt34_{m}.json", mt
    if arm == "V14" and blk == "hb": return U + f"hyst2/hb_{m}.json", htw
    return P[arm].format(blk, m), al
D = {(a, b, m): load(*files(a, b, m)) for a in ARM for b in ("roiv2", "hb", "mt34", "syn96b") for m in ARM[a]}
cutL = {"hb": {"DEB_S3": "0.995", "DEB_S4": "0.99", "DEB_S5": "0.99", "DEB_S6": "0.995"}, "roiv2": {f"DEB_S{k}": "0.995" for k in K}, "syn96b": {"DEB_S3": "0.99", "DEB_S4": "0.99", "DEB_S5": "0.99", "DEB_S6": "0.98"}}
ok = True
for b in cutL:
    for m, c in cutL[b].items():
        d = json.load(open(f"{U}lsw/L_{b}_{m}_t{c}.json")); s = np.zeros(3)
        for n, _, r in d["rows"]:
            if b == "hb" and source_of(n) != "htw": continue
            q = r["b60_t0"]; s += (q["tp"], q["fp"], q["fn"])
        h = sum(D["V16", b, m]["L"].values())[:3]; g = bool((s == h).all()); ok &= g
        if not g: print("GATE FAIL", b, m, s, h)
print("gate L = L json on DEB_S3..S6:", "PASS" if ok else "FAIL")
SEL, CONF, nS, nC = set(), set(), 0, 0
names = sorted(D["V14", "roiv2", "AA_S3"]["L"]); cnt = collections.Counter(gid.get(n, n) for n in names)
for g, c in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])):
    if nS <= nC: SEL.add(g); nS += c
    else: CONF.add(g); nC += c
def pool(a, b, k, keep=lambda n: True): return sum(v for m in ARM[a] for n, v in D[a, b, m][k].items() if keep(n))
def perm(a, b, k): return [f1(sum(D[a, b, m][k].values())) for m in ARM[a]]
print("cell             roi303  P     R     id    | htw    P     R    | MT-34 | synth | per-model roi303 / htw")
CELL = {"V14 + L": ("V14", "L"), "V14 + L+H": ("V14", KH), "V16 + L": ("V16", "L"), "V16 + L+H": ("V16", KH)}; S = {}
for lab, (a, k) in CELL.items():
    r, h, mm, sy = pool(a, "roiv2", k), pool(a, "hb", k), pool(a, "mt34", k), pool(a, "syn96b", k); S[lab] = (f1(r), f1(h), f1(mm), f1(sy), r[4] / max(r[3], 1))
    print(f"{lab:14s} {f1(r):.4f} {r[0] / (r[0] + r[1]):.3f} {r[0] / (r[0] + r[2]):.3f} {r[4] / max(r[3], 1):.3f} | {f1(h):.4f} {h[0] / (h[0] + h[1]):.3f} {h[0] / (h[0] + h[2]):.3f} | {f1(mm):.4f} | {f1(sy):.4f} | " + " ".join(f"{x:.4f}" for x in perm(a, "roiv2", k)) + " / " + " ".join(f"{x:.4f}" for x in perm(a, "hb", k)))
a, b = S["V16 + L+H"], S["V16 + L"]; dr, dh, ds = a[0] - b[0], a[1] - b[1], a[3] - b[3]; bm = (dr + dh) / 2
print(f"\nH on the V16 maps: roi303 {dr:+.4f} htw {dh:+.4f} MT-34 {a[2] - b[2]:+.4f} synth {ds:+.4f} block mean {bm:+.4f} identity {b[4]:.3f} -> {a[4]:.3f}")
for lab, half in (("SEL", SEL), ("CONF", CONF)):
    keep = lambda n: gid.get(n, n) in half
    print(f"   roi303 {lab}: V16+L {f1(pool('V16', 'roiv2', 'L', keep)):.4f} -> V16+L+H {f1(pool('V16', 'roiv2', KH, keep)):.4f}  (V14+L {f1(pool('V14', 'roiv2', 'L', keep)):.4f})")
prom = bm >= 0 and dr >= -0.005 and dh >= -0.005 and ds >= -0.005 and a[4] > b[4]
print("PROMOTED: V16 + L+H is the current best" if prom else "NOT PROMOTED: H does not hold on the V16 maps; candidate stays V16 + L")
print(f"V16 + L+H vs V14 + L: roi303 {a[0] - S['V14 + L'][0]:+.4f} htw {a[1] - S['V14 + L'][1]:+.4f} MT-34 {a[2] - S['V14 + L'][2]:+.4f} synth {a[3] - S['V14 + L'][3]:+.4f}")
print("FOLD_READ_DONE")
