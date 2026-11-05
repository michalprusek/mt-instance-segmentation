#!/usr/bin/env python3
"""Summary of anat_l.py (docs/TODO.md 2026-10-31 19:45): gate vs the L json at the same cut, class shares and count-based
prices (mean over models, min-max), FN by length x contact x map / assembly. usage: sum_anat.py"""
import json, sys, numpy as np
sys.path.insert(0, "scripts")
from step1_read import source_of
W = "/data2/prusek/mt_work/runs/ult/"
CUT = {"hb": {"AA_S3": 0.995, "AA_S4": 0.98, "AA_S5": 0.995, "AA_S6": 0.995}, "roiv2": {m: 0.995 for m in ("AA_S3", "AA_S4", "AA_S5", "AA_S6")}}
FPC = ["off_other", "off_border", "overshoot", "fragment", "frag_overshoot", "gap_bridge", "junction"]
FNC = ["covered_split", "partial", "missed"]
f1 = lambda tp, fp, fn: 2 * tp / max(2 * tp + fp + fn, 1)
LB = [(0, 12), (12, 20), (20, 35), (35, 60), (60, 120), (120, 1e9)]
def cbin(c): return "isolated" if c == 0 else ("touching" if c < 0.3 else "alongside")
ok = True
for blk in ("hb", "roiv2"):
    print(f"\n######## {blk} (htw VAL)" if blk == "hb" else "\n######## roiv2 (roi303 v2 VAL)")
    S = {}
    for m, cut in CUT[blk].items():
        A = json.load(open(f"{W}anat/{blk}_{m}.json"))
        L = json.load(open(f"{W}lsw/L_{blk}_{m}_t{cut}.json"))
        lt = np.zeros(3)
        for name, ngt, res in L["rows"]:
            if blk == "hb" and source_of(name) != "htw": continue
            q = res["b60_t0"]; lt += (q["tp"], q["fp"], q["fn"])
        at = np.array([A["total"][k] for k in ("tp", "fp", "fn")])
        g = bool((lt == at).all()); ok &= g
        rows = [r for _, r in A["rows"]]
        fp = {c: sum(r["fp_classes"][c] for r in rows) for c in FPC}
        fpl = {c: sum(l for r in rows for k, l in r["fps"] if k == c) for c in FPC}
        G = [x for r in rows for x in r["gts"]]
        fn = {c: sum(1 for x in G if x[0] == c) for c in FNC}
        S[m] = dict(t=at, fp=fp, fpl=fpl, fn=fn, G=G)
        print(f"  {m} cut {cut} anat tp/fp/fn {at.astype(int).tolist()} L json {lt.astype(int).tolist()} gate {'PASS' if g else 'FAIL'}"
              f"  F1 {f1(*at):.4f}  P {at[0] / (at[0] + at[1]):.3f} R {at[0] / (at[0] + at[2]):.3f}  FN classes sum {sum(fn.values())}")
    ms = list(S)
    def agg(vals): v = np.array(vals, float); return f"{v.mean():7.1f} [{v.min():.0f}-{v.max():.0f}]"
    def aggf(vals): v = np.array(vals, float); return f"{v.mean():+.4f} [{v.min():+.4f},{v.max():+.4f}]"
    print("  FP class      count (mean [min-max])  share   length native px  price if removed (F1 gain)")
    for c in FPC:
        n = [S[m]["fp"][c] for m in ms]; sh = np.mean([S[m]["fp"][c] / max(S[m]["t"][1], 1) for m in ms])
        pr = [f1(S[m]["t"][0], S[m]["t"][1] - S[m]["fp"][c], S[m]["t"][2]) - f1(*S[m]["t"]) for m in ms]
        print(f"  {c:14s} {agg(n)}  {sh:5.1%}  {np.mean([S[m]['fpl'][c] for m in ms]):8.0f}         {aggf(pr)}")
    print("  FN class      count                   share   price if turned TP")
    for c in FNC:
        n = [S[m]["fn"][c] for m in ms]; sh = np.mean([S[m]["fn"][c] / max(sum(S[m]["fn"].values()), 1) for m in ms])
        pr = [f1(S[m]["t"][0] + S[m]["fn"][c], S[m]["t"][1], S[m]["t"][2] - S[m]["fn"][c]) - f1(*S[m]["t"]) for m in ms]
        print(f"  {c:14s} {agg(n)}  {sh:5.1%}  {aggf(pr)}")
    print("  GT recall by native length (mean over models) | n GT | FN that are MAP-side (support < 0.8)")
    for lo, hi in LB:
        rec, n, mapside = [], 0, []
        for m in ms:
            sel = [x for x in S[m]["G"] if lo <= x[1] < hi]; n = len(sel)
            rec.append(np.mean([x[0] == "tp" for x in sel]) if sel else np.nan)
            f = [x for x in sel if x[0] != "tp"]; mapside.append(np.mean([x[3] < 0.8 for x in f]) if f else np.nan)
        print(f"   {lo:>4.0f}-{hi if hi < 1e8 else 'inf':<4}  recall {np.nanmean(rec):.3f} [{np.nanmin(rec):.3f}-{np.nanmax(rec):.3f}]  n {n:4d}  map-side {np.nanmean(mapside):.2f}")
    print("  GT recall by contact with another GT (5 eval px)")
    for cb in ("isolated", "touching", "alongside"):
        rec, n, mapside = [], 0, []
        for m in ms:
            sel = [x for x in S[m]["G"] if cbin(x[2]) == cb]; n = len(sel)
            rec.append(np.mean([x[0] == "tp" for x in sel]) if sel else np.nan)
            f = [x for x in sel if x[0] != "tp"]; mapside.append(np.mean([x[3] < 0.8 for x in f]) if f else np.nan)
        print(f"   {cb:9s}  recall {np.nanmean(rec):.3f} [{np.nanmin(rec):.3f}-{np.nanmax(rec):.3f}]  n {n:4d}  map-side {np.nanmean(mapside):.2f}")
    print("  FN split: map-side (support < 0.8) vs assembly-side, and by FN class (mean over models)")
    for c in FNC:
        a = [np.mean([x[3] < 0.8 for x in S[m]["G"] if x[0] == c]) if any(x[0] == c for x in S[m]["G"]) else np.nan for m in ms]
        print(f"   {c:14s} map-side {np.nanmean(a):.2f}")
    allfn = [np.mean([x[3] < 0.8 for x in S[m]["G"] if x[0] != "tp"]) for m in ms]
    print(f"   all FN         map-side {np.mean(allfn):.2f} [{min(allfn):.2f}-{max(allfn):.2f}]")
    tot_len = [sum(x[1] for x in S[m]["G"]) for m in ms]
    print(f"  GT n {len(S[ms[0]]['G'])}, total native length {tot_len[0]:.0f} px; median GT length {np.median([x[1] for x in S[ms[0]]['G']]):.1f}")
print("\nGATE", "PASS" if ok else "FAIL")
