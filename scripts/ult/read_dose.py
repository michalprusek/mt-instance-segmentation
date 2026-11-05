#!/usr/bin/env python3
"""ARM 64 repair rule (docs/TODO.md 2026-11-02 14:30). ADOPT iff, four pairs vs DEB_S3..S6 at L+H b80 (from read80_DOSE_S3_DEB_S3.json): htw and roi303 each >= -0.005, synth >= -0.010; AND recall of GT < 20 native px
(anat_l.py, L at the own cut, pooled over four runs) on htw and on roi303 each >= the AA_S3..S6 value - 0.02; AND htw FP (same source, mean over four runs) <= the midpoint of AA_S3..S6 and DEB_S3..S6. Report: dose-response 0 / 0.5 / 1."""
import json, numpy as np
U = "/data2/prusek/mt_work/runs/ult/"; K = (3, 4, 5, 6); AR = {n: [f"{n}_S{k}" for k in K] for n in ("AA", "DOSE", "DEB")}
def rows(m, b): return json.load(open(f"{U}anat/{b}_{m}.json"))
def short(n, b, lo=0, hi=20):
    x = [g[0] == "tp" for m in AR[n] for _, r in rows(m, b)["rows"] for g in r["gts"] if lo <= g[1] < hi]; return float(np.mean(x)), len(x) // 4
def fp(n, b): return float(np.mean([rows(m, b)["total"]["fp"] for m in AR[n]]))
R = json.load(open(U + "read80_DOSE_S3_DEB_S3.json")); ok = R["hb"][0] >= -0.005 and R["roiv2"][0] >= -0.005 and R["syn96b"][0] >= -0.010
print(f"== F1 at L+H b80, DOSE vs DEB: htw {R['hb'][0]:+.4f}, roi303 {R['roiv2'][0]:+.4f}, synth {R['syn96b'][0]:+.4f}, block mean {R['block_mean']:+.4f} -> floors {'PASS' if ok else 'FAIL'}")
print("== recall by native GT length at the own L cut, dose 0 (AA) / 0.5 (DOSE) / 1 (DEB)")
for b in ("hb", "roiv2"):
    for lo, hi in ((0, 12), (12, 20), (0, 20), (20, 35), (35, 1e9)):
        s = {n: short(n, b, lo, hi) for n in AR}; print(f"   {b:5s} {lo:.0f}-{hi if hi < 1e8 else 'inf'}: " + " / ".join(f"{s[n][0]:.3f}" for n in ("AA", "DOSE", "DEB")) + f"  (n {s['AA'][1]})")
    a, d = short("AA", b)[0], short("DOSE", b)[0]; g = d >= a - 0.02; ok &= g; print(f"   {b:5s} short recall {d:.3f} vs AA {a:.3f} - 0.02 -> {'PASS' if g else 'FAIL'}")
f = {n: fp(n, "hb") for n in AR}; mid = (f["AA"] + f["DEB"]) / 2; g = f["DOSE"] <= mid; ok &= g
print(f"== htw FP, mean over four runs: AA {f['AA']:.0f} / DOSE {f['DOSE']:.0f} / DEB {f['DEB']:.0f}; midpoint {mid:.1f} -> {'PASS' if g else 'FAIL'}"); print(f"== roi303 FP: AA {fp('AA', 'roiv2'):.0f} / DOSE {fp('DOSE', 'roiv2'):.0f} / DEB {fp('DEB', 'roiv2'):.0f}")
print("ARM 64 ADOPT" if ok else "ARM 64 NOT ADOPTED")
