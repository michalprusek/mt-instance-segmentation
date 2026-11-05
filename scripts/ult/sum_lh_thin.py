#!/usr/bin/env python3
"""Summary of anat_lh.py for V16 + L+H (docs/TODO.md 2026-11-02 07:00). Gate: tp / fp / fn = the fold-in jsons."""
import json, sys, collections, numpy as np
sys.path.insert(0, "scripts")
from step1_read import source_of
U = "/data2/prusek/mt_work/runs/ult/"; KH = "bridge0.98_22.5"; MS = [f"THIN_S{k}" for k in (3, 4, 5, 6)]
FPC = ["off_other", "off_border", "overshoot", "fragment", "frag_overshoot", "gap_bridge", "junction"]; FNC = ["covered_split", "partial", "missed"]
f1 = lambda tp, fp, fn: 2 * tp / max(2 * tp + fp + fn, 1); LB = [(0, 10), (10, 12), (12, 20), (20, 35), (35, 60), (60, 120), (120, 1e9)]
cb = lambda c: "isolated" if c == 0 else ("touching" if c < 0.3 else "alongside"); ok = True
for blk, ff in (("hb", "hball"), ("roiv2", "roiv2")):
    print(f"\n######## {blk}"); S = {}
    for m in MS:
        A = json.load(open(f"{U}anat/lh_{blk}_{m}.json")); at = np.array([A["total"][k] for k in ("tp", "fp", "fn")])
        J = json.load(open(f"{U}fold/{ff}_{m}.json")); lt = np.zeros(3)
        for n, _, r in J["rows"]:
            if blk == "hb" and source_of(n) != "htw": continue
            q = r[KH]; lt += (q["tp"], q["fp"], q["fn"])
        g = bool((lt == at).all()); ok &= g; rows = [r for _, r in A["rows"]]
        S[m] = dict(t=at, fp={c: sum(r["fp_classes"][c] for r in rows) for c in FPC}, fpl={c: sum(l for r in rows for k, l in r["fps"] if k == c) for c in FPC},
                    G=[x for r in rows for x in r["gts"]], B=[x for r in rows for x in r["breaks"]], W=[x for r in rows for x in r["gtwhy"]])
        print(f"  {m} anat {at.astype(int).tolist()} fold {lt.astype(int).tolist()} {'PASS' if g else 'FAIL'}  F1 {f1(*at):.4f} P {at[0] / (at[0] + at[1]):.3f} R {at[0] / (at[0] + at[2]):.3f}")
    mean = lambda f: float(np.mean([f(S[m]) for m in MS]))
    print("  FP classes: " + "  ".join(f"{c} {mean(lambda s: s['fp'][c]):.0f} ({mean(lambda s: s['fp'][c] / s['t'][1]):.0%}, +{mean(lambda s: f1(s['t'][0], s['t'][1] - s['fp'][c], s['t'][2]) - f1(*s['t'])):.3f})" for c in FPC))
    print("  FN classes: " + "  ".join(f"{c} {mean(lambda s: sum(x[0] == c for x in s['G'])):.0f} ({mean(lambda s: sum(x[0] == c for x in s['G']) / s['t'][2]):.0%}, +{mean(lambda s: f1(s['t'][0] + sum(x[0] == c for x in s['G']), s['t'][1], s['t'][2] - sum(x[0] == c for x in s['G'])) - f1(*s['t'])):.3f}, map-side {mean(lambda s: np.mean([x[3] < 0.8 for x in s['G'] if x[0] == c] or [np.nan])):.2f})" for c in FNC))
    print("  recall by length: " + "  ".join(f"{lo:.0f}-{hi if hi < 1e8 else 'inf'}: {mean(lambda s: np.mean([x[0] == 'tp' for x in s['G'] if lo <= x[1] < hi])):.3f} (n {sum(lo <= x[1] < hi for x in S[MS[0]]['G'])})" for lo, hi in LB))
    print("  recall by contact: " + "  ".join(f"{c}: {mean(lambda s: np.mean([x[0] == 'tp' for x in s['G'] if cb(x[2]) == c])):.3f} (n {sum(cb(x[2]) == c for x in S[MS[0]]['G'])})" for c in ("isolated", "touching", "alongside")))
    print(f"  all FN map-side share: {mean(lambda s: np.mean([x[3] < 0.8 for x in s['G'] if x[0] != 'tp'])):.2f}")
    w = collections.Counter(x for m in MS for x in S[m]["W"]); tot = sum(w.values())
    print("  covered_split GT, why (share over models, total %d = %.0f / model): " % (tot, tot / 4) + "  ".join(f"{k} {v / tot:.0%}" for k, v in w.most_common()))
    br = [x for m in MS for x in S[m]["B"]]; c = collections.Counter(x[0] for x in br); n = max(len(br), 1)
    print("  BREAKS in split GT (%.0f / model): " % (len(br) / 4) + "  ".join(f"{k} {v / n:.0%}" for k, v in c.most_common()))
    print("  at a crossing: %.0f%%   | median gap: %.1f px | by reason (at-crossing share, median gap, median score): " % (100 * np.mean([x[1] for x in br]), np.median([x[2] for x in br])) + "  ".join(f"{k}: {np.mean([x[1] for x in br if x[0] == k]):.0%}, {np.median([x[2] for x in br if x[0] == k]):.1f}, {np.median([x[3] for x in br if x[0] == k]):+.1f}" for k, _ in c.most_common()))
print("\nGATE", "PASS" if ok else "FAIL"); print("SUM_LH_DONE")
