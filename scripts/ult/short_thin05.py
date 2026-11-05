#!/usr/bin/env python3
"""Report only (docs/TODO.md 2026-11-02 22:55): recall by native GT length and FP, same bins and source as read_dose.py, for AA / DEB / DOSE / THIN05 at the own L cut. Moves no verdict."""
import json, os, numpy as np
U = "/data2/prusek/mt_work/runs/ult/"; K = (3, 4, 5, 6); AR = {n: [f"{n}_S{k}" for k in K] for n in ("AA", "DEB", "DOSE", "THIN05", "THIN")}
def rows(m, b): return json.load(open(f"{U}anat/{b}_{m}.json"))
have = lambda n, b: all(os.path.exists(f"{U}anat/{b}_{m}.json") for m in AR[n])
def rec(n, b, lo, hi):
    x = [g[0] == "tp" for m in AR[n] for _, r in rows(m, b)["rows"] for g in r["gts"] if lo <= g[1] < hi]; return float(np.mean(x)), len(x) // 4
for b in ("hb", "roiv2"):
    N = [n for n in AR if have(n, b)]; print(f"== {b}: " + " / ".join(N))
    for lo, hi in ((0, 12), (12, 20), (0, 20), (20, 35), (35, 1e9)):
        s = {n: rec(n, b, lo, hi) for n in N}; print(f"   {lo:.0f}-{hi if hi < 1e8 else 'inf'}: " + " / ".join(f"{s[n][0]:.3f}" for n in N) + f"  (n {s[N[0]][1]})")
    print("   FP mean over runs: " + " / ".join(f"{np.mean([rows(m, b)['total']['fp'] for m in AR[n]]):.0f}" for n in N))
    print("   per run < 20 px, THIN05: " + " ".join(f"{np.mean([g[0] == 'tp' for _, r in rows(m, b)['rows'] for g in r['gts'] if g[1] < 20]):.3f}" for m in AR["THIN05"]) + "  DEB: " + " ".join(f"{np.mean([g[0] == 'tp' for _, r in rows(m, b)['rows'] for g in r['gts'] if g[1] < 20]):.3f}" for m in AR["DEB"]))
print("SHORT_DONE")
