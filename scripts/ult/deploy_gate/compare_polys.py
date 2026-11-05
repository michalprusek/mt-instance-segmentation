#!/usr/bin/env python3
"""Deployment gate: two polyline sets per frame must be the same, same order. usage: compare_polys.py A.json B.json [TOL_PX]. Exit 0 only if identical within TOL_PX (default 0)."""
import json, sys
import numpy as np
A, B = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])); tol = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
assert set(A) == set(B), (len(A), len(B)); bad = []; worst = 0.0; npoly = 0; ncount = 0
for n in sorted(A):
    a, b = A[n], B[n]; npoly += len(a)
    if len(a) != len(b): ncount += 1; bad.append((n, f"count {len(a)} vs {len(b)}")); continue
    for i, (p, q) in enumerate(zip(a, b)):
        p, q = np.asarray(p), np.asarray(q)
        if p.shape != q.shape: bad.append((n, f"poly {i} shape {p.shape} vs {q.shape}")); break
        d = float(np.abs(p - q).max()); worst = max(worst, d)
        if d > tol: bad.append((n, f"poly {i} max |d| {d:.4f}")); break
print(f"{len(A)} frames, {npoly} polylines; frames differing {len(bad)} (count differs in {ncount}); max |d| over matched {worst:.6f}")
for b in bad[:12]: print("  ", b)
print("IDENTICAL" if not bad else "DIFFERENT"); sys.exit(0 if not bad else 1)
