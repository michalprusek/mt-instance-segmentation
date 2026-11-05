#!/usr/bin/env python3
"""ARM 65 R1, the MT-34 VAL row under Gaussian fusion (docs/TODO.md 2026-11-04 08:10): TMP_Si (t, t, t) against DEB_Si, L+H b80 at each model's own hb cut, split Alice / new-22. Report, no line. cwd mt34_work."""
import json, sys
import numpy as np
sys.path.insert(0, "scripts")
from step1_read import source_of
U = "/data2/prusek/mt_work/runs/ult/mt34g/"; f1 = lambda s: 2 * s[0] / max(2 * s[0] + s[1] + s[2], 1)
def load(m):
    d = json.load(open(f"{U}hball_{m}.json")); out = {}
    for n, _, r in d["rows"]:
        src = source_of(n)
        if src == "htw": continue
        out[n] = (("alice" if src == "alice" else "new22"), np.array([r["b80"][k] for k in ("tp", "fp", "fn")], float))
    return out, d["cut"]
rng = np.random.default_rng(0)
for part in ("alice", "new22", "all"):
    pairs = []
    for s in range(1, 7):
        a, ca = load(f"TMP_S{s}"); b, cb = load(f"DEB_S{s}")
        names = sorted(n for n in a if part == "all" or a[n][0] == part); pairs.append((np.array([a[n][1] for n in names]), np.array([b[n][1] for n in names])))
    per = [f1(x.sum(0)) - f1(y.sum(0)) for x, y in pairs]; X = sum(x for x, _ in pairs); Y = sum(y for _, y in pairs); k = len(X)
    bs = [f1(X[i].sum(0)) - f1(Y[i].sum(0)) for i in (rng.integers(0, k, k) for _ in range(5000))]
    print(f"MT-34 VAL gauss {part:6s} frames {k:2d}: TMP {f1(X.sum(0)):.4f} DEB {f1(Y.sum(0)):.4f} delta {f1(X.sum(0)) - f1(Y.sum(0)):+.4f} [{np.percentile(bs, 2.5):+.4f},{np.percentile(bs, 97.5):+.4f}]  per seed " + " ".join(f"{d:+.4f}" for d in per))
