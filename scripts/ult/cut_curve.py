#!/usr/bin/env python3
"""Report only: L F1 per cut (lsw sweeps) for THIN vs DEB, is the own cut at the grid edge? Also the probability saturation of the maps."""
import glob, json, sys, numpy as np
sys.path.insert(0, "scripts")
from step1_read import source_of
U = "/data2/prusek/mt_work/runs/ult/lsw/"
f1 = lambda s: 2 * s[0] / max(2 * s[0] + s[1] + s[2], 1)
for blk in ("roiv2", "hb"):
    print("==", blk)
    for m in ("DEB_S3", "DEB_S4", "DEB_S5", "DEB_S6", "THIN_S3", "THIN_S4", "THIN_S5", "THIN_S6"):
        o = {}
        for f in glob.glob(f"{U}L_{blk}_{m}_t*.json"):
            d = json.load(open(f)); s = [0, 0, 0]
            for n, _, r in d["rows"]:
                if blk == "hb" and source_of(n) != "htw": continue
                q = r["b60_t0"]; s[0] += q["tp"]; s[1] += q["fp"]; s[2] += q["fn"]
            o[d["cut"]] = (f1(s), s[0] / max(s[0] + s[1], 1), s[0] / max(s[0] + s[2], 1))
        print(f"  {m:8s} " + " ".join(f"{c:g}:{v[0]:.4f}" for c, v in sorted(o.items())[-6:]) + f" | own {max(o, key=lambda c: (o[c][0], c)):g}")
