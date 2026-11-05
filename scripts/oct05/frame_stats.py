#!/usr/bin/env python3
"""Model-blind FRAME-level support audit (2026-10-05): the same statistics on real blocks and on generator output.
    python frame_stats.py OUT.json name=glob[:split] ...   (split omitted = all frames)"""
import glob, json, sys, numpy as np
from scipy.ndimage import gaussian_filter
sys.path.insert(0, "src")
from mt_bench.cvat_import import read_frame_h5
def stats(fr):
    img = np.asarray(fr["image"], float); lo, hi = np.percentile(img, [1, 99]); z = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)
    hp = z - gaussian_filter(z, 3.0); nm = 1.4826 * np.median(np.abs(hp - np.median(hp)))
    a = hp - hp.mean(); c1 = float((a[:, 1:] * a[:, :-1]).mean() / (a.var() + 1e-12)); c2 = float((a[:, 2:] * a[:, :-2]).mean() / (a.var() + 1e-12))
    lp = gaussian_filter(z, 25.0); kurt = float(((a / (a.std() + 1e-12)) ** 4).mean())
    L = [float(np.linalg.norm(np.diff(np.asarray(q, float), axis=0), axis=1).sum()) for q in fr["polylines"] if len(q) >= 2]
    H, W = z.shape
    return dict(h=H, w=W, levels=int(len(np.unique(img))), sat=float(((z <= 0) | (z >= 1)).mean()), nmad=float(nm), ac1=c1, ac2=c2, kurt=kurt,
                bg_lp_std=float(lp.std()), n=len(L), len_per_mpx=float(sum(L) / (H * W) * 1e6), n_per_mpx=float(len(L) / (H * W) * 1e6))
out = {}
for spec in sys.argv[2:]:
    name, pat = spec.split("=", 1); split = None
    if ":" in pat: pat, split = pat.rsplit(":", 1)
    rows = []
    for p in sorted(glob.glob(pat)):
        fr = read_frame_h5(p)
        if split and str(fr["attrs"].get("split")) != split: continue
        rows.append(stats(fr))
    out[name] = rows
    q = lambda k: " ".join("%.3g" % v for v in np.percentile([r[k] for r in rows], [1, 10, 50, 90, 99]))
    print(f"== {name} ({len(rows)} frames)  p1 p10 p50 p90 p99")
    for k in ("levels", "sat", "nmad", "ac1", "ac2", "kurt", "bg_lp_std", "n_per_mpx", "len_per_mpx", "h"): print(f"   {k:12s} {q(k)}")
json.dump(out, open(sys.argv[1], "w"))
