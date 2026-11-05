#!/usr/bin/env python3
"""S1 overlays: per frame against lam 1 on frame t, curves of t that change in orange; where a link decision moved and whether it looks right.
usage: overlay_s1.py SP_TEMPORAL OUT_DIR"""
import json, os, sys
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
SP, OUT = sys.argv[1:3]
L = {o["vid"]: o for o in json.load(open(f"{SP}/list.json"))}; A, B = {}, {}
for k in (0, 1):
    A.update(json.load(open(f"{SP}/res/v16_{k}.json"))); B.update(json.load(open(f"{SP}/s1/s1_real_{k}.json"))["frames"])
for short in ("f83a4529", "5ca130c0", "94800a42"):
    vid = next(v for v in L if v.startswith(short)); f = L[vid]["frames"][9]
    img = np.asarray(Image.open(f"{SP}/ov/{f['path']}")).astype(float); lo, hi = np.percentile(img, [1, 99]); h, w = img.shape
    pa, pb = A[vid]["frames"][f["frame"]]["polys"], B[vid][f["frame"]]["lam1"]
    key = lambda p: (len(p), tuple(np.round(p[0], 2)), tuple(np.round(p[-1], 2)))
    ka = {key(p) for p in pa}; kb = {key(p) for p in pb}
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    for x, (ps, other, name) in zip(ax, ((pa, kb, "per frame"), (pb, ka, "lam 1"))):
        x.imshow(np.clip((img - lo) / (hi - lo), 0, 1), cmap="gray"); x.axis("off"); ch = 0
        for p in ps:
            q = np.asarray(p); c = key(p) in other
            ch += not c; x.plot(q[:, 1], q[:, 0], color="cyan" if c else "orange", lw=0.8 if c else 1.6, alpha=0.8 if c else 1.0)
        x.set_title(f"{name} {short} frame {f['frame']}: {len(ps)} curves, {ch} differ (orange)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, f"s1_{short}.png"), dpi=80); plt.close(fig); print("saved", short, len(pa), len(pb))
