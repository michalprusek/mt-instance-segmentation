#!/usr/bin/env python3
"""Temporal stability overlays: curves of frame t (cyan) and t+1 (magenta) on frame t, a central crop, SPARSE35 and V16.
usage: overlay.py SP_TEMPORAL OUT_DIR"""
import json, os, sys
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
SP, OUT = sys.argv[1:3]
L = {o["vid"]: o for o in json.load(open(f"{SP}/list.json"))}
D = {}
for a in ("sparse35", "v16"):
    D[a] = {}
    for k in (0, 1): D[a].update(json.load(open(f"{SP}/res/{a}_{k}.json")))
for short in ("f83a4529", "9caf72f4", "94800a42", "5ca130c0"):
    vid = next(v for v in L if v.startswith(short)); F = L[vid]["frames"]; f0, f1 = F[9], F[10]
    img = np.asarray(Image.open(f"{SP}/ov/{f0['path']}")).astype(float); lo, hi = np.percentile(img, [1, 99])
    h, w = img.shape; c = 700; r0, c0 = (h - c) // 2, (w - c) // 2
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    for x, a in zip(ax, ("sparse35", "v16")):
        x.imshow(np.clip((img - lo) / (hi - lo), 0, 1), cmap="gray"); x.set_xlim(c0, c0 + c); x.set_ylim(r0 + c, r0); x.set_title(f"{a} {short} frame {f0['frame']} cyan, {f1['frame']} magenta"); x.axis("off")
        for f, col in ((f0, "cyan"), (f1, "magenta")):
            for p in D[a][vid]["frames"][f["frame"]]["polys"]:
                p = np.asarray(p)
                if len(p) > 1: x.plot(p[:, 1], p[:, 0], color=col, lw=0.9, alpha=0.8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, f"pair_{short}.png"), dpi=90); plt.close(fig); print("saved", short)
