#!/usr/bin/env python3
"""S0 overlays: curves of frame t (cyan) and t+3 (magenta, windows share no frame) on frame t, a central crop, per frame against the fused arms.
usage: overlay_s0.py SP_TEMPORAL OUT_DIR"""
import json, os, sys
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
SP, OUT = sys.argv[1:3]
L = {o["vid"]: o for o in json.load(open(f"{SP}/list.json"))}
A, F, G = {}, {}, {}
for k in (0, 1):
    A.update(json.load(open(f"{SP}/res/v16_{k}.json"))); F.update(json.load(open(f"{SP}/fuse/fuse_{k}.json"))); G.update(json.load(open(f"{SP}/fuse/fuse2_{k}.json")))
ARMS = {"per frame": lambda v, f: A[v]["frames"][f]["polys"], "mean image": lambda v, f: F[v]["frames"][f]["meanimg"], "mean map, own cut": lambda v, f: G[v]["frames"][f]["meanmap_q"]}
for short in ("9caf72f4", "5ca130c0", "f83a4529"):
    vid = next(v for v in L if v.startswith(short)); Fr = L[vid]["frames"]; f0, f1 = Fr[9], Fr[12]
    img = np.asarray(Image.open(f"{SP}/ov/{f0['path']}")).astype(float); lo, hi = np.percentile(img, [1, 99])
    h, w = img.shape; c = 700; r0, c0 = (h - c) // 2, (w - c) // 2
    fig, ax = plt.subplots(1, 3, figsize=(24, 8))
    for x, (name, get) in zip(ax, ARMS.items()):
        x.imshow(np.clip((img - lo) / (hi - lo), 0, 1), cmap="gray"); x.set_xlim(c0, c0 + c); x.set_ylim(r0 + c, r0); x.set_title(f"{name} {short} frame {f0['frame']} cyan, {f1['frame']} magenta"); x.axis("off")
        for f, col in ((f0, "cyan"), (f1, "magenta")):
            for p in get(vid, f["frame"]):
                p = np.asarray(p)
                if len(p) > 1: x.plot(p[:, 1], p[:, 0], color=col, lw=0.9, alpha=0.8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, f"s0_{short}.png"), dpi=70); plt.close(fig); print("saved", short)
