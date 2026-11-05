#!/usr/bin/env python3
"""Old vs new production output over the frame: one crop per check frame, side by side. usage: overlay_prod.py PRODCHECK OUT"""
import json, os, sys
import numpy as np
from PIL import Image, ImageDraw
D, O = sys.argv[1], sys.argv[2]; A = json.load(open(D + "/prod_sparse35_before.json"))["frames"]; B = json.load(open(D + "/prod_v16_after.json"))["frames"]
for f in ("roi_median.png", "roi_densest.png", "htw_median.tif", "htw_densest.tif", "mt34_median.tif", "mt34_densest.png"):
    a = np.asarray(Image.open(os.path.join(D, f)), float); lo, hi = np.percentile(a, (1, 99)); g = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1) * 255
    H, W = g.shape; S = min(H, W, 900); y0, x0 = (H - S) // 2, (W - S) // 2; Z = 2 if S <= 512 else 1
    panels = []
    for tag, R, col in (("frame", None, None), ("before: SPARSE35 + A, n %d" % A[f]["n"], A[f], (255, 70, 70)), ("after: V16 + L+H, n %d" % B[f]["n"], B[f], (60, 230, 120))):
        im = Image.fromarray(g[y0:y0 + S, x0:x0 + S].astype(np.uint8)).convert("RGB").resize((S * Z, S * Z)); d = ImageDraw.Draw(im)
        if R:
            palette = [col, tuple(int(c * .75) for c in col[::-1]), (255, 210, 60), (90, 170, 255), (240, 120, 255)]
            for k, p in enumerate(R["polys"]):
                q = [((x - x0) * Z, (y - y0) * Z) for x, y in p]
                if len(q) > 1: d.line(q, fill=palette[k % 5], width=2)
        d.rectangle([0, 0, 330, 18], fill=(0, 0, 0)); d.text((4, 3), tag, fill=(255, 255, 255)); panels.append(im)
    c = Image.new("RGB", (sum(p.width for p in panels) + 8, panels[0].height), (255, 255, 255)); x = 0
    for p in panels: c.paste(p, (x, 0)); x += p.width + 4
    c.save(os.path.join(O, f.split(".")[0] + ".jpg"), quality=88); print(f, c.size)
