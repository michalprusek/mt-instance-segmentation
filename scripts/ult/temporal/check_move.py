#!/usr/bin/env python3
"""S0 check: fuse.py's no-op proof never moved a map. On the pairs with the largest shift, does the declared sign align the neighbour better than zero or the opposite sign?
usage: check_move.py PKG WEIGHTS FRAMES_DIR LIST_JSON TOP_JSON GEOM_DIR"""
import json, os, sys
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
PKG, W, ROOT, LST, TOP, GEOM = sys.argv[1:7]
sys.path[:0] = [os.path.dirname(PKG), PKG, os.path.join(PKG, "vendor"), GEOM]
from wrapper import MicrotubuleModel, UP
m = MicrotubuleModel().load_weights(W); L = {o["vid"]: o for o in json.load(open(LST))}
def move(a, d):
    dy, dx = int(np.rint(d[0])), int(np.rint(d[1]))
    if dy == 0 and dx == 0: return a
    H, W_ = a.shape[-2:]; y = np.clip(np.arange(H) - dy, 0, H - 1); x = np.clip(np.arange(W_) - dx, 0, W_ - 1)
    return a[..., y, :][..., :, x]
def mp(f): return m.infer_maps(np.asarray(Image.open(os.path.join(ROOT, f["path"]))).astype(np.float32))["prob_eval"].astype(np.float32)
for r in json.load(open(TOP)):
    F = L[r["vid"]]["frames"]; i = r["i"]; d = np.asarray(r["d"]); a, b = mp(F[i]) > 0.995, mp(F[i + 1]) > 0.995   # a point of i lands at + d in i + 1, so i + 1 moves back onto i by -d
    near = ndi.binary_dilation(a, iterations=2); out = []
    for name, s in (("declared -d", -d), ("zero", 0 * d), ("opposite +d", d)):
        out.append(f"{name} {(move(b, s * UP) & near).sum() / max(b.sum(), 1):.3f}")
    print(r["vid"][:8], i, np.round(d, 1).tolist(), " | ".join(out), flush=True)
print("CHECK_DONE")
