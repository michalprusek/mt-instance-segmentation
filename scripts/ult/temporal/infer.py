#!/usr/bin/env python3
"""Temporal stability (docs/TODO.md 2026-11-03 14:10): one model of the hub wrapper over the frame windows, no overrides.
usage: infer.py PKG WEIGHTS FRAMES_DIR LIST_JSON OUT_JSON k/K"""
import json, os, sys, time
import numpy as np
from PIL import Image
PKG, W, ROOT, LST, OUT = sys.argv[1:6]; k, K = (int(x) for x in sys.argv[6].split("/"))
sys.path[:0] = [os.path.dirname(PKG), PKG, os.path.join(PKG, "vendor")]
from wrapper import MicrotubuleModel
DY, DX = 5, 7   # control: same scene, the tile grid and the percentile window move
CONTROL_FRAMES = 5
m = MicrotubuleModel().load_weights(W)
print("params", m._params_path.name, "fusion", m.params.get("fusion"), "decoder", m.params.get("decoder"), "thr", m.params.get("prob_thr"), flush=True)
rows = json.load(open(LST))[k::K]
def read(p):
    a = np.asarray(Image.open(p))
    assert a.ndim == 2, (p, a.shape)
    return a.astype(np.float32)
def polys(img):
    o = m.predict(img)
    return [np.asarray(c).tolist() for c in o["centerlines_rc"]], o["decoder"]
R = {}
for r in rows:
    vid = r["vid"]; R[vid] = {"frames": {}, "control": {}}
    for i, f in enumerate(r["frames"]):
        img = read(os.path.join(ROOT, f["path"])); t0 = time.perf_counter()
        c, dec = polys(img)
        R[vid]["frames"][f["frame"]] = {"md5": f["md5"], "shape": list(img.shape), "decoder": dec, "polys": c, "t": time.perf_counter() - t0}
        if i < CONTROL_FRAMES:
            a, _ = polys(img[DY:, DX:]); b, _ = polys(img[:-DY, :-DX])
            R[vid]["control"][f["frame"]] = {"a": [(np.asarray(p) + [DY, DX]).tolist() for p in a], "b": b}
        print(vid[:8], f["frame"], img.shape, "n", len(c), dec, f"{time.perf_counter() - t0:.1f}s", flush=True)
    json.dump(R, open(OUT, "w"))
print("INFER_DONE", k, flush=True)
