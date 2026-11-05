#!/usr/bin/env python3
"""ARM 63 pre-launch test on kajman: V14, V15, V16 render bit-identically to the debris-patched backup (3 seeds); V17 differs from V16 only
in width_thin_prob, keeps the instances, changes the frame; share of thin frames as declared. usage: test_thin_k.py BACKUP.py"""
import dataclasses, glob, importlib.util, json, sys
import numpy as np
from PIL import Image
sys.path[:0] = [".", "src", "scripts", "synth", "third_party"]
import mt_generator as N
from gen_train import build_cfg
sp = importlib.util.spec_from_file_location("mt_deb", sys.argv[1]); O = importlib.util.module_from_spec(sp); sys.modules["mt_deb"] = O; sp.loader.exec_module(O)
def cfg_of(f):
    c = json.load(open(f)); return dataclasses.replace(build_cfg(c["best_params"], mask_hw=1.0), **c.get("gen_overrides", {}))
C = {v: cfg_of(f) for v, f in (("V14", "calib_V14_noisemix_aa.json"), ("V15", "calib_V15_rings.json"), ("V16", "calib_V16_debris.json"), ("V17", "calib_V17_thin.json"))}
bgs = sorted(glob.glob("/data2/prusek/mt_work/IRM_backgrounds_v2/*.tif")); fails = 0
def check(n, ok):
    global fails; fails += (not ok); print(("PASS " if ok else "FAIL ") + n, flush=True)
check("V17 differs from V16 only in width_thin_prob", [f.name for f in dataclasses.fields(N.GenConfig) if getattr(C["V16"], f.name) != getattr(C["V17"], f.name)] == ["width_thin_prob"])
for s in (31, 32, 33):
    b = np.asarray(Image.open(bgs[s * 11 % len(bgs)]).convert("F"), np.float32)[:640, :640]
    for v in ("V14", "V15", "V16"):
        o = O.GenConfig(**{f.name: getattr(C[v], f.name) for f in dataclasses.fields(O.GenConfig)})
        a = O.generate_frame(b, np.random.default_rng(s), o); c = N.generate_frame(b, np.random.default_rng(s), C[v])
        check(f"{v} bit-identical seed {s}", np.array_equal(a[0], c[0]) and all(np.array_equal(x["centerline"], y["centerline"]) for x, y in zip(a[1], c[1])))
    c = N.generate_frame(b, np.random.default_rng(s), C["V16"]); d = N.generate_frame(b, np.random.default_rng(s), C["V17"])
    check(f"V17 same instances seed {s}", all(np.array_equal(x["centerline"], y["centerline"]) for x, y in zip(c[1], d[1])))
ws = []
for s in range(200):
    b = np.asarray(Image.open(bgs[s % len(bgs)]).convert("F"), np.float32)[:160, :160]
    ws.append(N.generate_frame(b, np.random.default_rng(5000 + s), C["V17"])[2]["width_sigma"])
ws = np.array(ws); print("V17 width_sigma: share <= 1.2 = %.3f, min %.2f, max %.2f" % ((ws <= 1.2).mean(), ws.min(), ws.max()))
check("thin share near the declared 0.35", 0.25 <= (ws <= 1.2).mean() <= 0.50); check("upper range kept", ws.max() > 2.0)
print("ALL PASS" if not fails else f"{fails} FAILED")
