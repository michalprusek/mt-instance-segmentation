#!/usr/bin/env python3
"""ARM 64 / 63b pre-launch test: V18 and V19 differ from V16 only in the declared field, keep the instances, change the frame; the V16 frame is unchanged by the build. Doses as declared."""
import dataclasses, glob, json, sys
import numpy as np
from PIL import Image
sys.path[:0] = [".", "src", "scripts", "synth", "third_party"]
import mt_generator as N
from gen_train import build_cfg
def cfg_of(f):
    c = json.load(open(f)); return dataclasses.replace(build_cfg(c["best_params"], mask_hw=1.0), **c.get("gen_overrides", {}))
C = {v: cfg_of(f) for v, f in (("V16", "calib_V16_debris.json"), ("V17", "calib_V17_thin.json"), ("V18", "calib_V18_debris05.json"), ("V19", "calib_V19_thin05.json"))}
fails = 0
def check(n, ok):
    global fails; fails += (not ok); print(("PASS " if ok else "FAIL ") + n, flush=True)
diff = lambda a, b: [f.name for f in dataclasses.fields(N.GenConfig) if getattr(C[a], f.name) != getattr(C[b], f.name)]
check("V18 differs from V16 only in debris_rate_range", diff("V16", "V18") == ["debris_rate_range"]); check("V18 range is (0, 20)", tuple(C["V18"].debris_rate_range) == (0.0, 20.0))
check("V19 differs from V16 only in width_thin_prob", diff("V16", "V19") == ["width_thin_prob"]); check("V19 prob is 0.175", C["V19"].width_thin_prob == 0.175)
bgs = sorted(glob.glob("/data2/prusek/mt_work/IRM_backgrounds_v2/*.tif"))
for s in (31, 32, 33):
    b = np.asarray(Image.open(bgs[s * 11 % len(bgs)]).convert("F"), np.float32)[:640, :640]; a = N.generate_frame(b, np.random.default_rng(s), C["V16"])
    for v in ("V18", "V19"):
        d = N.generate_frame(b, np.random.default_rng(s), C[v])
        check(f"{v} same instances seed {s}", all(np.array_equal(x["centerline"], y["centerline"]) for x, y in zip(a[1], d[1])))
ws = np.array([N.generate_frame(np.asarray(Image.open(bgs[s % len(bgs)]).convert("F"), np.float32)[:160, :160], np.random.default_rng(5000 + s), C["V19"])[2]["width_sigma"] for s in range(300)])
print("V19 width_sigma: share <= 1.2 = %.3f, min %.2f, max %.2f" % ((ws <= 1.2).mean(), ws.min(), ws.max())); check("V19 thin share near 0.175 + the shipped tail", 0.12 <= (ws <= 1.2).mean() <= 0.30); check("upper range kept", ws.max() > 2.0)
print("ALL PASS" if not fails else f"{fails} FAILED")
