#!/usr/bin/env python3
"""ARM 62 pre-launch test on kajman (docs/TODO.md 2026-11-01 07:55): the debris-patched generator renders V14 and V15
bit-identically to the ring-patched backup (3 seeds, full backgrounds); V16 differs from V14 only in debris_rate_range,
keeps the instances, changes the frame; every rendered debris footprint is compact. usage: test_debris_k.py BACKUP.py"""
import dataclasses, glob, importlib.util, json, sys
import numpy as np
from PIL import Image
sys.path[:0] = [".", "src", "scripts", "synth", "third_party"]
import mt_generator as N
from gen_train import build_cfg
sp = importlib.util.spec_from_file_location("mt_ring", sys.argv[1]); O = importlib.util.module_from_spec(sp); sys.modules["mt_ring"] = O; sp.loader.exec_module(O)
def cfg_of(path):
    c = json.load(open(path)); return dataclasses.replace(build_cfg(c["best_params"], mask_hw=1.0), **c.get("gen_overrides", {}))
C = {v: cfg_of(f) for v, f in (("V14", "calib_V14_noisemix_aa.json"), ("V15", "calib_V15_rings.json"), ("V16", "calib_V16_debris.json"))}
bgs = sorted(glob.glob("/data2/prusek/mt_work/IRM_backgrounds_v2/*.tif")); fails = 0
def check(name, ok):
    global fails; fails += (not ok); print(("PASS " if ok else "FAIL ") + name, flush=True)
check("V16 differs from V14 only in debris_rate_range", [f.name for f in dataclasses.fields(N.GenConfig) if getattr(C["V14"], f.name) != getattr(C["V16"], f.name)] == ["debris_rate_range"])
for s in (21, 22, 23):
    b = np.asarray(Image.open(bgs[s * 13 % len(bgs)]).convert("F"), np.float32)
    for v in ("V14", "V15"):
        o = O.GenConfig(**{f.name: getattr(C[v], f.name) for f in dataclasses.fields(O.GenConfig)})
        a = O.generate_frame(b, np.random.default_rng(s), o); c = N.generate_frame(b, np.random.default_rng(s), C[v])
        check(f"{v} bit-identical seed {s}", np.array_equal(a[0], c[0]) and all(np.array_equal(x["centerline"], y["centerline"]) for x, y in zip(a[1], c[1])))
    d = N.generate_frame(b, np.random.default_rng(s), C["V16"]); c = N.generate_frame(b, np.random.default_rng(s), C["V14"])
    check(f"V16 same instances, different frame, seed {s}", all(np.array_equal(x["centerline"], y["centerline"]) for x, y in zip(d[1], c[1])) and not np.array_equal(d[0], c[0]))
rng = np.random.default_rng(7); r = [N._footprint_ratio(p) for p in (N._debris_patch(rng, C["V16"]) for _ in range(300)) if p is not None]
print("footprint axis ratio over %d draws: median %.2f max %.2f" % (len(r), np.median(r), max(r)))
check("all footprints compact", max(r) <= C["V16"].debris_max_axis_ratio)
n = [N.add_debris(np.zeros((2048, 2048)), np.random.default_rng(900 + s), C["V16"], 0.05) for s in range(30)]
print("debris objects per 2048^2 frame: mean", np.mean(n)); check("V16 draws debris", np.mean(n) > 5)
print("ALL PASS" if not fails else f"{fails} FAILED")
