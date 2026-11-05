#!/usr/bin/env python3
"""ARM 65 pre-flight on the host that trains: the patched item against the pre-patch code, on the V16 calib and the real backgrounds, its speed, and triplets to look at.
usage: check_item.py PRE65_DIR OUT_DIR  (cwd mt34_work, PYTHONPATH as in launch_ult.sh, MT_BG_DIR set)"""
import dataclasses, functools, glob, importlib.util, json, os, random, sys, time
import numpy as np
import torch
PRE, OUT = sys.argv[1:3]; os.makedirs(OUT, exist_ok=True)
def load(name, path):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sys.modules[name] = m; sp.loader.exec_module(m); return m
sys.argv = ["x"]
old_gen = load("old_gen", f"{PRE}/mt_generator.py")
old = load("old_train", f"{PRE}/train_v5.py"); new = load("new_train", "scripts/train_v5.py")
import mt_generator as gen
from gen_train import build_cfg
from mt_clip import clip_triplet, generate_clip, ClipConfig, REGIMES
d = json.load(open("calib_V16_debris.json"))
cfg = dataclasses.replace(build_cfg(d["best_params"], mask_hw=1.0), **d["gen_overrides"])
ocfg = dataclasses.replace(old.__dict__.get("GenConfig", old_gen.GenConfig)(), **{f.name: getattr(cfg, f.name) for f in dataclasses.fields(old_gen.GenConfig) if hasattr(cfg, f.name)})
bgs = sorted(glob.glob(os.environ["MT_BG_DIR"] + "/*")); print("backgrounds", len(bgs), flush=True)
kw = dict(crop=512, epoch_len=100, target_half_width=1.0, tirf_prob=0.35)
D = dict(old=old.OnlineDS(ocfg, bgs, old_gen.generate_frame, **kw), off=new.OnlineDS(cfg, bgs, gen.generate_frame, **kw),
         t0=new.OnlineDS(cfg, bgs, functools.partial(clip_triplet, p_single=0.0), **kw), t1=new.OnlineDS(cfg, bgs, functools.partial(clip_triplet, p_single=1.0), **kw))
def item(n, s, i):
    random.seed(s); return D[n][i]
bad = 0
for s in range(24):
    a, ga = item("old", s, s); b, gb = item("off", s, s); c, gc = item("t0", s, s); e, ge = item("t1", s, s)
    ok = dict(flag_off=torch.equal(a, b) and torch.equal(ga, gb), single=torch.equal(a, e) and torch.equal(ga, ge), target=torch.equal(ga, gc), centre=torch.equal(a[1], c[1]), differs=not torch.equal(c[0], c[1]) and not torch.equal(c[2], c[1]))
    if not all(ok.values()): bad += 1; print("FAIL item", s, ok, flush=True)
print("ITEM_CHECK", "OK" if bad == 0 else f"FAILED {bad}", "of 24", flush=True)
for name, ds in (("still", D["off"]), ("p_single 0", D["t0"]), ("recipe 0.35", new.OnlineDS(cfg, bgs, functools.partial(clip_triplet, p_single=0.35), **kw))):
    t0 = time.perf_counter()
    for i in range(40): ds[i + 100]
    print(f"seconds per item, one thread, {name}: {(time.perf_counter() - t0) / 40:.3f}", flush=True)
from PIL import Image
def to8(a):
    lo, hi = np.percentile(a, [1, 99]); return (np.clip((a - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)
rng = np.random.default_rng(20261104)
for r in REGIMES:
    for k in range(2):
        b = np.asarray(Image.open(bgs[int(rng.integers(len(bgs)))]).convert("F"), np.float32)[300:1100, 300:1100]
        imgs, per, m = generate_clip(b, np.random.default_rng(1000 + k), cfg, 1, 1, regime=r, ccfg=ClipConfig(glide_speed_px=(6.0, 6.0)) if r == "gliding" else None)
        u = [to8(x) for x in imgs]
        Image.fromarray(np.hstack([u[1], np.dstack(u)[:, :, 0] * 0 + u[1]])).save(f"{OUT}/{r}_{k}_t.png") if False else None
        Image.fromarray(np.dstack(u)).save(f"{OUT}/{r}_{k}_rgb.png"); Image.fromarray(u[1]).save(f"{OUT}/{r}_{k}_t.png")
        print(r, k, "instances", len(per[1]), "shift scale", round(m["shift_scale"], 2), "speed", round(m["glide_speed"], 1), "noise alt", m.get("noise_alt"), flush=True)
print("CHECK_ITEM_DONE", flush=True)
