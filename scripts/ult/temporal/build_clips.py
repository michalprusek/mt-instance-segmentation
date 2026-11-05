#!/usr/bin/env python3
"""R3 and S1's held-out synthetic clips (docs/TODO.md 2026-11-04 08:10): V16 calib, IRM, 1024 px, 12 clips per regime, 11 frames, seed 20261104. Truth clipped to the frame as build_synth_eval.py does.
usage: build_clips.py OUT_DIR  (cwd mt34_work, PYTHONPATH as in launch_ult.sh, MT_BG_DIR set)"""
import dataclasses, glob, json, os, sys
import numpy as np
from PIL import Image
from build_synth_eval import clip_polyline
from gen_train import build_cfg
from mt_clip import REGIMES, generate_clip
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); C = 1024; NB = 5; MIN_LEN = 15.0
d = json.load(open("calib_V16_debris.json")); cfg = dataclasses.replace(build_cfg(d["best_params"], mask_hw=1.0), **d["gen_overrides"])
bgs = sorted(glob.glob(os.environ["MT_BG_DIR"] + "/*"))
for ir, reg in enumerate(REGIMES):
    for i in range(12):
        rng = np.random.default_rng(20261104 + 1000 * ir + i)
        bg = np.asarray(Image.open(bgs[int(rng.integers(len(bgs)))]).convert("F"), np.float32)
        H, W = bg.shape; bg = bg[H // 2 - C // 2:H // 2 + C // 2, W // 2 - C // 2:W // 2 + C // 2]
        imgs, per, m = generate_clip(bg, rng, cfg, NB, NB, "irm", regime=reg)
        gt = []
        for inst in per:
            f = {}
            for z in inst:
                s = clip_polyline(np.asarray(z["centerline"], float), (C, C), MIN_LEN)
                if s is not None: f[int(z["inst_id"])] = s.tolist()
            gt.append(f)
        name = f"{reg}_{i:02d}"
        np.save(f"{OUT}/{name}.npy", np.stack(imgs).astype(np.float32))
        json.dump(dict(gt=gt, meta={k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in m.items() if isinstance(v, (int, float, str, bool, np.floating))}), open(f"{OUT}/{name}.json", "w"))
        print(name, "instances", len(per[NB]), "gt at t", len(gt[NB]), "speed", round(m["glide_speed"], 1), "shift", round(m["shift_scale"], 2), "alt", m.get("noise_alt"), flush=True)
print("CLIPS_DONE", flush=True)
