#!/usr/bin/env python3
"""ARM 62 mechanism reading (i) (docs/TODO.md 2026-11-01 01:45, reported, no line): debris-only synthetic frames — the
V16 debris renderer on 43 of the 428 backgrounds (every 10th), no filaments, fixed seeds, rate 20 per frame, PSF blur and
the recipe's IRM noise — through a model (gauss fusion, native). Reports the foreground fraction at 0.995 inside the debris
footprint (|blurred debris field| >= 0.1 x the frame's MT contrast) and on the SAME pixels of the bare background (no
rings, same noise seed): the excess is what the debris itself make the model fire on.
usage: debris_probe.py CKPT OUT.json"""
import dataclasses, glob, json, sys
import numpy as np, torch
from PIL import Image
from scipy.ndimage import gaussian_filter
sys.path[:0] = [".", "src", "scripts", "synth", "third_party"]
from train_v5 import TILE_FOR, build_model, norm01, predict
import mt_generator as N
from gen_train import build_cfg
ck, out = sys.argv[1:3]
c = json.load(open("calib_V16_debris.json"))
cfg = dataclasses.replace(build_cfg(c["best_params"], mask_hw=1.0), **c["gen_overrides"])
cfg = dataclasses.replace(cfg, debris_rate=20.0, debris_rate_range=None)
state = torch.load(ck, map_location="cuda")
seg = [v for k, v in state.items() if "decoder.seg_layers." in k and k.endswith(".weight")]
model = build_model("nnunet", out_ch=int(seg[0].shape[0]), in_ch=3).to("cuda").eval(); model.load_state_dict(state, strict=False)
bgs = sorted(glob.glob("/data2/prusek/mt_work/IRM_backgrounds_v2/*.tif"))[::10]
n_in = f_ring = f_bare = 0; nr = 0
for i, p in enumerate(bgs):
    b = np.asarray(Image.open(p).convert("F"), np.float64)
    rng = np.random.default_rng(61000 + i); contrast = float(rng.uniform(*cfg.contrast_range))
    f = np.zeros(b.shape); nr += N.add_debris(f, rng, cfg, contrast)
    f = gaussian_filter(f, float(rng.uniform(*cfg.psf_sigma_range))); foot = np.abs(f) >= 0.1 * contrast
    if not foot.any(): continue
    ims = [N.add_irm_noise(b * (1.0 + g), np.random.default_rng(62000 + i), cfg) for g in (f, 0.0 * f)]
    with torch.no_grad():
        fg = [predict(model, norm01(im), TILE_FOR["nnunet"], fusion="gauss")[0] > 0.995 for im in ims]
    n_in += int(foot.sum()); f_ring += int((fg[0] & foot).sum()); f_bare += int((fg[1] & foot).sum())
res = {"ckpt": ck, "frames": len(bgs), "debris": nr, "px_in": n_in, "fg_ring": f_ring / max(n_in, 1), "fg_bare": f_bare / max(n_in, 1)}
res["fg_excess"] = res["fg_ring"] - res["fg_bare"]
json.dump(res, open(out, "w"), indent=1); print(res)
