"""Render sample frames using the CALIBRATED GenConfig (synth/calibrate.py best params)."""
import os, glob, json, sys, dataclasses, argparse
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mt_generator import GenConfig, generate_frame

def build_cfg(p):
    return dataclasses.replace(
        GenConfig(),
        n_mt_range=(max(2, p["density_center"] - 6), p["density_center"] + 6),
        cluster_frac=p.get("cluster_frac", 0.0),
        contrast_range=(0.5 * p["contrast_center"], 1.5 * p["contrast_center"]),
        width_mean=p.get("width_mean", 1.3), width_std=0.35,
        curve_frac_range=(0.3 * p["curve_center"], p["curve_center"]),
        ina_range=(max(0.5, p.get("ina", 0.9) - 0.12), p.get("ina", 0.9) + 0.12),
        height_base_nm_range=(max(5.0, p.get("height_base", 40.0) - 20.0), p.get("height_base", 40.0) + 20.0),
        length_log_mean=p["length_log_mean"],
        waviness_frame_prob=p["waviness_frame_prob"],
        waviness_amp_range=(0.6 * p["waviness_amp_center"], 1.4 * p["waviness_amp_center"]),
        detach_frame_prob=p["detach_frame_prob"], detach_prob=p["detach_prob"],
        psf_sigma_range=(0.7 * p["psf_center"], 1.3 * p["psf_center"]),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", default="/home/prusek/mt_enc_exp/calib/best.json")
    ap.add_argument("--out", default="/home/prusek/mt_enc_exp/calib_render")
    ap.add_argument("--n", type=int, default=9); ap.add_argument("--crop", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=50)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = build_cfg(json.load(open(a.best))["best_params"])
    BG = sorted(glob.glob("/home/prusek/BIOCEV/datasets/microtubules/IRM_backgrounds_v2/*.tif"))
    c = a.crop
    for i in range(a.n):
        rng = np.random.default_rng(a.seed + i)
        bg = np.asarray(Image.open(BG[rng.integers(len(BG))]), float)
        while bg.ndim > 2: bg = bg[..., 0]
        H, W = bg.shape
        bg = bg[H//2-c//2:H//2+c//2, W//2-c//2:W//2+c//2]
        img, inst, meta = generate_frame(bg, rng, cfg)
        lo, hi = np.percentile(img, [1, 99]); v = np.clip((img-lo)/(hi-lo+1e-6), 0, 1)
        tag = ("wavy_" if meta["wavy"] else "") + ("inv" if meta["inverted"] else "norm")
        Image.fromarray((v*255).astype(np.uint8)).save(os.path.join(a.out, f"calib_{i}_{tag}_{len(inst)}mt.png"))
        print(f"  {i}: {len(inst)} mt, wavy={meta['wavy']} inv={meta['inverted']}", flush=True)
    print("DONE", a.out, flush=True)

if __name__ == "__main__":
    main()
