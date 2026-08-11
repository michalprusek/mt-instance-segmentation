"""Render sample synthetic IRM frames on real backgrounds for visual inspection.
Saves: full frame, a zoom crop (to see polarity flips), and a GT-overlay."""
import os, glob, argparse
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from mt_generator import GenConfig, generate_frame

def win(a, p=(1, 99)):
    lo, hi = np.percentile(a, p); return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)

def save_gray(a01, path):
    Image.fromarray((a01 * 255).astype(np.uint8)).save(path)

def overlay(img01, instances, crop=None):
    rgb = np.stack([img01] * 3, -1)
    for k, ins in enumerate(instances):
        cl = ins["centerline"]
        col = np.array([[1, 0.2, 0.2], [0.2, 0.6, 1], [0.3, 1, 0.3], [1, 1, 0.2]])[k % 4]
        ix = np.round(cl[:, 0]).astype(int); iy = np.round(cl[:, 1]).astype(int)
        ok = (ix >= 0) & (ix < rgb.shape[1]) & (iy >= 0) & (iy < rgb.shape[0])
        rgb[iy[ok], ix[ok]] = col
    return (rgb * 255).astype(np.uint8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg_dir", default="/home/prusek/BIOCEV/datasets/microtubules/IRM_backgrounds_v2")
    ap.add_argument("--out", default="/home/prusek/mt_enc_exp/synth_samples")
    ap.add_argument("--crop", type=int, default=900)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = GenConfig()
    bgs = sorted(glob.glob(os.path.join(args.bg_dir, "*.tif")))
    # spread across microscopes A/B/C
    pick = []
    for sc in ("A_", "B_", "C_"):
        pick += [p for p in bgs if os.path.basename(p).startswith(sc)][:2]
    pick = pick[:args.n]
    for i, bp in enumerate(pick):
        rng = np.random.default_rng(args.seed + i)
        bg = np.asarray(Image.open(bp)).astype(np.float64)
        while bg.ndim > 2: bg = bg[..., 0]
        H, W = bg.shape; c = args.crop
        bg = bg[H//2 - c//2:H//2 + c//2, W//2 - c//2:W//2 + c//2]
        img, inst, meta = generate_frame(bg, rng, cfg)
        i01 = win(img)
        tag = "inv" if meta["inverted"] else "norm"
        name = f"{os.path.basename(bp)[:12]}_{tag}"
        save_gray(i01, os.path.join(args.out, f"s{i}_{name}_full.png"))
        # zoom crop (top-left quadrant) to see polarity / detachment segments
        z = i01[:c//2, :c//2]
        save_gray(z, os.path.join(args.out, f"s{i}_{name}_zoom.png"))
        Image.fromarray(overlay(i01, inst)).save(os.path.join(args.out, f"s{i}_{name}_overlay.png"))
        print(f"s{i} {name}: {len(inst)} instances inverted={meta['inverted']}", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
