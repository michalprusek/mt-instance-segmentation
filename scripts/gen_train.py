"""Generate a CALIBRATED synthetic training set: image + binary foreground mask (free GT).
With --ori, also saves a K-channel ORIENTATION-KEYED centerline GT (ORION 'overpass': each channel =
centerlines whose local tangent falls in that 30deg bin, so crossings split across channels = amodal)."""
import os, glob, json, sys, dataclasses, argparse
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
sys.path.insert(0, "/home/prusek/mt_enc_exp/synth")
from mt_generator import GenConfig, generate_frame

K_ORI = 6                                              # orientation bins over [0,pi): 6 x 30deg (ORION)
def ori_channels(instances, shape, K=K_ORI, dilate=1):
    """K-channel thin centerline GT keyed by local tangent orientation (undirected, [0,pi))."""
    ch = np.zeros((K,) + shape, bool)
    for ins in instances:
        cl = ins["centerline"]
        if len(cl) < 2: continue
        d = np.gradient(cl, axis=0); ang = np.arctan2(d[:, 1], d[:, 0]) % np.pi
        b = (ang / (np.pi / K)).astype(int) % K
        for (x, y), bb in zip(cl, b):
            ix, iy = int(round(x)), int(round(y))
            if 0 <= iy < shape[0] and 0 <= ix < shape[1]: ch[bb, iy, ix] = True
    return np.stack([binary_dilation(ch[k], iterations=dilate) for k in range(K)])

def build_cfg(p, mask_hw=None):
    extra = {"mask_half_width": mask_hw} if mask_hw and mask_hw > 0 else {}
    return dataclasses.replace(
        GenConfig(),
        n_mt_range=(max(2, p["density_center"]-6), p["density_center"]+6),
        cluster_frac=p.get("cluster_frac", 0.0),
        contrast_range=(0.5*p["contrast_center"], 1.5*p["contrast_center"]),
        width_mean=p.get("width_mean", 1.3), width_std=0.35,
        curve_base=p.get("curve_base", p.get("curve_center", 0.05)),   # length-coupled stiffness (new)
        ina_range=(max(0.5, p.get("ina", 0.9)-0.12), p.get("ina", 0.9)+0.12),
        height_base_nm_range=(max(5.0, p.get("height_base", 40.0)-20.0), p.get("height_base", 40.0)+20.0),
        length_log_mean=p["length_log_mean"], length_log_sigma=p.get("length_log_sigma", 0.45),
        waviness_frame_prob=p["waviness_frame_prob"], waviness_amp_range=(0.6*p["waviness_amp_center"], 1.4*p["waviness_amp_center"]),
        detach_frame_prob=p["detach_frame_prob"], detach_prob=p["detach_prob"],
        orient_kappa_range=(0.0, p.get("orient_kappa", 0.0)),
        kink_prob=p.get("kink_prob", 0.12), short_prob=p.get("short_prob", 0.15),
        psf_sigma_range=(0.7*p["psf_center"], 1.3*p["psf_center"]),
        **extra,
    )

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=600); ap.add_argument("--crop", type=int, default=768)
    ap.add_argument("--mask_hw", type=float, default=-1.0)          # override mask_half_width (px); -1 = keep GenConfig default
    ap.add_argument("--out", default="train_synth")
    ap.add_argument("--ori", action="store_true")                   # also save K-channel orientation-keyed centerline GT
    ap.add_argument("--inst", action="store_true")                  # also save per-instance integer LABEL map (for affinity supervision)
    ap.add_argument("--calib", default="/home/prusek/mt_enc_exp/calib/best.json")  # which calibration best.json to use
    a = ap.parse_args()
    cfg = build_cfg(json.load(open(a.calib))["best_params"], mask_hw=a.mask_hw)
    BG = sorted(glob.glob("/home/prusek/BIOCEV/datasets/microtubules/IRM_backgrounds_v2/*.tif"))
    OUT = f"/home/prusek/mt_enc_exp/{a.out}"
    os.makedirs(OUT+"/images", exist_ok=True); os.makedirs(OUT+"/masks", exist_ok=True)
    if a.ori: os.makedirs(OUT+"/ori", exist_ok=True)
    if a.inst: os.makedirs(OUT+"/inst", exist_ok=True)
    c = a.crop
    for i in range(a.n):
        rng = np.random.default_rng(10000+i)
        bg = np.asarray(Image.open(BG[rng.integers(len(BG))]), float)
        while bg.ndim > 2: bg = bg[..., 0]
        H, W = bg.shape; bg = bg[H//2-c//2:H//2+c//2, W//2-c//2:W//2+c//2]
        img, inst, meta = generate_frame(bg, rng, cfg)
        fg = np.zeros((c, c), bool)
        for ins in inst: fg |= ins["mask"]
        lo, hi = np.percentile(img, [1, 99]); v = np.clip((img-lo)/(hi-lo+1e-6), 0, 1)
        Image.fromarray((v*255).astype(np.uint8)).save(f"{OUT}/images/{i:04d}.png")
        Image.fromarray((fg*255).astype(np.uint8)).save(f"{OUT}/masks/{i:04d}.png")
        if a.inst:
            L = np.zeros((c, c), np.int16)                          # per-instance label map (0=bg); overlaps: later wins
            for idx, ins in enumerate(inst, 1): L[ins["mask"]] = idx
            np.save(f"{OUT}/inst/{i:04d}.npy", L)
        if a.ori:
            np.save(f"{OUT}/ori/{i:04d}.npy", ori_channels(inst, (c, c)).astype(np.uint8))
        if i % 100 == 0: print(f"  {i}/{a.n}  fg%={100*fg.mean():.2f}", flush=True)
    print("DONE", OUT, flush=True)

if __name__ == "__main__":
    main()
