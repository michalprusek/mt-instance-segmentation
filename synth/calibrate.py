"""Label-free Optuna calibration loop for the synthetic MT generator.

Tunes GenConfig knobs so that synthetic frames match the REAL corpus DISTRIBUTION in DINOv2
(blocks 11/15/17) BACKGROUND-SUBTRACTED-RESIDUAL patch features, via linear MMD. No annotations —
only real IRM frames + the real empty-field background library. Knob ranges are seeded from the
BIOCEV `synthmt_irm` generator priors (bending, waviness amplitude, white-segment/detachment, noise).

Run on tulen (dinov3_env): python calibrate.py --trials 150 --m 16
"""
import sys
try:
    import pysqlite3; sys.modules["sqlite3"] = pysqlite3      # base python lacks _sqlite3
except Exception:
    pass
import os, glob, json, argparse, dataclasses
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, label as cc_label
import torch, optuna
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mt_generator import GenConfig, generate_frame

DEV = "cuda"; SIZE = 518; CROP = 768; BLOCKS = [11, 15, 17]
# OBJECTIVE: residual_mmd = OURS (foreground-aware: bg-subtracted residual, patch tokens, distributional
# MMD). global_cosine = SynthMT reproduction (raw image, DINOv2 layer-5 GLOBAL mean-pool, set cosine).
# residual_mmd = OURS-current (foreground-aware, FRAME-pooled). region_mmd = OURS-NEW (per-foreground-
# REGION pooling → recovers per-MT VARIANCE + no bg dilution). global_cosine = SynthMT reproduction.
CALIB_OBJECTIVE = os.environ.get("CALIB_OBJECTIVE", "residual_mmd")
GC_BLOCK = 5
FEAT_BLOCKS = [GC_BLOCK] if CALIB_OBJECTIVE == "global_cosine" else BLOCKS
BG   = sorted(glob.glob("/home/prusek/BIOCEV/datasets/microtubules/IRM_backgrounds_v2/*.tif"))
_RC = os.environ.get("REAL_DIR", "/home/prusek/BIOCEV/datasets/microtubules/real_corpus_v2")  # override via REAL_DIR
REAL = sorted(glob.glob(_RC + "/*.tif") + glob.glob(_RC + "/*.png")) \
    or sorted(glob.glob("/home/prusek/BIOCEV/datasets/microtubules/morphology_reference_frames/irm/*.tif"))  # fallback to base 320
OUT  = os.environ.get("CALIB_OUT", "/home/prusek/mt_enc_exp/calib" + {"global_cosine":"_gc","region_mmd":"_reg"}.get(os.environ.get("CALIB_OBJECTIVE","residual_mmd"),"")); os.makedirs(OUT, exist_ok=True)
IMA_M = torch.tensor([0.485,0.456,0.406]).view(3,1,1); IMA_S = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
CALIB_ENCODER = os.environ.get("CALIB_ENCODER", "dinov2")   # dinov2 | phikon (must match the segmenter backbone!)
if CALIB_ENCODER == "phikon":
    from transformers import AutoModel
    m = AutoModel.from_pretrained("owkin/phikon-v2").to(DEV).eval()
else:
    m = torch.hub.load('facebookresearch/dinov2','dinov2_vitl14').to(DEV).eval()

def crop(a):
    a = np.asarray(a, float)
    while a.ndim > 2: a = a[..., 0]
    H, W = a.shape; return a[H//2-CROP//2:H//2+CROP//2, W//2-CROP//2:W//2+CROP//2]
def norm01(a, p=(1,99)):
    lo, hi = np.percentile(a, p); return np.clip((a-lo)/(hi-lo+1e-6), 0, 1)
def resid(I):
    return I/(gaussian_filter(I, 40)+1e-6) - 1.0
def tens(img01):
    return (torch.from_numpy(np.asarray(Image.fromarray((img01*255).astype(np.uint8)).convert("RGB").resize((SIZE,SIZE),Image.BICUBIC))).float().permute(2,0,1)/255. - IMA_M)/IMA_S

@torch.no_grad()
def encode(imgs01):
    out = []
    for i in range(0, len(imgs01), 8):
        ts = torch.stack([tens(x) for x in imgs01[i:i+8]]).to(DEV)
        if CALIB_ENCODER == "phikon":
            hs = m(pixel_values=ts, output_hidden_states=True, interpolate_pos_encoding=True).hidden_states
            pooled = torch.cat([hs[L+1][:, 1:, :].mean(1) for L in BLOCKS], dim=1)
        else:
            outs = m.get_intermediate_layers(ts, n=FEAT_BLOCKS, reshape=False, return_class_token=False, norm=True)
            pooled = torch.cat([o.mean(1) for o in outs], dim=1)        # concat blocks (global mean-pool per block)
        out.append(torch.nn.functional.normalize(pooled, dim=1).cpu().numpy())
    return np.concatenate(out)

def prep_img(I):
    """Feature-input prep: OURS (residual_mmd & region_mmd) use the |RESIDUAL| (foreground-aware AND
    POLARITY-INVARIANT — dark IRM MTs and inverted-frame bright MTs look the same, so always-dark synth
    matches real frames of either polarity; inversion is handled by training augmentation, not calibration).
    SynthMT (global_cosine) uses the RAW normalized image."""
    return norm01(np.abs(resid(I)), p=(2, 98)) if CALIB_OBJECTIVE in ("residual_mmd", "region_mmd") else norm01(I)

def region_mask(img01, thr_pct=96.0):
    """Label-free FOREGROUND regions: strong-|deviation| pixels (polarity-robust — works for dark IRM
    MTs AND inverted frames). img01 is residual-normalized, so MTs sit in the tails."""
    dev = np.abs(img01 - np.median(img01))
    return dev > np.percentile(dev, thr_pct)

@torch.no_grad()
def encode_regions(imgs01):
    """PER-REGION features: pool patch tokens per connected FOREGROUND component (not the whole frame).
    The SPREAD across regions carries per-MT VARIANCE; only foreground contributes (no bg dilution)."""
    feats, dim = [], None
    for i in range(0, len(imgs01), 8):
        batch = imgs01[i:i+8]
        ts = torch.stack([tens(x) for x in batch]).to(DEV)
        outs = m.get_intermediate_layers(ts, n=FEAT_BLOCKS, reshape=True, return_class_token=False, norm=True)
        grid = torch.cat(outs, dim=1)                      # (B, C, h, w)
        B, C, h, w = grid.shape; dim = C
        g = grid.reshape(B, C, h*w).cpu().numpy()
        for b in range(B):
            fg = region_mask(batch[b])
            fgp = np.asarray(Image.fromarray(fg.astype(np.uint8)).resize((w, h), Image.NEAREST)) > 0
            lbl, n = cc_label(fgp); flat = lbl.reshape(-1)
            for r in range(1, n+1):
                idx = flat == r
                if idx.sum() < 1: continue
                v = g[b][:, idx].mean(1); feats.append(v / (np.linalg.norm(v) + 1e-9))
    return np.array(feats) if feats else np.zeros((1, dim or 1), np.float32)

def featurize(imgs01):
    return encode_regions(imgs01) if CALIB_OBJECTIVE == "region_mmd" else encode(imgs01)

def distance(X, real):
    """OURS = distributional RBF-MMD over patch features. SynthMT = 1 - cosine of SET-MEAN global embeddings."""
    if CALIB_OBJECTIVE == "global_cosine":
        xm, rm = X.mean(0), real.mean(0)
        return float(1 - (xm @ rm) / (np.linalg.norm(xm) * np.linalg.norm(rm) + 1e-9))
    return rbf_mmd2(X, real)

def rbf_mmd2(X, Y):
    """RBF-kernel MMD^2 (median-heuristic bandwidth) — matches the full feature DISTRIBUTION
    (diversity/covariance across frames), not just the mean like linear MMD."""
    Z = np.vstack([X, Y]); D = np.sum((Z[:, None] - Z[None]) ** 2, -1)
    sig = np.median(D[D > 0]); K = np.exp(-D / (sig + 1e-9)); n = len(X)
    return float(K[:n, :n].mean() + K[n:, n:].mean() - 2 * K[:n, n:].mean())

# ---- real target distribution (residual features) ----
rng0 = np.random.default_rng(0)
sel = [REAL[i] for i in rng0.choice(len(REAL), min(48, len(REAL)), replace=False)]
real_feats = featurize([prep_img(crop(Image.open(p))) for p in sel])

def build_cfg(t):
    dc = t.suggest_int("density_center", 6, 40)
    cf = t.suggest_float("cluster_frac", 0.0, 0.85)     # spatial clustering of MT placement
    cc = t.suggest_float("contrast_center", 0.01, 0.12, log=True)
    wc = t.suggest_float("width_mean", 0.9, 1.9)        # MEAN only (std = fixed prior)
    cb = t.suggest_float("curve_base", 0.01, 0.12, log=True)   # LENGTH-COUPLED curvature mean (Pampaloni)
    ina = t.suggest_float("ina", 0.6, 1.15)             # illumination NA — strongest IRM contrast knob (~2/3 obj NA)
    llm = t.suggest_float("length_log_mean", 5.0, 6.6)
    wfp = t.suggest_float("waviness_frame_prob", 0.0, 0.8)
    wa = t.suggest_float("waviness_amp_center", 1.0, 15.0)
    dfp = t.suggest_float("detach_frame_prob", 0.0, 0.9)
    dp = t.suggest_float("detach_prob", 0.1, 0.8)
    ok = t.suggest_float("orient_kappa", 0.0, 4.0)      # NEW nematic alignment (frame-level, research P4)
    kp = t.suggest_float("kink_prob", 0.0, 0.4)         # NEW sharp-bend fat-tail rate (research P3/P5)
    sp = t.suggest_float("short_prob", 0.0, 0.4)        # NEW short-seed population (research P6)
    psf = t.suggest_float("psf_center", 0.6, 2.2)
    hb = t.suggest_float("height_base", 15.0, 85.0)     # FRAME base MT height regime (dark ⇄ bright); interference
    return dataclasses.replace(
        GenConfig(),
        n_mt_range=(max(2, dc-6), dc+6),
        cluster_frac=cf,
        contrast_range=(0.5*cc, 1.5*cc),
        width_mean=wc, width_std=0.35,
        curve_base=cb,                                  # length-coupled stiffness (curve_frac_range = wide clamp)
        ina_range=(max(0.5, ina - 0.12), ina + 0.12),
        height_base_nm_range=(max(5.0, hb - 20.0), hb + 20.0),
        length_log_mean=llm,
        waviness_frame_prob=wfp, waviness_amp_range=(0.6*wa, 1.4*wa),
        detach_frame_prob=dfp, detach_prob=dp,
        orient_kappa_range=(0.0, ok), kink_prob=kp, short_prob=sp,
        psf_sigma_range=(0.7*psf, 1.3*psf),
    )

def render_set(cfg, M, seed):
    rng = np.random.default_rng(seed)
    imgs = []
    for _ in range(M):
        bg = crop(Image.open(BG[rng.integers(len(BG))]))
        img, _, _ = generate_frame(bg, rng, cfg)
        imgs.append(prep_img(img))
    return imgs

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--trials", type=int, default=400); ap.add_argument("--m", type=int, default=16)
    args = ap.parse_args()
    # baseline (default config) for reference
    base_mmd = distance(featurize(render_set(GenConfig(), args.m, 7)), real_feats)
    print(f"baseline (default GenConfig) [{CALIB_OBJECTIVE}] = {base_mmd:.5f}", flush=True)

    def objective(t):
        cfg = build_cfg(t)
        mmd = distance(featurize(render_set(cfg, args.m, 10000+t.number)), real_feats)
        if t.number % 10 == 0: print(f"  trial {t.number}: MMD={mmd:.5f}", flush=True)
        return mmd

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=1, n_startup_trials=20))
    study.optimize(objective, n_trials=args.trials)
    print(f"\nBEST MMD = {study.best_value:.5f}  (baseline {base_mmd:.5f}, {100*(1-study.best_value/base_mmd):.0f}% lower)")
    print("BEST params:", json.dumps(study.best_params, indent=2))
    json.dump({"best_value": study.best_value, "baseline": base_mmd, "best_params": study.best_params},
              open(os.path.join(OUT, "best.json"), "w"), indent=2)
    import csv
    with open(os.path.join(OUT, "trials.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["trial", "value"])
        for tr in study.trials:
            if tr.value is not None: w.writerow([tr.number, tr.value])
    # render calibrated samples + real crops for side-by-side
    best_cfg = build_cfg(optuna.trial.FixedTrial(study.best_params))
    rng = np.random.default_rng(123)
    for k in range(6):
        bg = crop(Image.open(BG[rng.integers(len(BG))]))
        img, _, meta = generate_frame(bg, rng, best_cfg)
        Image.fromarray((norm01(img)*255).astype(np.uint8)).save(os.path.join(OUT, f"calib_{k}_w{int(meta['wavy'])}_inv{int(meta['inverted'])}.png"))
    for k, p in enumerate(sel[:6]):
        Image.fromarray((norm01(crop(Image.open(p)))*255).astype(np.uint8)).save(os.path.join(OUT, f"real_{k}.png"))
    print("DONE — samples + best.json + trials.csv in", OUT, flush=True)

if __name__ == "__main__":
    main()
