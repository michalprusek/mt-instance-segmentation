#!/usr/bin/env python3
"""Does the trained model FIRE on invisible synthetic label?  RUNS ON TULEN (GPU, a few minutes).

Declared in docs/TODO.md 0.0-CAMPAIGN (2026-09-04 22:50) as check (i) before any label-validity arm
is built. Renders the same synthetic frames as scripts/diag_filament_contrast.py (same seed), scores
each frame's label visibility with that instrument (isolated in-frame samples, |c|/noise), runs the
checkpoint on the [1, 99]-stretched native frame exactly as the trainer normalises it, and reads:

  * p along the label (median, share >= 0.98, share >= 0.5) -- per frame and split by the frame's
    visibility mode (invisible: > 40 % of label below SNR 1; visible: < 10 %);
  * foreground fraction at 0.98 / 0.9 / 0.5, and its OFF-LABEL part (pixels further than 5 px from
    any label pixel) -- the texture-prior signature.

High p along invisible label = the model learned to fire on unlearnable label (a gate has teeth);
p near the background level = it already discounts it and a gate changes little.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:src:scripts:synth:third_party python scripts/diag_model_on_synth.py \
        --ckpt /disk2/prusek/mt_work/runs/SPARSE35/model.pth --calib calib_V6_c070_narrow.json \
        --n-synth 80 --out runs/diag_roi/model_on_synth_SPARSE35.json
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "src", "scripts", "synth", "third_party"):
    sys.path.insert(0, os.path.join(ROOT, d))

from diag_filament_contrast import MARGIN, frame_noise, polyline_rows, profile_stats, resample, stretch  # noqa: E402
from scipy.ndimage import gaussian_filter

SNR_EDGES = [0, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 1e9]
# real label recall at 0.98 by local SNR bin (break anatomy 2026-09-04 23:35, 139 frames)
REAL_RECALL = {"0-0.7": 0.343, "0.7-1.0": 0.589, "1.0-1.5": 0.725, "1.5-2.0": 0.805, "2.0-3.0": 0.868,
               "3.0-5.0": 0.891, "5.0+": 0.894}


def inject(img01, rng, mode):
    """Add noise to a stretched frame. 'corr': Gaussian-filtered white noise (kernel sigma U(0.8, 1.5))
    with amplitude set so the residual's lag-1 autocorrelation lands near a target U(0.2, 0.6) --
    the real 16-bit frames' band; 'white': white noise of the SAME amplitude (the control that moves
    SNR by the same amount without changing the texture). Re-stretched to [1, 99] afterwards."""
    if mode == "none":
        return img01, {}
    noise = frame_noise(img01)
    sig = float(rng.uniform(0.8, 1.5))
    target = float(rng.uniform(0.2, 0.6))
    rho = float(np.exp(-1.0 / (4.0 * sig * sig)))          # lag-1 ACF of the filtered field
    frac = min(target / rho, 0.95)                          # v_c / (v_w + v_c)
    a = noise * np.sqrt(frac / (1.0 - frac))
    z = rng.normal(0, 1, img01.shape)
    if mode == "corr":
        z = gaussian_filter(z, sig); z /= z.std() + 1e-9
    out = np.clip(img01 + a * z, 0, 1)
    lo, hi = np.percentile(out, [1, 99])
    return np.clip((out - lo) / (hi - lo + 1e-6), 0, 1), {"sigma": sig, "target_acf": target, "amp_over_noise": float(a / noise)}


def sample_pairs(img01, polys, prob):
    """(local snr, p) at every in-frame 2 px sample of every centerline."""
    img_s = gaussian_filter(img01, 1.0); noise = frame_noise(img01); H, W = img01.shape
    out = []
    for poly in polys:
        Q, t = resample(poly)
        if Q is None:
            continue
        inside = (Q[:, 0] >= MARGIN) & (Q[:, 0] < W - MARGIN) & (Q[:, 1] >= MARGIN) & (Q[:, 1] < H - MARGIN)
        if inside.sum() < 3:
            continue
        Q, t = Q[inside], t[inside]
        c = np.abs(profile_stats(img_s, Q, t)) / noise
        pv = map_coordinates(prob, [Q[:, 1], Q[:, 0]], order=1, mode="nearest")
        out.append(np.stack([c, pv], 1))
    return np.concatenate(out, 0) if out else np.zeros((0, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arch", default="nnunet")
    ap.add_argument("--calib", default="calib_V6_c070_narrow.json")
    ap.add_argument("--n-synth", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--out", required=True)
    ap.add_argument("--inject", default="none", choices=["none", "white", "corr"],
                    help="noise injected into the stretched render before inference (check 2 of the NOISE arm)")
    args = ap.parse_args()

    import torch
    from gen_train import build_cfg
    from mt_generator import generate_frame
    from PIL import Image
    from train_v5 import TILE_FOR, build_model, predict

    state = torch.load(args.ckpt, map_location="cuda")
    seg = [v for k, v in state.items() if "decoder.seg_layers." in k and k.endswith(".weight")]
    out_ch = int(seg[0].shape[0]) if seg else 1
    model = build_model(args.arch, out_ch=out_ch, in_ch=3).to("cuda").eval()
    model.load_state_dict(state, strict=False)

    cal = json.load(open(args.calib))
    cfg = build_cfg(cal["best_params"], mask_hw=1.0)
    if cal.get("gen_overrides"):
        cfg = dataclasses.replace(cfg, **cal["gen_overrides"])
    bgs = []
    for d in ("/home/prusek/mt_enc_exp/irm_backgrounds_v2",
              "/home/prusek/BIOCEV/datasets/microtubules/IRM_backgrounds_v2",
              os.path.join(ROOT, "data/real/irm_backgrounds_v2")):
        bgs = sorted(glob.glob(os.path.join(d, "*.tif")))
        if bgs:
            break
    rng = np.random.default_rng(args.seed)
    g = 640
    frames, pairs_all = [], []
    for k in range(args.n_synth):
        b = np.asarray(Image.open(bgs[int(rng.integers(len(bgs)))]), np.float32)
        if b.ndim == 3:
            b = b.mean(2)
        if b.shape[0] < g or b.shape[1] < g:
            b = np.asarray(Image.fromarray(b).resize((max(g, b.shape[1]), max(g, b.shape[0]))), np.float32)
        y0, x0 = int(rng.integers(0, b.shape[0] - g + 1)), int(rng.integers(0, b.shape[1] - g + 1))
        img, inst, meta = generate_frame(b[y0:y0 + g, x0:x0 + g], rng, cfg)
        img01 = stretch(img)
        img01, inj = inject(img01, np.random.default_rng(args.seed + 1000 + k), args.inject)
        rows = polyline_rows(img01, [z["centerline"] for z in inst], None, tag=f"synth_{k}")
        w = np.array([r["n_iso"] for r in rows], float)
        weak = float((w * np.array([r["iso_frac_weak_1"] or 0 for r in rows])).sum() / w.sum()) if w.sum() > 20 else None
        with torch.no_grad():
            ch = predict(model, img01.astype(np.float32), TILE_FOR[args.arch])
        prob = np.asarray(ch[0], np.float32)
        # label raster (in-frame centerline pixels) and its 5 px surround
        lab = np.zeros((g, g), bool)
        pts = []
        for z in inst:
            c = np.asarray(z["centerline"], float)
            ok = (c[:, 0] >= 0) & (c[:, 0] < g) & (c[:, 1] >= 0) & (c[:, 1] < g)
            c = c[ok]
            if len(c):
                iy = np.clip(np.round(c[:, 1]).astype(int), 0, g - 1)
                ix = np.clip(np.round(c[:, 0]).astype(int), 0, g - 1)
                lab[iy, ix] = True
                pts.append(c)
        pts = np.concatenate(pts, 0) if pts else np.zeros((0, 2))
        far = distance_transform_edt(~lab) > 5.0 if lab.any() else np.ones((g, g), bool)
        pl = map_coordinates(prob, [pts[:, 1], pts[:, 0]], order=1, mode="nearest") if len(pts) else np.zeros(0)
        rec = {"frame": f"synth_{k}", "weak_iso": weak, "n_inst": len(inst),
               "contrast": float(meta.get("contrast", np.nan)), "poisson_gain": float(meta.get("poisson_gain", np.nan)),
               "p_label_median": float(np.median(pl)) if len(pl) else None,
               "p_label_ge098": float(np.mean(pl >= 0.98)) if len(pl) else None,
               "p_label_ge05": float(np.mean(pl >= 0.5)) if len(pl) else None,
               "p_far_median": float(np.median(prob[far])), "p_far_p99": float(np.percentile(prob[far], 99))}
        for t in (0.98, 0.9, 0.5):
            rec[f"fg_{t}"] = float((prob > t).mean()); rec[f"fg_off_{t}"] = float(((prob > t) & far).mean())
        rec["inject"] = inj
        pairs_all.append(sample_pairs(img01, [z["centerline"] for z in inst], prob))
        frames.append(rec)
        print(f"synth_{k}: weak={weak} p_label_med={rec['p_label_median']} ge098={rec['p_label_ge098']} "
              f"fg98={rec['fg_0.98']:.4f} off98={rec['fg_off_0.98']:.4f} off90={rec['fg_off_0.9']:.4f} off50={rec['fg_off_0.5']:.4f}",
              flush=True)

    def med(rows, k):
        v = [r[k] for r in rows if r.get(k) is not None]
        return float(np.median(v)) if v else None
    inv = [r for r in frames if r["weak_iso"] is not None and r["weak_iso"] > 0.4]
    vis = [r for r in frames if r["weak_iso"] is not None and r["weak_iso"] < 0.1]
    summary = {}
    for name, rows in (("invisible_mode", inv), ("visible_mode", vis)):
        summary[name] = {"n": len(rows), **{k: med(rows, k) for k in
                         ("p_label_median", "p_label_ge098", "p_label_ge05", "p_far_median", "p_far_p99",
                          "fg_0.98", "fg_off_0.98", "fg_0.9", "fg_off_0.9", "fg_0.5", "fg_off_0.5")}}
    P = np.concatenate(pairs_all, 0)
    summary["label_recall_098_by_local_snr"] = {}
    for lo, hi in zip(SNR_EDGES[:-1], SNR_EDGES[1:]):
        m = (P[:, 0] >= lo) & (P[:, 0] < hi)
        key = f"{lo}-{hi}" if hi < 1e8 else f"{lo}+"
        if m.sum() > 100:
            summary["label_recall_098_by_local_snr"][key] = {"n": int(m.sum()), "recall_098": float(np.mean(P[m, 1] >= 0.98)),
                                                           "recall_09": float(np.mean(P[m, 1] >= 0.9)),
                                                           "real_recall_098": REAL_RECALL.get(key)}
    summary["inject"] = args.inject
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"args": vars(args), "summary": summary, "frames": frames}, open(args.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
