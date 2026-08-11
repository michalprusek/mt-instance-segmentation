#!/usr/bin/env python3
"""Run the v4b semantic model over MT-34 and save its K=6 orientation channels.

RUNS ON TULEN (needs ~/dinov3_env + the checkpoint). Loading mirrors ``amodal_eval2.py``
exactly -- same env vars, same ``zoom(norm01(img), 1.5, order=1)`` -- so the saved channels
sit in the same 1.5x eval frame as everything else and the numbers stay comparable.

    ssh prusek@tulen.utia.cas.cz
    cd ~/mt_enc_exp && SEG_WEIGHTS=$HOME/mt_enc_exp/dino_seg_ori_v4b.pth \
        ~/dinov3_env/bin/python scripts/predict_v4b_mt34.py
"""
import os
import sys

os.environ.setdefault("SEG_MODE", "ori")
os.environ.setdefault("SEG_BACKBONE", "dinov2")
os.environ.setdefault("SEG_INPUT", "raw")
os.environ.setdefault("SEG_ARCH", "base")
os.environ.setdefault("SEG_WEIGHTS", "/home/prusek/mt_enc_exp/dino_seg_ori_v4b.pth")

import glob  # noqa: E402

import numpy as np  # noqa: E402
import tifffile  # noqa: E402
from scipy.ndimage import zoom  # noqa: E402

sys.path.insert(0, "/home/prusek/mt_enc_exp/scripts")
sys.path.insert(0, os.environ.get("MT_SRC", "/home/prusek/mt_enc_exp/mt_src"))
from learn_amodal import norm01, prob_channels  # noqa: E402
from mt_bench.fov import fov_mask  # noqa: E402


def norm01_fov(img):
    """Percentile stretch computed INSIDE the field of view.

    ``learn_amodal.norm01`` takes its 1st/99th percentiles over the WHOLE frame. Frames with a
    field stop (or merely a dark border -- Alice has one too) put that p1 out on the stop, so
    the imaged interior is squashed into a small part of [0, 1]: measured on MT-34 the
    interior's post-normalisation spread is 5.6-6.4x smaller than with an FOV-based stretch.
    The synthetic training frames have no stop and use the full range, so this is a
    train/inference distribution mismatch in the INPUT, not a model or instancer problem.
    Fixing it is preprocessing fidelity, not an inference trick.
    """
    m = fov_mask(img)
    lo, hi = np.percentile(img[m], [1, 99])
    return np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)

TIF_DIR = os.environ.get("MT34_TIF", "/home/prusek/mt_enc_exp/mt34/tif")
OUT_DIR = os.environ.get("MT34_PRED", "/home/prusek/mt_enc_exp/mt34_pred")
UP = 1.5
THR = 0.35


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    tifs = sorted(glob.glob(os.path.join(TIF_DIR, "*.tif")))
    print(f"{len(tifs)} frames -> {OUT_DIR}", flush=True)
    fgs = []
    for t in tifs:
        raw = tifffile.imread(t).astype(float)
        nrm = norm01_fov if os.environ.get("FOV_NORM", "0") == "1" else norm01
        img = zoom(nrm(raw), UP, order=1)
        ch = prob_channels(img)                       # (K, H, W) float
        fg = float((ch.max(axis=0) > THR).mean())
        fgs.append(fg)
        out = os.path.join(OUT_DIR, os.path.basename(t).replace(".tif", ".npz"))
        np.savez_compressed(out, prob=ch.astype(np.float16))
        print(f"  {os.path.basename(t):48s} {ch.shape}  fg%={100 * fg:.2f}", flush=True)
    print(f"\nmean predicted fg% = {100 * np.mean(fgs):.2f}", flush=True)
    print("Over-firing gate: compare against in-domain synth fg%; <3x = clean.", flush=True)


if __name__ == "__main__":
    main()
