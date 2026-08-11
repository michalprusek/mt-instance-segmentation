#!/usr/bin/env python3
"""Run the synth-trained nnU-Net over MT-34 and emit eval-compatible probability maps.

RUNS ON TULEN. Two scale conventions are produced because the fair one is not obvious:
v4b is trained at the generator's native scale but applied to 1.5x-upscaled frames, so nnU-Net
is given BOTH -- ``--scale 1.5`` (v4b's convention) and ``--scale 1.0`` (its own training
scale, upsampled afterwards for scoring). Picking the better on VAL and reporting both is how
nnU-Net gets its best shot rather than v4b's hand-me-down.

Input normalisation matches training: an 8-bit percentile stretch, but computed INSIDE the
field of view -- the synthetic crops have no field stop, so a whole-frame stretch would hand
nnU-Net a 6x flatter image than it trained on (see mt_bench.fov and protocol 17f).

Output: one ``<stem>.npz`` per frame with ``prob`` of shape ``(1, H*1.5, W*1.5)``, the shape the
instancer evaluation expects. Instancer B cannot consume this -- nnU-Net's softmax head assigns
each pixel exactly one class and so cannot express the amodal orientation overlap -- which is an
architectural difference worth stating, not an oversight.

    ~/nnunet_env/bin/python scripts/nnunet_predict_mt34.py --scale 1.5
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys

import numpy as np
import tifffile
from PIL import Image
from scipy.ndimage import zoom

sys.path.insert(0, os.environ.get("MT_SRC", "/home/prusek/mt_enc_exp/mt34_work/src"))
from mt_bench.fov import fov_mask  # noqa: E402

ROOT = "/disk2/prusek/nnunet"
UP = 1.5


def norm8_fov(img: np.ndarray) -> np.ndarray:
    m = fov_mask(img)
    lo, hi = np.percentile(img[m], [1, 99])
    v = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)
    return (v * 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif-dir", default="/home/prusek/mt_enc_exp/mt34/tif")
    ap.add_argument("--scale", type=float, default=1.5,
                    help="1.5 = v4b's inference convention; 1.0 = nnU-Net's training scale")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tag = f"s{args.scale:g}".replace(".", "")
    out_dir = args.out or f"/home/prusek/mt_enc_exp/mt34_pred_nnunet_{tag}"
    stage = f"/disk2/prusek/nnunet/infer_{tag}"
    raw_in, raw_out = os.path.join(stage, "in"), os.path.join(stage, "out")
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(raw_in, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    tifs = sorted(glob.glob(os.path.join(args.tif_dir, "*.tif")))
    shapes = {}
    for t in tifs:
        stem = os.path.splitext(os.path.basename(t))[0]
        img = tifffile.imread(t).astype(float)
        shapes[stem] = img.shape
        v = norm8_fov(img)
        if args.scale != 1.0:
            v = zoom(v, args.scale, order=1)
        Image.fromarray(v).save(os.path.join(raw_in, f"{stem}_0000.png"))
    print(f"staged {len(tifs)} frames at scale {args.scale}", flush=True)

    env = dict(os.environ,
               nnUNet_raw=f"{ROOT}/nnUNet_raw",
               nnUNet_preprocessed=f"{ROOT}/nnUNet_preprocessed",
               nnUNet_results=f"{ROOT}/nnUNet_results",
               nnUNet_compile="f")
    subprocess.run([
        os.path.expanduser("~/nnunet_env/bin/nnUNetv2_predict"),
        "-i", raw_in, "-o", raw_out, "-d", "501", "-c", "2d",
        "-p", "nnUNetResEncUNetMPlans", "-tr", "nnUNetTrainer_250epochs",
        "-f", "0", "--save_probabilities",
    ], check=True, env=env)

    fgs = []
    for t in tifs:
        stem = os.path.splitext(os.path.basename(t))[0]
        h, w = shapes[stem]
        hi_shape = (int(round(h * UP)), int(round(w * UP)))
        npz = np.load(os.path.join(raw_out, f"{stem}.npz"))
        prob = np.asarray(npz["probabilities"])            # (C, ...) with C = 2 classes
        fg = np.squeeze(prob)[1] if prob.shape[0] == 2 else np.squeeze(prob)
        if fg.shape != hi_shape:
            fg = zoom(fg, (hi_shape[0] / fg.shape[0], hi_shape[1] / fg.shape[1]), order=1)
        np.savez_compressed(os.path.join(out_dir, f"{stem}.npz"),
                            prob=fg[None, ...].astype(np.float16))
        fgs.append(float((fg > 0.5).mean()))
        print(f"  {stem:44s} {fg.shape}  fg%={100 * fgs[-1]:.2f}", flush=True)
    print(f"\nmean predicted fg% = {100 * np.mean(fgs):.2f}  -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
