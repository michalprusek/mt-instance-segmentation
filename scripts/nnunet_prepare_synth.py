#!/usr/bin/env python3
"""Convert the synthetic training set into nnU-Net v2 raw format. RUNS ON TULEN.

Trains on exactly the data ``dino_seg`` (v4b) sees -- same generator, same calibration, same
``mask_hw=1.0`` -- so the comparison isolates the ARCHITECTURE. Note that v4b is trained at the
generator's native scale and applied to 1.5x-upscaled real frames; nnU-Net is given the same
deal rather than a corrected one, because changing the scale convention would compare two
things at once.

Binary foreground only. nnU-Net's segmentation head assigns each pixel exactly one class, so
it structurally cannot express the AMODAL overlap the K=6 orientation channels carry (a
crossing pixel belonging to two microtubules at once). That is a real architectural
difference, not an oversight: instancer A can consume an nnU-Net mask, instancer B cannot.

    ~/nnunet_env/bin/python scripts/nnunet_prepare_synth.py \
        --src ~/mt_enc_exp/nnunet_synth --root /disk2/prusek/nnunet
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import numpy as np
from PIL import Image

DATASET_ID = 501
DATASET_NAME = f"Dataset{DATASET_ID}_MTSynth"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/prusek/mt_enc_exp/nnunet_synth")
    ap.add_argument("--root", default="/disk2/prusek/nnunet")
    args = ap.parse_args()

    raw = os.path.join(args.root, "nnUNet_raw", DATASET_NAME)
    img_dir = os.path.join(raw, "imagesTr")
    lbl_dir = os.path.join(raw, "labelsTr")
    for d in (img_dir, lbl_dir,
              os.path.join(args.root, "nnUNet_preprocessed"),
              os.path.join(args.root, "nnUNet_results")):
        os.makedirs(d, exist_ok=True)

    images = sorted(glob.glob(os.path.join(args.src, "images", "*.png")))
    if not images:
        raise SystemExit(f"no images under {args.src}/images")

    n = 0
    for ip in images:
        stem = os.path.splitext(os.path.basename(ip))[0]
        mp = os.path.join(args.src, "masks", f"{stem}.png")
        if not os.path.exists(mp):
            continue
        shutil.copy(ip, os.path.join(img_dir, f"mt_{stem}_0000.png"))
        # nnU-Net wants label VALUES 0/1, not 0/255.
        m = (np.asarray(Image.open(mp)) > 127).astype(np.uint8)
        Image.fromarray(m).save(os.path.join(lbl_dir, f"mt_{stem}.png"))
        n += 1

    with open(os.path.join(raw, "dataset.json"), "w") as fh:
        json.dump({
            "channel_names": {"0": "IRM"},
            "labels": {"background": 0, "microtubule": 1},
            "numTraining": n,
            "file_ending": ".png",
            "overwrite_image_reader_writer": "NaturalImage2DIO",
        }, fh, indent=2)

    print(f"{DATASET_NAME}: {n} training cases -> {raw}")
    print("\nNext, on tulen:")
    print(f"  export nnUNet_raw={args.root}/nnUNet_raw")
    print(f"  export nnUNet_preprocessed={args.root}/nnUNet_preprocessed")
    print(f"  export nnUNet_results={args.root}/nnUNet_results")
    print(f"  ~/nnunet_env/bin/nnUNetv2_plan_and_preprocess -d {DATASET_ID} "
          f"-pl nnUNetPlannerResEncM -c 2d --verify_dataset_integrity")
    print("  # INSPECT the plan's patch_size / n_pool_per_axis before training: heavy")
    print("  #   downsampling coarsens 2-px filaments, which is exactly how the v8/ASPP")
    print("  #   decoder upgrade destroyed strict-tolerance localisation (Alice tol2")
    print("  #   0.940 -> 0.914 with fg% rising).")
    print(f"  ~/nnunet_env/bin/nnUNetv2_train {DATASET_ID} 2d 0 "
          f"-p nnUNetResEncUNetMPlans -tr nnUNetTrainer_250epochs")


if __name__ == "__main__":
    main()
