#!/usr/bin/env python3
"""Run the semantic model over a directory of benchmark h5 frames. RUNS ON TULEN.

``predict_v4b_mt34.py`` reads ``.tif``; the synthetic sets are written as h5 in the MT-34
layout, so this is the same predictor pointed at that layout. Preprocessing is copied from it
verbatim -- ``zoom(norm01(img), 1.5, order=1)`` -- so the saved channels sit in the same 1.5x
frame as every other number in the project and stay comparable.

    SEG_WEIGHTS=/home/prusek/mt_enc_exp/dino_seg_ori_v4b.pth \
    ~/dinov3_env/bin/python scripts/predict_h5_dir.py \
        --data data/synth_seq --out /home/prusek/mt_enc_exp/synth_seq_pred
"""
import argparse
import glob
import os
import sys

os.environ.setdefault("SEG_MODE", "ori")
os.environ.setdefault("SEG_BACKBONE", "dinov2")
os.environ.setdefault("SEG_INPUT", "raw")
os.environ.setdefault("SEG_ARCH", "base")
os.environ.setdefault("SEG_WEIGHTS", "/home/prusek/mt_enc_exp/dino_seg_ori_v4b.pth")

import numpy as np  # noqa: E402
from scipy.ndimage import zoom  # noqa: E402

sys.path.insert(0, "/home/prusek/mt_enc_exp/scripts")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))

from learn_amodal import norm01, prob_channels  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402

UP = 1.5
THR = 0.35


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.data, "*.h5")))
    if not paths:
        raise SystemExit(f"no h5 frames under {args.data}")
    print(f"{len(paths)} frames -> {args.out}", flush=True)

    fgs = []
    for p in paths:
        img = zoom(norm01(read_frame_h5(p)["image"].astype(float)), UP, order=1)
        ch = prob_channels(img)
        fg = float((ch.max(axis=0) > THR).mean())
        fgs.append(fg)
        np.savez_compressed(os.path.join(args.out, os.path.basename(p).replace(".h5", ".npz")),
                            prob=ch.astype(np.float16))
        print(f"  {os.path.basename(p):28s} fg%={100 * fg:.2f}", flush=True)
    print(f"\nmean predicted fg% = {100 * np.mean(fgs):.2f}  "
          f"(in-domain synth reference ~1.6; >3x would be over-firing)", flush=True)


if __name__ == "__main__":
    main()
