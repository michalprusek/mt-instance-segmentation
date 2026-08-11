#!/usr/bin/env python3
"""Build a SYNTHETIC evaluation set with exact ground truth. RUNS ON TULEN.

MT-34's ground truth is human-corrected v7+PySOAX output: it carries an agreement bias toward
PySOAX-family tracers and is demonstrably incomplete on sparse frames. Synthetic ground truth
has neither problem -- the centerlines ARE the objects that were drawn -- so this set answers a
question MT-34 cannot: how good is the instancer when the annotation is exactly right?

It is deliberately IN-DOMAIN (same calibrated generator, same real background pool as training,
different seeds). That is the point: combined with MT-34 it separates the two error sources.

    instancer error   <- measured here, on exact GT, in-domain
    + annotation error + domain gap  <- the rest of the MT-34 gap

Written in the MT-34 h5 layout (``image`` + ``polylines/pl_XXXX`` as ``(x=col, y=row)`` at
native 1x, with a ``split`` attr) so every existing script consumes it unchanged.

    ~/dinov3_env/bin/python scripts/build_synth_eval.py --n 40
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/prusek/mt_enc_exp/synth")
sys.path.insert(0, "/home/prusek/mt_enc_exp/scripts")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))

from gen_train import build_cfg  # noqa: E402
from mt_generator import generate_frame  # noqa: E402
from mt_bench.cvat_import import assign_split, write_frame_h5  # noqa: E402

# Well clear of gen_train's training seeds (10000 + i, i < 2000). These frames were never
# trained on, even though the generator and background pool are the same.
SEED_BASE = 900_000
BG_GLOB = "/home/prusek/BIOCEV/datasets/microtubules/IRM_backgrounds_v2/*.tif"


def clip_polyline(cl: np.ndarray, shape: tuple[int, int],
                  min_len: float) -> np.ndarray | None:
    """Keep the part of a centerline inside the crop; drop it if too little remains.

    A human annotator would not draw a 3-px stub poking into the corner, and neither should
    the reference -- otherwise the model is charged a false negative for something nobody
    would call a microtubule.
    """
    h, w = shape
    inside = ((cl[:, 0] >= 0) & (cl[:, 0] <= w - 1)
              & (cl[:, 1] >= 0) & (cl[:, 1] <= h - 1))
    if inside.sum() < 2:
        return None
    # Longest contiguous run inside the frame.
    idx = np.where(inside)[0]
    splits = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    run = max(splits, key=len)
    if len(run) < 2:
        return None
    seg = cl[run]
    length = float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum())
    return seg if length >= min_len else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--crop", type=int, default=768)
    ap.add_argument("--calib", default="/home/prusek/mt_enc_exp/calib_reg418_morph.json")
    ap.add_argument("--mask-hw", type=float, default=1.0)
    ap.add_argument("--min-len", type=float, default=15.0)
    ap.add_argument("--out", default="data/synth_eval")
    ap.add_argument("--png-dir", default="/home/prusek/mt_enc_exp/synth_eval/tif")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.png_dir, exist_ok=True)
    cfg = dataclasses.replace(
        build_cfg(json.load(open(args.calib))["best_params"], mask_hw=args.mask_hw))
    bgs = sorted(glob.glob(BG_GLOB))
    c = args.crop

    names, counts = [], []
    for i in range(args.n):
        rng = np.random.default_rng(SEED_BASE + i)
        bg = np.asarray(Image.open(bgs[rng.integers(len(bgs))]), float)
        while bg.ndim > 2:
            bg = bg[..., 0]
        H, W = bg.shape
        bg = bg[H // 2 - c // 2:H // 2 + c // 2, W // 2 - c // 2:W // 2 + c // 2]
        img, inst, _meta = generate_frame(bg, rng, cfg)

        polylines = []
        for ins in inst:
            cl = np.asarray(ins["centerline"], dtype=float)   # (x, y)
            if len(cl) < 2:
                continue
            seg = clip_polyline(cl, (c, c), args.min_len)
            if seg is not None:
                polylines.append(seg)

        name = f"synth_{i:04d}"
        names.append(name)
        counts.append(len(polylines))
        # Same 1/99 percentile stretch the training images use, so v4b sees what it trained on.
        lo, hi = np.percentile(img, [1, 99])
        v = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)
        import tifffile
        tifffile.imwrite(os.path.join(args.png_dir, f"{name}.tif"),
                         v.astype(np.float32))
        write_frame_h5(os.path.join(args.out, f"{name}.h5"), v, polylines,
                       {"source_task": "synth", "frame_id": i, "n_manual": len(polylines),
                        "reviewed": True, "sources": ",".join(["exact"] * len(polylines))})

    # Split written in a second pass so it is a deterministic function of the whole name list.
    import h5py
    split = assign_split(names)
    for name in names:
        with h5py.File(os.path.join(args.out, f"{name}.h5"), "r+") as h:
            h.attrs["split"] = split[name]

    n_val = sum(1 for n in names if split[n] == "val")
    print(f"synth_eval: {len(names)} frames, {sum(counts)} exact GT polylines "
          f"({np.mean(counts):.1f}/frame), {n_val} val / {len(names) - n_val} test")
    print(f"  h5 -> {args.out}   images -> {args.png_dir}")


if __name__ == "__main__":
    main()
