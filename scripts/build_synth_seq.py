#!/usr/bin/env python3
"""Build a synthetic SEQUENCE set with exact cross-frame correspondence. RUNS ON TULEN.

There is no real cross-frame ground truth for microtubules -- CVAT project 31 turned out to
hold static crop rectangles, not filament tracks -- so every quantitative tracking claim has to
rest on synthetic video. This writes that video.

Layout is deliberately the MT-34 h5 layout, one file per frame
(``seq000_f00.h5``, ``seq000_f01.h5``, ...), so `predict_v4b_mt34.py`, `run_oracle_eval.py` and
every other existing script consume it unchanged. The correspondence lives in two extra
attributes that older readers simply ignore:

* ``inst_ids``  -- one integer per polyline; the SAME integer in another frame is the SAME
  microtubule. This is the tracking ground truth.
* ``speeds``    -- the sampled gliding speed of each instance, px/frame, for velocity error.

**The split is assigned per SEQUENCE, not per frame.** Frames of one sequence are near-copies
of each other; splitting them individually would put a frame's own neighbour in the other half
and make every number meaningless.

    ~/dinov3_env/bin/python scripts/build_synth_seq.py --n-seq 24 --n-frames 5
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/prusek/mt_enc_exp/synth")
sys.path.insert(0, "/home/prusek/mt_enc_exp/scripts")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "synth"))

from mt_bench.cvat_import import write_frame_h5  # noqa: E402
from mt_sequence import MotionConfig, generate_sequence  # noqa: E402

BG_DIR = "/home/prusek/BIOCEV/datasets/microtubules/IRM_backgrounds_v2"


def _load_cfg(calib: str | None):
    """The calibrated generator config, exactly as training builds it."""
    from gen_train import build_cfg
    params = json.load(open(calib))["best_params"] if calib else {}
    return build_cfg(params, mask_hw=1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seq", type=int, default=24)
    ap.add_argument("--n-frames", type=int, default=5)
    ap.add_argument("--out", default="data/synth_seq")
    ap.add_argument("--calib", default="/home/prusek/mt_enc_exp/calib_reg418_morph.json")
    ap.add_argument("--bg-dir", default=BG_DIR)
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--drift-px", type=float, default=1.0,
                    help="stage drift per frame; a confound the tracker must not report "
                         "as motility")
    ap.add_argument("--crop", type=int, default=768,
                    help="centre crop of the background, matching build_synth_eval.py")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = _load_cfg(args.calib)
    mcfg = MotionConfig(drift_px_std=args.drift_px)
    bgs = sorted(glob.glob(os.path.join(args.bg_dir, "*.tif")))
    if not bgs:
        raise SystemExit(f"no backgrounds under {args.bg_dir}")

    rng = np.random.default_rng(args.seed)
    manifest = []
    for s in range(args.n_seq):
        bg = np.asarray(Image.open(bgs[int(rng.integers(len(bgs)))]), dtype=np.float64)
        while bg.ndim > 2:
            bg = bg[..., 0]
        # Centre-crop to the SAME size build_synth_eval.py uses, and do NOT rescale the raw
        # values. Both matter: the generator places a fixed number of filaments into whatever
        # frame it is handed, so a full 2048^2 background spreads them over seven times the
        # area and the frame comes out nearly empty -- measured as 0.01 % foreground against
        # the ~1.6 % this model expects, which looks exactly like a broken model and is not.
        c = args.crop
        H, W = bg.shape
        bg = bg[H // 2 - c // 2:H // 2 + c // 2, W // 2 - c // 2:W // 2 + c // 2]
        images, per_frame, meta = generate_sequence(bg, rng, cfg, n_frames=args.n_frames,
                                                   mcfg=mcfg)
        split = "val" if s % 2 == 0 else "test"      # per SEQUENCE, never per frame
        speeds = meta.get("speeds", {})
        for k, (img, inst) in enumerate(zip(images, per_frame)):
            ids = [int(i["inst_id"]) for i in inst]
            write_frame_h5(
                os.path.join(args.out, f"seq{s:03d}_f{k:02d}.h5"),
                img, [np.asarray(i["centerline"], dtype=np.float64) for i in inst],
                attrs={"split": split, "seq_id": s, "frame_idx": k,
                       "regime": meta["regime"], "n_frames": args.n_frames,
                       "inst_ids": np.asarray(ids, dtype=np.int32),
                       "speeds": np.asarray([float(speeds.get(i, 0.0)) for i in ids],
                                            dtype=np.float32),
                       "source_task": meta["regime"]})   # so per-source reporting still works
        manifest.append({"seq": s, "split": split, "regime": meta["regime"],
                         "n_instances": meta["n_instances"], "frames": len(images)})
        print(f"  seq{s:03d} {meta['regime']:8s} {len(images)} frames "
              f"{meta['n_instances']:3d} instances -> {split}", flush=True)

    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    n_val = sum(1 for m in manifest if m["split"] == "val")
    print(f"\nwrote {len(manifest)} sequences to {args.out} "
          f"({n_val} val / {len(manifest) - n_val} test)")
    print("Split is per SEQUENCE: neighbouring frames never straddle it.")


if __name__ == "__main__":
    main()
