#!/usr/bin/env python3
"""Does the trainer pick its checkpoint at the scale the model is deployed at?

Declared in docs/TODO.md 0.0-CAMPAIGN (diagnosis 30, 2026-09-07). The trainer scores every
fourth epoch with ``validate(...)``, which upscales the real VAL frame by 1.5x before the
forward pass -- but the synthetic training crops are never upscaled, and the campaign screens
at NATIVE inference. So the selection may rank epochs by behaviour at a scale the training
distribution does not contain. This scores the SAVED checkpoints of a run at both scales with
the trainer's own instrument and reports each curve's argmax.

Two checkpoints of the same run share a seed, so this comparison carries no seed noise.

    python scripts/diag_pick_scale.py --ckpt-dir runs/SPARSE35 --val-data data/real/mt34_eval \
        --val-n 16 --out runs/tune_s35/pick_scale_SPARSE35.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from mt_bench.cvat_import import read_frame_h5  # noqa: E402
import train_v5 as T  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--val-data", default="data/real/mt34_eval")
    ap.add_argument("--val-n", type=int, default=16)
    ap.add_argument("--arch", default="nnunet")
    ap.add_argument("--head", default="binary")
    ap.add_argument("--params", default="src/instance/params_a_model_v5H.json")
    ap.add_argument("--scales", default="1.0,1.5")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    frames = []
    for p in sorted(glob.glob(os.path.join(args.val_data, "*.h5"))):
        fr = read_frame_h5(p)
        if str(fr["attrs"].get("split")) == "val" and fr["polylines"]:
            frames.append(fr)
    frames = frames[: args.val_n]
    params = json.load(open(args.params))
    thr = params.get("prob_thr", 0.35)
    scales = [float(x) for x in args.scales.split(",")]
    cks = sorted(glob.glob(os.path.join(args.ckpt_dir, "ep*.pth")))
    print(f"{len(frames)} VAL frames, {len(cks)} checkpoints, scales {scales}, thr {thr}",
          flush=True)

    out_ch = {"binary": 1, "heat": 2}.get(args.head, T.K_ORI + 1)
    model = T.build_model(args.arch, out_ch).to(T.DEV) if hasattr(T, "build_model") else None
    if model is None:
        raise SystemExit("train_v5 has no build_model(); wire the arch factory it does use")

    rows = []
    for ck in cks:
        ep = int(os.path.basename(ck)[2:5])
        sd = torch.load(ck, map_location=T.DEV, weights_only=False)
        model.load_state_dict(sd, strict=False)
        rec = {"epoch": ep, "ckpt": ck}
        for up in scales:
            agg = T.validate(model, frames, params, thr, False,
                             T.TILE_FOR[args.arch], prob_channel=(0 if args.head == "heat" else -1),
                             up=up)
            rec[f"f1@{up}"] = float(agg["mean_f1"])
            rec[f"fg@{up}"] = float(agg["fg"])
        rows.append(rec)
        print("  ep%03d  " % ep + "  ".join(f"{up}x {rec[f'f1@{up}']:.4f} (fg {100*rec[f'fg@{up}']:.2f}%)"
                                            for up in scales), flush=True)

    print()
    for up in scales:
        b = max(rows, key=lambda r: r[f"f1@{up}"])
        print(f"argmax at {up}x: epoch {b['epoch']:3d}  f1 {b[f'f1@{up}']:.4f}"
              + "".join(f"   (at {o}x it scores {b[f'f1@{o}']:.4f})" for o in scales if o != up))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"rows": rows, "scales": scales, "val_data": args.val_data,
                   "n_frames": len(frames), "thr": thr}, open(args.out, "w"), indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
