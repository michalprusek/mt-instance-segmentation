#!/usr/bin/env python3
"""Validate the in-training gate against a KNOWN foreground. RUNS ON TULEN.

``train_gated.validate`` re-implements ``learn_amodal.prob_channels`` so it can score a live
model instead of a checkpoint. A re-implementation that silently disagrees would make every
selection decision meaningless, so it is checked against the numbers the standalone pipeline
already produced for v4b on MT-34 (whole-frame norm, thr 0.35, from `fg_metrics4.json`):

    cc_per_gt 10.69 | endp_per_kpx 23.57 | gaps_per_mt 1.56 | prec2 0.728 | rec2 0.971

Those were measured over all 33 GT frames; this runs the VAL half, so exact equality is not
expected -- agreement to within the val/test split difference is. A gross mismatch (an order of
magnitude, or an empty mask) means the preprocessing diverged.

    cd /home/prusek/mt_enc_exp/mt34_work
    SEG_MODE=ori PYTHONPATH=src ~/dinov3_env/bin/python scripts/check_gate_on_v4b.py
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SEG_MODE", "ori")
os.environ.setdefault("SEG_BACKBONE", "dinov2")
os.environ.setdefault("SEG_INPUT", "raw")

from train_gated import DEV, load_val, validate  # noqa: E402

REFERENCE = {"cc_per_gt": 10.69, "endp_per_kpx": 23.57, "gaps_per_mt": 1.56,
             "prec2": 0.728, "rec2": 0.971}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/home/prusek/mt_enc_exp/dino_seg_ori_v4b.pth")
    ap.add_argument("--val-data", default="data/real/mt34_eval")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    from dino_seg import DinoSeg
    model = DinoSeg().to(DEV)
    model.load_state_dict(torch.load(args.weights, map_location=DEV))

    frames, empty = load_val(args.val_data, args.split)
    print(f"{len(frames)} GT frames + {len(empty)} empty probe(s) from {args.split}",
          flush=True)
    props = validate(model, frames, empty)

    print(f"\n{'property':16s} {'measured':>10s} {'standalone':>11s}")
    for k, ref in REFERENCE.items():
        print(f"{k:16s} {props.get(k, float('nan')):10.3f} {ref:11.3f}")
    print(f"{'fg %':16s} {100 * props.get('fg', float('nan')):10.3f} "
          f"{'~1.93 (all)':>11s}")
    print(f"{'fg_empty %':16s} {100 * props.get('fg_empty', float('nan')):10.3f}")


if __name__ == "__main__":
    main()
