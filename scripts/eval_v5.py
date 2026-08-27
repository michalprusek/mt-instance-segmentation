#!/usr/bin/env python3
"""Score a v5 checkpoint properly: predict once, then sweep the decode threshold.

Two corrections this exists to make.

1. ``train_v5.py`` validates on the first 8 non-empty VAL frames, which is 6 Alice frames plus
   the two sparsest 586 frames -- the EASY half. Every crossing-dense frame in VAL is excluded.
   Selecting on that subset is acceptable (it is cheap and runs every few epochs); reporting on
   it is not, so this defaults to the full VAL split.
2. ``prob_thr`` in the shipped instancer params was tuned for v4b's probability calibration. A
   differently-trained model puts its mass somewhere else, and scoring it at v4b's threshold
   measures the calibration mismatch rather than the model. The sweep is a selection event and
   belongs on VAL, never on TEST.

    PYTHONPATH=src:third_party python scripts/eval_v5.py --ckpt .../v5A_ckpt/ep028.pth \
        --arch resunet --thrs 0.35,0.5,0.65,0.8
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import zoom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "third_party"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from instance.instancer_a import instance_a  # noqa: E402
from instance.metrics import aggregate_benchmark, centerline_f1  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.fov import fov_mask  # noqa: E402

UP, KAPPA_MAX, K_ORI = 1.5, 0.25, 6


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", help="checkpoint to run")
    # Sweeping a cached prediction directory through THIS code path is what makes the baseline
    # comparable: same instancer call, same FOV handling, same metric arguments. Re-deriving
    # the baseline from another script's numbers would compare two pipelines, not two models.
    src.add_argument("--pred-dir", help="cached <name>.npz with key 'prob' (e.g. v4b)")
    ap.add_argument("--arch", default="resunet", choices=["resunet", "dino", "nnunet"])
    ap.add_argument("--data", default="data/real/mt34_eval")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0, help="0 = all frames in the split")
    ap.add_argument("--include-empty", action="store_true",
                    help="keep frames with zero ground-truth polylines. Off by default so "
                         "the frame set matches every historical run; on for --dump-pred, "
                         "because the over-firing probe frame IS the excess being measured.")
    ap.add_argument("--thrs", default="0.35,0.45,0.55,0.65,0.75,0.85")
    ap.add_argument("--params", default="src/instance/params_a_model_synthtuned.json")
    ap.add_argument("--fg-head", action="store_true")
    ap.add_argument("--no-fov", action="store_true")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--dump-pred", default=None,
                    help="write <stem>.npz with key 'prob' so the instancer can be tuned "
                         "against this model's own masks. v4b has predict_h5_dir.py and "
                         "nnU-Net has its own predictor; v5 checkpoints had no dump path at "
                         "all, so they could be scored but never tuned for.")
    args = ap.parse_args()

    model = None
    if args.ckpt:
        import torch
        from train_v5 import TILE_FOR, build_model, norm01, predict
        state = torch.load(args.ckpt, map_location="cuda")
        # Read the head width off the checkpoint: pre-v5 models have K=6 channels, v5 adds the
        # binary union head, and --head binary has exactly 1. Assuming one shape makes the
        # others fail with a shape error, and re-scoring older checkpoints is the point here.
        #
        # Each architecture names its output layer differently, and getting this wrong fails
        # LOUDLY only by luck: the default of K+1 happened to match the amodal nnU-Net runs, so
        # the detection was never exercised until a 1-channel checkpoint hit it.
        seg = [v for k, v in state.items()
               if "decoder.seg_layers." in k and k.endswith(".weight")]   # nnU-Net ResEnc
        if seg:
            out_ch = int(seg[0].shape[0])
        else:
            hw = [v for k, v in state.items() if k.endswith("head.weight")
                  or k.endswith("head_ori.weight")]
            out_ch = int(hw[0].shape[0]) if hw else K_ORI + 1
            if any(k.endswith("head_fg.weight") for k in state):
                out_ch += 1
        model = build_model(args.arch, out_ch=out_ch).to("cuda").eval()
        model.load_state_dict(state, strict=False)
        print(f"loaded {out_ch}-channel head", flush=True)

    frames, names = [], []
    for p in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        fr = read_frame_h5(p)
        if str(fr["attrs"].get("split")) == args.split and (fr["polylines"] or args.include_empty):
            frames.append(fr)
            names.append(os.path.basename(p))
    if args.limit:
        frames, names = frames[:args.limit], names[:args.limit]
    params = json.load(open(args.params))
    tag = os.path.basename(args.ckpt) if args.ckpt else os.path.basename(args.pred_dir)
    print(f"{len(frames)} frames  arch={args.arch}  source={tag}", flush=True)

    # Predict once; the sweep is then pure CPU on cached probabilities.
    cache = []
    for fr, nm in zip(frames, names):
        if model is not None:
            ch = predict(model, zoom(norm01(np.asarray(fr["image"], float)), UP, order=1),
                         TILE_FOR[args.arch])
        else:
            ch = np.load(os.path.join(args.pred_dir,
                                      nm.replace(".h5", ".npz")))["prob"].astype(np.float32)
        H, W = ch.shape[1:]
        if args.no_fov:
            keep = np.ones((H, W), bool)
        else:
            # Field-stop handling must be byte-for-byte what run_oracle_eval.py does, because
            # every historical number in this project -- including TEST 0.457 -- came from that
            # path. This script originally eroded by 4 px and left the orientation channels
            # unmasked, which cost ~0.008 by deleting real microtubules near the rim and
            # changed what the w_ori term reads. Two harnesses differing by 0.008 is the size
            # of some of the margins being reported, so they must agree exactly.
            keep = zoom(fov_mask(fr["image"]).astype(np.uint8), UP, order=0).astype(bool)
            keep = np.pad(keep, ((0, max(0, H - keep.shape[0])),
                                 (0, max(0, W - keep.shape[1]))),
                          constant_values=True)[:H, :W]
        ch = ch * keep[None, :, :]
        if args.dump_pred:
            os.makedirs(args.dump_pred, exist_ok=True)
            np.savez_compressed(os.path.join(args.dump_pred, nm.replace(".h5", ".npz")),
                                prob=ch.astype(np.float16))
        cache.append((ch, keep, [np.asarray(q, float) * UP for q in fr["polylines"]]))
    print("predicted; sweeping", flush=True)

    results = {}
    for thr in [float(t) for t in args.thrs.split(",")]:
        rows = []
        for ch, keep, gt in cache:
            ori = ch[:K_ORI]
            prob = ch[K_ORI] if (args.fg_head and ch.shape[0] > K_ORI) else ori.max(0)
            mask = (prob > thr) & keep
            _, masks = instance_a(mask, KAPPA_MAX, params=params, channels=ori, prob=prob)
            r = centerline_f1(masks, gt, tol=5.0, length_coverage=0.95,
                              precision_coverage=0.95)
            r["fg"] = float(mask.mean())
            rows.append(r)
        a = aggregate_benchmark(rows)
        a["fg"] = float(np.mean([r["fg"] for r in rows]))
        # Per-frame rows travel with the aggregate so a paired bootstrap can be re-derived
        # offline: adding a confidence interval to a number must never mean re-running it.
        a["_frames"] = [{"name": n, "tp": r["tp"], "fp": r["fp"], "fn": r["fn"],
                         "f1": r["f1"]} for n, r in zip(names, rows)]
        results[thr] = a
        print("  thr=%.2f  F1=%.4f  tp=%3d fp=%3d fn=%3d  fg=%.2f%%"
              % (thr, a["mean_f1"], a["total_tp"], a["total_fp"], a["total_fn"],
                 100 * a["fg"]), flush=True)
    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump({"ckpt": args.ckpt or args.pred_dir, "arch": args.arch,
                       "split": args.split,
                       "n_frames": len(frames), "results": {str(k): v
                                                            for k, v in results.items()}},
                      fh, indent=1)


if __name__ == "__main__":
    main()
