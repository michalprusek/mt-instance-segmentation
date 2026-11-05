#!/usr/bin/env python3
"""Score a semantic checkpoint by the TARGET metric: strict centerline-F1 after instancing.

RUNS ON TULEN (needs ~/dinov3_env and a checkpoint).

Why this exists. Every selection rule the project has used is a PROXY -- tolerant coverage F1,
then the ``fg_quality`` continuity battery. Section 17p is the cost of that: the continuity
proxy improved 25 % and downstream F1 did not move, because ``cc_per_gt`` counts mask
components while the metric counts predicted instances, and the instancer sits between them.
The only way to stop paying that is to put the instancer inside the validation loop.

It is affordable now because the honest validation set is SYNTHETIC (exact ground truth, no
annotation noise, and -- per the annotation-free result of section 17q -- a better tuning
signal than real VAL anyway). Twenty 768 px frames cost a few GPU-seconds each plus the
instancer on CPU, which is minutes, not hours.

    # score a checkpoint directly
    PYTHONPATH=src:third_party ~/dinov3_env/bin/python scripts/val_instance_f1.py \
        --ckpt /home/prusek/mt_enc_exp/dino_seg_ori_v4b.pth

    # or score predictions somebody else already wrote
    PYTHONPATH=src:third_party ~/dinov3_env/bin/python scripts/val_instance_f1.py \
        --pred-dir /home/prusek/mt_enc_exp/synth_pred_v4b
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

from instance.instancer_a import instance_a  # noqa: E402
from instance.metrics import aggregate_benchmark, centerline_f1  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402

UP = 1.5
KAPPA_MAX = 0.25
TOL, LEN_COV, PREC_COV = 5.0, 0.95, 0.95


def norm01(a: np.ndarray, p=(1, 99)) -> np.ndarray:
    """Whole-frame percentile stretch -- ``learn_amodal.norm01`` verbatim.

    Must stay identical to the prediction path used everywhere else, or the number this
    returns is not comparable with any other number in the project.
    """
    lo, hi = np.percentile(a, p)
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)


def load_split(data_dir: str, split: str) -> list[dict]:
    frames = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.h5"))):
        fr = read_frame_h5(path)
        if str(fr["attrs"].get("split")) != split:
            continue
        frames.append({"name": os.path.basename(path), "image": fr["image"],
                       "polylines": fr["polylines"]})
    return frames


def score_frame(chans: np.ndarray, polylines, params: dict, prob_thr: float) -> dict:
    """Instance the K-channel probabilities and score them against the GT polylines.

    ``chans`` is already in the ``UP`` frame; the GT is at native 1x, so it is scaled here --
    the same convention as ``run_oracle_eval``. Getting this backwards silently collapses the
    metric, which is why it is done in exactly one place.
    """
    prob = chans.max(axis=0)
    mask = prob > prob_thr
    _, masks = instance_a(mask, KAPPA_MAX, params=params, channels=chans, prob=prob)
    gt = [np.asarray(p, dtype=float) * UP for p in polylines]
    res = centerline_f1(masks, gt, tol=TOL, length_coverage=LEN_COV,
                        precision_coverage=PREC_COV)
    res["n_pred"] = len(masks)
    res["n_gt"] = len(gt)
    res["fg"] = float(mask.mean())
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", help="semantic checkpoint to run")
    src.add_argument("--pred-dir", help="directory of pre-computed <name>.npz with key 'prob'")
    ap.add_argument("--data", default="data/synth_eval")
    ap.add_argument("--split", default="val", choices=["val", "test"],
                    help="TEST is scored once, at the end of a project -- not from here")
    ap.add_argument("--params", default="src/instance/params_a_model_synthtuned.json")
    ap.add_argument("--prob-thr", type=float, default=None,
                    help="overrides the params file; default is the file's own prob_thr")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N frames")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    params = json.load(open(args.params)) if os.path.exists(args.params) else {}
    prob_thr = args.prob_thr if args.prob_thr is not None else params.get("prob_thr", 0.35)

    frames = load_split(args.data, args.split)
    if args.limit:
        frames = frames[:args.limit]
    if not frames:
        raise SystemExit(f"no {args.split} frames under {args.data}")

    predict = None
    if args.ckpt:
        os.environ.setdefault("SEG_MODE", "ori")
        os.environ.setdefault("SEG_BACKBONE", "dinov2")
        os.environ.setdefault("SEG_INPUT", "raw")
        os.environ.setdefault("SEG_ARCH", "base")
        import torch

        sys.path.insert(0, "/home/prusek/mt_enc_exp/scripts")
        from dino_seg import DinoSeg

        from train_gated import prob_channels as _pc
        model = DinoSeg().to("cuda").eval()
        state = torch.load(args.ckpt, map_location="cuda")
        # Per-epoch checkpoints hold trainable params only (train_gated strips the frozen
        # backbone), so strict loading would reject them; the backbone is identical anyway.
        model.load_state_dict(state, strict=False)

        def predict(img01):
            return _pc(model, img01)

    rows = []
    for fr in frames:
        if predict is not None:
            chans = predict(zoom(norm01(np.asarray(fr["image"], dtype=float)), UP, order=1))
        else:
            npz = np.load(os.path.join(args.pred_dir, fr["name"].replace(".h5", ".npz")))
            chans = npz["prob"].astype(np.float32)
        r = score_frame(chans, fr["polylines"], params, prob_thr)
        r["name"] = fr["name"]
        rows.append(r)
        print(f"  {r['name']:24s} gt={r['n_gt']:3d} pred={r['n_pred']:3d} "
              f"tp={r['tp']:3d} fp={r['fp']:3d} fn={r['fn']:3d} "
              f"f1={r['f1']:.3f} fg={100 * r['fg']:.2f}%", flush=True)

    agg = aggregate_benchmark(rows)
    print(f"\nINSTANCE-F1 ({args.split}, n={len(rows)}): mean={agg['mean_f1']:.4f} "
          f"micro={agg['micro_f1']:.4f} "
          f"tp={agg['total_tp']} fp={agg['total_fp']} fn={agg['total_fn']}", flush=True)
    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump({"agg": agg, "frames": rows, "prob_thr": prob_thr,
                       "params": args.params, "split": args.split}, fh, indent=1)
        print(f"wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
