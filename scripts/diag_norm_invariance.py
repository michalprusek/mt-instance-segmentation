#!/usr/bin/env python3
"""Is this network invariant to the percentile-normalisation SUPPORT, as InstanceNorm implies?

Training normalises a 768 render whole; evaluation normalises a 1024-2048 real frame whole. The
same real pixels therefore reach the network at amplitudes differing by ~3.2x at the median
(measured). That mismatch was the premise of arm 42 -- but `build_model` uses InstanceNorm2d and
inference is tiled at the training crop size, and conv -> InstanceNorm over matched tile support is
invariant to a GLOBAL AFFINE rescale of the input. If so, the network cannot see absolute
amplitude at all, only SNR and texture, and the whole amplitude line was measuring the wrong thing.

Feeds each frame twice -- normalised over the full frame, and over a size-matched window applied
globally -- and diffs the probability maps.

    KILL LINE (pre-declared): max |dp| < 0.01 over the frame => amplitude-invariant, the support
    mismatch is INERT for the model, and arm 42's stated mechanism is void.
    max |dp| > 0.05 => the mismatch is live and a support-matched inference sweep is warranted.

    PYTHONPATH=src:scripts python scripts/diag_norm_invariance.py --ckpt <pth> --data <dir> --n 4
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("src", "scripts"):
    sys.path.insert(0, os.path.join(ROOT, sub))

from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from eval_v5 import checkpoint_in_channels  # noqa: E402
from train_v5 import TILE_FOR, build_model, predict  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--arch", default="nnunet")
    ap.add_argument("--split", default="val")
    ap.add_argument("--support", type=int, default=768)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location="cuda")
    state = state.get("model", state) if isinstance(state, dict) and "model" in state else state
    hw = [v for k, v in state.items() if k.endswith("head.weight") and v.ndim == 4]
    out_ch = int(hw[0].shape[0]) if hw else 1
    model = build_model(args.arch, out_ch=out_ch,
                        in_ch=checkpoint_in_channels(state)).to("cuda").eval()
    model.load_state_dict(state, strict=False)
    print(f"loaded {out_ch}-channel head", flush=True)

    rows = []
    for p in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        fr = read_frame_h5(p)
        if str(fr["attrs"].get("split")) != args.split:
            continue
        img = np.asarray(fr["image"], float)
        h, w = img.shape
        if min(h, w) < args.support:
            continue
        def scale(lo, hi):
            return np.clip((img - lo) / (hi - lo + 1e-9), 0, 1).astype(np.float32)
        lo_f, hi_f = np.percentile(img, [1, 99])
        y, x = (h - args.support) // 2, (w - args.support) // 2
        win = img[y:y + args.support, x:x + args.support]
        lo_w, hi_w = np.percentile(win, [1, 99])
        # `predict` takes a 2-D image and returns (C, H, W); channel 0 is existence.
        a = predict(model, scale(lo_f, hi_f), TILE_FOR[args.arch])[0]
        b = predict(model, scale(lo_w, hi_w), TILE_FOR[args.arch])[0]
        d = np.abs(a - b)
        rows.append({"frame": os.path.basename(p)[:-3],
                     "span_ratio": float((hi_w - lo_w) / max(hi_f - lo_f, 1e-9)),
                     "max_abs_dp": float(d.max()), "mean_abs_dp": float(d.mean()),
                     "share_dp_gt_0.05": float((d > 0.05).mean()),
                     "fg_full": float((a > 0.98).mean()), "fg_win": float((b > 0.98).mean())})
        print(f"  {rows[-1]['frame']}: span x{rows[-1]['span_ratio']:.2f}  "
              f"max|dp| {rows[-1]['max_abs_dp']:.4f}  mean {rows[-1]['mean_abs_dp']:.5f}  "
              f"fg {rows[-1]['fg_full']:.4f} -> {rows[-1]['fg_win']:.4f}", flush=True)
        if len(rows) >= args.n:
            break

    mx = max(r["max_abs_dp"] for r in rows) if rows else float("nan")
    verdict = ("AMPLITUDE-INVARIANT (support mismatch inert, arm 42's mechanism void)" if mx < 0.01
               else "LIVE (support-matched inference warranted)" if mx > 0.05 else "AMBIGUOUS")
    summ = {"n": len(rows), "support": args.support, "max_abs_dp_over_frames": round(mx, 5),
            "verdict": verdict, "rows": rows}
    if args.out:
        json.dump(summ, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
