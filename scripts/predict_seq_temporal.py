#!/usr/bin/env python3
"""Predict over sequences with TEMPORAL context: the three input channels carry (t-1, t, t+1).

``predict_h5_dir.py`` replicates one frame into all three channels, which is the model's
single-frame mode. This is the other mode of the same network: neighbouring frames of the same
sequence go into the channels instead. Frames at a sequence boundary reuse the edge frame, so
the first and last frame of a sequence degrade to single-frame rather than reaching into
another sequence -- mixing sequences would silently invent motion.

Both modes are exposed by ``--mode`` so the SAME script produces the comparison, rather than
two scripts whose preprocessing could drift apart.

    SEG_WEIGHTS=/home/prusek/mt_enc_exp/dino_seg_ori_temporal.pth \
    PYTHONPATH=src ~/dinov3_env/bin/python scripts/predict_seq_temporal.py \
        --data data/synth_seq --out /home/prusek/mt_enc_exp/synth_seq_pred_temporal --mode temporal
"""
import argparse
import collections
import glob
import os
import sys

os.environ.setdefault("SEG_MODE", "ori")
os.environ.setdefault("SEG_BACKBONE", "dinov2")
os.environ.setdefault("SEG_INPUT", "raw")
os.environ.setdefault("SEG_ARCH", "base")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.ndimage import zoom  # noqa: E402

sys.path.insert(0, "/home/prusek/mt_enc_exp/scripts")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from dino_seg import IMA_M, IMA_S, DinoSeg  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402

UP, THR = 1.5, 0.35
TILE, STRIDE = 518, 392
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def norm01(a, p=(1, 99)):
    lo, hi = np.percentile(a, p)
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)


@torch.no_grad()
def predict_stack(model, stack: np.ndarray) -> np.ndarray:
    """Tiled prediction from a (3, H, W) input stack."""
    _, H, W = stack.shape
    ys = list(range(0, max(1, H - TILE + 1), STRIDE))
    xs = list(range(0, max(1, W - TILE + 1), STRIDE))
    if ys[-1] != H - TILE:
        ys.append(max(0, H - TILE))
    if xs[-1] != W - TILE:
        xs.append(max(0, W - TILE))
    acc = cnt = None
    for y in ys:
        for x in xs:
            t = stack[:, y:y + TILE, x:x + TILE]
            th, tw = t.shape[1], t.shape[2]
            tt = ((torch.from_numpy(np.ascontiguousarray(t)) - IMA_M) / IMA_S)[None].to(DEV)
            o = model(tt)
            if isinstance(o, (tuple, list)):
                o = o[0]
            o = torch.sigmoid(o)[0].cpu().numpy()
            if acc is None:
                acc = np.zeros((o.shape[0], H, W), dtype=np.float32)
                cnt = np.zeros((H, W), dtype=np.float32)
            acc[:, y:y + th, x:x + tw] += o[:, :th, :tw]
            cnt[y:y + th, x:x + tw] += 1
    return acc / np.maximum(cnt, 1)[None]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="temporal", choices=["temporal", "single"])
    ap.add_argument("--weights", default=os.environ.get(
        "SEG_WEIGHTS", "/home/prusek/mt_enc_exp/dino_seg_ori_temporal.pth"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model = DinoSeg().to(DEV).eval()
    model.load_state_dict(torch.load(args.weights, map_location=DEV))

    # Group by sequence so neighbours are real neighbours.
    seqs = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        fr = read_frame_h5(p)
        a = fr["attrs"]
        seqs[int(a.get("seq_id", 0))].append((int(a.get("frame_idx", 0)), p, fr))
    for s in seqs:
        seqs[s].sort(key=lambda t: t[0])

    n_done, fgs = 0, []
    for sid, frames in sorted(seqs.items()):
        imgs = [zoom(norm01(np.asarray(fr["image"], dtype=float)), UP, order=1)
                for _, _, fr in frames]
        for k, (_, path, _fr) in enumerate(frames):
            if args.mode == "single":
                stack = np.repeat(imgs[k][None], 3, axis=0)
            else:
                # Clamp at the sequence edge: a first or last frame degrades to single-frame
                # rather than borrowing from a different sequence.
                prev = imgs[max(k - 1, 0)]
                nxt = imgs[min(k + 1, len(imgs) - 1)]
                stack = np.stack([prev, imgs[k], nxt])
            ch = predict_stack(model, stack.astype(np.float32))
            fg = float((ch.max(axis=0) > THR).mean())
            fgs.append(fg)
            np.savez_compressed(
                os.path.join(args.out, os.path.basename(path).replace(".h5", ".npz")),
                prob=ch.astype(np.float16))
            n_done += 1
            if n_done % 20 == 0:
                print(f"  {n_done} frames, fg%={100 * np.mean(fgs):.2f}", flush=True)
    print(f"\n{n_done} frames -> {args.out}\nmode={args.mode}  "
          f"mean fg% = {100 * np.mean(fgs):.2f}  (in-domain reference ~1.6)", flush=True)


if __name__ == "__main__":
    main()
