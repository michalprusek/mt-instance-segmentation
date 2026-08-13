#!/usr/bin/env python3
"""Render a tracked sequence as a video: one colour per TRACK, held across frames.

The point of the colouring is that it makes identity legible without a legend. A microtubule
that keeps its colour from the first frame to the last was followed correctly; a filament that
changes colour mid-sequence was lost and re-acquired as a new object, which is the
fragmentation this project is fighting. Track ids are assigned in order of first appearance,
so the colour is stable even as tracks are born and die.

Works on any directory of consecutively-named frames -- real video frames extracted from a
recording, or a synthetic sequence -- so real and synthetic can be compared side by side.

    SEG_WEIGHTS=/home/prusek/mt_enc_exp/dino_seg_ori_v4b.pth PYTHONPATH=src \
    python scripts/viz_tracks_video.py --frames /path/to/frames --out tracks.mp4
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

os.environ.setdefault("SEG_MODE", "ori")
os.environ.setdefault("SEG_BACKBONE", "dinov2")
os.environ.setdefault("SEG_INPUT", "raw")
os.environ.setdefault("SEG_ARCH", "base")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import zoom  # noqa: E402

sys.path.insert(0, "/home/prusek/mt_enc_exp/scripts")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from instance.instancer_a import instance_a  # noqa: E402
from instance.tracker import track_sequence  # noqa: E402

UP, KAPPA_MAX = 1.5, 0.25


def display01(img):
    """Contrast from the CENTRAL crop: many IRM frames have a saturated surround outside the
    field stop that would otherwise drive the whole image to black."""
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape[:2]
    core = img[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
    lo, hi = np.percentile(core, [1.0, 99.0])
    return np.clip((img - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


def norm01(a, p=(1, 99)):
    """Model input normalisation -- the one it was trained with. Not the display stretch."""
    lo, hi = np.percentile(a, p)
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)


def load_frames(pattern_dir: str):
    paths = sorted(sum([glob.glob(os.path.join(pattern_dir, e))
                        for e in ("*.png", "*.tif", "*.tiff", "*.jpg")], []))
    if not paths:
        raise SystemExit(f"no frames under {pattern_dir}")
    out = []
    for p in paths:
        if p.lower().endswith((".tif", ".tiff")):
            import tifffile
            im = tifffile.imread(p)
        else:
            im = np.asarray(Image.open(p).convert("L"))
        im = np.asarray(im, dtype=np.float64)
        while im.ndim > 2:
            im = im[..., 0]
        out.append(im)
    return paths, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="directory of consecutive frames")
    ap.add_argument("--out", required=True, help="output .mp4 or .gif")
    ap.add_argument("--weights", default=os.environ.get(
        "SEG_WEIGHTS", "/home/prusek/mt_enc_exp/dino_seg_ori_v4b.pth"))
    ap.add_argument("--params", default="src/instance/params_a_model_synthtuned.json")
    ap.add_argument("--mode", default="single", choices=["single", "temporal"])
    ap.add_argument("--fps", type=float, default=1.5)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    paths, raws = load_frames(args.frames)
    print(f"{len(paths)} frames from {args.frames}", flush=True)

    import torch
    from dino_seg import IMA_M, IMA_S, DinoSeg
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = DinoSeg().to(dev).eval()
    model.load_state_dict(torch.load(args.weights, map_location=dev))

    imgs = [zoom(norm01(r), UP, order=1) for r in raws]
    TILE, STRIDE = 518, 392

    @torch.no_grad()
    def predict(k):
        if args.mode == "single":
            stack = np.repeat(imgs[k][None], 3, axis=0)
        else:
            stack = np.stack([imgs[max(k - 1, 0)], imgs[k], imgs[min(k + 1, len(imgs) - 1)]])
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
                t = np.ascontiguousarray(stack[:, y:y + TILE, x:x + TILE])
                tt = ((torch.from_numpy(t).float() - IMA_M) / IMA_S)[None].to(dev)
                o = model(tt)
                if isinstance(o, (tuple, list)):
                    o = o[0]
                o = torch.sigmoid(o)[0].cpu().numpy()
                if acc is None:
                    acc = np.zeros((o.shape[0], H, W), np.float32)
                    cnt = np.zeros((H, W), np.float32)
                acc[:, y:y + t.shape[1], x:x + t.shape[2]] += o[:, :t.shape[1], :t.shape[2]]
                cnt[y:y + t.shape[1], x:x + t.shape[2]] += 1
        return acc / np.maximum(cnt, 1)[None]

    params = json.load(open(args.params))
    params.pop("kappa_max", None)
    thr = params.get("prob_thr", 0.35)

    per_frame = []
    for k in range(len(imgs)):
        ch = predict(k)
        pls, _ = instance_a(ch.max(axis=0) > thr, KAPPA_MAX, params,
                            channels=ch, prob=ch.max(axis=0))
        per_frame.append(pls)
        print(f"  frame {k}: {len(pls)} instances", flush=True)

    tracks = track_sequence(per_frame)
    full = sum(1 for t in tracks if t.length == len(imgs))
    print(f"{len(tracks)} tracks, {full} spanning every frame", flush=True)

    # polyline -> track id, per frame, so the colour follows identity rather than position
    owner = [dict() for _ in per_frame]
    for tr in tracks:
        for k, poly in zip(tr.frames, tr.polylines):
            for j, p in enumerate(per_frame[k]):
                if np.array_equal(np.asarray(p), np.asarray(poly)):
                    owner[k][j] = tr.track_id
                    break

    disp = [zoom(display01(r), UP, order=1) for r in raws]
    H, W = disp[0].shape
    cmap = plt.get_cmap("hsv")
    fig, ax = plt.subplots(figsize=(W / 130, H / 130))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.axis("off")

    def draw(k):
        ax.clear()
        ax.axis("off")
        ax.imshow(disp[k], cmap="gray", interpolation="nearest")
        for j, p in enumerate(per_frame[k]):
            tid = owner[k].get(j, -1)
            col = cmap((tid * 0.37) % 1.0) if tid >= 0 else (0.5, 0.5, 0.5, 1.0)
            p = np.asarray(p, float)
            ax.plot(p[:, 0], p[:, 1], "-", lw=1.3, color=col, alpha=0.95)
        head = args.title or os.path.basename(os.path.normpath(args.frames))
        ax.text(0.01, 0.99, f"{head}\nframe {k + 1}/{len(disp)} · "
                            f"{len(per_frame[k])} instances · {len(tracks)} tracks",
                transform=ax.transAxes, va="top", ha="left", color="white", fontsize=9,
                bbox=dict(facecolor="black", alpha=0.55, pad=3, edgecolor="none"))
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)

    anim = animation.FuncAnimation(fig, draw, frames=len(disp), interval=1000 / args.fps)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    if args.out.lower().endswith(".gif"):
        anim.save(args.out, writer=animation.PillowWriter(fps=args.fps))
    else:
        try:
            anim.save(args.out, writer=animation.FFMpegWriter(fps=args.fps, bitrate=2400))
        except Exception as exc:                       # noqa: BLE001
            alt = os.path.splitext(args.out)[0] + ".gif"
            print(f"ffmpeg unavailable ({exc}); writing {alt}", flush=True)
            anim.save(alt, writer=animation.PillowWriter(fps=args.fps))
            args.out = alt
    plt.close(fig)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
