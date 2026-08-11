#!/usr/bin/env python3
"""Microtubule instance segmentation: image in, per-instance centerlines out.

Single entry point for the packaged model. Two stages:

  1. **semantic** -- a frozen DINOv2 ViT-L/14 backbone with a light high-resolution decoder
     predicts K=6 orientation-keyed foreground channels ("overpass" channels: at a crossing
     the two filaments land in different channels, so the representation stays amodal);
  2. **instancing** -- a curvature-bounded instancer groups the foreground into individual
     microtubules, enforcing kappa <= 0.25 rad/px as a HARD constraint. That bound is derived
     from data, not tuned: it sits just above the 0.239 rad/px maximum over 957
     human-annotated microtubules measured at an 8 px baseline. Microtubules cannot kink.

No human annotation enters either stage: the semantic model is trained purely on synthetic
frames and the instancer's hyperparameters are fitted on synthetic data with exact ground
truth.

Usage
-----
    python predict.py --input frame.tif --out-dir results/
    python predict.py --input folder/ --out-dir results/ --overlay

Outputs per frame: ``<name>.json`` with one polyline per instance (vertices are
``[x, y]`` = ``[col, row]`` in ORIGINAL image pixels) and, with ``--overlay``, a PNG.

Scale
-----
Inference runs at 1.5x upscale internally because that is the scale the model was trained and
evaluated at. Output coordinates are mapped back to the input resolution, so callers never see
the 1.5x.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import zoom

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "model"))

os.environ.setdefault("SEG_MODE", "ori")
os.environ.setdefault("SEG_BACKBONE", "dinov2")
os.environ.setdefault("SEG_INPUT", "raw")
os.environ.setdefault("SEG_ARCH", "base")

# Run fully offline. The DINOv2 repository is vendored under model/hub; pointing torch.hub at
# that directory makes it reuse the cached copy ("Using cache found in ...") instead of
# contacting GitHub. The backbone is then built WITHOUT pretrained weights, because our
# checkpoint is a full state_dict carrying every one of them (see model/dino_seg.py).
# Nothing is fetched at run time.
_HUB = os.path.join(_HERE, "model", "hub")
if os.path.isdir(os.path.join(_HUB, "facebookresearch_dinov2_main")):
    import torch as _torch_for_hub
    _torch_for_hub.hub.set_dir(_HUB)

UP = 1.5
KAPPA_MAX = 0.25
TILE, STRIDE = 518, 392
DEFAULT_WEIGHTS = os.path.join(_HERE, "weights", "dino_seg_ori_v4b.pth")
DEFAULT_PARAMS = os.path.join(_HERE, "params", "params_a_model_synthtuned.json")


def norm01(a, p=(1, 99)):
    """Percentile stretch over the whole frame -- exactly what training and eval used.

    An FOV-restricted variant was tested and lost on validation (0.412 vs 0.438); do not
    "improve" this without re-measuring, since the model was fitted to this input
    distribution.
    """
    lo, hi = np.percentile(a, p)
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)


def load_image(path: str) -> np.ndarray:
    """Read a 2D grayscale frame. Multi-channel input is reduced to its first channel."""
    if path.lower().endswith((".tif", ".tiff")):
        import tifffile
        img = tifffile.imread(path)
    else:
        from PIL import Image
        img = np.asarray(Image.open(path))
    img = np.asarray(img)
    while img.ndim > 2:
        img = img[0] if img.shape[0] <= 4 else img[..., 0]
    return img.astype(np.float64)


class Segmenter:
    """The semantic stage. Loads once, then predicts many frames."""

    def __init__(self, weights: str = DEFAULT_WEIGHTS, device: str | None = None):
        import torch
        from dino_seg import DinoSeg, IMA_M, IMA_S
        self._torch = torch
        self._ima = (IMA_M, IMA_S)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if not os.path.exists(weights):
            raise SystemExit(
                f"weights not found: {weights}\n"
                f"The checkpoint is ~1.2 GB and ships alongside this script under weights/.")
        self.model = DinoSeg().to(self.device).eval()
        self.model.load_state_dict(torch.load(weights, map_location=self.device))

    def channels(self, img01: np.ndarray) -> np.ndarray:
        """Tiled K-channel prediction; returns (K, H, W) in [0, 1]."""
        torch = self._torch
        IMA_M, IMA_S = self._ima
        H, W = img01.shape
        ys = list(range(0, max(1, H - TILE + 1), STRIDE))
        xs = list(range(0, max(1, W - TILE + 1), STRIDE))
        if ys[-1] != H - TILE:
            ys.append(max(0, H - TILE))
        if xs[-1] != W - TILE:
            xs.append(max(0, W - TILE))
        acc = cnt = None
        with torch.no_grad():
            for y in ys:
                for x in xs:
                    t = img01[y:y + TILE, x:x + TILE]
                    th, tw = t.shape
                    tt = torch.from_numpy(t.astype(np.float32))[None].repeat(3, 1, 1)
                    tt = ((tt - IMA_M) / IMA_S)[None].to(self.device)
                    o = self.model(tt)
                    if isinstance(o, (tuple, list)):
                        o = o[0]
                    o = torch.sigmoid(o)[0].cpu().numpy()
                    if acc is None:
                        acc = np.zeros((o.shape[0], H, W), dtype=np.float32)
                        cnt = np.zeros((H, W), dtype=np.float32)
                    acc[:, y:y + th, x:x + tw] += o[:, :th, :tw]
                    cnt[y:y + th, x:x + tw] += 1
        return acc / np.maximum(cnt, 1)[None]


def instance_polylines(channels: np.ndarray, params: dict, prob_thr: float):
    """The instancing stage. Returns polylines in the 1.5x frame."""
    from instance.instancer_a import instance_a
    prob = channels.max(axis=0)
    polylines, _ = instance_a(prob > prob_thr, KAPPA_MAX, params,
                              channels=channels, prob=prob)
    return polylines


def display01(img: np.ndarray) -> np.ndarray:
    """Display stretch from the CENTRAL crop — for the overlay only, never for the model.

    Many IRM frames have a saturated bright surround outside an octagonal field stop. Whole
    frame percentiles are then set by the surround and the imaged interior collapses to black,
    which makes the overlay unreadable even when the prediction is perfect. The central 60 % is
    inside the field stop for every frame we have seen.

    The model keeps :func:`norm01` regardless: it was trained on that input distribution and
    changing it would change the predictions, not just the picture.
    """
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape[:2]
    core = img[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
    lo, hi = np.percentile(core, [1.0, 99.0])
    return np.clip((img - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


def save_overlay(path: str, img01: np.ndarray, polylines) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(img01.shape[1] / 110, img01.shape[0] / 110))
    ax.imshow(img01, cmap="gray", interpolation="nearest")
    cmap = plt.get_cmap("hsv")
    for i, p in enumerate(polylines):
        p = np.asarray(p, dtype=float)
        ax.plot(p[:, 0], p[:, 1], "-", lw=1.1, color=cmap((i * 0.37) % 1.0), alpha=0.95)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="image file or a directory of images")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--params", default=DEFAULT_PARAMS)
    ap.add_argument("--prob-thr", type=float, default=None,
                    help="foreground threshold; default comes from the params file")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    ap.add_argument("--overlay", action="store_true", help="also write a PNG per frame")
    args = ap.parse_args()

    paths = ([args.input] if os.path.isfile(args.input) else
             sorted(sum([glob.glob(os.path.join(args.input, e))
                         for e in ("*.tif", "*.tiff", "*.png", "*.jpg")], [])))
    if not paths:
        raise SystemExit(f"no images found at {args.input}")
    os.makedirs(args.out_dir, exist_ok=True)

    params = json.load(open(args.params))
    params.pop("kappa_max", None)      # derived, never read from a params file
    thr = args.prob_thr if args.prob_thr is not None else params.get("prob_thr", 0.35)

    seg = Segmenter(args.weights, args.device)
    print(f"device={seg.device} | {len(paths)} frame(s) | threshold={thr:.3f}", flush=True)

    for path in paths:
        t0 = time.time()
        raw = load_image(path)
        img01 = zoom(norm01(raw), UP, order=1)
        chans = seg.channels(img01)
        polylines = instance_polylines(chans, params, thr)

        name = os.path.splitext(os.path.basename(path))[0]
        # Back to input resolution: callers should never have to know about the 1.5x.
        out = [(np.asarray(p, dtype=float) / UP).round(2).tolist() for p in polylines]
        with open(os.path.join(args.out_dir, f"{name}.json"), "w") as fh:
            json.dump({"image": os.path.basename(path),
                       "shape": [int(raw.shape[0]), int(raw.shape[1])],
                       "n_instances": len(out),
                       "coordinate_order": "x=col, y=row, in original image pixels",
                       "polylines": out}, fh)
        if args.overlay:
            save_overlay(os.path.join(args.out_dir, f"{name}.png"),
                         zoom(display01(raw), UP, order=1)[:img01.shape[0], :img01.shape[1]],
                         polylines)
        print(f"  {name}: {len(out)} instances  ({time.time() - t0:.1f}s)", flush=True)

    print(f"wrote {args.out_dir}/", flush=True)


if __name__ == "__main__":
    main()
