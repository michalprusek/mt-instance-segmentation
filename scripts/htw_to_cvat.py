#!/usr/bin/env python3
"""Prepare the HTW frames for human review in CVAT.

Why this exists: the HTW ground truth is measurably incomplete. The model's foreground covers
100 % of the annotated centerlines, yet 82 % of the excess of its over-long predictions lands
nowhere near any annotation, on structures carrying 49x the contrast of a displaced-curve null.
Those are real microtubules nobody drew, and under the metric's precision criterion they do not
merely add false positives -- they also break the instances that would otherwise match annotated
ground truth. Reviewing the frames by hand is the only way to settle how large that is.

Two conversions are needed and both are easy to get wrong:

  1. The tif files are float32. CVAT renders them poorly or not at all, and the filaments are
     low-amplitude, so they are written out as 8-bit PNG under a percentile stretch. The stretch
     is per frame, which is what makes faint filaments visible.

  2. The .npy ground truth is stored at 1.5x the tif resolution (coordinates reach ~768 on a
     512-px image). It is divided by 1.5 on the way into the XML, or every polyline would sit at
     the wrong place and the reviewer would be shown nonsense.

Only the EXISTING HUMAN ground truth is exported. Model predictions are deliberately not
pre-loaded: correcting model output scales agreement bias along with the annotation, and the
whole point of this review is an estimate that does not depend on the model.

    python scripts/htw_to_cvat.py --out-dir /tmp/htw_cvat
"""
from __future__ import annotations

import argparse
import glob
import os
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

GT_SCALE = 1.5          # the .npy vertices are stored at 1.5x the tif resolution
LABEL = "microtubule"


def stretch_to_uint8(a: np.ndarray, lo_mad: float, hi_mad: float) -> np.ndarray:
    """Robust stretch about the background: median - lo_mad*MAD .. median + hi_mad*MAD.

    A plain percentile stretch puts both ends deep in the noise tail, and these frames are
    noise-dominated, so it renders as heavy speckle with the filaments barely above it. Anchoring
    on the background's own median and spread instead suppresses the speckle and leaves the dark
    filaments clearly visible -- which is the whole point, since a human has to spot the ones
    nobody annotated.
    """
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med))) * 1.4826
    if mad <= 0:
        mad = max(float(a.std()), 1e-6)
    lo, hi = med - lo_mad * mad, med + hi_mad * mad
    return (np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif-dir", default="data/real/htw_eval/tif")
    ap.add_argument("--gt-dir", default="data/real/htw_eval/gt")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--lo-mad", type=float, default=6.0,
                    help="stretch lower bound, in MADs BELOW the background median (dark filaments)")
    ap.add_argument("--hi-mad", type=float, default=3.0,
                    help="stretch upper bound, in MADs above the background median")
    args = ap.parse_args()

    img_dir = os.path.join(args.out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"

    tifs = sorted(glob.glob(os.path.join(args.tif_dir, "*.tif")))
    if not tifs:
        raise SystemExit(f"no tif files under {args.tif_dir}")

    n_poly = 0
    for i, tp in enumerate(tifs):
        stem = os.path.splitext(os.path.basename(tp))[0]
        a = np.asarray(Image.open(tp), dtype=np.float64)
        while a.ndim > 2:
            a = a[..., 0]
        H, W = a.shape
        png = f"htw_{stem}.png"
        Image.fromarray(stretch_to_uint8(a, args.lo_mad, args.hi_mad)).save(os.path.join(img_dir, png))

        im = ET.SubElement(root, "image", id=str(i), name=png, width=str(W), height=str(H))

        gp = os.path.join(args.gt_dir, stem + ".npy")
        if not os.path.isfile(gp):
            # A frame silently arriving with no ground truth would look like "nothing annotated
            # here", which is exactly the conclusion this review is trying to test.
            raise SystemExit(f"missing ground truth {gp} for {tp}")
        polys = np.load(gp, allow_pickle=True)
        for p in polys:
            p = np.asarray(p, dtype=float)
            if p.ndim != 2 or len(p) < 2:
                continue
            q = p / GT_SCALE                      # 1.5x -> native tif pixels
            q[:, 0] = np.clip(q[:, 0], 0, W - 1)
            q[:, 1] = np.clip(q[:, 1], 0, H - 1)
            pts = ";".join(f"{x:.2f},{y:.2f}" for x, y in q)
            ET.SubElement(im, "polyline", label=LABEL, occluded="0", source="manual",
                          points=pts, z_order="0")
            n_poly += 1

    xml_path = os.path.join(args.out_dir, "annotations.xml")
    ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)

    print(f"{len(tifs)} frames -> {img_dir}")
    print(f"{n_poly} existing ground-truth polylines -> {xml_path}")
    print(f"label = {LABEL!r}, coordinates divided by {GT_SCALE} into native tif pixels")

    # A cheap self-check the reviewer can trust: the annotated centerline must be DARKER than a
    # displaced copy of itself. If the scale conversion were wrong this would come out flat.
    a = np.asarray(Image.open(tifs[0]), dtype=np.float64)
    polys = np.load(os.path.join(args.gt_dir,
                                 os.path.splitext(os.path.basename(tifs[0]))[0] + ".npy"),
                    allow_pickle=True)
    on, off = [], []
    rng = np.random.default_rng(0)
    for p in polys:
        p = np.asarray(p, float) / GT_SCALE
        if len(p) < 2:
            continue
        c = np.clip(np.round(p[:, 0]).astype(int), 0, a.shape[1] - 1)
        r = np.clip(np.round(p[:, 1]).astype(int), 0, a.shape[0] - 1)
        on.append(a[r, c].mean())
        dx, dy = rng.integers(60, 200, 2)
        off.append(a[np.clip(r + dy, 0, a.shape[0] - 1), np.clip(c + dx, 0, a.shape[1] - 1)].mean())
    print(f"scale check on {os.path.basename(tifs[0])}: mean intensity under GT "
          f"{np.mean(on):.1f} vs displaced {np.mean(off):.1f} "
          f"({'OK, GT sits on dark filaments' if np.mean(on) < np.mean(off) else 'WRONG SCALE'})")


if __name__ == "__main__":
    main()
