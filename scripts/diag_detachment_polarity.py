#!/usr/bin/env python3
"""Are the microtubule segments nobody annotated DETACHED ones?

In interference reflection microscopy the contrast of a filament is -cos(2*k*h) in its height h
above the coverslip, so a microtubule in contact is DARK, and a segment that lifts off passes
through zero at h = lambda/(8*n_water) = 56.4 nm and becomes BRIGHT. An annotator tracing dark
lines therefore has no visual cue that a bright stretch is the same filament, and will stop at the
detachment.

If that is what is happening, the segments our model finds and the annotation lacks must have the
OPPOSITE contrast sign from the annotated ones. If they merely have the same sign, the annotation
is plainly incomplete instead, which is a different problem with a different fix.

The distinction matters for what may be done about it. Adding filaments because our own model
found them fits the benchmark to the model. Adding segments because a stated physical rule says a
microtubule does not stop being a microtubule when it lifts off the glass is a correction to the
annotation CONVENTION, and it is defensible independently of any model.

    python scripts/diag_detachment_polarity.py --pred-dir preds/LEN_treatment --thr 0.997 --min-length 20
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.ndimage import zoom
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from instance.instancer_a import instance_a  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.fov import fov_mask  # noqa: E402

UP = 1.5
KAPPA_MAX = 0.25


def source_of(name):
    b = os.path.basename(name)
    return "htw" if b.startswith("htw") else ("585" if b.startswith("alice") else "586")


def densify(p, step=1.0):
    seg = np.sqrt(((p[1:] - p[:-1]) ** 2).sum(1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return p
    t = np.arange(0.0, s[-1], step)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)


def signed_contrast(img, poly, halfwin=7.0):
    """SIGNED filament-minus-flank contrast along a centerline. Negative = darker than its
    surroundings (in contact), positive = brighter (lifted past the interference zero)."""
    q = densify(poly)
    if len(q) < 5:
        return None
    t = np.gradient(q, axis=0)
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    H, W = img.shape

    def take(pts):
        c = np.round(pts[:, 0]).astype(int)
        r = np.round(pts[:, 1]).astype(int)
        ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
        out = np.full(len(pts), np.nan)
        out[ok] = img[r[ok], c[ok]]
        return out

    core = take(q)
    flank = np.nanmean(np.stack([take(q + n * halfwin), take(q - n * halfwin)]), axis=0)
    v = core - flank
    v = v[np.isfinite(v)]
    return v if len(v) >= 5 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--data-dir", default="data/real/mt34_htw_eval")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--params", default="src/instance/params_a_model_v5H.json")
    ap.add_argument("--thr", type=float, default=0.997)
    ap.add_argument("--min-length", type=float, default=20.0)
    ap.add_argument("--out", default="detachment_polarity.json")
    args = ap.parse_args()
    if args.split == "test":
        raise SystemExit("refusing to touch TEST: it carries a declared exposure budget")

    params = dict(json.load(open(args.params)))
    params["min_length"] = args.min_length
    params["prob_thr"] = args.thr

    acc = defaultdict(lambda: defaultdict(list))
    for path in sorted(glob.glob(os.path.join(args.data_dir, "*.h5"))):
        fr = read_frame_h5(path)
        if str(fr["attrs"].get("split")) != args.split:
            continue
        name = os.path.basename(path).replace(".h5", ".npz")
        src = source_of(name)
        img = fr["image"].astype(np.float64)
        gt = [np.asarray(p, float) for p in fr["polylines"] if len(p) >= 2]
        if not gt:
            continue
        npz = os.path.join(args.pred_dir, name)
        if not os.path.isfile(npz):
            raise SystemExit(f"missing prediction {npz} for a frame that has ground truth")
        prob = np.load(npz)["prob"].astype(np.float32)
        if prob.ndim == 3:
            prob = prob.max(axis=0)
        hi = (int(round(img.shape[0] * UP)), int(round(img.shape[1] * UP)))
        fv = zoom(fov_mask(fr["image"]).astype(np.uint8), UP, order=0).astype(bool)
        fv = np.pad(fv, ((0, max(0, hi[0] - fv.shape[0])), (0, max(0, hi[1] - fv.shape[1]))),
                    constant_values=True)[:hi[0], :hi[1]]

        # the frame's own reference polarity, from the annotated filaments
        gt_sign = []
        for g in gt:
            v = signed_contrast(img, g)
            if v is not None:
                gt_sign.append(float(np.median(v)))
        if not gt_sign:
            continue
        ref = float(np.median(gt_sign))
        acc[src]["gt"].extend(gt_sign)

        polys, _masks = instance_a((prob > args.thr) & fv, KAPPA_MAX, params=params,
                                   channels=prob[None], prob=prob)
        tree = cKDTree(np.concatenate([densify(g) for g in gt], 0))
        for pl in polys:
            q = densify(np.asarray(pl, float) / UP)
            if len(q) < 5:
                continue
            d = tree.query(q, k=1)[0]
            far = q[d > 5.0]                      # the part of this prediction on no annotation
            if len(far) < 8:
                continue
            v = signed_contrast(img, far)
            if v is None:
                continue
            m = float(np.median(v))
            acc[src]["far"].append(m)
            # same sign as the frame's annotated filaments, or opposite?
            acc[src]["same_sign"].append(1.0 if np.sign(m) == np.sign(ref) else 0.0)

    print(f"SIGNED CONTRAST of annotated vs unannotated structure  "
          f"(prob_thr={args.thr}, min_length={args.min_length}, {args.split})\n")
    print("negative = DARKER than surroundings (microtubule in contact with the glass)")
    print("positive = BRIGHTER (lifted past the 56.4 nm interference zero -> DETACHED)\n")
    print(f'{"block":>6s} {"n GT":>6s} {"GT contrast p50":>16s} {"% GT dark":>10s} | '
          f'{"n far":>6s} {"far contrast p50":>17s} {"% far dark":>11s} {"% OPPOSITE sign to GT":>22s}')
    out = {}
    for s in ("585", "586", "htw"):
        g = np.array(acc[s]["gt"]); f = np.array(acc[s]["far"])
        if not len(g) or not len(f):
            continue
        same = np.array(acc[s]["same_sign"])
        out[s] = dict(n_gt=len(g), gt_p50=float(np.median(g)),
                      gt_dark=float((g < 0).mean()), n_far=len(f),
                      far_p50=float(np.median(f)), far_dark=float((f < 0).mean()),
                      opposite=float(1 - same.mean()))
        print(f'{s:>6s} {len(g):6d} {np.median(g):16.2f} {(g < 0).mean() * 100:9.1f}% | '
              f'{len(f):6d} {np.median(f):17.2f} {(f < 0).mean() * 100:10.1f}% '
              f'{(1 - same.mean()) * 100:21.1f}%')

    print()
    print("A high 'OPPOSITE sign' fraction means the unannotated structure is DETACHED microtubule")
    print("that the annotation convention skips. A low one means the annotation is simply missing")
    print("filaments of the same appearance, which is ordinary incompleteness.")
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
