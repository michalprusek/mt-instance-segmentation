#!/usr/bin/env python3
"""How much of each benchmark source can never be scored correct, because real microtubules
were not annotated?

The user's decision on 2026-08-27 is to keep the ground truth as it is and report progress
against a STATED ceiling. That makes this number the denominator of every later claim, so it
gets two independent estimates rather than one:

  A. MODEL-BASED. Take the detections the metric calls false positives, keep those far from
     every annotation and carrying real filament contrast, and count how many DISTINCT
     filaments they form (not how many fragments) by connected components on their union.

  B. RIDGE-BASED, model-independent. A classical Meijering ridge filter, thresholded so it
     recovers ~90 % of the ANNOTATED centerline in that same frame (which calibrates it to the
     lab's own contrast), then everything near an annotation removed. What survives, is
     filament-shaped, and beats the null is an unannotated filament.

Estimate A can be corrupted by the very model it measures; B cannot, and B needs no
predictions at all. Where they agree the number is trustworthy; where they disagree the
disagreement is the finding.

Both use the same control: the annotated curves rigidly displaced inside the same frame, which
carries the frame's texture and illumination but is aligned to nothing.

    python scripts/benchmark_ceiling.py --data-dir data/real/mt34_htw_eval --split val \
        --pred-dir preds/v5H_combined_val --out ceiling_val.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import zoom
from scipy.spatial import cKDTree
from skimage.filters import meijering

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from instance.instancer_a import instance_a  # noqa: E402
from instance.metrics import centerline_f1  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.fov import fov_mask  # noqa: E402

UP = 1.5
KAPPA_MAX = 0.25


def source_of(path, attrs):
    if os.path.basename(path).startswith("htw"):
        return "htw"
    return "585" if str(attrs.get("source_task")) == "585" else "586"


def densify(poly, step=1.0):
    if len(poly) < 2:
        return poly
    seg = np.sqrt(((poly[1:] - poly[:-1]) ** 2).sum(1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return poly
    t = np.arange(0.0, s[-1], step)
    return np.stack([np.interp(t, s, poly[:, 0]), np.interp(t, s, poly[:, 1])], axis=1)


def stamp(polys, shape, r=1):
    m = np.zeros(shape, bool)
    for p in polys:
        q = densify(p, 0.5)
        if len(q) < 2:
            continue
        c = np.round(q[:, 0]).astype(int)
        rr = np.round(q[:, 1]).astype(int)
        ok = (c >= 0) & (c < shape[1]) & (rr >= 0) & (rr < shape[0])
        m[rr[ok], c[ok]] = True
    if r > 0:
        m = ndi.binary_dilation(m, ndi.generate_binary_structure(2, 2), iterations=r)
    return m


def contrast_along(img, poly, dark):
    """Mean |filament - flanking background| under a centerline, native pixels."""
    q = densify(poly)
    if len(q) < 3:
        return float("nan")
    t = np.gradient(q, axis=0)
    n = np.stack([-t[:, 1], t[:, 0]], axis=1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    H, W = img.shape

    def take(pts):
        c = np.round(pts[:, 0]).astype(int)
        r = np.round(pts[:, 1]).astype(int)
        ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
        return img[r[ok], c[ok]] if ok.sum() >= 3 else None

    core = take(q)
    if core is None:
        return float("nan")
    fl = [take(q + n * o) for o in (-7.0, 7.0)]
    fl = [f for f in fl if f is not None]
    if not fl:
        return float("nan")
    v = float(core.mean()) - float(np.mean([f.mean() for f in fl]))
    return -v if dark else v


def null_curves(polys, shape, rng, k=1):
    H, W = shape
    out = []
    for p in polys:
        for _ in range(k):
            for _try in range(12):
                q = p + np.array([rng.uniform(-.45, .45) * W, rng.uniform(-.45, .45) * H])
                if (q[:, 0] >= 0).all() and (q[:, 0] < W).all() and \
                   (q[:, 1] >= 0).all() and (q[:, 1] < H).all():
                    out.append(q)
                    break
    return out


def component_contrasts(comp_mask, img, dark, min_px):
    """Split a binary mask into components and return each one's mean contrast.

    A component is scored by comparing its own pixels against a locally dilated ring, which is
    the 2-d analogue of the flanking-profile estimate used for curves.
    """
    lab, n = ndi.label(comp_mask, np.ones((3, 3)))
    out = []
    if n == 0:
        return out
    sizes = ndi.sum(comp_mask, lab, range(1, n + 1))
    for i in np.where(sizes >= min_px)[0] + 1:
        sel = lab == i
        ring = ndi.binary_dilation(sel, iterations=6) & ~ndi.binary_dilation(sel, iterations=2)
        if ring.sum() < 10:
            continue
        v = float(img[sel].mean()) - float(img[ring].mean())
        out.append(-v if dark else v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/real/mt34_htw_eval")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--pred-dir", default=None, help="omit to run the ridge estimate only")
    ap.add_argument("--params", default="src/instance/params_a_model_v5H.json")
    ap.add_argument("--prob-thr", type=float, default=0.99)
    ap.add_argument("--min-length", type=float, default=20.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="ceiling.json")
    args = ap.parse_args()
    if args.split == "test":
        raise SystemExit("refusing to touch TEST: it carries a declared exposure budget")

    params = dict(json.load(open(args.params)))
    params["min_length"] = args.min_length
    rng = np.random.default_rng(20260827)
    acc = defaultdict(lambda: defaultdict(float))

    paths = [p for p in sorted(glob.glob(os.path.join(args.data_dir, "*.h5")))]
    frames = []
    for p in paths:
        fr = read_frame_h5(p)
        if str(fr["attrs"].get("split")) == args.split:
            frames.append((p, fr))
    if args.limit:
        frames = frames[:args.limit]

    for path, fr in frames:
        src = source_of(path, fr["attrs"])
        img = fr["image"].astype(np.float64)
        H, W = img.shape
        gt = [np.asarray(q, float) for q in fr["polylines"] if len(q) >= 2]
        if not gt:
            continue
        fov = fov_mask(fr["image"])

        gt_on = stamp(gt, (H, W), r=1)
        dark = bool(img[gt_on].mean() < np.median(img))
        near = stamp(gt, (H, W), r=6)

        nulls = null_curves(gt, (H, W), rng)
        c_null = [contrast_along(img, q, dark) for q in nulls]
        c_null = np.array([c for c in c_null if np.isfinite(c)])
        cut = float(np.percentile(c_null, 95)) if len(c_null) else np.inf

        acc[src]["n_gt"] += len(gt)
        acc[src]["frames"] += 1

        # ---------- A. model-based, de-fragmented -------------------------------------
        if args.pred_dir:
            npz = os.path.join(args.pred_dir, os.path.basename(path).replace(".h5", ".npz"))
            if not os.path.isfile(npz):
                raise SystemExit(f"missing prediction {npz}")
            prob = np.load(npz)["prob"].astype(np.float32)
            if prob.ndim == 3:
                prob = prob.max(axis=0)
            hi = (int(round(H * UP)), int(round(W * UP)))
            fv = zoom(fov.astype(np.uint8), UP, order=0).astype(bool)
            fv = np.pad(fv, ((0, max(0, hi[0] - fv.shape[0])), (0, max(0, hi[1] - fv.shape[1]))),
                        constant_values=True)[:hi[0], :hi[1]]
            polys, masks = instance_a((prob > args.prob_thr) & fv, KAPPA_MAX, params=params,
                                      channels=prob[None], prob=prob)
            res = centerline_f1(masks, [g * UP for g in gt], tol=5.0,
                                length_coverage=0.95, precision_coverage=0.95)
            matched = {p["pred"] for p in res["pairs"] if p["tp"]}
            gt_pts = np.concatenate([densify(g) for g in gt], axis=0)
            tree = cKDTree(gt_pts)
            far_real = []
            for i, pl in enumerate(polys):
                if i in matched:
                    continue
                q = densify(np.asarray(pl, float) / UP)
                if len(q) < 3:
                    continue
                if float(np.median(tree.query(q, k=1)[0])) <= 5.0:
                    continue                       # a fragment of an annotated microtubule
                c = contrast_along(img, q, dark)
                if np.isfinite(c) and c > cut:
                    far_real.append(q)
            acc[src]["A_fragments"] += len(far_real)
            # De-fragment: several pieces of ONE unannotated filament must count once.
            if far_real:
                u = stamp(far_real, (H, W), r=3)
                acc[src]["A_distinct"] += int(ndi.label(u, np.ones((3, 3)))[1])

        # ---------- B. ridge-based, model-independent ---------------------------------
        # Thresholding on ridge response and then testing brightness would be circular: the
        # detector selects bright things, so every survivor "beats" a background null. The
        # control has to go through the SAME detector. A phase-randomised surrogate keeps the
        # image's power spectrum -- hence its noise amplitude and correlation length -- while
        # destroying every coherent filament, so whatever the pipeline finds there is its own
        # false-discovery rate at this operating point.
        bg = float(np.median(img))
        v = (bg - img) if dark else (img - bg)

        def ridge_count(field):
            R = meijering(field, sigmas=[1, 1.5, 2], black_ridges=False)
            t = float(np.percentile(R[gt_on], 10))   # recovers ~90 % of annotated centerline
            m = ndi.binary_opening((R > t) & fov & ~near, np.ones((3, 3)))
            lab, n = ndi.label(m, np.ones((3, 3)))
            if n == 0:
                return 0
            sizes = ndi.sum(m, lab, range(1, n + 1))
            keep = np.where(sizes >= 30)[0] + 1
            # a microtubule is elongated; a noise clump is not
            out = 0
            for i in keep:
                ys, xs = np.where(lab == i)
                pts = np.stack([xs - xs.mean(), ys - ys.mean()], 1)
                ev = np.linalg.eigvalsh(np.cov(pts.T)) if len(pts) > 2 else np.array([1.0, 1.0])
                if ev[0] > 1e-6 and (ev[1] / ev[0]) >= 9.0:     # axis ratio >= 3
                    out += 1
            return out

        ph = np.angle(np.fft.fft2(rng.standard_normal(v.shape)))
        F = np.fft.fft2(v)
        surrogate = np.real(np.fft.ifft2(np.abs(F) * np.exp(1j * ph)))
        n_real, n_surr = ridge_count(v), ridge_count(surrogate)
        acc[src]["B_all"] += n_real
        acc[src]["B_surr"] += n_surr
        acc[src]["B_real"] += max(0, n_real - n_surr)   # excess over the detector's own FDR
        print(f"  {os.path.basename(path):55s} {src:4s} gt={len(gt):4d} "
              f"A_dist={acc[src][chr(39)+chr(39)] if False else acc[src]["A_distinct"]:.0f} "
              f"B_real={acc[src]['B_real']:.0f}", flush=True)

    print()
    print(f"BENCHMARK CEILING, split={args.split}  (unannotated real filaments per source)")
    print(f'{"src":5s} {"frames":>6s} {"n GT":>6s} | {"A frags":>8s} {"A distinct":>11s} '
          f'{"ceiling A":>10s} | {"B ridge":>8s} {"ceiling B":>10s}')
    tot = defaultdict(float)
    out = {}
    for s in ("585", "586", "htw"):
        d = acc.get(s)
        if not d:
            continue
        g = d["n_gt"]
        ca = 2 * g / (2 * g + d["A_distinct"]) if d["A_distinct"] else 1.0
        cb = 2 * g / (2 * g + d["B_real"]) if d["B_real"] else 1.0
        out[s] = dict(n_gt=g, A_fragments=d["A_fragments"], A_distinct=d["A_distinct"],
                      ceiling_A=ca, B_real=d["B_real"], ceiling_B=cb, frames=d["frames"])
        for k in ("n_gt", "A_distinct", "B_real"):
            tot[k] += d[k]
        print(f'{s:5s} {d["frames"]:6.0f} {g:6.0f} | {d["A_fragments"]:8.0f} '
              f'{d["A_distinct"]:11.0f} {ca:10.3f} | {d["B_real"]:8.0f} {cb:10.3f}')
    g = tot["n_gt"]
    ca = 2 * g / (2 * g + tot["A_distinct"])
    cb = 2 * g / (2 * g + tot["B_real"])
    print(f'{"POOL":5s} {"":6s} {g:6.0f} | {"":8s} {tot["A_distinct"]:11.0f} {ca:10.3f} | '
          f'{tot["B_real"]:8.0f} {cb:10.3f}')
    out["POOL"] = dict(n_gt=g, A_distinct=tot["A_distinct"], ceiling_A=ca,
                       B_real=tot["B_real"], ceiling_B=cb)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
