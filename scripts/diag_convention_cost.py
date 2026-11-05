#!/usr/bin/env python3
"""What does the annotation CONVENTION cost us?

Confirmed by the user 2026-08-28: in the collaborating lab's data a microtubule segment that lifts
off the coverslip is not annotated. Interference contrast is -cos(2*k*h) in the height above the
glass, so an attached segment is dark and a lifted one, past h = lambda/(8*n) = 56.4 nm, is bright.
An annotator tracing dark lines stops at the detachment; the filament continues.

The generator, meanwhile, DOES label detached segments -- `detach_frame_prob`, `detach_prob` and
`height_detach_nm_range` exist precisely to render them. So the network is trained to find something
the ground truth then scores as a false positive. This measures how much that costs.

Two bounds, because the convention hurts in two different ways:

  LOWER  a predicted instance that matches nothing and lies mostly on opposite-polarity structure is
         a correctly found detached segment counted as a false positive. Removing those from the
         false-positive count is the conservative correction.
  UPPER  a predicted instance that DOES match a ground-truth filament but fails the 95 % precision
         criterion because it continues into a detached stretch. Treating that stretch as
         don't-care -- the standard ignore-region protocol -- turns those into true positives.

Neither correction adds anything the model did not already predict, and neither uses the model to
decide what is a microtubule: the classification is by contrast SIGN against the frame's own
annotated filaments, which is a property of the physics.

    python scripts/diag_convention_cost.py --pred-dir preds/LEN_treatment --thr 0.997 --min-length 20
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
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from instance.instancer_a import instance_a  # noqa: E402
from instance.metrics import _polyline_pixels, _skel_points  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.fov import fov_mask  # noqa: E402

UP = 1.5
KAPPA_MAX = 0.25
TOL = 5.0
COV = 0.95


def source_of(name):
    b = os.path.basename(name)
    return "htw" if b.startswith("htw") else ("585" if b.startswith("alice") else "586")


def signed_contrast_at(img, pts_native, halfwin=7.0):
    """Signed core-minus-flank contrast at each point of a curve, in native pixels.

    The normal is estimated from the local point ordering, so this needs an ORDERED curve.
    """
    if len(pts_native) < 5:
        return None
    t = np.gradient(pts_native, axis=0)
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    H, W = img.shape

    def take(p):
        c = np.round(p[:, 0]).astype(int)
        r = np.round(p[:, 1]).astype(int)
        ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
        out = np.full(len(p), np.nan)
        out[ok] = img[r[ok], c[ok]]
        return out

    core = take(pts_native)
    flank = np.nanmean(np.stack([take(pts_native + n * halfwin),
                                 take(pts_native - n * halfwin)]), axis=0)
    return core - flank


def densify(p, step=1.0):
    seg = np.sqrt(((p[1:] - p[:-1]) ** 2).sum(1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return p
    t = np.arange(0.0, s[-1], step)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)


def score_frame(pred_pts, gt_pts, ignore_masks=None):
    """Hungarian centerline-F1 for one frame, optionally ignoring some predicted points.

    ``ignore_masks[i]`` is a boolean over ``pred_pts[i]`` marking points that count neither for
    nor against precision -- the standard don't-care protocol.
    """
    Np, Ng = len(pred_pts), len(gt_pts)
    if Np == 0:
        return 0, 0, Ng, []
    if Ng == 0:
        return 0, Np, 0, []
    cost = np.ones((Np, Ng), dtype=np.float32)
    grid = [[(0.0, 0.0)] * Ng for _ in range(Np)]
    gt_trees = [cKDTree(g) for g in gt_pts]
    for i, p in enumerate(pred_pts):
        if len(p) == 0:
            continue
        keep = np.ones(len(p), bool) if ignore_masks is None else ~ignore_masks[i]
        pk = p[keep] if keep.any() else p
        ptree = cKDTree(p)
        for j, g in enumerate(gt_pts):
            prec = float((gt_trees[j].query(pk, k=1)[0] <= TOL).mean()) if len(pk) else 0.0
            rec = float((ptree.query(g, k=1)[0] <= TOL).mean())
            grid[i][j] = (prec, rec)
            f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
            cost[i, j] = 1.0 - f1
    ri, ci = linear_sum_assignment(cost)
    tp, matched_p, matched_g, pairs = 0, set(), set(), []
    for i, j in zip(ri, ci):
        prec, rec = grid[i][j]
        ok = (rec >= COV) and (prec >= COV)
        pairs.append((i, j, prec, rec, ok))
        if ok:
            tp += 1
            matched_p.add(i)
            matched_g.add(j)
    return tp, Np - len(matched_p), Ng - len(matched_g), pairs


def f1(tp, fp, fn):
    return 2 * tp / max(2 * tp + fp + fn, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--data-dir", default="data/real/mt34_htw_eval")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--params", default="src/instance/params_a_model_v5H.json")
    ap.add_argument("--thr", type=float, default=0.997)
    ap.add_argument("--min-length", type=float, default=20.0)
    ap.add_argument("--frac", type=float, default=0.5,
                    help="fraction of a prediction's far points that must be opposite-polarity "
                         "for it to count as a detached segment")
    ap.add_argument("--out", default="convention_cost.json")
    args = ap.parse_args()
    if args.split == "test":
        raise SystemExit("refusing to touch TEST: it carries a declared exposure budget")

    params = dict(json.load(open(args.params)))
    params["min_length"] = args.min_length
    params["prob_thr"] = args.thr

    per = defaultdict(lambda: defaultdict(list))
    for path in sorted(glob.glob(os.path.join(args.data_dir, "*.h5"))):
        fr = read_frame_h5(path)
        if str(fr["attrs"].get("split")) != args.split:
            continue
        name = os.path.basename(path).replace(".h5", ".npz")
        src = source_of(name)
        img = fr["image"].astype(np.float64)
        gt_native = [np.asarray(p, float) for p in fr["polylines"] if len(p) >= 2]
        if not gt_native:
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

        # the frame's own polarity reference, from its annotated filaments
        ref = []
        for g in gt_native:
            v = signed_contrast_at(img, densify(g))
            if v is not None and np.isfinite(v).any():
                ref.append(float(np.nanmedian(v)))
        if not ref:
            continue
        ref_sign = np.sign(np.median(ref))

        polys, masks = instance_a((prob > args.thr) & fv, KAPPA_MAX, params=params,
                                  channels=prob[None], prob=prob)
        gt_up = [g * UP for g in gt_native]
        gt_pts = [_polyline_pixels(g, hi) for g in gt_up]
        pred_pts = [_skel_points(m) for m in masks]
        gt_tree = cKDTree(np.concatenate([densify(g) for g in gt_up], 0))

        # per predicted instance: which of its points are far from every annotation AND carry the
        # opposite contrast sign -- i.e. detached microtubule the convention omits
        ignore, detached_inst = [], []
        for pl, pp in zip(polys, pred_pts):
            q = densify(np.asarray(pl, float))
            far_q = gt_tree.query(q, k=1)[0] > TOL
            v = signed_contrast_at(img, q / UP)
            opp = np.zeros(len(q), bool)
            if v is not None:
                opp = far_q & np.isfinite(v) & (np.sign(v) == -ref_sign)
            det = bool(far_q.sum() >= 8 and opp.sum() / max(far_q.sum(), 1) >= args.frac)
            detached_inst.append(det)
            # map the flag onto the skeleton points used for scoring
            if len(pp) and opp.any():
                t = cKDTree(q)
                d, idx = t.query(pp, k=1)
                ignore.append((d <= 2.0) & opp[idx])
            else:
                ignore.append(np.zeros(len(pp), bool))

        tp0, fp0, fn0, _ = score_frame(pred_pts, gt_pts)
        # LOWER bound: unmatched detached instances stop counting as false positives
        tpL, fpL, fnL, _ = score_frame(pred_pts, gt_pts)
        _, _, _, pairs0 = score_frame(pred_pts, gt_pts)
        matched = {i for i, j, p, r, ok in pairs0 if ok}
        fpL = fp0 - sum(1 for i, d in enumerate(detached_inst) if d and i not in matched)
        # UPPER bound: detached stretches are don't-care for precision as well
        tpU, fpU, fnU, _ = score_frame(pred_pts, gt_pts, ignore_masks=ignore)
        fpU = fpU - sum(1 for i, d in enumerate(detached_inst) if d and i not in matched)

        per[src]["base"].append((tp0, fp0, fn0))
        per[src]["lower"].append((tpL, max(fpL, 0), fnL))
        per[src]["upper"].append((tpU, max(fpU, 0), fnU))
        per[src]["n_detached"].append(sum(detached_inst))
        per[src]["n_pred"].append(len(polys))

    print(f"COST OF THE ANNOTATION CONVENTION  (prob_thr={args.thr}, min_length={args.min_length}, "
          f"{args.split})\n")
    print("A detached segment is scored by CONTRAST SIGN against the frame's own annotated")
    print("filaments -- not by the model. Neither correction adds a prediction.\n")
    print(f'{"block":>6s} {"frames":>7s} {"detached preds":>15s} {"F1 as scored":>13s} '
          f'{"F1, detached not FP":>20s} {"F1, also don\'t-care":>20s} {"cost of convention":>19s}')
    out = {}
    for s in ("585", "586", "htw"):
        if not per[s]["base"]:
            continue
        def mean_f1(key):
            return float(np.mean([f1(*t) for t in per[s][key]]))
        b, lo, up = mean_f1("base"), mean_f1("lower"), mean_f1("upper")
        nd, npd = int(np.sum(per[s]["n_detached"])), int(np.sum(per[s]["n_pred"]))
        out[s] = dict(frames=len(per[s]["base"]), base=b, lower=lo, upper=up,
                      n_detached=nd, n_pred=npd)
        print(f'{s:>6s} {len(per[s]["base"]):7d} {nd:6d} / {npd:<6d} {b:13.4f} '
              f'{lo:20.4f} {up:20.4f} {up - b:+19.4f}')
    allb = [f1(*t) for s in per for t in per[s]["base"]]
    alll = [f1(*t) for s in per for t in per[s]["lower"]]
    allu = [f1(*t) for s in per for t in per[s]["upper"]]
    print(f'{"POOL":>6s} {len(allb):7d} {"":15s} {np.mean(allb):13.4f} '
          f'{np.mean(alll):20.4f} {np.mean(allu):20.4f} {np.mean(allu) - np.mean(allb):+19.4f}')
    out["POOL"] = dict(base=float(np.mean(allb)), lower=float(np.mean(alll)),
                       upper=float(np.mean(allu)), frames=len(allb))
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
