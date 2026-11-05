#!/usr/bin/env python3
"""Score TIRF asking the only question this ground truth can answer: does the model find what TIRF
actually SHOWS?

Measured on `data/real/tirf12_eval`: the median signal-carrying polyline is bright over 11 % of its
length. The annotation was drawn on the co-registered IRM channel, where the whole microtubule is
visible; TIRF shows only the labelled stretches. So the project's strict criterion -- 95 % of the
polyline's length covered -- is UNREACHABLE on this block even for a perfect model, and the near-
zero strict scores measure the labelling ratio rather than the method. That is why they are not
quoted as a TIRF result.

This instrument is asymmetric, and deliberately so:

  * RECALL is computed over the BRIGHT samples of the polyline only. The dark stretches are
    don't-care -- the microtubule is there, but TIRF gives no evidence of it, and demanding them
    would be demanding a hallucination.
  * PRECISION is computed over the WHOLE polyline. A prediction running along a dark stretch of a
    real microtubule is correct and must not be punished; only predictions away from any annotated
    filament count against it.

This is the ignore-region protocol applied INSIDE a curve rather than between curves. It is a
different instrument from `centerline_f1` and is reported separately -- never pooled with the IRM
number, and never described as the project's strict F1.

    python scripts/score_tirf_visible.py --pred-dir preds/SPARSE35_tirf --thr 0.99 --label SPARSE35
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from build_tirf_eval import frame_noise_sigma, visible_mask  # noqa: E402
from instance.instancer_a import instance_a  # noqa: E402
from instance.metrics import _polyline_pixels, _skel_points, aggregate_benchmark  # noqa: E402

UP, TOL, KAPPA_MAX = 1.5, 5.0, 0.25


def densify(p, step=1.0):
    d = np.sqrt(((p[1:] - p[:-1]) ** 2).sum(1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 0:
        return p
    t = np.arange(0.0, s[-1], step)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)


def masked_f1(pred_masks, gt_full, gt_visible, tol=TOL,
              length_coverage=0.95, precision_coverage=0.95):
    """centerline_f1 with recall over gt_visible and precision over gt_full.

    Same Hungarian assignment and the same two coverage thresholds as the project metric, so the
    only difference from `centerline_f1` is which point set each side is measured against."""
    P, G = list(pred_masks), list(gt_full)
    if not P and not G:
        return {"tp": 0, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not P:
        return {"tp": 0, "fp": 0, "fn": len(G), "precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not G:
        return {"tp": 0, "fp": len(P), "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_pts = [_skel_points(m) for m in P]
    cost = np.ones((len(P), len(G)), np.float32)
    ok = np.zeros((len(P), len(G)), bool)
    for i, pp in enumerate(pred_pts):
        if not len(pp):
            continue
        for j, (gf, gv) in enumerate(zip(gt_full, gt_visible)):
            if not len(gf) or not len(gv):
                continue
            prec = float((cKDTree(gf).query(pp, k=1)[0] <= tol).mean())     # against the WHOLE curve
            rec = float((cKDTree(pp).query(gv, k=1)[0] <= tol).mean())      # against the BRIGHT part
            f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
            cost[i, j] = 1.0 - f1
            ok[i, j] = rec >= length_coverage and prec >= precision_coverage
    ri, ci = linear_sum_assignment(cost)
    tp = int(sum(ok[i, j] for i, j in zip(ri, ci)))
    fp, fn = len(P) - tp, len(G) - tp
    prec = tp / max(len(P), 1)
    rec = tp / max(len(G), 1)
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--data-dir", default="data/real/tirf12_eval")
    ap.add_argument("--params", default="src/instance/params_a_model_v5H.json")
    ap.add_argument("--thr", type=float, required=True)
    ap.add_argument("--min-length", type=float, default=44.74)
    ap.add_argument("--min-visible-px", type=float, default=20.0,
                    help="a curve with less bright length than this carries too little evidence to "
                         "ask about. It is DON'T-CARE: not required, and a prediction that follows "
                         "it is dropped rather than counted as a false positive -- it is a real "
                         "microtubule, so firing there is not an error.")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    params = dict(json.load(open(args.params)))
    params["min_length"] = args.min_length
    params["prob_thr"] = args.thr

    rows, per_frame = [], []
    n_scored = n_dropped = 0
    for path in sorted(glob.glob(os.path.join(args.data_dir, "*.h5"))):
        with h5py.File(path, "r") as f:
            img = f["image"][()].astype(np.float64)
            polys = [np.asarray(f["polylines"][k][()], float) for k in sorted(f["polylines"])]
        polys = [q for q in polys if len(q) >= 2]
        if not polys:
            continue
        sigma = frame_noise_sigma(img, polys)
        npz = os.path.join(args.pred_dir, os.path.basename(path).replace(".h5", ".npz"))
        if not os.path.isfile(npz):
            raise SystemExit(f"missing prediction {npz}")
        prob = np.load(npz)["prob"].astype(np.float32)
        if prob.ndim == 3:
            prob = prob.max(axis=0)
        _p, masks = instance_a(prob > args.thr, KAPPA_MAX, params=params,
                               channels=prob[None], prob=prob)
        shape = masks[0].shape if masks else (int(img.shape[0] * UP), int(img.shape[1] * UP))

        gt_full, gt_vis, dont_care = [], [], []
        for q in polys:
            vm = visible_mask(img, q, sigma)
            if vm is None or vm.sum() < args.min_visible_px:
                n_dropped += 1
                dont_care.append(q * UP)
                continue
            d = densify(q)                       # 1 px spacing, same as visible_mask
            n = min(len(d), len(vm))
            # _skel_points and _polyline_pixels both return (ROW, COL). The polylines in the h5
            # are (x=col, y=row) -- the transpose. Building the visible point set in (x, y) makes
            # recall come out at exactly zero while precision stays high, which reads like a model
            # failure rather than a bug. That trap has now cost this project three times, so the
            # assertion below refuses to score a mismatch instead of reporting one.
            full = _polyline_pixels(q * UP, shape)
            vis_xy = d[:n][vm[:n]] * UP
            vis = np.unique(np.round(vis_xy[:, ::-1]).astype(int), axis=0).astype(float)
            if len(vis) and len(full):
                near = cKDTree(full).query(vis, k=1)[0]
                if float((near <= 2.0).mean()) < 0.9:
                    raise SystemExit(
                        "the visible point set does not lie on its own polyline "
                        f"({100 * (near <= 2.0).mean():.0f} % within 2 px) -- coordinate convention "
                        "mismatch; refusing to produce a score")
            gt_full.append(full)
            gt_vis.append(vis)
            n_scored += 1
        # A prediction lying on a curve we declined to score is neither credited nor blamed. These
        # are REAL microtubules whose TIRF evidence is too thin to ask about; counting a prediction
        # there as a false positive punishes the model for being right, and with 164 of 231 curves
        # in that state it dominated the score -- the first version of this instrument reported an
        # F1 near zero for that reason alone.
        kept = masks
        if dont_care and gt_full:
            tdc = cKDTree(np.concatenate([_polyline_pixels(g, shape) for g in dont_care], 0))
            tsc = cKDTree(np.concatenate(gt_full, 0))
            kept = []
            for m in masks:
                pts = _skel_points(m)
                if not len(pts):
                    continue
                if (tdc.query(pts, k=1)[0] <= TOL).mean() > (tsc.query(pts, k=1)[0] <= TOL).mean():
                    continue
                kept.append(m)
        r = masked_f1(kept, gt_full, gt_vis)
        r["n_gt"] = len(gt_full)
        rows.append(r)
        per_frame.append((os.path.basename(path), len(polys), len(gt_full), r["f1"], len(kept)))

    a = aggregate_benchmark(rows)
    lab = args.label or os.path.basename(args.pred_dir)
    print(f"TIRF visible-extent score  {lab}  thr={args.thr}\n")
    print(f'{"frame":46s} {"curves":>7s} {"scored":>7s} {"F1":>7s} {"pred":>6s}')
    for nm, n, k, f1, npd in per_frame:
        print(f'{nm[:46]:46s} {n:7d} {k:7d} {f1:7.3f} {npd:6d}')
    print(f'\n  mean-F1 {a["mean_f1"]:.4f}   tp={a["total_tp"]} fp={a["total_fp"]} fn={a["total_fn"]}')
    print(f'  {n_scored} curves scored; {n_dropped} carried under {args.min_visible_px:.0f} px of '
          f"visible filament and are DON'T-CARE (not required, and predictions on them dropped)")
    print("\n  Recall is over the bright part of each curve, precision over the whole curve.")
    print("  This is NOT the project's strict centerline-F1 and must not be pooled with it.")
    if args.out:
        json.dump(dict(label=lab, thr=args.thr, agg=a, n_scored=n_scored, n_dropped=n_dropped,
                       per_frame=[dict(frame=n, n_curves=c, n_scored=k, f1=f, n_pred=p)
                                  for n, c, k, f, p in per_frame]),
                  open(args.out, "w"), indent=1, default=float)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
