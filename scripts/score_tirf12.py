#!/usr/bin/env python3
"""Score a model on the TIRF block, without punishing it for not finding what is not there.

`data/real/tirf12_eval` has 231 polylines, but only ~100 of them carry TIRF signal: the annotation
was drawn on the co-registered IRM channel and carried across, and IRM sees whatever lies against
the coverslip while TIRF sees only labelled tubulin inside the evanescent field. Scoring against all
231 would count every unlabelled microtubule as a miss -- a penalty for correct behaviour, and a
number that says more about the labelling ratio than about the model.

The signal-free polylines are therefore DON'T-CARE, the same ignore-region protocol
`score_by_provenance.py` uses for model-drawn ground truth: they are neither required nor
penalised, and a prediction that follows one is dropped from scoring rather than counted as a false
positive. Both numbers are reported, because the gap between them is itself informative.

Three frames carry almost no signal at all. They are kept deliberately: with every curve
don't-care they become zero-ground-truth frames that still contribute false positives, which is a
direct test of the project's first principle -- does the model fire where nothing is visible?

    python scripts/score_tirf12.py --pred-dir preds/SPARSE35_tirf --thr 0.99 --label SPARSE35
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np
from scipy.ndimage import zoom
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from instance.instancer_a import instance_a  # noqa: E402
from instance.metrics import aggregate_benchmark, centerline_f1  # noqa: E402

UP = 1.5
TOL = 5.0
KAPPA_MAX = 0.25


def densify(p, step=1.0):
    d = np.sqrt(((p[1:] - p[:-1]) ** 2).sum(1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 0:
        return p
    t = np.arange(0.0, s[-1], step)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)


def skel_pts(mask):
    from skimage.morphology import skeletonize
    r, c = np.nonzero(skeletonize(mask))
    return np.stack([c, r], 1).astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--data-dir", default="data/real/tirf12_eval")
    ap.add_argument("--params", default="src/instance/params_a_model_v5H.json")
    ap.add_argument("--thr", type=float, required=True)
    ap.add_argument("--min-length", type=float, default=44.74)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    params = dict(json.load(open(args.params)))
    params["min_length"] = args.min_length
    params["prob_thr"] = args.thr

    rows_all, rows_sig, per_frame = [], [], []
    for path in sorted(glob.glob(os.path.join(args.data_dir, "*.h5"))):
        with h5py.File(path, "r") as f:
            img = f["image"][()].astype(np.float32)
            polys = [np.asarray(f["polylines"][k][()], float)
                     for k in sorted(f["polylines"])]
            has = np.asarray(f.attrs["has_tirf_signal"]).astype(bool)
        keep = [i for i, p in enumerate(polys) if len(p) >= 2]
        if len(has) != len(polys):
            raise SystemExit(f"{path}: {len(has)} flags for {len(polys)} polylines")
        gt = [polys[i] * UP for i in keep]
        flag = has[keep]
        if not gt:
            continue

        npz = os.path.join(args.pred_dir, os.path.basename(path).replace(".h5", ".npz"))
        if not os.path.isfile(npz):
            raise SystemExit(f"missing prediction {npz}")
        prob = np.load(npz)["prob"].astype(np.float32)
        if prob.ndim == 3:
            prob = prob.max(axis=0)
        _p, masks = instance_a(prob > args.thr, KAPPA_MAX, params=params,
                               channels=prob[None], prob=prob)

        # (a) every transferred polyline required -- the number that flatters nobody but is wrong
        r_all = centerline_f1(masks, gt, tol=TOL, length_coverage=0.95, precision_coverage=0.95)
        r_all["n_gt"] = len(gt)
        rows_all.append(r_all)

        # (b) only the polylines that carry TIRF signal; the rest are DON'T-CARE
        sig = [g for g, h in zip(gt, flag) if h]
        dc = [g for g, h in zip(gt, flag) if not h]
        kept = masks
        if dc:
            tm = cKDTree(np.concatenate([densify(g) for g in dc], 0))
            ts = cKDTree(np.concatenate([densify(g) for g in sig], 0)) if sig else None
            kept = []
            for m in masks:
                pts = skel_pts(m)
                if not len(pts):
                    continue
                near_dc = (tm.query(pts, k=1)[0] <= TOL).mean()
                near_sig = (ts.query(pts, k=1)[0] <= TOL).mean() if ts is not None else 0.0
                if near_dc > near_sig:
                    continue        # follows an unlabelled microtubule: neither credited nor blamed
                kept.append(m)
        r_sig = centerline_f1(kept, sig, tol=TOL, length_coverage=0.95, precision_coverage=0.95)
        r_sig["n_gt"] = len(sig)
        rows_sig.append(r_sig)
        per_frame.append((os.path.basename(path), len(gt), int(flag.sum()),
                          r_all["f1"], r_sig["f1"], len(masks), len(kept)))

    a_all, a_sig = aggregate_benchmark(rows_all), aggregate_benchmark(rows_sig)
    lab = args.label or os.path.basename(args.pred_dir)
    print(f"TIRF12  {lab}  thr={args.thr}  min_length={args.min_length}\n")
    print(f'{"frame":46s} {"gt":>4s} {"sig":>4s} {"F1 all":>7s} {"F1 sig":>7s} {"pred":>5s} {"kept":>5s}')
    for nm, n, s, f1a, f1s, np_, nk in per_frame:
        print(f'{nm[:46]:46s} {n:4d} {s:4d} {f1a:7.3f} {f1s:7.3f} {np_:5d} {nk:5d}')
    print(f'\n  all 231 curves required : mean-F1 {a_all["mean_f1"]:.4f}  '
          f'tp={a_all["total_tp"]} fp={a_all["total_fp"]} fn={a_all["total_fn"]}')
    print(f'  signal-carrying only    : mean-F1 {a_sig["mean_f1"]:.4f}  '
          f'tp={a_sig["total_tp"]} fp={a_sig["total_fp"]} fn={a_sig["total_fn"]}')
    print("\n  The second is the honest one. The first counts every microtubule that carries no")
    print("  fluorophore as a miss, which measures the labelling ratio, not the model.")

    if args.out:
        json.dump(dict(label=lab, thr=args.thr, min_length=args.min_length,
                       all=a_all, signal_only=a_sig,
                       per_frame=[dict(frame=n, n_gt=g, n_sig=s, f1_all=a, f1_sig=b,
                                       n_pred=p, n_kept=k)
                                  for n, g, s, a, b, p, k in per_frame]),
                  open(args.out, "w"), indent=1, default=float)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
