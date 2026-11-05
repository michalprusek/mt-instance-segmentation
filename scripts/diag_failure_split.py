#!/usr/bin/env python3
"""Which of the metric's TWO criteria does each ground-truth microtubule actually fail?

A true positive needs BOTH: >=95 % of the ground truth covered within 5 px (coverage), AND >=95 % of
the matched prediction lying within 5 px of that ground truth (precision). They fail for opposite
reasons -- coverage fails when the prediction is too SHORT or absent, precision when it is too LONG,
spanning several microtubules or running onto a filament nobody annotated.

This matters because the semantic diagnostic says HTW's predicted foreground covers 100 % of the
ground-truth centerline at tol 2, while its instance recall is about 0.5. Both cannot be a detection
problem. The split below says which criterion is actually binding, per block, and therefore whether
the remaining loss is something a better model can fix or something the incomplete ground truth
imposes.

    python scripts/diag_failure_split.py --pred-dir preds/LEN_treatment --thr 0.997 --min-length 20
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from instance.instancer_a import instance_a  # noqa: E402
from instance.metrics import centerline_f1  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.fov import fov_mask  # noqa: E402

UP = 1.5
KAPPA_MAX = 0.25


def source_of(name):
    b = os.path.basename(name)
    if b.startswith("roi_img_"):
        return "roi303"
    return "htw" if b.startswith("htw") else ("585" if b.startswith("alice") else "586")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--data-dir", default="data/real/mt34_htw_eval")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--params", default="src/instance/params_a_model_v5H.json")
    ap.add_argument("--thr", type=float, required=True)
    ap.add_argument("--min-length", type=float, required=True)
    ap.add_argument("--cov", type=float, default=0.95, help="coverage AND precision rule; 0.95 = "
                    "strict (every earlier read), 0.80 = the 2026-09-15 declared convention")
    ap.add_argument("--no-fov", action="store_true", help="skip the field-stop mask, as the "
                    "eval_v5.py sweeps on roi303 / the 100-frame block do (--no-fov)")
    ap.add_argument("--out", default="failure_split.json")
    args = ap.parse_args()
    if args.split == "test":
        raise SystemExit("refusing to touch TEST: it carries a declared exposure budget")

    params = dict(json.load(open(args.params)))
    params["min_length"] = args.min_length
    params["prob_thr"] = args.thr

    stat = defaultdict(lambda: defaultdict(int))
    prec_of_failed = defaultdict(list)
    len_ratio = defaultdict(list)

    for path in sorted(glob.glob(os.path.join(args.data_dir, "*.h5"))):
        fr = read_frame_h5(path)
        if str(fr["attrs"].get("split")) != args.split:
            continue
        name = os.path.basename(path).replace(".h5", ".npz")
        src = source_of(name)
        gt = [np.asarray(p, float) * UP for p in fr["polylines"] if len(p) >= 2]
        if not gt:
            continue
        npz = os.path.join(args.pred_dir, name)
        if not os.path.isfile(npz):
            raise SystemExit(f"missing prediction {npz} for a frame that has ground truth")
        prob = np.load(npz)["prob"].astype(np.float32)
        if prob.ndim == 3:
            prob = prob.max(axis=0)
        img = fr["image"]
        hi = (int(round(img.shape[0] * UP)), int(round(img.shape[1] * UP)))
        if args.no_fov:
            fv = np.ones(hi, bool)
        else:
            fv = zoom(fov_mask(img).astype(np.uint8), UP, order=0).astype(bool)
            fv = np.pad(fv, ((0, max(0, hi[0] - fv.shape[0])), (0, max(0, hi[1] - fv.shape[1]))),
                        constant_values=True)[:hi[0], :hi[1]]
        from scipy.spatial import cKDTree
        def _dens(q):
            seg = np.sqrt(((q[1:] - q[:-1]) ** 2).sum(1)); c = np.concatenate([[0.], np.cumsum(seg)])
            if c[-1] <= 0: return q
            t = np.arange(0., c[-1], 1.0)
            return np.stack([np.interp(t, c, q[:, 0]), np.interp(t, c, q[:, 1])], 1)
        dens = [_dens(g) for g in gt]
        own_tree = [cKDTree(d) for d in dens]
        all_tree = cKDTree(np.concatenate(dens, 0))
        polys, masks = instance_a((prob > args.thr) & fv, KAPPA_MAX, params=params,
                                  channels=prob[None], prob=prob)
        r = centerline_f1(masks, gt, tol=5.0, length_coverage=args.cov, precision_coverage=args.cov)

        matched_gt = set()
        for pr in r["pairs"]:
            j = pr["gt"]
            matched_gt.add(j)
            cov_ok = pr["recall"] >= args.cov
            prec_ok = pr["precision"] >= args.cov
            if cov_ok and prec_ok:
                stat[src]["tp"] += 1
            elif cov_ok and not prec_ok:
                stat[src]["precision_only"] += 1
                prec_of_failed[src].append(pr["precision"])
            elif prec_ok and not cov_ok:
                stat[src]["coverage_only"] += 1
            else:
                stat[src]["both"] += 1
            # Where does the EXCESS of a too-long prediction go? Onto another ANNOTATED
            # microtubule (a merge, which a better assembler could fix) or onto a filament
            # nobody drew (the ground truth's own incompleteness, which nothing can fix)?
            if not prec_ok:
                ys, xs = np.where(masks[pr["pred"]])
                if len(xs):
                    P = np.stack([xs, ys], 1).astype(float)
                    d_own = own_tree[j].query(P, k=1)[0]
                    excess = P[d_own > 5.0]
                    if len(excess):
                        d_any = all_tree.query(excess, k=1)[0]
                        # "near another GT" = within tol of SOME annotation that is not this one
                        len_ratio[src].append(float((d_any <= 5.0).mean()))
        stat[src]["unmatched_gt"] += len(gt) - len(matched_gt)
        stat[src]["n_gt"] += len(gt)
        stat[src]["n_pred"] += len(masks)

    print(f"FAILURE SPLIT at prob_thr={args.thr}, min_length={args.min_length}  "
          f"({os.path.basename(args.pred_dir)}, {args.split})\n")
    print(f'{"block":>6s} {"n GT":>6s} {"TP":>6s} {"coverage-only":>14s} {"precision-only":>15s} '
          f'{"both":>6s} {"unmatched":>10s} | {"excess on ANOTHER GT":>19s}')
    out = {}
    for s in ("585", "586", "htw"):
        d = stat[s]
        if not d:
            continue
        n = d["n_gt"]
        out[s] = dict(d)
        lr = np.mean(len_ratio[s]) if len_ratio[s] else float("nan")
        out[s]["excess_on_another_annotation"] = float(lr) if lr == lr else None
        print(f'{s:>6s} {n:6d} {d["tp"]:6d} {d["coverage_only"]:14d} {d["precision_only"]:15d} '
              f'{d["both"]:6d} {d["unmatched_gt"]:10d} | {lr * 100:18.1f}%')
    print()
    print("coverage-only = the prediction is too SHORT or absent  -> a model/detection problem")
    print("precision-only = the prediction is too LONG, running past this microtubule onto")
    print("                 another one or onto a filament nobody annotated -> assembly, or the")
    print("                 ground truth's own incompleteness")
    for s in ("585", "586", "htw"):
        if prec_of_failed[s]:
            a = np.array(prec_of_failed[s])
            print(f'  {s}: among precision-only failures, achieved precision '
                  f'p25/p50/p75 = {np.percentile(a,25):.2f}/{np.percentile(a,50):.2f}/'
                  f'{np.percentile(a,75):.2f}  (needs {args.cov})')
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
