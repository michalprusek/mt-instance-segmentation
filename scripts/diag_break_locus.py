#!/usr/bin/env python3
"""WHERE do the residual fragments break? Ends of fragment predictions located relative to GT
crossings and GT endpoints.

Step 2f' (2026-09-16): under the "band + nothing off the annotation" ceiling the model-owned
residual is FRAGMENTS (a prediction >= 95 % on one GT covering < 80 % of it), not junction
chimeras, and the width profile says the mask is not wider than the band on htw. So the spurs
that break the assembler's traversal are either the model's foreground filling the WEDGE of a
crossing (the break sits at a crossing) or side strands along a plain stretch (mid-filament
break). This locates every fragment end:

  at_gt_end      <= 10 eval px from an endpoint of the GT it lies on   (a legitimate end)
  near_crossing  <= 15 eval px from a GT crossing                       (wedge / junction)
  near_other_gt  <= --other-px (8) eval px from a DIFFERENT GT filament  (close parallel)
  mid            elsewhere                                              (along-filament spur)

Run on a dumped map dir (`diag_ceiling_union.py --dump-dir`) with the same threshold; the same
statistics on the oracle dump are the reference. VAL only.

    PYTHONPATH=src python scripts/diag_break_locus.py --pred-dir preds/CEIL_roi303_all_delfar5 \\
        --data data/real/roi303_eval --split val --thr 0.98 --out brk.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from instance.geometry import resample  # noqa: E402
from instance.instancer_a import instance_a  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.gt_stats import count_crossings  # noqa: E402

UP, KAPPA_MAX, TOL = 1.5, 0.25, 5.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--thr", type=float, required=True)
    ap.add_argument("--params", default="src/instance/params_a_derived.json")
    ap.add_argument("--cov", type=float, default=0.80)
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--end-px", type=float, default=10.0)
    ap.add_argument("--cross-px", type=float, default=15.0)
    ap.add_argument("--other-px", type=float, default=8.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.split == "test":
        raise SystemExit("VAL only")
    params = json.load(open(args.params))
    ends = {"fragment": collections.Counter(), "all": collections.Counter()}
    n_pred = n_frag = 0
    for path in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        name = os.path.basename(path)
        if args.prefix and not name.startswith(args.prefix):
            continue
        fr = read_frame_h5(path)
        if str(fr["attrs"].get("split")) != args.split or not fr["polylines"]:
            continue
        npz = os.path.join(args.pred_dir, name.replace(".h5", ".npz"))
        if not os.path.exists(npz):
            continue
        ch = np.load(npz)["prob"].astype(np.float32)
        prob = ch.max(axis=0)
        gts = [resample(np.asarray(q, float), ds=1.0) * UP for q in fr["polylines"] if len(q) >= 2]
        gts = [g for g in gts if len(g) >= 2]
        if not gts:
            continue
        gt_pts = np.concatenate(gts, 0)
        gt_id = np.concatenate([np.full(len(g), i) for i, g in enumerate(gts)])
        tree = cKDTree(gt_pts)
        gt_trees = [cKDTree(g) for g in gts]
        gt_ends = np.array([g[0] for g in gts] + [g[-1] for g in gts])
        end_tree = cKDTree(gt_ends)
        cr = count_crossings([np.asarray(q, float) for q in fr["polylines"]], tol=2.0, ds=1.0, endpoint_margin=8.0)
        cr_pts = np.array([[c["x"] * UP, c["y"] * UP] for c in cr]) if cr else np.zeros((0, 2))
        cr_tree = cKDTree(cr_pts) if len(cr_pts) else None
        polys, _ = instance_a(prob > args.thr, KAPPA_MAX, params=params, channels=ch, prob=prob)
        for pl in polys:
            pl = np.asarray(pl, float)
            if len(pl) < 2:
                continue
            pl = resample(pl, ds=1.0)
            if len(pl) < 2:
                continue
            n_pred += 1
            d, idx = tree.query(pl)
            on = d <= TOL
            cls = "other"
            if on.mean() >= 0.95:
                g = int(np.bincount(gt_id[idx[on]]).argmax())
                cov = float((gt_trees[g].query(pl)[0] <= TOL).mean())  # share of prediction near that GT
                gcov = float((cKDTree(pl).query(gts[g])[0] <= TOL).mean())  # share of that GT covered
                if gcov < args.cov:
                    cls = "fragment"; n_frag += 1
            own = int(np.bincount(gt_id[idx[on]]).argmax()) if on.any() else -1
            for e in (pl[0], pl[-1]):
                de = end_tree.query(e)[0]
                dc = cr_tree.query(e)[0] if cr_tree is not None else np.inf
                # nearest GT sample that belongs to a different filament than the one the prediction lies on
                dd, ii = tree.query(e, k=min(64, len(gt_pts)))
                dd, ii = np.atleast_1d(dd), np.atleast_1d(ii)
                other = dd[gt_id[ii] != own]
                do = float(other.min()) if other.size else np.inf
                loc = ("at_gt_end" if de <= args.end_px else "near_crossing" if dc <= args.cross_px
                       else "near_other_gt" if do <= args.other_px else "mid")
                ends["all"][loc] += 1
                if cls == "fragment":
                    ends["fragment"][loc] += 1
    out = {"pred_dir": args.pred_dir, "thr": args.thr, "n_pred": n_pred, "n_fragment": n_frag,
           "ends": {k: dict(v) for k, v in ends.items()}}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    for k in ("fragment", "all"):
        c = ends[k]; tot = sum(c.values()) or 1
        print(f"{k:<9} n_ends={tot}  at_gt_end {c['at_gt_end'] / tot:.3f}  near_crossing {c['near_crossing'] / tot:.3f}  "
              f"near_other_gt {c['near_other_gt'] / tot:.3f}  mid {c['mid'] / tot:.3f}")
    print(f"predictions {n_pred}, fragments {n_frag}")


if __name__ == "__main__":
    main()
