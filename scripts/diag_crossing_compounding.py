#!/usr/bin/env python3
"""Does a microtubule's chance of being segmented correctly decay with the number of crossings
it passes through?

The crossing-dense block fails the metric's COVERAGE criterion on 171 of its 359 ground-truth
filaments -- the matched prediction does not span the whole microtubule -- while the semantic
foreground covers 99 % of the annotated centerline at 2 px. So the pixels are present and the
assembler is not joining them. In that block a filament crosses many others, and the junction
matcher preserves identity at a measured 0.752 on model masks against 0.917 on oracle masks.

If each junction is an independent chance to break, the probability that a filament survives k
crossings intact goes as p^k, and a filament through 5 crossings would come out at 0.75^5 = 0.24.
That would make 586's coverage failures a COMPOUNDING junction problem, not a detection problem,
and it would say the work belongs in the junction matcher rather than in the foreground.

This measures the decay directly: for every ground-truth filament, count the crossings it passes
through and record whether it ended up a true positive.

    python scripts/diag_crossing_compounding.py --pred-dir preds/LEN_treatment --thr 0.997 --min-length 20
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
from instance.metrics import centerline_f1  # noqa: E402
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


def crossings_per_gt(gt_dense, tol=3.0, endpoint_margin=8.0):
    """For each ground-truth filament, how many OTHER filaments it touches within ``tol``.

    Endpoints are excluded by a margin so that two filaments merely ending near each other are not
    counted as a crossing -- a crossing is an interior contact.
    """
    trees = [cKDTree(g) for g in gt_dense]
    counts = []
    for i, g in enumerate(gt_dense):
        m = endpoint_margin
        interior = g[int(m):len(g) - int(m)] if len(g) > 2 * m + 4 else g
        n = 0
        for j, t in enumerate(trees):
            if i == j or len(interior) == 0:
                continue
            oj = gt_dense[j]
            o_int = oj[int(m):len(oj) - int(m)] if len(oj) > 2 * m + 4 else oj
            if len(o_int) == 0:
                continue
            if cKDTree(o_int).query(interior, k=1)[0].min() <= tol:
                n += 1
        counts.append(n)
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--data-dir", default="data/real/mt34_htw_eval")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--params", default="src/instance/params_a_model_v5H.json")
    ap.add_argument("--thr", type=float, default=0.997)
    ap.add_argument("--min-length", type=float, default=20.0)
    ap.add_argument("--out", default="crossing_compounding.json")
    args = ap.parse_args()
    if args.split == "test":
        raise SystemExit("refusing to touch TEST: it carries a declared exposure budget")

    params = dict(json.load(open(args.params)))
    params["min_length"] = args.min_length
    params["prob_thr"] = args.thr

    rows = defaultdict(list)      # src -> (n_crossings, is_tp, gt_length)
    for path in sorted(glob.glob(os.path.join(args.data_dir, "*.h5"))):
        fr = read_frame_h5(path)
        if str(fr["attrs"].get("split")) != args.split:
            continue
        name = os.path.basename(path).replace(".h5", ".npz")
        src = source_of(name)
        gt = [np.asarray(p, float) * UP for p in fr["polylines"] if len(p) >= 2]
        if len(gt) < 2:
            continue
        npz = os.path.join(args.pred_dir, name)
        if not os.path.isfile(npz):
            raise SystemExit(f"missing prediction {npz} for a frame that has ground truth")
        prob = np.load(npz)["prob"].astype(np.float32)
        if prob.ndim == 3:
            prob = prob.max(axis=0)
        img = fr["image"]
        hi = (int(round(img.shape[0] * UP)), int(round(img.shape[1] * UP)))
        fv = zoom(fov_mask(img).astype(np.uint8), UP, order=0).astype(bool)
        fv = np.pad(fv, ((0, max(0, hi[0] - fv.shape[0])), (0, max(0, hi[1] - fv.shape[1]))),
                    constant_values=True)[:hi[0], :hi[1]]
        _polys, masks = instance_a((prob > args.thr) & fv, KAPPA_MAX, params=params,
                                   channels=prob[None], prob=prob)
        r = centerline_f1(masks, gt, tol=5.0, length_coverage=0.95, precision_coverage=0.95)
        tp_gt = {p["gt"] for p in r["pairs"] if p["tp"]}

        dense = [densify(g) for g in gt]
        xs = crossings_per_gt(dense)
        for j, (g, k) in enumerate(zip(dense, xs)):
            L = float(np.sqrt(((g[1:] - g[:-1]) ** 2).sum(1)).sum())
            rows[src].append((k, 1.0 if j in tp_gt else 0.0, L))

    print(f"TRUE-POSITIVE RATE vs CROSSINGS, STRATIFIED BY LENGTH  "
          f"(thr={args.thr}, min_length={args.min_length})\n")
    print("Crossing count and filament length are correlated -- a long filament meets more of them")
    print("and also has more chances to break anywhere else -- so the marginal decay cannot")
    print("separate the two. Within a length band it can.\n")
    LEN_BINS = [(0, 200), (200, 450), (450, 900), (900, 1e9)]
    out = {}
    for s in ("585", "586", "htw"):
        R = np.array(rows[s]) if rows[s] else None
        if R is None or len(R) < 20:
            continue
        print(f'--- {s} ---   {len(R)} filaments, overall TP rate {R[:, 1].mean():.3f}')
        print(f'  {"length band":>14s} {"n":>5s} | ' +
              ''.join(f'{lbl:>16s}' for lbl in ("0 crossings", "1", "2+")))
        strat = {}
        for lo, hi_ in LEN_BINS:
            m = (R[:, 2] >= lo) & (R[:, 2] < hi_)
            if m.sum() < 12:
                continue
            band = R[m]
            cells = []
            for cl, ch in ((0, 0), (1, 1), (2, 999)):
                q = (band[:, 0] >= cl) & (band[:, 0] <= ch)
                cells.append(f'{band[q, 1].mean():.3f} (n={int(q.sum())})' if q.sum() >= 4 else '-')
            lbl = f"{int(lo)}-{'inf' if hi_ > 1e8 else int(hi_)}"
            strat[lbl] = cells
            print(f'  {lbl:>14s} {int(m.sum()):5d} | ' + ''.join(f'{c:>16s}' for c in cells))
        # and the pure LENGTH effect at zero crossings, which isolates the other axis
        z = R[R[:, 0] == 0]
        if len(z) >= 20:
            print(f'  at ZERO crossings, TP rate by length:', end=' ')
            for lo, hi_ in LEN_BINS:
                m = (z[:, 2] >= lo) & (z[:, 2] < hi_)
                if m.sum() >= 6:
                    print(f'{int(lo)}-{"inf" if hi_ > 1e8 else int(hi_)}: {z[m, 1].mean():.3f}', end='  ')
            print()
        out[s] = dict(n=len(R), tp_rate=float(R[:, 1].mean()), stratified=strat)
        print()
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
