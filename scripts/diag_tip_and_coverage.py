#!/usr/bin/env python3
"""Is HTW's 70 % false-negative mass a DETECTION failure, or filaments found and truncated at the tips?

The strict metric needs 95 % of a ground-truth centerline covered within 5 px, so a systematic
per-end truncation of ``t`` px leaves ``max(0, t - 5)`` uncovered at each end and the whole instance
survives only while ``t <= 5 + 0.025 * L``. At the 1.5x eval scale HTW's median ground truth is about
42 px, so its budget is roughly 6 px -- while 585's, at 605 px, is 20 px. A 4 px tip deficit, which is
the default behaviour of any thresholded probability map and is maximised by the shipped
``prob_thr = 0.99``, therefore costs 585 nothing and could cost HTW half its score.

That would mean the recorded "HTW is semantic-limited / never detected" diagnosis is measuring the
wrong thing: in this metric a false negative is equally "found, but short at both ends". The
diagnosis is also v4b-era -- no per-block semantic coverage has ever been measured on v5H.

Three measurements, all on cached predictions, no GPU:

  A. Per-block SEMANTIC coverage: what fraction of each ground-truth centerline is within 2 px of
     predicted foreground, at each threshold. This separates "never fired" from "fired and trimmed".
  B. TIP GAP: distance from each ground-truth ENDPOINT to the nearest predicted foreground pixel, at
     each threshold. If tips are present at 0.90 but missing at 0.99, one global threshold is paying
     for endpoints with false positives elsewhere, and hysteresis recovers them for free.
  C. The metric's own sensitivity: score the ground truth against a TRUNCATED copy of itself, which
     is the upper bound on what any tip-faithful method could lose.

    python scripts/diag_tip_and_coverage.py --pred-dir preds/v5H_combined_val --split val
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.ndimage import distance_transform_edt, zoom
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from instance.metrics import centerline_f1  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.fov import fov_mask  # noqa: E402

UP = 1.5


def source_of(name):
    b = os.path.basename(name)
    return "htw" if b.startswith("htw") else ("585" if b.startswith("alice") else "586")


def densify(p, step=1.0):
    seg = np.sqrt(((p[1:] - p[:-1]) ** 2).sum(1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return p, 0.0
    t = np.arange(0.0, s[-1], step)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1), float(s[-1])


def truncate(p, t):
    """Drop ``t`` px of arc length from each end -- the shape a tip-shy predictor produces."""
    q, L = densify(p)
    if L <= 2 * t + 4:
        return None
    k = int(round(t))
    return q[k:len(q) - k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--data-dir", default="data/real/mt34_htw_eval")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--thrs", type=float, nargs="+", default=[0.5, 0.7, 0.9, 0.97, 0.99, 0.995])
    ap.add_argument("--out", default="tip_coverage.json")
    args = ap.parse_args()
    if args.split == "test":
        raise SystemExit("refusing to touch TEST: it carries a declared exposure budget")

    frames = []
    for p in sorted(glob.glob(os.path.join(args.data_dir, "*.h5"))):
        fr = read_frame_h5(p)
        if str(fr["attrs"].get("split")) == args.split:
            frames.append((p, fr))

    cov = defaultdict(lambda: defaultdict(list))     # [thr][src] -> per-GT coverage fraction
    tip = defaultdict(lambda: defaultdict(list))     # [thr][src] -> per-endpoint distance
    budget = defaultdict(list)

    for path, fr in frames:
        name = os.path.basename(path).replace(".h5", ".npz")
        src = source_of(name)
        npz = os.path.join(args.pred_dir, name)
        gt = [np.asarray(p, float) * UP for p in fr["polylines"] if len(p) >= 2]
        if not gt:
            continue
        for g in gt:
            _q, L = densify(g)
            budget[src].append(5.0 + 0.025 * L)      # the metric's own per-end tolerance
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

        for thr in args.thrs:
            m = (prob > thr) & fv
            if not m.any():
                continue
            # distance to the nearest foreground pixel, everywhere
            dist = distance_transform_edt(~m)
            for g in gt:
                q, _L = densify(g)
                c = np.clip(np.round(q[:, 0]).astype(int), 0, hi[1] - 1)
                r = np.clip(np.round(q[:, 1]).astype(int), 0, hi[0] - 1)
                d = dist[r, c]
                cov[thr][src].append(float((d <= 2.0).mean()))
                tip[thr][src].append(float(d[0]))
                tip[thr][src].append(float(d[-1]))

    print("A/B. PER-BLOCK SEMANTIC COVERAGE and TIP GAP, cached predictions, VAL")
    print("   coverage = fraction of the GT centerline within 2 px of predicted foreground")
    print("   tip gap  = distance from a GT ENDPOINT to the nearest predicted foreground pixel\n")
    print(f'{"thr":>7s} {"block":>6s} {"n GT":>6s} {"coverage p50":>13s} {"cov<0.5":>8s} '
          f'{"tip gap p50":>12s} {"tip p75":>8s} {"% tips >6 px":>13s}')
    out = {}
    for thr in args.thrs:
        for s in ("585", "586", "htw"):
            c = np.array(cov[thr][s]); t = np.array(tip[thr][s])
            if not len(c):
                continue
            out[f"thr{thr}_{s}"] = dict(n=len(c), cov_p50=float(np.median(c)),
                                        cov_lt_half=float((c < 0.5).mean()),
                                        tip_p50=float(np.median(t)),
                                        tip_p75=float(np.percentile(t, 75)),
                                        tip_gt6=float((t > 6).mean()))
            print(f'{thr:7.3f} {s:>6s} {len(c):6d} {np.median(c):13.3f} '
                  f'{(c < 0.5).mean() * 100:7.1f}% {np.median(t):12.2f} '
                  f'{np.percentile(t, 75):8.2f} {(t > 6).mean() * 100:12.1f}%')
        print()

    print("Metric budget: a per-end truncation survives only while t <= 5 + 0.025*L")
    print(f'{"block":>6s} {"median budget px":>17s} {"% of GT with budget < 7 px":>28s}')
    for s in ("585", "586", "htw"):
        b = np.array(budget[s])
        print(f'{s:>6s} {np.median(b):17.1f} {(b < 7).mean() * 100:27.1f}%')

    print("\nC. THE METRIC AGAINST ITSELF: ground truth scored against a truncated copy")
    print(f'{"truncation":>12s} {"585":>8s} {"586":>8s} {"htw":>8s} {"pooled":>8s}')
    for t in (2, 3, 4, 5):
        per = defaultdict(list)
        for path, fr in frames:
            src = source_of(os.path.basename(path))
            gt = [np.asarray(p, float) * UP for p in fr["polylines"] if len(p) >= 2]
            if not gt:
                continue
            H = int(max(max(g[:, 1].max() for g in gt) + 50, 64))
            W = int(max(max(g[:, 0].max() for g in gt) + 50, 64))
            pred = []
            for g in gt:
                q = truncate(g, t)
                if q is None:
                    continue
                m = np.zeros((H, W), bool)
                c = np.clip(np.round(q[:, 0]).astype(int), 0, W - 1)
                r = np.clip(np.round(q[:, 1]).astype(int), 0, H - 1)
                m[r, c] = True
                pred.append(m)
            if not pred:
                per[src].append(0.0)
                continue
            r = centerline_f1(pred, gt, tol=5.0, length_coverage=0.95, precision_coverage=0.95)
            per[src].append(r["f1"])
        allv = [v for s in per for v in per[s]]
        print(f'{t:9d} px {np.mean(per["585"]):8.3f} {np.mean(per["586"]):8.3f} '
              f'{np.mean(per["htw"]):8.3f} {np.mean(allv):8.3f}')

    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
