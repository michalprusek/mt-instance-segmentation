#!/usr/bin/env python3
"""Score predict.py's output against the MT-34 ground truth. An acceptance test.

Run it to confirm a deployed package reproduces the number it was shipped with. It is a
REPRODUCTION check, not a research measurement: the expected values in README.md were produced
by this same script on this same host, so a match means the installation is sound and a
mismatch means something in the environment differs.

    python predict.py --input mt34-benchmark/tif --out-dir /tmp/pred
    python score.py --pred /tmp/pred --data mt34-benchmark/gt --split val

Two conventions this script exists to get right, both of which have silently broken things in
this project before:

* **Scale.** ``predict.py`` reports polylines in ORIGINAL image pixels; the benchmark metric is
  defined in the 1.5x evaluation frame. Both predictions and ground truth are scaled here, in
  one place, so the two can never drift apart.
* **Coordinate order.** Everything is ``[x, y] = [col, row]``, the transpose of NumPy indexing.
  That is what ``predict.py`` writes and what the h5 stores.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

UP = 1.5
TOL, LEN_COV, PREC_COV = 5.0, 0.95, 0.95


def read_gt(path: str):
    """(image_shape, polylines, attrs) for one benchmark frame, from JSON or h5.

    JSON is the primary format: the production container has no ``h5py`` (verified against
    ``cell-segmentation-hub-ml``), so ground truth readable only from h5 would be unreadable
    exactly where it is meant to be checked. The h5 branch is kept for hosts that do have it.
    """
    if path.endswith(".json"):
        d = json.load(open(path))
        return tuple(d["shape"]), [np.asarray(p, float) for p in d["polylines"]], d
    import h5py
    with h5py.File(path, "r") as h:
        attrs = dict(h.attrs)
        shape = (int(attrs["height"]), int(attrs["width"]))
        g = h["polylines"]
        polys = [np.asarray(g[k], dtype=float) for k in sorted(g.keys())]
    return shape, polys, attrs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="directory of predict.py JSON outputs")
    ap.add_argument("--data", required=True,
                    help="directory of ground truth: gt/*.json (preferred) or h5/*.h5")
    ap.add_argument("--split", default="val", choices=["val", "test", "all"],
                    help="'test' is the project's held-out split -- see README before using it")
    ap.add_argument("--quiet", action="store_true", help="aggregate only")
    args = ap.parse_args()

    from instance.metrics import aggregate_benchmark, centerline_f1
    from instance.oracle import oracle_instance_masks

    rows = []
    files = sorted(glob.glob(os.path.join(args.data, "*.json"))) or \
        sorted(glob.glob(os.path.join(args.data, "*.h5")))
    if not files:
        raise SystemExit(f"no ground truth (*.json or *.h5) under {args.data}")
    for path in files:
        shape, gt_polys, attrs = read_gt(path)
        if args.split != "all" and str(attrs.get("split")) != args.split:
            continue
        if not gt_polys:
            continue                      # zero-GT frames are over-firing probes, not scorable
        name = os.path.splitext(os.path.basename(path))[0]
        jf = os.path.join(args.pred, name + ".json")
        if not os.path.exists(jf):
            print(f"  {name}: NO PREDICTION -- skipped")
            continue
        pred = [np.asarray(p, dtype=float) for p in json.load(open(jf))["polylines"]]
        pred = [p for p in pred if len(p) >= 2]

        # Both sides into the 1.5x evaluation frame, together, here and nowhere else.
        masks = oracle_instance_masks(pred, shape, half_width=1.0, up=UP)
        gt = [q * UP for q in gt_polys]
        r = centerline_f1(masks, gt, tol=TOL, length_coverage=LEN_COV,
                          precision_coverage=PREC_COV)
        r["name"] = name
        r["task"] = str(attrs.get("source_task", "?"))
        rows.append(r)
        if not args.quiet:
            print("  %-46s gt=%3d pred=%3d tp=%3d fp=%3d fn=%3d f1=%.3f"
                  % (name[:46], len(gt), len(pred), r["tp"], r["fp"], r["fn"], r["f1"]))

    if not rows:
        raise SystemExit("nothing scored -- check --pred and --split")
    agg = aggregate_benchmark(rows)
    print("\n%s split, %d frames: mean F1 = %.4f  (micro %.4f, tp=%d fp=%d fn=%d)"
          % (args.split, len(rows), agg["mean_f1"], agg["micro_f1"],
             agg["total_tp"], agg["total_fp"], agg["total_fn"]))
    for t in sorted({r["task"] for r in rows}):
        sub = [r for r in rows if r["task"] == t]
        print("   task %-4s n=%2d  mean F1 = %.4f"
              % (t, len(sub), float(np.mean([r["f1"] for r in sub]))))


if __name__ == "__main__":
    main()
