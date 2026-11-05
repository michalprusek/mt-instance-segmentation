#!/usr/bin/env python3
"""Paired per-frame bootstrap between ANY two methods, each at its own operating point.

The existing paired test compares two of our own arms, both delivered as probability maps. A
baseline delivers finished instances instead, so a comparison against one has to accept both
forms -- and it has to score them through the SAME code, or the comparison differs for reasons
other than the methods.

Each side is given as either
    prob:<dir>:<thr>:<min_length>     our probability maps, instanced here
    inst:<dir>                        another method's instances (baseline_mt34.py --dump-inst)

    python scripts/paired_any.py \
        --a prob:preds/LEN_treatment_roi:0.99:44.74 --a-label ours \
        --b inst:inst_sam3_roi/conf0.4 --b-label sam3 \
        --data-dir data/real/roi303_eval --split val
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import zoom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from instance.instancer_a import instance_a  # noqa: E402
from instance.metrics import aggregate_benchmark, centerline_f1  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.fov import fov_mask  # noqa: E402

UP = 1.5
KAPPA_MAX = 0.25


def parse_side(spec):
    parts = spec.split(":")
    if parts[0] == "prob":
        if len(parts) != 4:
            raise SystemExit(f"prob spec needs dir:thr:min_length, got {spec!r}")
        return dict(kind="prob", dir=parts[1], thr=float(parts[2]), ml=float(parts[3]))
    if parts[0] == "inst":
        if len(parts) != 2:
            raise SystemExit(f"inst spec needs a directory, got {spec!r}")
        return dict(kind="inst", dir=parts[1])
    raise SystemExit(f"unknown side kind in {spec!r}; use prob: or inst:")


def masks_for(side, fr, name, base_params):
    img = fr["image"]
    hi = (int(round(img.shape[0] * UP)), int(round(img.shape[1] * UP)))
    if side["kind"] == "prob":
        p = os.path.join(side["dir"], name)
        if not os.path.isfile(p):
            raise SystemExit(f"missing prediction {p}")
        prob = np.load(p)["prob"].astype(np.float32)
        if prob.ndim == 3:
            prob = prob.max(axis=0)
        fv = zoom(fov_mask(img).astype(np.uint8), UP, order=0).astype(bool)
        fv = np.pad(fv, ((0, max(0, hi[0] - fv.shape[0])), (0, max(0, hi[1] - fv.shape[1]))),
                    constant_values=True)[:hi[0], :hi[1]]
        params = dict(base_params)
        params["min_length"] = side["ml"]
        params["prob_thr"] = side["thr"]
        _p, m = instance_a((prob > side["thr"]) & fv, KAPPA_MAX, params=params,
                           channels=prob[None], prob=prob)
        return m
    p = os.path.join(side["dir"], name)
    if not os.path.isfile(p):
        raise SystemExit(f"missing instances {p}")
    z = np.load(p)
    shp = tuple(int(v) for v in z["shape"]) if int(z["n"]) else hi
    out = []
    for k in range(int(z["n"])):
        mm = np.zeros(shp, bool)
        mm[z[f"r{k}"], z[f"c{k}"]] = True
        out.append(mm)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--a-label", default="A")
    ap.add_argument("--b-label", default="B")
    ap.add_argument("--data-dir", default="data/real/roi303_eval")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--params", default="src/instance/params_a_model_v5H.json")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--allow-test", action="store_true",
                    help="required to score TEST; it carries a declared exposure budget")
    ap.add_argument("--out", default="paired_any.json")
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("scoring TEST needs --allow-test: it carries a declared exposure budget")

    A, B = parse_side(args.a), parse_side(args.b)
    base = json.load(open(args.params))

    fa, fb, names = [], [], []
    for path in sorted(glob.glob(os.path.join(args.data_dir, "*.h5"))):
        fr = read_frame_h5(path)
        if str(fr["attrs"].get("split")) != args.split:
            continue
        gt = [np.asarray(p, float) * UP for p in fr["polylines"] if len(p) >= 2]
        if not gt:
            continue
        nm = os.path.basename(path).replace(".h5", ".npz")
        ra = centerline_f1(masks_for(A, fr, nm, base), gt, tol=5.0,
                           length_coverage=0.95, precision_coverage=0.95)
        rb = centerline_f1(masks_for(B, fr, nm, base), gt, tol=5.0,
                           length_coverage=0.95, precision_coverage=0.95)
        ra["n_gt"] = rb["n_gt"] = len(gt)
        fa.append(ra)
        fb.append(rb)
        names.append(nm)

    aa, bb = aggregate_benchmark(fa), aggregate_benchmark(fb)
    print(f"PAIRED, {args.split}, {len(names)} frames, each side at its own operating point\n")
    print(f'{"method":14s} {"mean-F1":>9s} {"micro":>8s} {"TP":>7s} {"FP":>7s} {"FN":>7s}')
    for lab, v in ((args.a_label, aa), (args.b_label, bb)):
        print(f'{lab:14s} {v["mean_f1"]:9.4f} {v["micro_f1"]:8.4f} '
              f'{v["total_tp"]:7d} {v["total_fp"]:7d} {v["total_fn"]:7d}')

    d = np.array([r["f1"] for r in fb]) - np.array([r["f1"] for r in fa])
    rng = np.random.default_rng(20260828)
    boot = d[rng.integers(0, len(d), size=(args.boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    print(f"\n  {args.b_label} - {args.a_label} = {d.mean():+.4f}   "
          f"95% CI [{lo:+.4f}, {hi:+.4f}]   p={p:.4f}   "
          f"{args.b_label} better on {int((d > 0).sum())}/{len(d)} frames")
    print(f"  verdict: {'SEPARATED' if (lo > 0 or hi < 0) else 'not separated'}")

    json.dump(dict(a=dict(label=args.a_label, spec=args.a, mean_f1=aa["mean_f1"]),
                   b=dict(label=args.b_label, spec=args.b, mean_f1=bb["mean_f1"]),
                   delta=float(d.mean()), ci=[float(lo), float(hi)], p=float(p),
                   n=len(d), split=args.split), open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
