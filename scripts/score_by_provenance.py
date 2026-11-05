#!/usr/bin/env python3
"""Score any method on the HUMAN-DRAWN ground truth only, and report it beside the mixed number.

Measured 2026-08-28: a prediction scores +0.20 to +0.28 higher against ground-truth polylines that
an older v7+PySOAX pipeline drew than against human-drawn ones, after controlling for length, in
5 of 5 comparable bands. MT-34's VAL ground truth is 69 % model-drawn in the 585 block and 25 % in
586, so every number reported on it is inflated by an unknown amount.

Simply dropping the model-drawn polylines would be wrong: a prediction that correctly follows one of
them would then count as a false positive, which penalises the method for being right. They are
treated as DON'T-CARE instead -- the standard ignore-region protocol. A prediction is dropped from
scoring when its best match among the model-drawn polylines is better than its best match among the
human-drawn ones.

This works for ANY method's predicted instances, so the same script scores the baselines. That
matters more than the headline: our instancer is PySOAX-family and the model-drawn ground truth was
made by v7+PySOAX, so the bias plausibly flatters US more than a foundation model, and the recorded
comparative claim rests on it.

    python scripts/score_by_provenance.py --pred-dir preds/LEN_treatment --thr 0.997 --min-length 20
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
from instance.metrics import aggregate_benchmark, centerline_f1  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.fov import fov_mask  # noqa: E402

UP = 1.5
KAPPA_MAX = 0.25
TOL = 5.0


def block_of(name):
    b = os.path.basename(name)
    return "htw" if b.startswith("htw") else ("585" if b.startswith("alice") else "586")


def densify(p, step=1.0):
    seg = np.sqrt(((p[1:] - p[:-1]) ** 2).sum(1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return p
    t = np.arange(0.0, s[-1], step)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)


def skel_pts(mask):
    ys, xs = np.where(mask)
    return np.stack([xs, ys], 1).astype(float) if len(ys) else np.zeros((0, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default=None,
                   help="cached probability maps; instances are built with our instancer")
    ap.add_argument("--inst-dir", default=None,
                   help="pre-computed instances from ANOTHER method (baseline_mt34.py --dump-inst); "
                        "scored through this identical path so the comparison cannot differ for "
                        "reasons other than the methods themselves")
    ap.add_argument("--data-dir", default="data/real/mt34_htw_eval")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--params", default="src/instance/params_a_model_v5H.json")
    ap.add_argument("--thr", type=float, default=0.997)
    ap.add_argument("--min-length", type=float, default=20.0)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="score_by_provenance.json")
    args = ap.parse_args()
    if args.split == "test":
        raise SystemExit("refusing to touch TEST: it carries a declared exposure budget")
    if bool(args.pred_dir) == bool(args.inst_dir):
        raise SystemExit("give exactly one of --pred-dir (probability maps) or --inst-dir "
                         "(another method's instances)")

    params = dict(json.load(open(args.params)))
    params["min_length"] = args.min_length
    params["prob_thr"] = args.thr

    mixed, human = defaultdict(list), defaultdict(list)
    counts = defaultdict(lambda: defaultdict(int))
    for path in sorted(glob.glob(os.path.join(args.data_dir, "*.h5"))):
        fr = read_frame_h5(path)
        if str(fr["attrs"].get("split")) != args.split:
            continue
        b = os.path.basename(path)
        blk = block_of(b)
        raw = str(fr["attrs"].get("sources", ""))
        srcs = raw.split(",") if raw else []
        polys = fr["polylines"]
        if len(srcs) != len(polys):
            print(f"  SKIP {b}: {len(srcs)} provenance entries for {len(polys)} polylines")
            continue
        keep = [i for i, p in enumerate(polys) if len(p) >= 2]
        gt = [np.asarray(polys[i], float) * UP for i in keep]
        srcs = [srcs[i] for i in keep]
        if not gt:
            continue
        img = fr["image"]
        hi = (int(round(img.shape[0] * UP)), int(round(img.shape[1] * UP)))
        if args.pred_dir:
            npz = os.path.join(args.pred_dir, b.replace(".h5", ".npz"))
            if not os.path.isfile(npz):
                raise SystemExit(f"missing prediction {npz} for a frame that has ground truth")
            prob = np.load(npz)["prob"].astype(np.float32)
            if prob.ndim == 3:
                prob = prob.max(axis=0)
            fv = zoom(fov_mask(img).astype(np.uint8), UP, order=0).astype(bool)
            fv = np.pad(fv, ((0, max(0, hi[0] - fv.shape[0])), (0, max(0, hi[1] - fv.shape[1]))),
                        constant_values=True)[:hi[0], :hi[1]]
            _p, masks = instance_a((prob > args.thr) & fv, KAPPA_MAX, params=params,
                                   channels=prob[None], prob=prob)
        else:
            ip = os.path.join(args.inst_dir, b.replace(".h5", ".npz"))
            if not os.path.isfile(ip):
                raise SystemExit(f"missing instances {ip} for a frame that has ground truth")
            z = np.load(ip)
            shp = tuple(int(v) for v in z["shape"]) if int(z["n"]) else hi
            masks = []
            for k in range(int(z["n"])):
                m = np.zeros(shp, bool)
                m[z[f"r{k}"], z[f"c{k}"]] = True
                masks.append(m)

        # (a) the number as usually reported: all ground truth, mixed provenance
        r_all = centerline_f1(masks, gt, tol=TOL, length_coverage=0.95, precision_coverage=0.95)
        r_all["n_gt"] = len(gt)
        mixed[blk].append(r_all)

        # (b) human-drawn ground truth only, with the model-drawn polylines as DON'T-CARE
        hum = [g for g, s in zip(gt, srcs) if s == "manual"]
        mod = [g for g, s in zip(gt, srcs) if s != "manual"]
        counts[blk]["manual"] += len(hum)
        counts[blk]["file"] += len(mod)
        if not hum:
            continue
        if mod:
            th = cKDTree(np.concatenate([densify(g) for g in hum], 0))
            tm = cKDTree(np.concatenate([densify(g) for g in mod], 0))
            kept = []
            for m in masks:
                pts = skel_pts(m)
                if not len(pts):
                    continue
                # a prediction that follows a model-drawn polyline more closely than any
                # human-drawn one is neither credited nor penalised
                near_h = (th.query(pts, k=1)[0] <= TOL).mean()
                near_m = (tm.query(pts, k=1)[0] <= TOL).mean()
                if near_m > near_h and near_m > 0.5:
                    continue
                kept.append(m)
        else:
            kept = list(masks)
        r_h = centerline_f1(kept, hum, tol=TOL, length_coverage=0.95, precision_coverage=0.95)
        r_h["n_gt"] = len(hum)
        human[blk].append(r_h)

    tag = args.label or os.path.basename(args.pred_dir)
    print(f"SCORE BY GROUND-TRUTH PROVENANCE — {tag}  "
          f"(thr={args.thr}, min_length={args.min_length}, {args.split})\n")
    print("  mixed  = every ground-truth polyline, as reported so far")
    print("  human  = human-drawn polylines only; model-drawn ones are don't-care\n")
    print(f'{"block":>6s} {"GT manual":>10s} {"GT model":>9s} | {"mixed F1":>9s} {"human F1":>9s} '
          f'{"inflation":>10s}')
    out = {}
    for blk in ("585", "586", "htw"):
        if not mixed[blk]:
            continue
        a = aggregate_benchmark(mixed[blk])
        h = aggregate_benchmark(human[blk]) if human[blk] else None
        hv = h["mean_f1"] if h else float("nan")
        out[blk] = dict(mixed=a["mean_f1"], human=hv,
                        n_manual=counts[blk]["manual"], n_file=counts[blk]["file"])
        print(f'{blk:>6s} {counts[blk]["manual"]:10d} {counts[blk]["file"]:9d} | '
              f'{a["mean_f1"]:9.4f} {hv:9.4f} {a["mean_f1"] - hv:+10.4f}')
    allm = [r for blk in mixed for r in mixed[blk]]
    allh = [r for blk in human for r in human[blk]]
    A, H = aggregate_benchmark(allm), aggregate_benchmark(allh)
    print(f'{"POOL":>6s} {sum(counts[b]["manual"] for b in counts):10d} '
          f'{sum(counts[b]["file"] for b in counts):9d} | '
          f'{A["mean_f1"]:9.4f} {H["mean_f1"]:9.4f} {A["mean_f1"] - H["mean_f1"]:+10.4f}')
    out["POOL"] = dict(mixed=A["mean_f1"], human=H["mean_f1"])
    print()
    print("  worst block, mixed: "
          f'{min(out[b]["mixed"] for b in ("585", "586", "htw") if b in out):.4f}'
          "    worst block, human: "
          f'{min(out[b]["human"] for b in ("585", "586", "htw") if b in out):.4f}')
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
