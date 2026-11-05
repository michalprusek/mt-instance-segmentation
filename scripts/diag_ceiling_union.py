#!/usr/bin/env python3
"""What is curing a given part of the model's map WORTH, on the declared instrument?

Advisor request 2026-09-16: before anyone prices a lever by intuition, bound it. Take the cached
probability map, paint the ground-truth band INTO it in a chosen region only, and score the result
through the unchanged instancer under the declared convention (0.80/0.80, derived constant). The
model's own false positives are kept, so the number is "recall-side ceiling with this defect cured",
not an oracle:

  region = cross20    GT band within 20 native px (30 eval px) of a GT crossing   -> the crossing erosion
  region = cross40    GT band within 40 native px (60 eval px) of a GT crossing   -> the wider halo
  region = all        GT band everywhere                                          -> the whole along-filament deficit
  region = none       nothing painted                                             -> the baseline, same code path

Painting = union of the thresholded mask with the band, and the probability raised to 1.0 on the
band, so both the mask and the evidence the instancer reads are cured there. Everything else
(threshold, instancer, FOV handling, matching) is byte-for-byte eval_v5.py's --no-fov path.

    PYTHONPATH=src:scripts python scripts/diag_ceiling_union.py --pred-dir <dump> --data data/real/roi303_eval \\
        --split val --thr 0.98 --params src/instance/params_a_derived.json --region cross20 --out ceil_cross20.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from instance.instancer_a import instance_a  # noqa: E402
from instance.metrics import aggregate_benchmark, centerline_f1  # noqa: E402
from instance.oracle import oracle_ori_channels  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.gt_stats import count_crossings  # noqa: E402

UP, KAPPA_MAX = 1.5, 0.25
REGIONS = {"none": None, "cross20": 20.0, "cross40": 40.0, "all": np.inf}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--data", default="data/real/roi303_eval")
    ap.add_argument("--split", default="val")
    ap.add_argument("--thr", type=float, required=True)
    ap.add_argument("--params", default="src/instance/params_a_derived.json")
    ap.add_argument("--region", choices=list(REGIONS), required=True)
    ap.add_argument("--cov", type=float, default=0.80)
    ap.add_argument("--delete-far", type=float, default=0.0,
                    help="eval px; if > 0, model foreground farther than this from the GT band is set to 0 "
                         "(removes every off-annotation prediction, keeps near-filament thickening and merges)")
    ap.add_argument("--tol", type=float, default=5.0)
    ap.add_argument("--band-hw", type=float, default=1.0,
                    help="half-width (native px) of the painted GT band; 1.0 = the training target. The montage of "
                         "2026-09-16 showed the 1.0 band running BESIDE the model's ridge (annotation offset within the "
                         "metric's tolerance) and spawning spurs; 2.0 swallows the offset")
    ap.add_argument("--paint-channels", type=int, default=0,
                    help="oracle variant only (--delete-far <= 0.01): paint the K orientation channels of the GT band "
                         "and hand them to the instancer as its channel stack (a crossing pixel lit in two channels), "
                         "so the +0.027 the step-1 oracle shows over the one-channel same-path oracle can be attributed "
                         "to the channels rather than to the step-1 mask construction")
    ap.add_argument("--dump-dir", default=None,
                    help="write the MODIFIED probability maps here as <frame>.npz (key 'prob', (1,H,W)), the "
                         "layout eval_v5.py --dump-pred uses, so diag_fp_anatomy.py can name what remains")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.split == "test":
        raise SystemExit("refusing TEST: ceilings are read on VAL")
    params = json.load(open(args.params))
    radius_native = REGIONS[args.region]
    if args.paint_channels and not (args.region == "all" and 0 < args.delete_far <= 0.01):
        raise SystemExit("--paint-channels is defined for the same-path oracle only (--region all --delete-far 0.01)")

    rows, names = [], []
    n_painted_px = 0
    for path in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        fr = read_frame_h5(path)
        if str(fr["attrs"].get("split")) != args.split or not fr["polylines"]:
            continue
        name = os.path.basename(path)
        npz = os.path.join(args.pred_dir, name.replace(".h5", ".npz"))
        if not os.path.exists(npz):
            raise SystemExit(f"missing {npz}")
        ch = np.load(npz)["prob"].astype(np.float32)
        prob = ch.max(axis=0)
        H, W = prob.shape
        shape_native = (int(fr["attrs"]["height"]), int(fr["attrs"]["width"]))
        gt = [np.asarray(q, float) * UP for q in fr["polylines"]]

        full_band = oracle_ori_channels(fr["polylines"], shape_native, K=6, half_width=args.band_hw, up=UP).max(axis=0) > 0.5
        full_band = np.pad(full_band, ((0, max(0, H - full_band.shape[0])), (0, max(0, W - full_band.shape[1]))))[:H, :W]
        if radius_native is not None:
            band = full_band.copy()
            if np.isfinite(radius_native):
                cr = count_crossings([np.asarray(q, float) for q in fr["polylines"]], tol=2.0, ds=1.0,
                                     endpoint_margin=8.0)
                seed = np.zeros((H, W), bool)
                for c in cr:
                    x, y = int(round(c["x"] * UP)), int(round(c["y"] * UP))
                    if 0 <= y < H and 0 <= x < W:
                        seed[y, x] = True
                region = (distance_transform_edt(~seed) <= radius_native * UP) if seed.any() else np.zeros((H, W), bool)
                band &= region
            n_painted_px += int(band.sum())
            prob = np.where(band, 1.0, prob).astype(np.float32)
            ch = prob[None]
        if args.delete_far > 0:
            far = distance_transform_edt(~full_band) > args.delete_far
            prob = np.where(far, 0.0, prob).astype(np.float32)
            ch = prob[None]
        if args.paint_channels:
            chans = oracle_ori_channels(fr["polylines"], shape_native, K=args.paint_channels,
                                        half_width=args.band_hw, up=UP).astype(np.float32)
            chans = np.pad(chans, ((0, 0), (0, max(0, H - chans.shape[1])), (0, max(0, W - chans.shape[2]))))[:, :H, :W]
            ch = chans
            prob = chans.max(axis=0)
        if args.dump_dir:
            os.makedirs(args.dump_dir, exist_ok=True)
            np.savez_compressed(os.path.join(args.dump_dir, name.replace(".h5", ".npz")), prob=prob[None].astype(np.float32))
        mask = prob > args.thr
        _, masks = instance_a(mask, KAPPA_MAX, params=params, channels=ch, prob=prob)
        r = centerline_f1(masks, gt, tol=args.tol, length_coverage=args.cov, precision_coverage=args.cov)
        rows.append(r); names.append(name)
        if args.limit and len(rows) >= args.limit:
            break

    a = aggregate_benchmark(rows)
    big = [r["f1"] for r in rows if r["tp"] + r["fn"] >= 10]
    a["macro_ge10"] = float(np.mean(big)) if big else float("nan")
    a["painted_px"] = n_painted_px
    a["_frames"] = [{"name": n, "tp": r["tp"], "fp": r["fp"], "fn": r["fn"], "f1": r["f1"]} for n, r in zip(names, rows)]
    out = {"pred_dir": args.pred_dir, "split": args.split, "thr": args.thr, "region": args.region, "delete_far": args.delete_far, "band_hw": args.band_hw, "paint_channels": args.paint_channels,
           "params": args.params, "cov": args.cov, "n_frames": len(rows), "result": a}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"region={args.region} delete_far={args.delete_far} band_hw={args.band_hw} thr={args.thr} n={len(rows)}  micro={a['micro_f1']:.4f} "
          f"P={a['micro_precision']:.3f} R={a['micro_recall']:.3f}  macro>=10={a['macro_ge10']:.4f}  "
          f"tp={a['total_tp']} fp={a['total_fp']} fn={a['total_fn']}  painted_px={n_painted_px}")


if __name__ == "__main__":
    main()
