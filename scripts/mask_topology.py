#!/usr/bin/env python3
"""Connectivity of a predicted foreground: the quantity cbDice is meant to move.

Protocol 20 located the bottleneck by elimination. Coverage (99.2 % of ground-truth length
within the metric's own 5 px), localisation (89.3 % of skeleton pixels within 5 px), width
(4.00 px, identical to the oracle) and branch topology (2621 branch points against 2689) are
all fine. What differs is that the mask is **shattered**: 2043 connected components and 4302
endpoints where the oracle has 294 and 968, for the same 494 microtubules. The breaks are
sub-pixel, which is exactly why every coverage-based metric called the foreground healthy.

So this is the number to look at first, before F1. A change in F1 without a change here would
mean the intervention worked for some other reason than the one it was chosen for, and a
change here without one in F1 is still informative -- it says the instancer, not the mask, is
now the limit.

    PYTHONPATH=src python scripts/mask_topology.py --pred-dir <npz dir> [--pred-dir <another>]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import convolve, label, zoom
from skimage.morphology import skeletonize

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.fov import fov_mask  # noqa: E402
from instance.oracle import oracle_mask  # noqa: E402

UP = 1.5
K8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])


def topology(mask: np.ndarray) -> dict:
    sk = skeletonize(mask)
    nb = convolve(sk.astype(np.uint8), K8, mode="constant") * sk
    _, ncc = label(mask, structure=np.ones((3, 3)))
    return {"components": int(ncc), "endpoints": int((nb == 1).sum()),
            "branch_points": int((nb >= 3).sum()), "skel_px": int(sk.sum()),
            "fg_frac": float(mask.mean())}


def measure(data_dir: str, split: str, pred_dir: str | None, thr: float,
            use_fov: bool = True) -> dict:
    tot = {"components": 0, "endpoints": 0, "branch_points": 0, "skel_px": 0}
    n_gt = n_frames = 0
    fg = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.h5"))):
        fr = read_frame_h5(path)
        a = fr["attrs"]
        if str(a.get("split")) != split or not fr["polylines"]:
            continue
        shape = (int(a["height"]), int(a["width"]))
        if pred_dir is None:
            m = oracle_mask(fr["polylines"], shape, half_width=1.0, up=UP)
        else:
            npz = os.path.join(pred_dir, os.path.basename(path).replace(".h5", ".npz"))
            if not os.path.exists(npz):
                continue
            ch = np.load(npz)["prob"].astype(np.float32)
            m = ch.max(axis=0) > thr
            if use_fov:
                H, W = m.shape
                fv = zoom(fov_mask(fr["image"]).astype(np.uint8), UP, order=0).astype(bool)
                fv = np.pad(fv, ((0, max(0, H - fv.shape[0])), (0, max(0, W - fv.shape[1]))),
                            constant_values=True)[:H, :W]
                m = m & fv
        t = topology(m)
        for k in tot:
            tot[k] += t[k]
        fg.append(t["fg_frac"])
        n_gt += len(fr["polylines"])
        n_frames += 1
    if not n_frames:
        return {}
    tot.update({"n_frames": n_frames, "n_gt": n_gt,
                "components_per_mt": tot["components"] / max(n_gt, 1),
                "endpoints_per_mt": tot["endpoints"] / max(n_gt, 1),
                "fg_pct": 100 * float(np.mean(fg))})
    return tot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/mt34_eval")
    ap.add_argument("--split", default="test")
    ap.add_argument("--pred-dir", action="append", default=[],
                    help="repeat for each model; label with name=path")
    ap.add_argument("--thr", type=float, default=None,
                    help="foreground threshold; default from params_a_model_synthtuned.json")
    ap.add_argument("--params", default="src/instance/params_a_model_synthtuned.json")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    thr = args.thr
    if thr is None:
        thr = json.load(open(args.params)).get("prob_thr", 0.35)

    rows = {"oracle": measure(args.data, args.split, None, thr)}
    for spec in args.pred_dir:
        name, _, p = spec.partition("=")
        if not p:
            name, p = os.path.basename(os.path.normpath(spec)), spec
        rows[name] = measure(args.data, args.split, p, thr)

    print(f"threshold {thr:.3f} | {args.split} split\n")
    print(f"{'model':16s} {'comp/MT':>8s} {'endpts/MT':>10s} {'branch':>8s} "
          f"{'skel px':>9s} {'fg %':>6s}")
    for name, r in rows.items():
        if not r:
            print(f"{name:16s}   (no predictions found)")
            continue
        print(f"{name:16s} {r['components_per_mt']:8.2f} {r['endpoints_per_mt']:10.2f} "
              f"{r['branch_points']:8d} {r['skel_px']:9d} {r['fg_pct']:6.2f}")
    o = rows.get("oracle")
    if o:
        print(f"\nThe oracle is the target: {o['components_per_mt']:.2f} components and "
              f"{o['endpoints_per_mt']:.2f} endpoints per microtubule.")
        print("Read connectivity FIRST. F1 moving without these moving means the intervention")
        print("worked for a reason other than the one it was chosen for.")
    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump(rows, fh, indent=2, default=float)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
