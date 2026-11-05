"""Can the mask REPRESENT the gap between two close parallels at all?

Before training anything to separate bundles, check whether the target representation can express
the separation. Two filaments 2-6 px apart, each rasterised at half_width=1.0, leave a gap of about
one pixel; if the oracle mask itself cannot be instanced back into two objects at that width, then
no network trained to predict that mask can either, and the fix is the REPRESENTATION rather than
the loss.

Training-free: rasterise the ground truth at a range of half-widths, run instancer A on it, and
read bundle_recovery. The half_width that maximises it is an upper bound on what a perfectly
trained foreground of that width could deliver.

    PYTHONPATH=src:scripts python scripts/parallel_screen.py --data data/real/mt34_eval --split val
"""
from __future__ import annotations

import argparse, glob, json, os, sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from instance.instancer_a import instance_a                      # noqa: E402
from instance.metrics import (bundle_recovery, centerline_f1,    # noqa: E402
                              fragmentation, junction_identity)
from instance.oracle import oracle_instance_masks, oracle_mask   # noqa: E402
from mt_bench.cvat_import import read_frame_h5                   # noqa: E402
from mt_bench.gt_stats import count_crossings                    # noqa: E402
from run_oracle_eval import parallel_pairs                       # noqa: E402

UP, KAPPA_MAX = 1.5, 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/mt34_eval")
    ap.add_argument("--split", default="val")
    ap.add_argument("--params", default="src/instance/params_a_model_v5H.json")
    ap.add_argument("--half-widths", default="0.4,0.5,0.75,1.0,1.5")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()
    if args.split == "test":
        raise SystemExit("refusing to touch TEST")

    params = json.load(open(args.params))
    params.pop("kappa_max", None); params.pop("prob_thr", None)
    hws = [float(x) for x in args.half_widths.split(",")]

    frames = []
    for p in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        fr = read_frame_h5(p)
        a = fr["attrs"]
        if str(a.get("split")) != args.split or len(fr["polylines"]) < 2:
            continue
        frames.append((os.path.basename(p), fr, a))
    print(f"{len(frames)} frames", flush=True)

    out = {}
    print(f"\n{'half_width':>10} {'fg%':>6} {'bundle':>7} {'pairs':>6} {'junction':>9} "
          f"{'frag':>6} {'F1':>7}")
    for hw in hws:
        brs, jis_u, jis_p, frags, f1s, fgs, npair = [], 0, 0, [], [], [], 0
        for name, fr, a in frames:
            shape = (int(a["height"]), int(a["width"]))
            himg = (int(round(shape[0] * UP)), int(round(shape[1] * UP)))
            gt = [np.asarray(q, float) * UP for q in fr["polylines"]]
            par = parallel_pairs(gt)
            cr = count_crossings(gt, tol=2.0 * UP, endpoint_margin=8.0 * UP)
            mask = oracle_mask(fr["polylines"], shape, half_width=hw, up=UP)
            fgs.append(float(mask.mean()))
            polys, _ = instance_a(mask, KAPPA_MAX, params, channels=None, prob=None)
            masks = oracle_instance_masks(polys, himg, half_width=1.0, up=1.0)
            b = bundle_recovery(masks, gt, par, tol=1.5 * UP)
            if b == b:
                brs.append(float(b)); npair += len(par)
            ji = junction_identity(masks, gt, cr, tol=5.0, arm_offset=15.0 * UP)
            jis_u += ji["n_crossings"]; jis_p += ji["n_preserved"]
            f = fragmentation(masks, gt)
            if f == f:
                frags.append(float(f))
            r = centerline_f1(masks, gt, tol=5.0, length_coverage=0.95, precision_coverage=0.95)
            f1s.append(r["f1"])
        row = {"fg": float(np.mean(fgs)), "bundle": float(np.mean(brs)) if brs else float("nan"),
               "pairs": npair, "junction": jis_p / jis_u if jis_u else float("nan"),
               "frag": float(np.mean(frags)) if frags else float("nan"),
               "f1": float(np.mean(f1s))}
        out[f"{hw}"] = row
        print(f"{hw:10.2f} {100*row['fg']:6.2f} {row['bundle']:7.3f} {row['pairs']:6d} "
              f"{row['junction']:9.3f} {row['frag']:6.3f} {row['f1']:7.4f}")
    if args.out_json:
        json.dump(out, open(args.out_json, "w"), indent=1, default=float)
        print("\nwrote", args.out_json)


if __name__ == "__main__":
    main()
