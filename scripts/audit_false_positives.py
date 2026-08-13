#!/usr/bin/env python3
"""Are the false positives wrong, or is the ground truth incomplete?

On the crossing-dense half of MT-34 the system produces 326 false positives against 193 true
positives, and precision is the larger loss there. Before optimising against that number it is
worth knowing what it counts, because "false positive" is three different things stacked
together:

  fragment      -- a piece of a real, ANNOTATED microtubule that failed the 95 % coverage rule.
                   That is the fragmentation cost, already measured elsewhere, not an error of
                   detection.
  unannotated   -- a filament the model found and the annotator did not draw. MT-34's ground
                   truth is human-corrected model output and is demonstrably incomplete on
                   sparse frames, so these are expected to exist. Every one of them lowers the
                   measurable ceiling: no model can score them as correct.
  hallucination -- a detection with no image evidence under it. The only category that is
                   actually the model's fault.

They are separated here by two measurements: distance to the nearest ground-truth centerline,
and the local image contrast under the detection compared with what a true positive carries and
with what an equally-long curve dropped on empty background carries.

    PYTHONPATH=src python scripts/audit_false_positives.py --pred-dir <npz dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from scipy.spatial import cKDTree

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from instance.geometry import arclength, resample  # noqa: E402
from instance.instancer_a import instance_a  # noqa: E402
from instance.metrics import centerline_f1  # noqa: E402
from instance.oracle import oracle_instance_masks  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.fov import fov_mask  # noqa: E402

UP, KAPPA_MAX, BG_SIGMA = 1.5, 0.25, 12.0


def contrast_along(resid: np.ndarray, poly: np.ndarray) -> float:
    q = resample(np.asarray(poly, float), ds=1.0)
    H, W = resid.shape
    r = np.clip(np.rint(q[:, 1]).astype(int), 0, H - 1)
    c = np.clip(np.rint(q[:, 0]).astype(int), 0, W - 1)
    return float(np.median(resid[r, c]))


def random_curves(resid: np.ndarray, like: np.ndarray, rng, n: int = 8):
    """The null: curves of the same shape dropped at random positions and orientations.

    A detection is only evidence of a filament if it carries more contrast than the same curve
    would carry by accident.
    """
    H, W = resid.shape
    p = np.asarray(like, float)
    p = p - p.mean(axis=0)
    out = []
    for _ in range(n):
        a = rng.uniform(0, 2 * np.pi)
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        q = p @ rot.T + [rng.uniform(0.15 * W, 0.85 * W), rng.uniform(0.15 * H, 0.85 * H)]
        out.append(contrast_along(resid, q))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/mt34_eval")
    ap.add_argument("--split", default="test")
    ap.add_argument("--pred-dir", default="/home/prusek/mt_enc_exp/mt34_pred")
    ap.add_argument("--params", default="src/instance/params_a_model_synthtuned.json")
    ap.add_argument("--near-px", type=float, default=8.0,
                    help="an unmatched detection within this of a GT centerline is a FRAGMENT")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    params = json.load(open(args.params))
    params.pop("kappa_max", None)
    thr = params.get("prob_thr", 0.35)
    rng = np.random.default_rng(0)

    tp_c, frag_c, unann_c, bg_c = [], [], [], []
    n_tp = n_frag = n_unann = 0
    per_task = {}

    for path in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        fr = read_frame_h5(path)
        a = fr["attrs"]
        if str(a.get("split")) != args.split or not fr["polylines"]:
            continue
        npz = os.path.join(args.pred_dir, os.path.basename(path).replace(".h5", ".npz"))
        if not os.path.exists(npz):
            continue
        ch = np.load(npz)["prob"].astype(np.float32)
        H, W = ch.shape[1:]
        fv = zoom(fov_mask(fr["image"]).astype(np.uint8), UP, order=0).astype(bool)
        fv = np.pad(fv, ((0, max(0, H - fv.shape[0])), (0, max(0, W - fv.shape[1]))),
                    constant_values=True)[:H, :W]
        ch = ch * fv[None]
        prob = ch.max(axis=0)
        pls, _ = instance_a(prob > thr, KAPPA_MAX, params, channels=ch, prob=prob)
        if not pls:
            continue

        gt = [np.asarray(p, float) * UP for p in fr["polylines"]]
        masks = oracle_instance_masks(pls, (H, W), half_width=1.0, up=1.0)
        res = centerline_f1(masks, gt, tol=5.0, length_coverage=0.95, precision_coverage=0.95)
        matched = {p["pred"] for p in res["pairs"] if p["tp"]}

        raw = zoom(np.asarray(fr["image"], float), UP, order=1)[:H, :W]
        resid = np.abs(raw - gaussian_filter(raw, BG_SIGMA))
        gt_tree = cKDTree(np.concatenate([resample(g, ds=1.0) for g in gt]))
        task = str(a.get("source_task"))
        per_task.setdefault(task, {"tp": 0, "frag": 0, "unann": 0})

        for i, pl in enumerate(pls):
            cval = contrast_along(resid, pl)
            if i in matched:
                tp_c.append(cval)
                n_tp += 1
                per_task[task]["tp"] += 1
                bg_c.extend(random_curves(resid, pl, rng))
                continue
            q = resample(np.asarray(pl, float), ds=2.0)
            d = float(np.median(gt_tree.query(q, k=1)[0]))
            if d <= args.near_px:
                frag_c.append(cval)
                n_frag += 1
                per_task[task]["frag"] += 1
            else:
                unann_c.append(cval)
                n_unann += 1
                per_task[task]["unann"] += 1

    def med(x):
        return float(np.median(x)) if len(x) else float("nan")

    total_fp = n_frag + n_unann
    print(f"{args.split} split | {n_tp} true positives, {total_fp} false positives\n")
    print(f"{'category':32s} {'count':>7s} {'share of FP':>12s} {'median contrast':>16s}")
    print(f"{'true positive (matched)':32s} {n_tp:7d} {'-':>12s} {med(tp_c):16.2f}")
    print(f"{'FP: fragment of an annotated MT':32s} {n_frag:7d} "
          f"{100*n_frag/max(total_fp,1):11.1f}% {med(frag_c):16.2f}")
    print(f"{'FP: away from any annotation':32s} {n_unann:7d} "
          f"{100*n_unann/max(total_fp,1):11.1f}% {med(unann_c):16.2f}")
    print(f"{'null: same curve, random place':32s} {len(bg_c):7d} {'-':>12s} {med(bg_c):16.2f}")

    if len(unann_c) and len(bg_c) and len(tp_c):
        r_bg = med(unann_c) / max(med(bg_c), 1e-9)
        r_tp = med(unann_c) / max(med(tp_c), 1e-9)
        print(f"\n  detections away from any annotation carry {r_bg:.2f}x the contrast of the "
              f"null\n  and {r_tp:.2f}x that of a confirmed true positive.")
        if r_bg > 1.5 and r_tp > 0.6:
            print("\n  => They sit on real image structure. A large share are microtubules the")
            print("     annotator did not draw, so the MEASURABLE F1 CEILING IS BELOW 1.0 and")
            print("     optimising precision against this benchmark optimises against its gaps.")
        elif r_bg < 1.2:
            print("\n  => They carry no more evidence than empty background: these are genuine")
            print("     hallucinations and precision work is well aimed.")
        else:
            print("\n  => Intermediate: neither clearly real nor clearly invented. Inspect them.")

    print(f"\n{'task':10s} {'TP':>6s} {'fragment':>10s} {'unannotated':>12s}")
    for t, c in sorted(per_task.items()):
        print(f"{t:10s} {c['tp']:6d} {c['frag']:10d} {c['unann']:12d}")

    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump({"n_tp": n_tp, "n_fragment": n_frag, "n_unannotated": n_unann,
                       "median_contrast": {"tp": med(tp_c), "fragment": med(frag_c),
                                           "unannotated": med(unann_c), "null": med(bg_c)},
                       "per_task": per_task}, fh, indent=2)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
