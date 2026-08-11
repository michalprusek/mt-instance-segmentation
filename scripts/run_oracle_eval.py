#!/usr/bin/env python3
"""Benchmark instancers on MT-34: PySOAX (tuned baseline) vs the curvature-bounded A.

Runs on ORACLE masks by default -- rasterised ground truth -- so the instancer is measured
without the segmenter's errors in the loop. ``--masks model`` scores the same instancers on
predicted foreground instead.

Everything happens in the 1.5x eval frame: the mask is built at ``up=1.5`` and the GT
polylines are multiplied by 1.5, matching ``amodal_eval2.py`` on tulen.

    python scripts/run_oracle_eval.py --split val
    python scripts/run_oracle_eval.py --split val --methods a
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import zoom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "third_party"))

from instance.instancer_a import KAPPA_DS, instance_a  # noqa: E402
from instance.instancer_b import instance_b  # noqa: E402
from instance.metrics import (aggregate_benchmark, bootstrap_ci,  # noqa: E402
                              bundle_recovery, centerline_f1, fragmentation,
                              junction_identity, max_curvature, paired_bootstrap)
from instance.oracle import (oracle_instance_masks, oracle_mask,  # noqa: E402
                             oracle_ori_channels)
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.fov import fov_mask  # noqa: E402
from mt_bench.gt_stats import count_crossings  # noqa: E402

UP = 1.5
KAPPA_MAX = 0.25   # just above the largest curvature in the 957 MT-34 GT microtubules

# The Optuna-tuned parameters actually used to pre-annotate CVAT tasks 585/586
# (from tulen:~/run_v7_pysoax_predictions.py) -- the honest tuned baseline, not defaults.
PYSOAX_PARAMS = {
    "min_snake_length": 17, "gaussian_std": 0.8784, "grouping_distance": 17.1466,
    "direction_threshold": 0.87, "orient_weight": 0.0, "embed_weight": 0.0,
    "ridge_threshold": 0.0354, "stretch_factor": 0.998, "alpha": 0.25, "beta": 0.06,
    "gamma": 4.597, "external_factor": 1.918, "max_iterations": 5000,
    "change_threshold": 0.001, "check_period": 100, "point_spacing": 1.0,
}


def run_pysoax(mask: np.ndarray, params: dict | None = None) -> list[np.ndarray]:
    """Return PySOAX centerlines as (x=col, y=row) polylines."""
    import pysoax
    inst = pysoax.extract_soax_instances((mask.astype(np.uint8) * 255),
                                         params or PYSOAX_PARAMS)
    out = []
    for d in inst:
        cl = np.asarray(d["centerline"], dtype=float)   # (row, col)
        if len(cl) >= 2:
            out.append(np.stack([cl[:, 1], cl[:, 0]], axis=1))
    return out


def parallel_pairs(polylines, gap_lo=2.0 * UP, gap_hi=6.0 * UP,
                   min_len=20.0 * UP, ds=2.0) -> list[tuple[int, int]]:
    """Indices of GT pairs running side by side at a close gap, in the eval frame."""
    from scipy.spatial import cKDTree

    from instance.geometry import resample
    res = [resample(np.asarray(p, float), ds=ds) for p in polylines]
    trees = [cKDTree(p) if len(p) > 1 else None for p in res]
    pairs = []
    for i in range(len(res)):
        if trees[i] is None:
            continue
        for j in range(i + 1, len(res)):
            if trees[j] is None:
                continue
            d, _ = trees[j].query(res[i], k=1)
            if float(np.sum((d >= gap_lo) & (d <= gap_hi)) * ds) >= min_len:
                pairs.append((i, j))
    return pairs


def frame_polarity(image: np.ndarray, polylines) -> str:
    """Are the microtubules darker or brighter than their local background?

    IRM contrast flips sign with height above the coverslip (the two-beam interference
    zero-crossing), and MT-34 contains both polarities while Alice alone does not. Splitting
    recall by polarity is the first real test of the training-time inversion augmentation.
    """
    from scipy.ndimage import gaussian_filter
    img = np.asarray(image, dtype=float)
    res = img / np.maximum(gaussian_filter(img, 25), 1e-6) - 1.0
    h, w = res.shape
    vals = []
    for p in polylines:
        q = np.asarray(p, float)
        cc = np.clip(np.rint(q[:, 0]).astype(int), 0, w - 1)
        rr = np.clip(np.rint(q[:, 1]).astype(int), 0, h - 1)
        vals.append(res[rr, cc])
    if not vals:
        return "unknown"
    return "dark" if float(np.median(np.concatenate(vals))) < 0 else "bright"


def score(polylines, gt, shape, crossings, par_pairs, half_width=1.0) -> dict:
    masks = oracle_instance_masks(polylines, shape, half_width=half_width, up=1.0)
    f1 = centerline_f1(masks, gt, tol=5.0, length_coverage=0.95, precision_coverage=0.95)
    return {
        **f1,
        "n_pred": len(polylines),
        "junction": junction_identity(masks, gt, crossings, tol=5.0, arm_offset=15.0 * UP),
        "fragmentation": fragmentation(masks, gt, tol=5.0),
        "bundle": bundle_recovery(masks, gt, par_pairs, tol=1.5 * UP),
        "max_kappa": max_curvature(polylines, ds=KAPPA_DS),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/mt34_eval")
    ap.add_argument("--split", default="val", choices=["val", "test", "all"])
    ap.add_argument("--masks", default="oracle", choices=["oracle", "model"])
    ap.add_argument("--pred-dir", default=None, help="npz channel stacks, for --masks model")
    ap.add_argument("--methods", default="pysoax,a,b")
    ap.add_argument("--prob-thr", type=float, default=0.35)
    ap.add_argument("--no-fov", action="store_true",
                    help="do NOT mask the field of view (model masks only)")
    ap.add_argument("--kappa-max", type=float, default=KAPPA_MAX)
    ap.add_argument("--params-a", default=None,
                    help="JSON of instancer-A params (e.g. src/instance/params_a_model.json)")
    ap.add_argument("--params-b", default=None)
    ap.add_argument("--params-pysoax", default=None,
                    help="tuned PySOAX params; without it the SHIPPED params are used")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--n-boot", type=int, default=5000,
                    help="bootstrap replicates for the CIs on mean F1")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    pa = json.load(open(args.params_a)) if args.params_a else None
    pb = json.load(open(args.params_b)) if args.params_b else None
    pp = json.load(open(args.params_pysoax)) if args.params_pysoax else None
    if pa:
        pa.pop("kappa_max", None)
    if pb:
        pb.pop("kappa_max", None)
    per_method: dict[str, dict[str, list]] = {m: {} for m in methods}
    timings: dict[str, float] = {m: 0.0 for m in methods}

    for path in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        fr = read_frame_h5(path)
        attrs = fr["attrs"]
        if args.split != "all" and str(attrs.get("split")) != args.split:
            continue
        task = str(attrs.get("source_task", "?"))
        name = os.path.basename(path)

        shape = (int(attrs["height"]), int(attrs["width"]))
        hi_shape = (int(round(shape[0] * UP)), int(round(shape[1] * UP)))
        gt = [np.asarray(p, float) * UP for p in fr["polylines"]]
        crossings = count_crossings(gt, tol=2.0 * UP, endpoint_margin=8.0 * UP)
        par = parallel_pairs(gt)
        polarity = frame_polarity(fr["image"], fr["polylines"])

        if args.masks == "oracle":
            chans = oracle_ori_channels(fr["polylines"], shape, K=6,
                                        half_width=1.0, up=UP)
            mask = chans.max(axis=0) > 0.5
        else:
            npz = np.load(os.path.join(args.pred_dir, name.replace(".h5", ".npz")))
            chans = npz["prob"].astype(np.float32)
            # The 1.5x zoom and the backbone's /14 padding can each shift a frame by a few
            # pixels; clipping a mismatched prediction against the GT frame would shift every
            # centerline silently rather than fail.
            if chans.shape[1:] != hi_shape:
                raise SystemExit(
                    f"{name}: prediction {chans.shape[1:]} != eval frame {hi_shape}")
            mask = chans.max(axis=0) > args.prob_thr
            if not args.no_fov:
                # The task-586 frames have an octagonal field stop the segmenter fires on
                # (see mt_bench.fov). Predictions outside the imaged field are not
                # microtubules; the annotators never drew there either.
                fv = zoom(fov_mask(fr["image"]).astype(np.uint8), UP, order=0).astype(bool)
                fv = np.pad(fv, ((0, max(0, hi_shape[0] - fv.shape[0])),
                                 (0, max(0, hi_shape[1] - fv.shape[1]))),
                            constant_values=True)[:hi_shape[0], :hi_shape[1]]
                mask &= fv
                chans = chans * fv[None, :, :]

        thr_a = (pa or {}).get("prob_thr", args.prob_thr)
        mask_a = (mask if args.masks == "oracle" or thr_a == args.prob_thr
                  else (chans.max(axis=0) > thr_a))

        for m in methods:
            t0 = time.time()
            if m == "pysoax":
                pls = run_pysoax(mask, pp)
            elif m == "a":
                pls, _ = instance_a(mask_a, args.kappa_max, pa,
                                    channels=chans, prob=chans.max(axis=0))
            elif m == "b":
                # The params FILE wins over the CLI default: a tuned prob_thr that a
                # positional default silently overrides is the same class of bug as a tuned
                # kappa_max sitting unread in a JSON.
                pls, _ = instance_b(chans, args.kappa_max,
                                    {"prob_thr": args.prob_thr, **(pb or {})})
            else:
                raise SystemExit(f"unknown method {m}")
            timings[m] += time.time() - t0
            per_method[m].setdefault(task, []).append(
                {"name": name, "n_gt": len(gt), "n_cross": len(crossings),
                 "n_par": len(par), "polarity": polarity,
                 **score(pls, gt, hi_shape, crossings, par)})

    def _angle_rates(rows) -> dict:
        acc = {"shallow": [0, 0], "oblique": [0, 0], "steep": [0, 0]}
        for r in rows:
            ja, jn = r["junction"]["by_angle"], r["junction"]["by_angle_n"]
            for k in acc:
                if jn[k]:
                    acc[k][0] += ja[k] * jn[k]
                    acc[k][1] += jn[k]
        return {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in acc.items()}

    def _n_by_angle(rows) -> dict:
        out = {"shallow": 0, "oblique": 0, "steep": 0}
        for r in rows:
            for k in out:
                out[k] += r["junction"]["by_angle_n"][k]
        return out

    report = {}
    frames_by_method: dict = {}
    print(f"\n=== MT-34 {args.split.upper()} | {args.masks} masks | "
          f"kappa_max={args.kappa_max} ===")
    for m in methods:
        print(f"\n--- {m} ---   ({timings[m]:.0f}s total)")
        allrows = []
        for task in sorted(per_method[m]):
            rows = per_method[m][task]
            for r in rows:
                r["task"] = task
            allrows += rows
            agg = aggregate_benchmark(rows)
            jr = [r["junction"]["rate"] for r in rows
                  if r["junction"]["n_crossings"] > 0]
            fg = [r["fragmentation"] for r in rows if r["fragmentation"] == r["fragmentation"]]
            bd = [r["bundle"] for r in rows if r["bundle"] == r["bundle"]]
            label = "Alice" if task == "585" else "new-22"
            ang, angn = _angle_rates(rows), _n_by_angle(rows)
            print(f"  task {task} ({label}, n={len(rows)}, scored={agg['n_scored']}): "
                  f"F1={agg['mean_f1']:.3f} micro={agg['micro_f1']:.3f} "
                  f"TP={agg['total_tp']} FP={agg['total_fp']} FN={agg['total_fn']} | "
                  f"junction-id={np.mean(jr) if jr else float('nan'):.3f} | "
                  f"frag={np.mean(fg) if fg else float('nan'):.2f} | "
                  f"bundle={np.mean(bd) if bd else float('nan'):.3f} | "
                  f"maxk={max(r['max_kappa'] for r in rows):.3f} | "
                  f"inst/frame={np.mean([r['n_pred'] for r in rows]):.0f}")
            print(f"      junction-id by crossing angle: "
                  f"shallow<30deg {ang['shallow']:.3f} (n={angn['shallow']}) | "
                  f"oblique30-60 {ang['oblique']:.3f} (n={angn['oblique']}) | "
                  f"steep>60deg {ang['steep']:.3f} (n={angn['steep']})")
            report[f"{m}/{task}"] = {"agg": agg,
                                     "junction": float(np.mean(jr)) if jr else None,
                                     "junction_by_angle": ang, "n_by_angle": angn,
                                     "frag": float(np.mean(fg)) if fg else None,
                                     "bundle": float(np.mean(bd)) if bd else None}
        pooled = aggregate_benchmark(allrows)
        print(f"  POOLED: F1={pooled['mean_f1']:.3f} micro={pooled['micro_f1']:.3f} "
              f"(scored {pooled['n_scored']}/{pooled['n_images']} frames; "
              f"zero-GT frames count in micro only)")
        report[f"{m}/pooled"] = pooled
        frames_by_method[m] = allrows

        if args.masks == "model":
            for pol in ("dark", "bright"):
                sub = [r for r in allrows if r.get("polarity") == pol]
                if sub:
                    a = aggregate_benchmark(sub)
                    print(f"  polarity {pol:6s} (n={len(sub)}): F1={a['mean_f1']:.3f} "
                          f"micro-recall={a['micro_recall']:.3f}")
                    report[f"{m}/polarity_{pol}"] = a

    # --- uncertainty -------------------------------------------------------------------
    # A pooled MT-34 split is 17 frames (6 Alice + 11 new-22). Differences of a few points
    # between methods are indistinguishable from frame-sampling noise at that size, so every
    # head-to-head claim gets a PAIRED, task-stratified interval on the difference rather
    # than two marginal intervals that would overlap for almost any pair.
    if len(methods) >= 2:
        ref = methods[0]
        print(f"\n=== paired bootstrap vs '{ref}' "
              f"({args.n_boot} replicates, stratified by task, 95% CI on the difference) ===")
        for m in methods[1:]:
            r = paired_bootstrap(frames_by_method[m], frames_by_method[ref],
                                 stat="mean_f1", n_boot=args.n_boot,
                                 stratum_key="task", frame_key="name", seed=args.seed)
            verdict = "significant" if r["significant"] else "NOT separable at n=%d" % r["n_frames"]
            print(f"  {m} - {ref}: {r['diff']:+.3f} "
                  f"[{r['lo']:+.3f}, {r['hi']:+.3f}]  p={r['p_two_sided']:.3f}  {verdict}")
            report[f"boot/{m}_vs_{ref}"] = r
    for m in methods:
        report[f"boot/{m}"] = bootstrap_ci(frames_by_method[m], stat="mean_f1",
                                           n_boot=args.n_boot, stratum_key="task",
                                           seed=args.seed)

    if args.out_json:
        # Per-frame rows travel WITH the aggregates: re-deriving an interval later must never
        # require re-running the eval (which would risk re-scoring TEST under changed code).
        report["_frames"] = frames_by_method
        with open(args.out_json, "w") as fh:
            json.dump(report, fh, indent=2, default=float)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
