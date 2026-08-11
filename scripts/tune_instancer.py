#!/usr/bin/env python3
"""Optuna tuning of an instancer on ORACLE masks, MT-34 VAL split only.

Every method gets the SAME budget on the SAME objective. That is not politeness -- this
project has already been burned once by comparing its own tuned pipeline against an
under-tuned baseline (TODO W10, the SAM 3 confidence sweep), and PySOAX's shipped parameters
were tuned months ago on different data at 1x scale while we now run it at 1.5x.

``kappa_max`` is NOT a search dimension. It is derived from the ground truth -- 0.25 rad/px,
just above the largest curvature in 957 human-annotated microtubules -- and the whole claim is
that it encodes physics rather than a fit. Letting Optuna raise it to ~0.34 would buy a little
F1 while allowing bends no real microtubule exhibits, which is exactly the argument the method
is supposed to make. Use ``--tune-kappa`` to reproduce that as an explicit ablation.

    python scripts/tune_instancer.py --method a --n-trials 120 --n-jobs 5
    python scripts/tune_instancer.py --method pysoax --n-trials 120 --n-jobs 5
    python scripts/tune_instancer.py --method b --n-trials 120 --n-jobs 5
"""
from __future__ import annotations

import sys as _sys
try:                                                    # base python on tulen lacks _sqlite3
    import pysqlite3                                    # noqa: E402
    _sys.modules["sqlite3"] = pysqlite3
except Exception:
    pass

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import optuna
from scipy.ndimage import zoom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "third_party"))

from instance.instancer_a import instance_a  # noqa: E402
from instance.instancer_b import instance_b  # noqa: E402
from instance.metrics import aggregate_benchmark, centerline_f1  # noqa: E402
from instance.oracle import (oracle_instance_masks, oracle_ori_channels)  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.fov import fov_mask  # noqa: E402

UP = 1.5
KAPPA_MAX = 0.25


def prepare(data_dir: str, split: str, masks: str = "oracle",
            pred_dir: str | None = None, prob_thr: float = 0.35) -> list[dict]:
    """Cache the per-frame inputs once; a trial then costs only the instancer + the metric."""
    frames = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.h5"))):
        fr = read_frame_h5(path)
        a = fr["attrs"]
        if str(a.get("split")) != split:
            continue
        shape = (int(a["height"]), int(a["width"]))
        hi = (int(round(shape[0] * UP)), int(round(shape[1] * UP)))
        if masks == "oracle":
            chans = oracle_ori_channels(fr["polylines"], shape, K=6, half_width=1.0, up=UP)
        else:
            npz = np.load(os.path.join(pred_dir,
                                       os.path.basename(path).replace(".h5", ".npz")))
            chans = npz["prob"].astype(np.float32)
            if chans.shape[1:] != hi:
                raise SystemExit(f"{path}: prediction {chans.shape[1:]} != {hi}")
            fv = zoom(fov_mask(fr["image"]).astype(np.uint8), UP, order=0).astype(bool)
            fv = np.pad(fv, ((0, max(0, hi[0] - fv.shape[0])),
                             (0, max(0, hi[1] - fv.shape[1]))),
                        constant_values=True)[:hi[0], :hi[1]]
            chans = chans * fv[None, :, :]
        frames.append({
            "name": os.path.basename(path),
            "chans": chans,
            "mask": chans.max(axis=0) > (0.5 if masks == "oracle" else prob_thr),
            "gt": [np.asarray(p, float) * UP for p in fr["polylines"]],
            "hi_shape": (int(round(shape[0] * UP)), int(round(shape[1] * UP))),
        })
    return frames


#: Wall-clock ceiling per trial. Instancer B's cost explodes combinatorially with
#: beam x K_out x max_step, and a configuration that cannot process 17 frames inside this is
#: not a usable instancer regardless of its F1 -- so it is pruned rather than waited on.
TRIAL_TIMEOUT_S = 400.0


def evaluate(frames, method: str, kappa_max: float, params: dict) -> float:
    rows = []
    t0 = time.time()
    for f in frames:
        if time.time() - t0 > TRIAL_TIMEOUT_S:
            raise optuna.TrialPruned(
                f"exceeded {TRIAL_TIMEOUT_S:.0f}s after {len(rows)}/{len(frames)} frames")
        if method == "a":
            m = (f["mask"] if "prob_thr" not in params
                 else f["chans"].max(axis=0) > params["prob_thr"])
            pls, _ = instance_a(m, kappa_max, params,
                                channels=f["chans"], prob=f["chans"].max(axis=0))
        elif method == "b":
            pls, _ = instance_b(f["chans"], kappa_max, params)
        elif method == "pysoax":
            import pysoax
            inst = pysoax.extract_soax_instances(
                (f["mask"].astype(np.uint8) * 255), params)
            pls = [np.stack([np.asarray(d["centerline"], float)[:, 1],
                             np.asarray(d["centerline"], float)[:, 0]], axis=1)
                   for d in inst if len(d["centerline"]) >= 2]
        else:
            raise SystemExit(f"unknown method {method}")
        masks = oracle_instance_masks(pls, f["hi_shape"], half_width=1.0, up=1.0)
        r = centerline_f1(masks, f["gt"], tol=5.0,
                          length_coverage=0.95, precision_coverage=0.95)
        rows.append({**r, "n_gt": len(f["gt"])})
    return float(aggregate_benchmark(rows)["mean_f1"])


def space_a(t: optuna.Trial, masks: str = "oracle") -> dict:
    extra = {}
    if masks == "model":
        # On thick, noisy predicted foreground the BINARISATION threshold controls skeleton
        # quality more than any matching weight -- the project's own SAM 3 record has a
        # confidence sweep moving F1 0.39 -> 0.71. Leaving it fixed would spend the budget
        # on the wrong knob.
        extra["prob_thr"] = t.suggest_float("prob_thr", 0.2, 0.75)
    return {
        **extra,
        "merge_radius": t.suggest_float("merge_radius", 2.0, 10.0),
        "bridge_max_len": t.suggest_float("bridge_max_len", 6.0, 50.0),
        "window": t.suggest_float("window", 6.0, 36.0),
        "w_theta": t.suggest_float("w_theta", 0.2, 5.0),
        "w_kappa": t.suggest_float("w_kappa", 0.0, 40.0),
        "w_gap": t.suggest_float("w_gap", 0.0, 0.2),
        "c_open": t.suggest_float("c_open", 0.3, 4.0),
        "min_length": t.suggest_float("min_length", 6.0, 50.0),
        "smooth_size": t.suggest_int("smooth_size", 3, 15, step=2),
        # Levers added in protocol 17k. gap_floor doubles as the displacement-cost switch:
        # a very large value forces the old direct-turn behaviour, so the ablation is a
        # parameter setting rather than a second code path.
        "gap_floor": t.suggest_float("gap_floor", 2.0, 12.0),
        "w_ori": t.suggest_float("w_ori", 0.0, 6.0),
        "link_max_gap": t.suggest_float("link_max_gap", 0.0, 40.0),
        "c_open_link": t.suggest_float("c_open_link", 0.3, 4.0),
        "bridge_thr": t.suggest_float("bridge_thr", 0.0, 0.5),
        "min_arc_len": 3, "ds": 2.0, "half_width": 1.0,
    }


def space_b(t: optuna.Trial, masks: str = "oracle") -> dict:
    return {
        "K_out": t.suggest_categorical("K_out", [6, 12, 18]),
        "prob_thr": t.suggest_float("prob_thr", 0.15, 0.6),
        "max_step": t.suggest_float("max_step", 1.5, 3.2),
        "dir_tol_deg": t.suggest_float("dir_tol_deg", 40.0, 85.0),
        "lam": t.suggest_float("lam", 0.5, 20.0),
        "beam": t.suggest_int("beam", 1, 5),
        "consume_deg": t.suggest_float("consume_deg", 10.0, 45.0),
        "consume_radius": t.suggest_float("consume_radius", 1.0, 4.0),
        "min_length": t.suggest_float("min_length", 6.0, 50.0),
        "ds": 2.0, "half_width": 1.0, "max_instances": 600,
    }


def space_pysoax(t: optuna.Trial, masks: str = "oracle") -> dict:
    return {
        "min_snake_length": t.suggest_int("min_snake_length", 5, 45),
        "grouping_distance": t.suggest_float("grouping_distance", 1.0, 30.0),
        "direction_threshold": t.suggest_float("direction_threshold", 0.2, 0.98),
        "point_spacing": t.suggest_float("point_spacing", 0.5, 3.0),
        "ridge_threshold": t.suggest_float("ridge_threshold", 0.005, 0.12),
        "gaussian_std": t.suggest_float("gaussian_std", 0.4, 2.0),
        "alpha": t.suggest_float("alpha", 0.05, 0.6),
        "beta": t.suggest_float("beta", 0.01, 0.3),
        "stretch_factor": 0.998, "gamma": 4.597, "external_factor": 1.918,
        "max_iterations": 5000, "change_threshold": 0.001, "check_period": 100,
        "orient_weight": 0.0, "embed_weight": 0.0,
    }


SPACES = {"a": space_a, "b": space_b, "pysoax": space_pysoax}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["a", "b", "pysoax"])
    ap.add_argument("--data", default="data/real/mt34_eval")
    ap.add_argument("--split", default="val")
    ap.add_argument("--masks", default="oracle", choices=["oracle", "model"])
    ap.add_argument("--pred-dir", default=None)
    ap.add_argument("--prob-thr", type=float, default=0.35)
    ap.add_argument("--n-trials", type=int, default=120)
    ap.add_argument("--n-jobs", type=int, default=5)
    ap.add_argument("--tune-kappa", action="store_true",
                    help="ABLATION ONLY: let Optuna fit the curvature bound too")
    ap.add_argument("--out-dir", default="src/instance")
    ap.add_argument("--log", default="data/enc_sensitivity_testset/instancer_tuning")
    args = ap.parse_args()

    os.makedirs(args.log, exist_ok=True)
    frames = prepare(args.data, args.split, args.masks, args.pred_dir, args.prob_thr)
    n_empty = sum(1 for f in frames if not f["gt"])
    print(f"prepared {len(frames)} {args.split} frames "
          f"({n_empty} with zero GT -> micro only)", flush=True)

    space = SPACES[args.method]

    def objective(trial: optuna.Trial) -> float:
        params = space(trial, args.masks)
        kappa = (trial.suggest_float("kappa_max", 0.15, 0.45)
                 if args.tune_kappa else KAPPA_MAX)
        return evaluate(frames, args.method, kappa, params)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=1234))
    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs,
                   show_progress_bar=False)

    # kappa_max is never persisted: instance_a/instance_b take it as an explicit argument,
    # so a stray dict entry would silently do nothing while looking authoritative.
    FIXED = {
        "a": {"min_arc_len": 3, "ds": 2.0, "half_width": 1.0},
        "b": {"ds": 2.0, "half_width": 1.0, "max_instances": 600},
        "pysoax": {"stretch_factor": 0.998, "gamma": 4.597, "external_factor": 1.918,
                   "max_iterations": 5000, "change_threshold": 0.001,
                   "check_period": 100, "orient_weight": 0.0, "embed_weight": 0.0},
    }
    best = dict(study.best_params)
    best.pop("kappa_max", None)
    best.update(FIXED[args.method])

    tag = (f"params_{args.method}"
           + ("_model" if args.masks == "model" else "")
           + ("_kappatuned" if args.tune_kappa else ""))
    out = os.path.join(args.out_dir, f"{tag}.json")
    with open(out, "w") as fh:
        json.dump(best, fh, indent=2)
    with open(os.path.join(args.log, f"study_{tag}_oracle_{args.split}.json"), "w") as fh:
        json.dump({"best_value": study.best_value, "best_params": best,
                   "n_trials": len(study.trials),
                   "kappa_max": ("tuned" if args.tune_kappa else KAPPA_MAX)}, fh, indent=2)
    print(f"\nBEST oracle-{args.split} mean F1 ({args.method}) = {study.best_value:.4f}")
    print(json.dumps(best, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
