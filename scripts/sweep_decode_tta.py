#!/usr/bin/env python3
"""D3 items 5-6, scoring half: instance-score the TTA / multi-scale probability maps.

Same discipline as ``sweep_decode.py``: only the BINARY MASK changes. ``channels`` and
``prob`` handed to ``instance_a`` remain the raw identity-transform maps from the original
prediction directory, so ``w_ori`` and ``bridge_thr`` see what they saw at the baseline.
The alternative map (dihedral TTA, polarity averaging, multi-scale) is used for binarisation
only, and each row carries the over-firing gate measured on that same mask.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

import h5py
import numpy as np
from scipy.ndimage import zoom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import run_oracle_eval as roe  # noqa: E402
import sweep_decode as sd  # noqa: E402
from instance.instancer_a import instance_a  # noqa: E402
from instance.metrics import aggregate_benchmark  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.fg_quality import (foreground_quality, mean_properties,  # noqa: E402
                                 passes_overfiring_gate, quality_score)
from mt_bench.fov import fov_mask  # noqa: E402
from mt_bench.gt_stats import count_crossings  # noqa: E402

UP = roe.UP
BASE = sd.BASE_THR
KEYS = ["pmax_id", "pmax_tta8", "pmax_tta8_perm", "pmax_inv", "pmax_tta8pol",
        "pmax_polavg", "pmax_s125", "pmax_s200", "pmax_ms"]
THRS = [0.30, 0.35, 0.40, BASE, 0.50, 0.55]

_CFG: dict = {}


def _init(cfg):
    global _CFG
    _CFG = cfg


def run_frame(path: str) -> dict:
    cfg = _CFG
    fr = read_frame_h5(path)
    attrs = fr["attrs"]
    name = os.path.basename(path)
    shape = (int(attrs["height"]), int(attrs["width"]))
    hi_shape = (int(round(shape[0] * UP)), int(round(shape[1] * UP)))
    gt = [np.asarray(p, float) * UP for p in fr["polylines"]]
    crossings = count_crossings(gt, tol=2.0 * UP, endpoint_margin=8.0 * UP)
    par = roe.parallel_pairs(gt)
    task = str(attrs.get("source_task", "?"))

    chans = np.load(os.path.join(cfg["pred_dir"],
                                 name.replace(".h5", ".npz")))["prob"].astype(np.float32)
    alt = np.load(os.path.join(cfg["alt_dir"], name.replace(".h5", ".npz")))
    alt = {k: alt[k].astype(np.float32) for k in alt.files}
    fv = zoom(fov_mask(fr["image"]).astype(np.uint8), UP, order=0).astype(bool)
    fv = np.pad(fv, ((0, max(0, hi_shape[0] - fv.shape[0])),
                     (0, max(0, hi_shape[1] - fv.shape[1]))),
                constant_values=True)[:hi_shape[0], :hi_shape[1]]
    chans = chans * fv[None, :, :]
    alt = {k: v * fv for k, v in alt.items()}
    prob = chans.max(axis=0)

    out = {"name": name, "task": task, "rows": {}}
    for spec in cfg["grid"]:
        m = alt[spec["key"]]
        if spec["kind"] == "flat":
            mask = m > spec["thr"]
        else:
            mask = sd._hysteresis(m > spec["lo"], m > spec["hi"])
        pls, _ = instance_a(mask, roe.KAPPA_MAX, cfg["params_a"], channels=chans, prob=prob)
        row = roe.score(pls, gt, hi_shape, crossings, par)
        props = foreground_quality(mask, fr["polylines"], up=UP, tol=2.0)
        out["rows"][spec["name"]] = {"name": name, "task": task, "n_gt": len(gt),
                                     **row, "props": props}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--alt-dir", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--params-a", default="src/instance/params_a_model_synthtuned.json")
    ap.add_argument("--jobs", type=int, default=17)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    pa = json.load(open(args.params_a))
    pa.pop("kappa_max", None)
    pa.pop("prob_thr", None)

    grid = []
    for k in KEYS:
        for t in THRS:
            grid.append({"name": f"{k}/flat/{t:.4f}", "kind": "flat", "key": k, "thr": t})
    for k in ("pmax_tta8", "pmax_tta8pol", "pmax_ms"):
        for lo in (0.15, 0.20, 0.25):
            grid.append({"name": f"{k}/hyst/{BASE:.3f}->{lo:.2f}", "kind": "hyst",
                         "key": k, "hi": BASE, "lo": lo})

    paths = []
    for p in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        with h5py.File(p) as f:
            if args.split == "all" or str(f.attrs.get("split")) == args.split:
                paths.append(p)
    print(f"{len(paths)} frames x {len(grid)} settings", flush=True)

    cfg = {"pred_dir": args.pred_dir, "alt_dir": args.alt_dir, "params_a": pa, "grid": grid}
    t0 = time.time()
    with Pool(min(args.jobs, len(paths)), initializer=_init, initargs=(cfg,)) as pool:
        frames = pool.map(run_frame, paths)
    print(f"done in {time.time() - t0:.0f}s", flush=True)

    report = {"settings": {}}
    for spec in grid:
        rows = [fr["rows"][spec["name"]] for fr in frames]
        agg = aggregate_benchmark(rows)
        props = mean_properties([r["props"] for r in rows])
        entry = {"spec": spec, "pooled": agg, "props": props,
                 "quality": quality_score(props),
                 "gate": bool(passes_overfiring_gate(props)),
                 "frag": float(np.nanmean([r["fragmentation"] for r in rows])),
                 "n_pred": float(np.mean([r["n_pred"] for r in rows])),
                 "by_task": {t: aggregate_benchmark([r for r in rows if r["task"] == t])
                             for t in sorted({r["task"] for r in rows})},
                 "frames": [{k: v for k, v in r.items() if k != "props"} for r in rows]}
        report["settings"][spec["name"]] = entry
        print(f"{spec['name']:34s} F1={agg['mean_f1']:.3f} micro={agg['micro_f1']:.3f} "
              f"TP={agg['total_tp']} FP={agg['total_fp']} FN={agg['total_fn']} "
              f"fg%={100 * props.get('fg', float('nan')):.2f} "
              f"rec2={props.get('rec2', float('nan')):.3f} "
              f"cc/gt={props.get('cc_per_gt', float('nan')):.2f} "
              f"gate={'PASS' if entry['gate'] else 'FAIL'}", flush=True)

    with open(args.out_json, "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(f"wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
