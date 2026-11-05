#!/usr/bin/env python3
"""How much of the number is the METRIC's strictness?  RUNS ON TULEN (CPU, ~10 min).

Scores cached maps at one threshold through the same instancer, then the same predictions under
several (length_coverage, precision_coverage, tol) settings of the strict centerline-F1, so the
share of the gap that the 95 % / 95 % / 5 px rule itself costs is a number, not an opinion.

    PYTHONPATH=.:src:scripts python scripts/metric_strictness.py \
        --pred-dir /disk2/prusek/mt_work/preds/SPARSE35_roi_native --thr 0.99 --out runs/diag_roi/metric_strictness.json
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "src", "scripts"):
    sys.path.insert(0, os.path.join(ROOT, d))
from instance.instancer_a import instance_a  # noqa: E402
from instance.metrics import _skel_points, aggregate_benchmark, centerline_f1  # noqa: E402
from scipy.ndimage import binary_dilation, gaussian_filter, label as cc_label, zoom  # noqa: E402


def dark_region(image, shape, dilate_px):
    """The frame's unilluminated region in the eval frame: sigma-8 smoothed native image below
    black level + 12 % of the (black, p99) range, connected components >= 0.5 % of the frame,
    dilated by ``dilate_px``. Declared 2026-09-05 (docs/TODO.md 0.0-CAMPAIGN)."""
    im = gaussian_filter(image.astype(np.float32), 8.0)
    b = float(im.min()); rng = float(np.percentile(im, 99) - b)
    if rng <= 0:
        return np.zeros(shape, bool)
    dark = im < b + 0.12 * rng
    lab, k = cc_label(dark)
    if k == 0:
        return np.zeros(shape, bool)
    # components >= 0.5 % of the frame: a 5-px band along one edge of a 1024-px frame qualifies (the
    # SPARSE35 gallery showed edge bands thinner than the 2 % first written down)
    sizes = np.bincount(lab.ravel()); keep = np.zeros(k + 1, bool); keep[1:] = sizes[1:] >= 0.005 * dark.size
    dark = keep[lab]
    if not dark.any():
        return np.zeros(shape, bool)
    dark = zoom(dark.astype(np.float32), (shape[0] / dark.shape[0], shape[1] / dark.shape[1]), order=0) > 0.5
    r = int(round(dilate_px)); yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    return binary_dilation(dark, structure=(yy * yy + xx * xx) <= r * r)
from mt_bench.cvat_import import read_frame_h5  # noqa: E402

UP, KAPPA_MAX, K_ORI = 1.5, 0.25, 6
# (length_coverage, precision_coverage, tol[, end_tol]); the 4th element is the declared symmetric end
# tolerance (2026-09-05, E = 25 eval px fixed from SPARSE35's overshoot distribution before any arm was read)
SETTINGS = [(0.95, 0.95, 5.0), (0.90, 0.90, 5.0), (0.80, 0.80, 5.0), (0.70, 0.70, 5.0),
            (0.95, 0.95, 8.0), (0.80, 0.80, 8.0), (0.95, None, 5.0), (0.80, None, 5.0),
            (0.95, 0.95, 5.0, 25.0), (0.95, 0.95, 5.0, 15.0), (0.80, 0.80, 5.0, 25.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/roi303_eval"); ap.add_argument("--split", default="val")
    ap.add_argument("--pred-dir", required=True); ap.add_argument("--thr", type=float, default=0.99)
    ap.add_argument("--channel", type=int, default=0)
    ap.add_argument("--frames", default="", help="comma list of stems (chunking)")
    ap.add_argument("--fov", type=float, default=0.0,
                    help="declared illumination-edge rule (2026-09-05): drop predictions with >= 50 %% of their skeleton inside "
                         "the frame's black region dilated by this many eval px (0 = off). Scored as a second set of rows.")
    ap.add_argument("--params", default="src/instance/params_a_model_v5H.json"); ap.add_argument("--out", required=True)
    ap.add_argument("--allow-test", action="store_true",
                    help="permit --split test. The guard exists so TEST is never touched by "
                         "accident; pass this ONLY as part of a read pre-declared in "
                         "docs/TODO.md 0.0-CAMPAIGN (final read 34, 2026-09-08).")
    a = ap.parse_args()
    if a.split == "test" and not a.allow_test:
        raise SystemExit("refusing to touch TEST (pass --allow-test for a pre-declared read)")
    params = json.load(open(a.params)); params.pop("kappa_max", None); params["prob_thr"] = a.thr
    want = set(a.frames.split(",")) if a.frames else None
    rows = {(s, suf): [] for s in SETTINGS for suf in ("", "_fov")}
    fov_rows = []
    n = 0
    for p in sorted(glob.glob(os.path.join(a.data, "*.h5"))):
        fr = read_frame_h5(p)
        if str(fr["attrs"].get("split")) != a.split:
            continue
        npz = os.path.join(a.pred_dir, os.path.basename(p).replace(".h5", ".npz"))
        if not os.path.exists(npz):
            continue
        stem = os.path.basename(p)[:-3]
        if want and stem not in want:
            continue
        ch = np.load(npz)["prob"].astype(np.float32); prob = ch[min(a.channel, ch.shape[0] - 1)]
        gt = [np.asarray(q, float) * UP for q in fr["polylines"] if len(q) >= 2]
        _, masks = instance_a(prob > a.thr, KAPPA_MAX, params=params, channels=ch[:K_ORI], prob=prob)
        variants = {"": masks}
        if a.fov > 0:
            dark = dark_region(fr["image"], prob.shape, a.fov)
            keep = []
            for m in masks:
                sk = _skel_points(m)
                inside = float(dark[sk[:, 0].astype(int), sk[:, 1].astype(int)].mean()) if len(sk) else 0.0
                keep.append(inside < 0.5)
            variants["_fov"] = [m for m, k in zip(masks, keep) if k]
            gt_in = [float(dark[np.clip(np.round(g[:, 1]).astype(int), 0, prob.shape[0] - 1),
                                np.clip(np.round(g[:, 0]).astype(int), 0, prob.shape[1] - 1)].mean()) for g in gt]
            fov_rows.append({"name": stem, "dark_frac": float(dark.mean()), "n_pred": len(masks),
                             "n_dropped": int(len(masks) - sum(keep)), "gt_inside_ge50": int(sum(x >= 0.5 for x in gt_in))})
        for suffix, mk in variants.items():
            for s in SETTINGS:
                lc, pc, tol = s[:3]
                r = centerline_f1(mk, gt, tol=tol, length_coverage=lc, precision_coverage=pc, end_tol=(s[3] if len(s) > 3 else 0.0))
                r.pop("pairs", None); r["n_gt"] = len(gt); r["name"] = stem; rows[(s, suffix)].append(r)
        n += 1
        print(n, os.path.basename(p), {f"{s[0]}/{s[1]}/{s[2]:g}": round(rows[(s, "")][-1]["f1"], 3) for s in SETTINGS[:4]}, flush=True)
    out, per_frame = {}, {}
    for (s, suf), rr in rows.items():
        if not rr:
            continue
        agg = aggregate_benchmark(rr); key = f"cov{s[0]}_prec{s[1]}_tol{s[2]:g}" + (f"_end{s[3]:g}" if len(s) > 3 and s[3] else "") + suf
        out[key] = {"mean_f1": agg["mean_f1"], "micro_f1": agg["micro_f1"], "tp": agg["total_tp"], "fp": agg["total_fp"], "fn": agg["total_fn"]}
        per_frame[key] = rr
        print(f"{key:34s} mean F1 {agg['mean_f1']:.4f}  micro {agg['micro_f1']:.4f}  TP {agg['total_tp']} FP {agg['total_fp']} FN {agg['total_fn']}")
    if fov_rows:
        print(f"FOV rule: {sum(r['n_dropped'] for r in fov_rows)} predictions dropped on {sum(r['n_dropped'] > 0 for r in fov_rows)} frames; "
              f"GT polylines >= 50 % inside the dark region: {sum(r['gt_inside_ge50'] for r in fov_rows)}")
    json.dump({"pred_dir": a.pred_dir, "thr": a.thr, "n_frames": n, "fov_px": a.fov, "results": out, "per_frame": per_frame,
               "fov_rows": fov_rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
