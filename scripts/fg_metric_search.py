#!/usr/bin/env python3
"""Which foreground property actually predicts downstream instance F1? RUNS ON TULEN.

The puzzle this answers: nnU-Net and v4b score almost identically on the tolerant
centerline-coverage F1 the generator and segmenter are tuned on (Alice tol2 0.947 vs 0.950),
yet nnU-Net's foreground instances markedly better (pooled 0.418 vs 0.379, junction identity
0.720 vs 0.511). So the metric we optimise does not rank foregrounds by the thing we care
about. This computes a battery of candidate foreground properties per frame, then correlates
each against the per-frame instance F1 that instancer A achieves on that same foreground.

A property that correlates is a candidate training objective; the coverage F1 we currently use
is included as the control.

    PYTHONPATH=src ~/dinov3_env/bin/python scripts/fg_metric_search.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import convolve, label
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))
from instance.geometry import resample  # noqa: E402
from instance.instancer_a import instance_a  # noqa: E402
from instance.metrics import centerline_f1  # noqa: E402
from instance.oracle import oracle_instance_masks  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402

UP = 1.5
KAPPA_MAX = 0.25
_K8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])


def foreground_properties(pm: np.ndarray, gt_polys) -> dict | None:
    """Candidate descriptors of foreground QUALITY, all instancing-motivated."""
    ys, xs = np.where(pm)
    if len(ys) == 0:
        return None
    tree = cKDTree(np.stack([ys, xs], axis=1))
    gt_pts = np.concatenate([resample(np.asarray(p, float) * UP, 1.0) for p in gt_polys])
    gt_rc = np.stack([gt_pts[:, 1], gt_pts[:, 0]], axis=1)
    gt_tree = cKDTree(gt_rc)

    skel = skeletonize(pm)
    sy, sx = np.where(skel)
    n_skel = max(len(sy), 1)
    nb = convolve(skel.astype(np.uint8), _K8, mode="constant") * skel

    # Gaps: how often the foreground DROPS OUT along a real microtubule. Every dropout is a
    # break the instancer has to bridge or lose.
    gaps, cov_frac = [], []
    for p in gt_polys:
        q = resample(np.asarray(p, float) * UP, 1.0)
        cov = tree.query(np.stack([q[:, 1], q[:, 0]], axis=1), k=1)[0] <= 2.0
        if len(cov) < 5:
            continue
        gaps.append(int((np.diff(cov.astype(int)) == -1).sum()))
        cov_frac.append(float(cov.mean()))

    _, n_cc = label(pm, structure=np.ones((3, 3)))
    return {
        # the control: what we currently tune on
        "rec2": float((tree.query(gt_rc, k=1)[0] <= 2.0).mean()),
        "prec2": float((gt_tree.query(np.stack([sy, sx], axis=1), k=1)[0] <= 2.0).mean()),
        # topology of the skeleton the instancer actually consumes
        "junc_per_kpx": float(1000 * (nb >= 3).sum() / n_skel),
        "endp_per_kpx": float(1000 * (nb == 1).sum() / n_skel),
        # continuity along real microtubules
        "gaps_per_mt": float(np.mean(gaps)) if gaps else float("nan"),
        "cov_per_mt": float(np.mean(cov_frac)) if cov_frac else float("nan"),
        # gross over/under-segmentation of the mask itself
        "cc_per_gt": n_cc / max(len(gt_polys), 1),
        "fg": float(pm.mean()),
        "skel_px_per_gt_px": n_skel / max(len(gt_pts), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/mt34_eval")
    ap.add_argument("--models", nargs="+", default=[
        "v4b=/home/prusek/mt_enc_exp/mt34_pred",
        "nnunet=/home/prusek/mt_enc_exp/mt34_pred_nnunet_s15",
    ])
    ap.add_argument("--thr", type=float, default=0.35)
    ap.add_argument("--out", default="data/enc_sensitivity_testset/fg_metrics.json")
    args = ap.parse_args()

    models = dict(m.split("=", 1) for m in args.models)
    rows = []
    for path in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        fr = read_frame_h5(path)
        a = fr["attrs"]
        if not fr["polylines"]:
            continue
        shape = (int(a["height"]), int(a["width"]))
        hi = (int(round(shape[0] * UP)), int(round(shape[1] * UP)))
        gt = [np.asarray(p, float) * UP for p in fr["polylines"]]

        for name, d in models.items():
            f = os.path.join(d, os.path.basename(path).replace(".h5", ".npz"))
            if not os.path.exists(f):
                continue
            pm = np.load(f)["prob"].astype(np.float32).max(axis=0) > args.thr
            props = foreground_properties(pm, fr["polylines"])
            if props is None:
                continue
            pls, _ = instance_a(pm, KAPPA_MAX)
            masks = oracle_instance_masks(pls, hi, half_width=1.0, up=1.0)
            r = centerline_f1(masks, gt, tol=5.0,
                              length_coverage=0.95, precision_coverage=0.95)
            rows.append({"model": name, "frame": os.path.basename(path),
                         "task": str(a.get("source_task")), "split": str(a.get("split")),
                         "n_gt": len(gt), "inst_f1": r["f1"], **props})
        print(f"  {os.path.basename(path)}", flush=True)

    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=1)

    keys = [k for k in rows[0] if k not in
            ("model", "frame", "task", "split", "inst_f1")]
    y = np.array([r["inst_f1"] for r in rows])
    print(f"\n=== per-model means (n={len(rows)} frame-model pairs) ===")
    for name in models:
        sub = [r for r in rows if r["model"] == name]
        print(f"{name:8s} inst_f1={np.mean([r['inst_f1'] for r in sub]):.3f}  " +
              "  ".join(f"{k}={np.nanmean([r[k] for r in sub]):.3f}" for k in keys))

    print("\n=== Spearman correlation of each foreground property with instance F1 ===")
    from scipy.stats import spearmanr
    out = []
    for k in keys:
        x = np.array([r[k] for r in rows], dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 8:
            continue
        rho, p = spearmanr(x[ok], y[ok])
        out.append((abs(rho), rho, p, k))
    for arho, rho, p, k in sorted(out, reverse=True):
        flag = "  <-- the metric we currently tune on" if k in ("rec2", "prec2") else ""
        print(f"  {k:20s} rho={rho:+.3f}  p={p:.2g}{flag}")


if __name__ == "__main__":
    main()
