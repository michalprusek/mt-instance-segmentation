#!/usr/bin/env python3
"""What ARE the false positives that force the operating point to 0.98-0.99?  RUNS ON TULEN (CPU).

Declared in docs/TODO.md 0.0-CAMPAIGN (2026-09-04 22:50) as check (ii) before any label-validity
arm is built. From cached maps, at each threshold: run instancer A exactly as eval_v5.py does, match
with the strict metric, and classify every UNMATCHED prediction by where it sits:

  A  on annotated GT   -- >= 50 % of its skeleton within tol px of some GT pixel: a fragment or a
                          chimera of an annotated filament (a coverage / assembly failure, not
                          over-firing);
  B  off-GT, visible   -- a ridge the image supports (median |c|/noise >= 1 along its skeleton): an
                          UNANNOTATED filament or debris -- annotation incompleteness; no training
                          change can remove it without also removing true positives;
  C  off-GT, invisible -- nothing in the image (median |c|/noise < 1): texture the model was taught
                          to fire on -- the only class a label-validity fix can move.

`|c|` here is isotropic (smoothed image at the point minus the median on a 4-6 px ring), so it needs
no local direction and can be read on a predicted skeleton as well as on GT.

    PYTHONPATH=.:src:scripts:synth:third_party python scripts/diag_fp_nature.py \
        --pred-dir /disk2/prusek/mt_work/preds/SPARSE35_roi_native --thrs 0.9,0.98 \
        --exclude roi_img_8,roi_img_149 --out runs/diag_roi/fp_nature_SPARSE35.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "src", "scripts", "synth", "third_party"):
    sys.path.insert(0, os.path.join(ROOT, d))

from instance.instancer_a import instance_a  # noqa: E402
from instance.metrics import _polyline_pixels, _skel_points, centerline_f1  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402

UP, KAPPA_MAX, K_ORI, TOL = 1.5, 0.25, 6, 5.0
RING = [(r * np.cos(a), r * np.sin(a)) for r in (4.0, 5.0, 6.0) for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)]


def stretch(img):
    lo, hi = np.percentile(img, [1, 99])
    return np.clip((np.asarray(img, np.float64) - lo) / (hi - lo + 1e-6), 0, 1)


def frame_noise(img):
    hp = img - gaussian_filter(img, 2.0)
    return 1.4826 * np.median(np.abs(hp - np.median(hp))) + 1e-9


def point_snr(img_s, noise, pts_xy):
    """Isotropic |contrast| / noise at native-image points (x, y)."""
    c = map_coordinates(img_s, [pts_xy[:, 1], pts_xy[:, 0]], order=1, mode="nearest")
    ring = np.stack([map_coordinates(img_s, [pts_xy[:, 1] + dy, pts_xy[:, 0] + dx], order=1, mode="nearest")
                     for dx, dy in RING], 1)
    return np.abs(c - np.median(ring, axis=1)) / noise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/roi303_eval")
    ap.add_argument("--split", default="val")
    ap.add_argument("--exclude", default="")
    ap.add_argument("--frames", default="", help="comma list of stems (chunking)")
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--channel", type=int, default=0)
    ap.add_argument("--thrs", default="0.9,0.98")
    ap.add_argument("--params", default="src/instance/params_a_model_v5H.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.split == "test":
        raise SystemExit("refusing to touch TEST")
    excl = set(args.exclude.split(",")) if args.exclude else set()
    want = set(args.frames.split(",")) if args.frames else None
    params = json.load(open(args.params))
    params.pop("kappa_max", None)
    thrs = [float(t) for t in args.thrs.split(",")]

    out = {"thrs": thrs, "frames": []}
    for p in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        stem = os.path.basename(p)[:-3]
        if stem in excl or (want and stem not in want):
            continue
        fr = read_frame_h5(p)
        if str(fr["attrs"].get("split")) != args.split:
            continue
        npz = os.path.join(args.pred_dir, stem + ".npz")
        if not os.path.exists(npz):
            continue
        allch = np.load(npz)["prob"].astype(np.float32)
        prob = allch[min(args.channel, allch.shape[0] - 1)]
        chans = allch[:K_ORI]
        H, W = prob.shape
        gt = [np.asarray(q, float) * UP for q in fr["polylines"] if len(q) >= 2]
        gt_px = [_polyline_pixels(g, (H, W)) for g in gt]
        gt_all = np.concatenate([g for g in gt_px if len(g)], 0) if any(len(g) for g in gt_px) else None
        gt_tree = cKDTree(gt_all) if gt_all is not None else None
        img01 = stretch(fr["image"]); img_s = gaussian_filter(img01, 1.0); noise = frame_noise(img01)
        rec = {"frame": stem, "n_gt": len(gt), "noise": float(noise), "per_thr": {}}
        for thr in thrs:
            mask = prob > thr
            pp = dict(params); pp["prob_thr"] = thr
            _, masks = instance_a(mask, KAPPA_MAX, params=pp, channels=chans, prob=prob)
            r = centerline_f1(masks, gt, tol=TOL, length_coverage=0.95, precision_coverage=0.95)
            matched = {q["pred"] for q in r["pairs"] if q["tp"]}
            fps = []
            for i, m in enumerate(masks):
                if i in matched:
                    continue
                sk = _skel_points(m)                     # (N, 2) as (y, x), like _polyline_pixels
                if len(sk) == 0:
                    continue
                sk = np.asarray(sk, float)
                on_gt = float((gt_tree.query(sk)[0] <= TOL).mean()) if gt_tree is not None else 0.0
                snr = float(np.median(point_snr(img_s, noise, sk[:, ::-1] / UP)))   # (x, y), native px
                cls = "A_on_gt" if on_gt >= 0.5 else ("B_visible_ridge" if snr >= 1.0 else "C_invisible")
                fps.append({"len_px": int(len(sk)), "on_gt": on_gt, "snr": snr, "cls": cls})
            agg = {c: {"n": sum(f["cls"] == c for f in fps), "len": int(sum(f["len_px"] for f in fps if f["cls"] == c))}
                   for c in ("A_on_gt", "B_visible_ridge", "C_invisible")}
            rec["per_thr"][str(thr)] = {"tp": r["tp"], "fp": r["fp"], "fn": r["fn"], "f1": r["f1"],
                                        "n_pred": len(masks), "fp_classes": agg, "fps": fps}
            print(f"{stem} thr={thr}: tp={r['tp']} fp={r['fp']} fn={r['fn']} "
                  + " ".join(f"{c[0]}={agg[c]['n']}" for c in agg), flush=True)
        out["frames"].append(rec)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"))
    # summary
    for thr in thrs:
        tot = {c: [0, 0] for c in ("A_on_gt", "B_visible_ridge", "C_invisible")}
        tp = fp = fn = 0
        for rec in out["frames"]:
            q = rec["per_thr"][str(thr)]; tp += q["tp"]; fp += q["fp"]; fn += q["fn"]
            for c in tot:
                tot[c][0] += q["fp_classes"][c]["n"]; tot[c][1] += q["fp_classes"][c]["len"]
        print(f"thr {thr}: TP {tp} FP {fp} FN {fn} | FP classes (count, skeleton px): "
              + ", ".join(f"{c} {v[0]} ({v[1]})" for c, v in tot.items()), flush=True)


if __name__ == "__main__":
    main()
