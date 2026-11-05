#!/usr/bin/env python3
"""Are the annotator gaps that an arm bridges VISIBLE in the image? RUNS ON TULEN (CPU).

Reads the per-frame output of diag_fp_anatomy.py, re-instances only the frames that contain a
`gap_bridge` false positive, and for each such prediction measures the local |contrast| / noise
(diag_fp_nature.point_snr, native frame, sigma-1 smoothed profile core minus a 4-6 px ring) of the
prediction's samples that lie BETWEEN the two GT footprints (further than 5 px from any GT), next
to the same statistic on the parts of the prediction that lie ON the GT pieces. A bridged stretch
with median SNR >= 1 is a visible continuation the annotator skipped; below ~0.7 the model
connected two pieces through evidence a human could not see (still possibly right, not verifiable).

    PYTHONPATH=.:src:scripts python scripts/diag_gap_contrast.py --anat "runs/diag_roi/anat/NOISE_roi_ep40_native_*.json" \
        --pred-dir preds/NOISE_roi_ep40_native --thr 0.99 --out runs/diag_roi/anat/gapc_NOISE.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from diag_fp_anatomy import UP, KAPPA_MAX, K_ORI, TOL, resample  # noqa: E402
from diag_fp_nature import frame_noise, point_snr, stretch  # noqa: E402
from instance.instancer_a import instance_a  # noqa: E402
from instance.metrics import _polyline_pixels  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/roi303_eval")
    ap.add_argument("--anat", required=True, help="glob of diag_fp_anatomy outputs")
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--thr", type=float, default=0.99)
    ap.add_argument("--params", default="src/instance/params_a_model_v5H.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    params = json.load(open(args.params)); params.pop("kappa_max", None); params["prob_thr"] = args.thr
    todo = {}
    for f in glob.glob(args.anat):
        for rec in json.load(open(f))["frames"]:
            gb = [x for x in rec["fps"] if x["cls"] == "gap_bridge"]
            if gb:
                todo[rec["frame"]] = gb
    print(len(todo), "frames with gap bridges", flush=True)
    rows = []
    for stem in sorted(todo):
        fr = read_frame_h5(os.path.join(args.data, stem + ".h5"))
        if str(fr["attrs"].get("split")) == "test":
            raise SystemExit("refusing to touch TEST")
        allch = np.load(os.path.join(args.pred_dir, stem + ".npz"))["prob"].astype(np.float32)
        prob = allch[0]; chans = allch[:K_ORI]; H, W = prob.shape
        gt = [np.asarray(q, float) * UP for q in fr["polylines"] if len(q) >= 2]
        gt_px = [_polyline_pixels(g, (H, W)) for g in gt]
        gt_all = np.concatenate([g for g in gt_px if len(g)], 0)
        gt_tree = cKDTree(gt_all)
        polys, _ = instance_a(prob > args.thr, KAPPA_MAX, params=params, channels=chans, prob=prob)
        img01 = stretch(fr["image"]); img_s = gaussian_filter(img01, 1.0); noise = frame_noise(img01)
        for x in todo[stem]:
            i = x["pred"]
            if i >= len(polys):
                continue
            c = resample(np.asarray(polys[i], float))          # (x, y) eval px
            d = gt_tree.query(c[:, ::-1])[0]
            on = d <= TOL
            # the between-footprint samples: off-GT samples that have on-GT samples on both sides
            idx_on = np.where(on)[0]
            if len(idx_on) < 2:
                continue
            between = ~on; between[: idx_on[0]] = False; between[idx_on[-1] + 1:] = False
            if between.sum() < 5:
                continue
            s_between = point_snr(img_s, noise, c[between] / UP)
            s_on = point_snr(img_s, noise, c[on] / UP)
            rows.append({"frame": stem, "pred": i, "n_between": int(between.sum()), "n_on": int(on.sum()),
                         "snr_between_med": float(np.median(s_between)), "snr_between_q25": float(np.percentile(s_between, 25)),
                         "snr_on_med": float(np.median(s_on)), "frac_between_ge1": float(np.mean(s_between >= 1.0)),
                         "p_between_med": float(np.median(prob[np.clip(np.round(c[between][:, 1]).astype(int), 0, H - 1),
                                                                np.clip(np.round(c[between][:, 0]).astype(int), 0, W - 1)]))})
            print(f"{stem} pred {i}: between {int(between.sum())} px SNR med {rows[-1]['snr_between_med']:.2f} "
                  f"(on-GT {rows[-1]['snr_on_med']:.2f}) frac>=1 {rows[-1]['frac_between_ge1']:.2f} p {rows[-1]['p_between_med']:.3f}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(rows, open(args.out, "w"), indent=1)
    if rows:
        sb = np.array([r["snr_between_med"] for r in rows]); so = np.array([r["snr_on_med"] for r in rows])
        w = np.array([r["n_between"] for r in rows], float)
        print(f"SUMMARY {len(rows)} bridges: between-SNR median {np.median(sb):.2f} (q25 {np.percentile(sb, 25):.2f}, q75 {np.percentile(sb, 75):.2f}); "
              f"on-GT median {np.median(so):.2f}; bridges with between-SNR >= 1: {np.mean(sb >= 1):.2f}, >= 0.7: {np.mean(sb >= 0.7):.2f}; "
              f"gap length median {np.median(w):.0f} px (eval)")


if __name__ == "__main__":
    main()
