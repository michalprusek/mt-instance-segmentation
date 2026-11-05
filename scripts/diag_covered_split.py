#!/usr/bin/env python3
"""The largest recoverable FN pool: GT that is >= 95 % covered by the prediction UNION yet matched
by no single instance (`covered_split` in diag_fp_anatomy). Is it the MAP (the thresholded mask
breaks along the GT) or the INSTANCER (one continuous mask component, cut into pieces or merged
into a chimera)? RUNS ON TULEN (CPU, chunked).

For every unmatched, union-covered GT polyline, in the eval frame at the given threshold:
  n_mask_cc    connected components of (prob > thr) touched along the GT (within 5 px)
  mask_gap_px  total GT length not within 5 px of the mask
  n_inst       distinct predicted instances touched along the GT (>= 10 px each)
  chimera      the longest touching instance also touches >= 50 % of another GT
Classes: MAP_BREAK (n_mask_cc >= 2), INSTANCER_CUT (1 component, >= 2 instances, no chimera),
CHIMERA (1 component, an instance shared with another GT), OTHER.

    PYTHONPATH=.:src:scripts python scripts/diag_covered_split.py --pred-dir preds/NOISE_roi_ep40_native \
        --thr 0.99 --frames a,b --out runs/diag_roi/csplit/NOISE_0.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import label as cc_label
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from diag_fp_anatomy import UP, KAPPA_MAX, K_ORI, TOL, resample  # noqa: E402
from instance.instancer_a import instance_a  # noqa: E402
from instance.metrics import _polyline_pixels, centerline_f1  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/roi303_eval")
    ap.add_argument("--split", default="val")
    ap.add_argument("--exclude", default="roi_img_8,roi_img_149")
    ap.add_argument("--frames", default="")
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--thr", type=float, default=0.99)
    ap.add_argument("--params", default="src/instance/params_a_model_v5H.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.split == "test":
        raise SystemExit("refusing to touch TEST")
    excl = set(args.exclude.split(",")); want = set(args.frames.split(",")) if args.frames else None
    params = json.load(open(args.params)); params.pop("kappa_max", None); params["prob_thr"] = args.thr
    out = {"args": vars(args), "frames": []}
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
        prob = allch[0]; chans = allch[:K_ORI]; H, W = prob.shape
        mask = prob > args.thr
        lab, ncc = cc_label(mask, structure=np.ones((3, 3)))
        gt = [np.asarray(q, float) * UP for q in fr["polylines"] if len(q) >= 2]
        gt_px = [_polyline_pixels(g, (H, W)) for g in gt]
        polys, masks = instance_a(mask, KAPPA_MAX, params=params, channels=chans, prob=prob)
        r = centerline_f1(masks, gt, tol=TOL, length_coverage=0.95, precision_coverage=0.95) if gt else {"tp": 0, "fp": len(masks), "fn": 0, "pairs": []}
        matched_g = {q["gt"] for q in r["pairs"] if q.get("tp")}
        pr_yx = [resample(np.asarray(q, float))[:, ::-1] for q in polys]
        pr_trees = [cKDTree(c) if len(c) else None for c in pr_yx]
        pr_all = np.concatenate([c for c in pr_yx if len(c)], 0) if any(len(c) for c in pr_yx) else None
        pr_tree = cKDTree(pr_all) if pr_all is not None else None
        mask_yx = np.stack(np.where(mask), 1).astype(np.float32)
        mask_tree = cKDTree(mask_yx) if len(mask_yx) else None
        # which GT does each instance touch (>= 50 % of that GT within 5 px)?
        inst_touch = {}
        for i, t in enumerate(pr_trees):
            if t is None:
                continue
            for j, g in enumerate(gt_px):
                if len(g) and (t.query(g)[0] <= TOL).mean() >= 0.5:
                    inst_touch.setdefault(i, set()).add(j)
        recs, cls_n = [], {"MAP_BREAK": 0, "INSTANCER_CUT": 0, "CHIMERA": 0, "OTHER": 0}
        for j, g in enumerate(gt_px):
            if j in matched_g or not len(g):
                continue
            cov = float((pr_tree.query(g)[0] <= TOL).mean()) if pr_tree is not None else 0.0
            if cov < 0.95:
                continue
            # mask components along the GT
            if mask_tree is not None:
                d, idx = mask_tree.query(g)
                near = d <= TOL
                comps = set(int(lab[int(mask_yx[i, 0]), int(mask_yx[i, 1])]) for i in idx[near])
                gap = int((~near).sum())
            else:
                comps, gap = set(), len(g)
            touching = [i for i, t in enumerate(pr_trees) if t is not None and (t.query(g)[0] <= TOL).sum() >= 10]
            touching_len = {i: int((pr_trees[i].query(g)[0] <= TOL).sum()) for i in touching}
            longest = max(touching_len, key=touching_len.get) if touching_len else None
            chim = bool(longest is not None and len(inst_touch.get(longest, set()) - {j}) >= 1)
            if len(comps) >= 2:
                cls = "MAP_BREAK"
            elif chim:
                cls = "CHIMERA"
            elif len(touching) >= 2:
                cls = "INSTANCER_CUT"
            else:
                cls = "OTHER"
            cls_n[cls] += 1
            recs.append({"gt": j, "len": int(len(g)), "n_mask_cc": len(comps), "mask_gap_px": gap, "n_inst": len(touching),
                         "longest_frac": (touching_len[longest] / len(g)) if longest is not None else 0.0, "chimera": chim, "cls": cls})
        out["frames"].append({"frame": stem, "n_gt": len(gt), "tp": r["tp"], "fn": r["fn"], "n_covered_split": len(recs), "classes": cls_n, "rows": recs})
        print(f"{stem}: nGT {len(gt)} tp {r['tp']} fn {r['fn']} covered_split {len(recs)} " + " ".join(f"{k}={v}" for k, v in cls_n.items() if v), flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"))
    tot = {k: sum(f["classes"][k] for f in out["frames"]) for k in ("MAP_BREAK", "INSTANCER_CUT", "CHIMERA", "OTHER")}
    print("TOTAL covered_split", sum(tot.values()), tot)


if __name__ == "__main__":
    main()
