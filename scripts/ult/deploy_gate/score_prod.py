#!/usr/bin/env python3
"""Production check: score what the production API returned, old and new model, with the research metric against the human GT of the same VAL frames, and compare the new model's curves with the research reference. VAL frames only; no TEST frame is involved. usage: score_prod.py MANIFEST OLD.json NEW.json REF_DIR"""
import json, os, sys
import numpy as np
sys.path[:0] = [".", "src", "scripts", "scripts/lpc"]
from lpc_lib import UP, resample, rasterise_polyline
from instance.metrics import centerline_f1
from mt_bench.cvat_import import read_frame_h5
from scipy.spatial import cKDTree
MAN, OLD, NEW, REF = sys.argv[1:5]; M = json.load(open(MAN))["frames"]; O = json.load(open(OLD))["frames"]; N = json.load(open(NEW))["frames"]
SRC = {"roi": "data/real/roi303_v2/", "htw": "data/real/mt34_htw_eval/", "mt34": "data/real/mt34_htw_eval/"}
def score(polys, shape, gt):
    masks = [rasterise_polyline(resample(np.asarray(p, float) * UP, ds=0.5), shape) for p in polys if len(p) >= 2]
    r = centerline_f1(masks, gt, tol=5.0, length_coverage=0.8, precision_coverage=0.8, end_tol=0.0); return np.array((r["tp"], r["fp"], r["fn"]))
f1 = lambda s: 2 * s[0] / max(2 * s[0] + s[1] + s[2], 1); TO, TN = np.zeros(3), np.zeros(3)
print(f"{'frame':14s} {'n gt':>5s} | old n, tp/fp/fn, F1 | new n, tp/fp/fn, F1 | new vs research reference")
for name, m in M.items():
    fr = read_frame_h5(SRC[name.split("_")[0]] + m["source"]); gt = [np.asarray(q, float) * UP for q in fr["polylines"]]; H, W = (int(round(s * UP)) for s in np.asarray(fr["image"]).shape[:2])
    a, b = O[m["file"]], N[m["file"]]; sa, sb = score(a["polys"], (H, W), gt), score(b["polys"], (H, W), gt); TO += sa; TN += sb
    ref = json.load(open(os.path.join(REF, f"reference_{name}.json")))["polylines_xy_eval_scale"]; rp = np.concatenate([resample(np.asarray(p, float), ds=1.0) for p in ref]); np_ = np.concatenate([resample(np.asarray(p, float) * UP, ds=1.0) for p in b["polys"]])
    d1, _ = cKDTree(rp).query(np_); d2, _ = cKDTree(np_).query(rp)
    print(f"{name:14s} {len(gt):5d} | {a['n']:4d} {sa[0]:4d}/{sa[1]:4d}/{sa[2]:4d} {f1(sa):.3f} | {b['n']:4d} {sb[0]:4d}/{sb[1]:4d}/{sb[2]:4d} {f1(sb):.3f} | n ref {len(ref)}, new->ref p99 {np.percentile(d1, 99):.2f} px, ref->new p99 {np.percentile(d2, 99):.2f} px (1.5x frame)")
print(f"pooled | old {TO.astype(int).tolist()} F1 {f1(TO):.3f} | new {TN.astype(int).tolist()} F1 {f1(TN):.3f}")
