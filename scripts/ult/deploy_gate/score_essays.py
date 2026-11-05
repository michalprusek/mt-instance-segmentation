#!/usr/bin/env python3
"""Essays production check (docs/TODO.md 2026-11-03 10:40): score the annotation JSON of two Essays jobs against the MT-34 VAL GT of frame 0, and compare the later job with the research reference on the same frames. usage: score_essays.py BEFORE AFTER REFERENCE.json"""
import glob, json, os, sys
import numpy as np
sys.path[:0] = [".", "src", "scripts", "scripts/lpc"]
from lpc_lib import UP, resample, rasterise_polyline
from instance.metrics import centerline_f1
from mt_bench.cvat_import import read_frame_h5
from scipy.spatial import cKDTree
BEFORE, AFTER, REF = sys.argv[1:4]; R = json.load(open(REF))
WELLS = {"pll336_100x_atp_ch1_I__060": "060", "pll338_500x_atp_ch4_I__064": "064", "pll336_100x_NOatp_ch4_II__066": "066", "pll338_500x_NOatp_ch2_III__06": "069"}
def polys(job, key):
    p = glob.glob(f"{job}/annotations/*{key}*_pos0.json"); assert len(p) == 1, (key, p)
    return [np.array([[q["x"], q["y"]] for q in pl["points"]], float) for pl in json.load(open(p[0]))["polylines"]]
def score(ps, shape, gt):
    masks = [rasterise_polyline(resample(q * UP, ds=0.5), shape) for q in ps if len(q) >= 2]
    r = centerline_f1(masks, gt, tol=5.0, length_coverage=0.8, precision_coverage=0.8, end_tol=0.0); return np.array((r["tp"], r["fp"], r["fn"]))
f1 = lambda s: 2 * s[0] / max(2 * s[0] + s[1] + s[2], 1); TO, TN = np.zeros(3), np.zeros(3)
print(f"{'well':5s} {'n gt':>5s} | old n, tp/fp/fn, F1 | new n, tp/fp/fn, F1 | new vs research reference")
for key, w in WELLS.items():
    h5 = f"alice_2026_02_06_{key}_f00_irm.h5"; fr = read_frame_h5("data/real/mt34_htw_eval/" + h5)
    assert fr.get("split", "val") != "test"; gt = [np.asarray(q, float) * UP for q in fr["polylines"]]
    H, W = (int(round(s * UP)) for s in fr["image"].shape[:2]); a, b = polys(BEFORE, key), polys(AFTER, key)
    sa, sb = score(a, (H, W), gt), score(b, (H, W), gt); TO += sa; TN += sb
    ref = R[h5]; rp = np.concatenate([resample(np.asarray(p, float), ds=0.5) for p in ref]); np_ = np.concatenate([resample(q * UP, ds=0.5) for q in b])
    d1 = cKDTree(rp).query(np_)[0]; d2 = cKDTree(np_).query(rp)[0]
    print(f"{w:5s} {len(gt):5d} | {len(a):4d} {sa[0]:3d}/{sa[1]:3d}/{sa[2]:3d} {f1(sa):.3f} | {len(b):4d} {sb[0]:3d}/{sb[1]:3d}/{sb[2]:3d} {f1(sb):.3f} | ref n {len(ref)} p99 {max(np.percentile(d1,99),np.percentile(d2,99)):.2f} px at 1.5x, max {max(d1.max(),d2.max()):.2f}")
print(f"pooled | old {TO.astype(int).tolist()} F1 {f1(TO):.3f} | new {TN.astype(int).tolist()} F1 {f1(TN):.3f}")
