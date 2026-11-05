#!/usr/bin/env python3
"""Edge-band lines: how many unmatched predictions run ALONG the frame edge (the dark band a few px
wide at the bottom/left of cropped ROI frames, traced as a filament), what an edge-band rule
would remove, and what it would cost on the GT. RUNS ON TULEN (CPU; instancing only on the
frames given, the GT census on every VAL frame).

Rule under test (declared 2026-09-05, replaces the darkness-based FOV rule that failed its smoke):
a prediction is an edge-band line iff >= --frac of its skeleton samples lie within --band eval px
of ONE frame edge and its principal direction is within --angle degrees of that edge.
The same test on every GT polyline gives the rule's cost (GT that would be unreachable).
Also reports, per frame, the mean intensity of the outer 20 native px along each edge relative to
the frame median, so the band itself is measured.

    PYTHONPATH=.:src:scripts python scripts/diag_edge_band.py --pred-dir preds/NOISE_roi_ep40_native \
        --frames "$(cat chunk.txt)" --thr 0.99 --out runs/diag_roi/edge_NOISE.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from diag_fp_anatomy import UP, KAPPA_MAX, K_ORI, TOL, resample  # noqa: E402
from instance.instancer_a import instance_a  # noqa: E402
from instance.metrics import centerline_f1  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402


def edge_test(c, H, W, band, frac, angle):
    """c: (N, 2) as (x, y) eval px. Returns (is_edge_band, which_edge, frac_in_band, angle_deg)."""
    if len(c) < 3:
        return False, None, 0.0, 90.0
    d = {"left": c[:, 0], "right": W - 1 - c[:, 0], "top": c[:, 1], "bottom": H - 1 - c[:, 1]}
    best = max(d, key=lambda k: float((d[k] <= band).mean()))
    f = float((d[best] <= band).mean())
    v = c - c.mean(0); u = np.linalg.svd(v, full_matrices=False)[2][0]
    ang = np.degrees(np.arctan2(abs(u[1]), abs(u[0])))            # 0 = horizontal, 90 = vertical
    edge_ang = 0.0 if best in ("top", "bottom") else 90.0
    dev = abs(ang - edge_ang)
    return bool(f >= frac and dev <= angle), best, f, float(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/roi303_eval")
    ap.add_argument("--split", default="val")
    ap.add_argument("--exclude", default="roi_img_8,roi_img_149")
    ap.add_argument("--frames", default="", help="frames to instance (empty = GT census only)")
    ap.add_argument("--pred-dir", default="")
    ap.add_argument("--thr", type=float, default=0.99)
    ap.add_argument("--params", default="src/instance/params_a_model_v5H.json")
    ap.add_argument("--band", type=float, default=30.0)
    ap.add_argument("--frac", type=float, default=0.8)
    ap.add_argument("--angle", type=float, default=20.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.split == "test":
        raise SystemExit("refusing to touch TEST")
    excl = set(args.exclude.split(",")); want = set(args.frames.split(",")) if args.frames else set()
    params = json.load(open(args.params)); params.pop("kappa_max", None); params["prob_thr"] = args.thr
    out = {"args": vars(args), "gt": [], "pred": []}
    for p in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        stem = os.path.basename(p)[:-3]
        if stem in excl:
            continue
        fr = read_frame_h5(p)
        if str(fr["attrs"].get("split")) != args.split:
            continue
        img = fr["image"].astype(np.float32); h, w = img.shape
        med = float(np.median(img)); rng = float(np.percentile(img, 99) - np.percentile(img, 1)) or 1.0
        prof = {"left": float(img[:, :20].mean()), "right": float(img[:, -20:].mean()),
                "top": float(img[:20].mean()), "bottom": float(img[-20:].mean())}
        prof = {k: (v - med) / rng for k, v in prof.items()}
        H, W = int(round(h * UP)), int(round(w * UP))
        gt = [np.asarray(q, float) * UP for q in fr["polylines"] if len(q) >= 2]
        n_gt_edge = 0
        for g in gt:
            ok, e, f, dev = edge_test(resample(g), H, W, args.band, args.frac, args.angle)
            n_gt_edge += int(ok)
        out["gt"].append({"frame": stem, "n_gt": len(gt), "n_gt_edge": n_gt_edge, "edge_profile": prof})
        if stem not in want or not args.pred_dir:
            continue
        allch = np.load(os.path.join(args.pred_dir, stem + ".npz"))["prob"].astype(np.float32)
        prob = allch[0]; chans = allch[:K_ORI]
        polys, masks = instance_a(prob > args.thr, KAPPA_MAX, params=params, channels=chans, prob=prob)
        r = centerline_f1(masks, gt, tol=TOL, length_coverage=0.95, precision_coverage=0.95) if gt else \
            {"tp": 0, "fp": len(masks), "fn": 0, "f1": float("nan"), "pairs": []}
        matched = {q["pred"] for q in r["pairs"] if q.get("tp")}
        rows, keep = [], []
        for i, q in enumerate(polys):
            ok, e, f, dev = edge_test(resample(np.asarray(q, float)), H, W, args.band, args.frac, args.angle)
            rows.append({"pred": i, "matched": i in matched, "edge": ok, "which": e, "frac": round(f, 3), "dev": round(dev, 1),
                         "len": int(len(resample(np.asarray(q, float))))})
            keep.append(not ok)
        r2 = centerline_f1([m for m, k in zip(masks, keep) if k], gt, tol=TOL, length_coverage=0.95, precision_coverage=0.95) if gt else \
            {"tp": 0, "fp": int(sum(keep)), "fn": 0, "f1": float("nan")}
        out["pred"].append({"frame": stem, "n_gt": len(gt), "tp": r["tp"], "fp": r["fp"], "fn": r["fn"], "f1": r["f1"],
                            "rule_tp": r2["tp"], "rule_fp": r2["fp"], "rule_fn": r2["fn"], "rule_f1": r2["f1"],
                            "n_edge_pred": int(sum(x["edge"] for x in rows)), "n_edge_matched": int(sum(x["edge"] and x["matched"] for x in rows)),
                            "rows": rows})
        print(f"{stem}: nGT {len(gt)} f1 {r['f1']:.3f} -> rule {r2['f1']:.3f}  edge preds {out['pred'][-1]['n_edge_pred']} "
              f"(matched {out['pred'][-1]['n_edge_matched']})  bands L/R/T/B {prof['left']:+.2f}/{prof['right']:+.2f}/{prof['top']:+.2f}/{prof['bottom']:+.2f}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    g = out["gt"]; print(f"GT census: {sum(x['n_gt'] for x in g)} polylines on {len(g)} frames; edge-band GT by the rule: {sum(x['n_gt_edge'] for x in g)}")
    dark = [x for x in g if min(x["edge_profile"].values()) < -0.3]
    print(f"frames with an edge band darker than 30 % of range: {len(dark)} ({[x['frame'] for x in dark][:12]}...)")
    if out["pred"]:
        P = out["pred"]; f1 = np.nanmean([x["f1"] for x in P if x["n_gt"]]); f2 = np.nanmean([x["rule_f1"] for x in P if x["n_gt"]])
        print(f"PRED ({len(P)} frames): edge-band predictions {sum(x['n_edge_pred'] for x in P)} (matched TPs among them {sum(x['n_edge_matched'] for x in P)}); "
              f"macro F1 {f1:.4f} -> {f2:.4f} under the rule; TP {sum(x['tp'] for x in P)} -> {sum(x['rule_tp'] for x in P)}, FP {sum(x['fp'] for x in P)} -> {sum(x['rule_fp'] for x in P)}")
    json.dump(out, open(args.out, "w"))


if __name__ == "__main__":
    main()
