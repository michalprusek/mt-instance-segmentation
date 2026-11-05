#!/usr/bin/env python3
"""Anatomy of the strict metric's failures on real frames, per arm. RUNS ON TULEN (CPU, chunked).

The overlays of 2026-09-05 (data/enc_sensitivity_testset/noise_arm/pairs) showed the NOISE arm's
extra false positives on near-empty frames are not hallucinations: a prediction that follows a
filament PAST the end of its annotation (roi_img_280), a prediction that BRIDGES two GT polylines
the annotator left as pieces of one filament (roi_img_285: 1 TP -> 0 TP + 3 FP), traced bright
junk (roi_img_159), and a line hugging the frame border in BOTH arms (roi_img_44, 214). This
instrument counts those cases so the reading is a number, not three pictures.

Every unmatched prediction at the given threshold is classified, in the eval frame (1.5x), by
where its own centerline lies relative to the GT:
  off_border     < 50 % of its samples within 5 px of any GT and >= 70 % within 8 px of the frame edge
  off_other      < 50 % on GT, not a border line (unannotated visible filament, junk, texture)
  overshoot      on one GT only, covering >= 95 % of it, but extending beyond it (precision fail)
  fragment       on one GT only (>= 95 % of its own samples on GT) and covering < 95 % of it
  frag_overshoot on one GT only, both short of the GT and extending off it
  gap_bridge     runs through >= 2 GT polylines END TO END (their footprints along the prediction
                 are disjoint intervals, each GT >= 50 % covered): the annotator's pieces of one line
  junction       >= 2 GT polylines with overlapping footprints or a turn: a real identity error
Every unmatched GT is classified: covered_split (>= 95 % of it within 5 px of the prediction union
but no single prediction matched), partial (50-95 %), missed (< 50 %).

Model-free census per frame: pairs of GT polylines whose ends lie within --gap-d px with outward
tangents that continue into each other (cos >= --cos with the join direction on both sides) --
the geometric signature of one filament annotated as two -- and, per arm, whether one prediction
covers >= 50 % of both (the arm bridged the annotator's gap).

    PYTHONPATH=.:src:scripts python scripts/diag_fp_anatomy.py --pred-dir preds/NOISE_roi_ep40_native \
        --thr 0.99 --exclude roi_img_8,roi_img_149 --frames a,b --out runs/diag_roi/anat/NOISE_0.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from instance.instancer_a import instance_a  # noqa: E402
from instance.metrics import _polyline_pixels, centerline_f1  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402

UP, KAPPA_MAX, K_ORI, TOL = 1.5, 0.25, 6, 5.0
FP_CLASSES = ("off_border", "off_other", "overshoot", "fragment", "frag_overshoot", "gap_bridge", "junction")
FN_CLASSES = ("covered_split", "partial", "missed")


def resample(poly, step=1.0):
    c = np.asarray(poly, float)
    if len(c) < 2:
        return c
    seg = np.hypot(*np.diff(c, axis=0).T); L = seg.sum()
    if L < step:
        return c
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.arange(0.0, L, step)
    return np.stack([np.interp(t, cum, c[:, 0]), np.interp(t, cum, c[:, 1])], 1)


def end_tangent(c, which, n=15):
    """Outward unit tangent at the start (which=0) or end (which=-1) of resampled polyline c."""
    if which == 0:
        a, b = c[min(n, len(c) - 1)], c[0]
    else:
        a, b = c[max(len(c) - 1 - n, 0)], c[-1]
    v = b - a; nv = np.hypot(*v)
    return v / nv if nv > 0 else np.array([1.0, 0.0])


def gap_pairs(gts, gap_d, cos_min):
    """Model-free: (i, j, gap_px) for GT pairs whose ends continue into each other."""
    out = []
    ends = []
    for i, g in enumerate(gts):
        if len(g) < 3:
            continue
        for w in (0, -1):
            ends.append((i, w, g[w], end_tangent(g, w)))
    for a in range(len(ends)):
        ia, wa, pa, ta = ends[a]
        for b in range(a + 1, len(ends)):
            ib, wb, pb, tb = ends[b]
            if ia == ib:
                continue
            u = pb - pa; d = float(np.hypot(*u))
            if d > gap_d or d < 1e-6:
                continue
            u = u / d
            if float(ta @ u) >= cos_min and float(tb @ (-u)) >= cos_min:
                out.append((ia, ib, d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/roi303_eval")
    ap.add_argument("--split", default="val")
    ap.add_argument("--exclude", default="")
    ap.add_argument("--frames", default="")
    ap.add_argument("--pred-dir", default="")
    ap.add_argument("--channel", type=int, default=0,
                    help="dump channel used as the probability; -1 = max over the first K_ORI channels, which is "
                         "eval_v5.py's convention and the only correct choice for an amodal (K+1 channel) dump")
    ap.add_argument("--thr", type=float, default=0.99)
    ap.add_argument("--params", default="src/instance/params_a_model_v5H.json")
    ap.add_argument("--gap-d", type=float, default=60.0, help="eval px (40 native)")
    ap.add_argument("--cos", type=float, default=0.85)
    ap.add_argument("--border", type=float, default=8.0)
    ap.add_argument("--census-only", action="store_true", help="model-free gap census only (no maps)")
    ap.add_argument("--cov", type=float, default=0.95, help="length AND precision coverage of the "
                    "matching rule; 0.95 = the strict instrument every earlier read used, 0.80 = "
                    "the convention declared on 2026-09-15. Also the class boundary for "
                    "overshoot / fragment / covered_split, so the classes stay consistent with the rule")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.split == "test":
        raise SystemExit("refusing to touch TEST")
    excl = set(args.exclude.split(",")) if args.exclude else set()
    want = set(args.frames.split(",")) if args.frames else None
    params = json.load(open(args.params)); params.pop("kappa_max", None); params["prob_thr"] = args.thr
    out = {"args": vars(args), "frames": []}
    for p in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        stem = os.path.basename(p)[:-3]
        if stem in excl or (want and stem not in want):
            continue
        fr = read_frame_h5(p)
        if str(fr["attrs"].get("split")) != args.split:
            continue
        gt = [np.asarray(q, float) * UP for q in fr["polylines"] if len(q) >= 2]
        if args.census_only:
            gt_rs = [resample(g) for g in gt]
            pairs = gap_pairs(gt_rs, args.gap_d, args.cos)
            out["frames"].append({"frame": stem, "n_gt": len(gt), "gap_pairs": len(pairs),
                                  "gap_px": [round(d, 1) for _, _, d in pairs],
                                  "gap_ids": [[int(a), int(b)] for a, b, _ in pairs]})
            continue
        npz = os.path.join(args.pred_dir, stem + ".npz")
        if not os.path.exists(npz):
            continue
        allch = np.load(npz)["prob"].astype(np.float32)
        chans = allch[:K_ORI]
        prob = chans.max(axis=0) if args.channel < 0 else allch[min(args.channel, allch.shape[0] - 1)]
        H, W = prob.shape
        gt = [np.asarray(q, float) * UP for q in fr["polylines"] if len(q) >= 2]
        gt_rs = [resample(g) for g in gt]
        gt_px = [_polyline_pixels(g, (H, W)) for g in gt]          # (N, 2) as (y, x)
        gt_trees = [cKDTree(g) if len(g) else None for g in gt_px]
        polys, masks = instance_a(prob > args.thr, KAPPA_MAX, params=params, channels=chans, prob=prob)
        r = centerline_f1(masks, gt, tol=TOL, length_coverage=args.cov, precision_coverage=args.cov) if gt else \
            {"tp": 0, "fp": len(masks), "fn": 0, "f1": float("nan"), "pairs": []}
        matched_p = {q["pred"] for q in r["pairs"] if q.get("tp")}
        matched_g = {q["gt"] for q in r["pairs"] if q.get("tp")}
        pr_rs = [resample(np.asarray(q, float)) for q in polys]
        pr_yx = [c[:, ::-1] for c in pr_rs]
        pr_all = np.concatenate([c for c in pr_yx if len(c)], 0) if any(len(c) for c in pr_yx) else None
        pr_tree = cKDTree(pr_all) if pr_all is not None else None
        # ---- unmatched predictions
        fp_cls = {c: 0 for c in FP_CLASSES}; fp_rows = []
        for i, c in enumerate(pr_yx):
            if i in matched_p or len(c) < 2:
                continue
            n = len(c)
            near = np.zeros(n, bool); footprint = {}
            for j, t in enumerate(gt_trees):
                if t is None:
                    continue
                d = t.query(c)[0] <= TOL
                if d.sum() >= max(10, 0.15 * n):
                    idx = np.where(d)[0]
                    cov_j = float((cKDTree(c).query(gt_px[j])[0] <= TOL).mean()) if len(gt_px[j]) else 0.0
                    footprint[j] = (int(idx.min()), int(idx.max()), float(d.mean()), cov_j)
                near |= d
            on_gt = float(near.mean())
            edge = np.minimum(np.minimum(c[:, 0], H - 1 - c[:, 0]), np.minimum(c[:, 1], W - 1 - c[:, 1]))
            border = float((edge <= args.border).mean())
            if on_gt < 0.5:
                cls = "off_border" if border >= 0.7 else "off_other"
            elif len(footprint) <= 1:
                cov = max([f[3] for f in footprint.values()], default=0.0)
                cls = "overshoot" if cov >= args.cov else ("fragment" if on_gt >= args.cov else "frag_overshoot")
            else:
                items = sorted(footprint.items(), key=lambda kv: kv[1][0])
                disjoint = all(items[k][1][1] < items[k + 1][1][0] + 3 for k in range(len(items) - 1))
                strong = sum(f[3] >= 0.5 for _, f in items) >= 2
                cls = "gap_bridge" if (disjoint and strong) else "junction"
            fp_cls[cls] += 1
            fp_rows.append({"pred": i, "len": n, "on_gt": round(on_gt, 3), "border": round(border, 3), "cls": cls,
                            "gts": {str(j): [f[2], f[3]] for j, f in footprint.items()}})
        # ---- unmatched GT
        fn_cls = {c: 0 for c in FN_CLASSES}
        for j, g in enumerate(gt_px):
            if j in matched_g or not len(g):
                continue
            cov = float((pr_tree.query(g)[0] <= TOL).mean()) if pr_tree is not None else 0.0
            fn_cls["covered_split" if cov >= args.cov else ("partial" if cov >= 0.5 else "missed")] += 1
        # ---- model-free gap census + did this arm bridge it
        pairs = gap_pairs(gt_rs, args.gap_d, args.cos)
        bridged = 0
        for ia, ib, d in pairs:
            ok = False
            for c in pr_yx:
                if len(c) < 2:
                    continue
                tc = cKDTree(c)
                ca = (tc.query(gt_px[ia])[0] <= TOL).mean() if len(gt_px[ia]) else 0
                cb = (tc.query(gt_px[ib])[0] <= TOL).mean() if len(gt_px[ib]) else 0
                if ca >= 0.5 and cb >= 0.5:
                    ok = True; break
            bridged += int(ok)
        rec = {"frame": stem, "n_gt": len(gt), "n_pred": len(polys), "tp": r["tp"], "fp": r["fp"], "fn": r["fn"],
               "f1": r["f1"], "fp_classes": fp_cls, "fn_classes": fn_cls, "fps": fp_rows,
               "gap_pairs": len(pairs), "gap_pairs_bridged": bridged,
               "gap_px": [round(d, 1) for _, _, d in pairs]}
        out["frames"].append(rec)
        print(f"{stem}: nGT={len(gt)} tp={r['tp']} fp={r['fp']} fn={r['fn']} | FP "
              + " ".join(f"{k}={v}" for k, v in fp_cls.items() if v) + " | FN "
              + " ".join(f"{k}={v}" for k, v in fn_cls.items() if v)
              + f" | gaps {len(pairs)} bridged {bridged}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"))
    if args.census_only:
        g = np.array([d for f in out["frames"] for d in f["gap_px"]], float)
        print("CENSUS frames", len(out["frames"]), "GT", sum(f["n_gt"] for f in out["frames"]), "gap pairs", len(g),
              "| by gap (eval px) <=30:", int((g <= 30).sum()), "<=60:", int((g <= 60).sum()), "<=120:", int((g <= 120).sum()),
              "<=200:", int((g <= 200).sum()), "| frames with >=1 pair", sum(f["gap_pairs"] > 0 for f in out["frames"]))
        return
    tot_fp = {c: sum(f["fp_classes"][c] for f in out["frames"]) for c in FP_CLASSES}
    tot_fn = {c: sum(f["fn_classes"][c] for f in out["frames"]) for c in FN_CLASSES}
    print("TOTAL FP", tot_fp, "FN", tot_fn, "gaps", sum(f["gap_pairs"] for f in out["frames"]),
          "bridged", sum(f["gap_pairs_bridged"] for f in out["frames"]))


if __name__ == "__main__":
    main()
