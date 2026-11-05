#!/usr/bin/env python3
"""D2/Q1 -- WHERE do coverage failures live?  RUNS ON TULEN (CPU).

For every ground-truth microtubule that instancer A does NOT recover as a true positive, this
localises the failure: it finds the best-covering predicted instance, walks the GT centerline,
and records every place where that instance stops covering.  Each break is then given the
nearest geometric excuse:

  * crossing        -- a GT-GT crossing (``mt_bench.gt_stats.count_crossings``) nearby;
  * parallel        -- a stretch where this GT runs 2-6 px (native) from another GT;
  * neither         -- mid-filament, no geometric excuse at all.

Two extra columns decide WHO is to blame, which is the whole point of running it twice:

  * ``mask_hole``   -- is the binary foreground itself absent at the break?  If yes the
                       instancer never had the pixels; if no, the instancer declined to link.
  * ``deg_to_bin_boundary`` -- the GT tangent's distance to a 30-degree bin edge at the break,
                       against a control drawn from the COVERED part of the same microtubule.
                       This is the bridge to Q2a: if breaks concentrate near bin boundaries the
                       target representation is implicated.

The identical analysis runs on the ORACLE mask (rasterised GT), which separates "the instancer
cannot do it" from "the mask does not support it".

``--fuse`` swaps the inference rule ``max_k p_k > thr`` for ``sum_k p_k > thr`` or the
noisy-OR ``1 - prod_k (1 - p_k) > thr``.  That is the causal test for the bin-splitting
hypothesis: if the model divides its mass between two adjacent orientation channels, the max
dips below threshold where the sum does not.

    PYTHONPATH=src /home/prusek/dinov3_env/bin/python scripts/diag_failure_geometry.py \
        --data data/synth_eval --masks model \
        --pred-dir /home/prusek/mt_enc_exp/synth_pred_v4b \
        --params-a src/instance/params_a_model.json \
        --out data/enc_sensitivity_testset/diag_fail_synth_model.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import label, zoom
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from instance.geometry import arclength, resample  # noqa: E402
from instance.instancer_a import instance_a  # noqa: E402
from instance.metrics import centerline_f1  # noqa: E402
from instance.oracle import oracle_instance_masks, oracle_ori_channels  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.fov import fov_mask  # noqa: E402
from mt_bench.gt_stats import count_crossings  # noqa: E402

UP = 1.5
KAPPA_MAX = 0.25
K = 6
BIN_W_DEG = 30.0
CONN8 = np.ones((3, 3), bool)
TOL = 5.0                       # the metric's tolerance, eval-frame px
PAR_LO, PAR_HI = 2.0 * UP, 6.0 * UP


def tangent_deg(pts: np.ndarray, win: float = 12.0) -> np.ndarray:
    s = arclength(pts)
    lo = np.clip(np.searchsorted(s, s - win / 2.0, "left"), 0, len(pts) - 1)
    hi = np.clip(np.searchsorted(s, s + win / 2.0, "right") - 1, 0, len(pts) - 1)
    bad = hi <= lo
    lo[bad] = np.maximum(0, hi[bad] - 1)
    hi[bad] = np.minimum(len(pts) - 1, lo[bad] + 1)
    d = pts[hi] - pts[lo]
    return np.rad2deg(np.arctan2(d[:, 1], d[:, 0])) % 180.0


def deg_to_boundary(ang_deg: np.ndarray) -> np.ndarray:
    m = ang_deg % BIN_W_DEG
    return np.minimum(m, BIN_W_DEG - m)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth_eval")
    ap.add_argument("--split", default="val")
    ap.add_argument("--masks", default="model", choices=["oracle", "model"])
    ap.add_argument("--pred-dir", default="/home/prusek/mt_enc_exp/synth_pred_v4b")
    ap.add_argument("--params-a", default=None)
    ap.add_argument("--prob-thr", type=float, default=0.35)
    ap.add_argument("--fuse", default="max", choices=["max", "sum", "noisyor"])
    ap.add_argument("--no-fov", action="store_true")
    ap.add_argument("--near", type=float, default=10.0,
                    help="excuse radius in EVAL-frame px (10 px at 1.5x = 6.7 native)")
    ap.add_argument("--near2", type=float, default=15.0, help="second excuse radius")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pa = json.load(open(args.params_a)) if args.params_a else None
    if pa:
        pa.pop("kappa_max", None)

    breaks, gts, frames = [], [], []
    t0 = time.time()
    for path in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        fr = read_frame_h5(path)
        if args.split != "all" and str(fr["attrs"].get("split")) != args.split:
            continue
        name = os.path.basename(path)
        shape = (int(fr["attrs"]["height"]), int(fr["attrs"]["width"]))
        hi_shape = (int(round(shape[0] * UP)), int(round(shape[1] * UP)))
        gt = [np.asarray(p, float) * UP for p in fr["polylines"]]
        if not gt:
            continue
        cr = count_crossings(gt, tol=2.0 * UP, endpoint_margin=8.0 * UP)
        ctree = cKDTree(np.array([[c["x"], c["y"]] for c in cr])) if cr else None

        # ---------------- mask ----------------
        if args.masks == "oracle":
            chans = oracle_ori_channels(fr["polylines"], shape, K=K, half_width=1.0, up=UP)
            fused = chans.max(axis=0)
            mask = fused > 0.5
        else:
            chans = np.load(os.path.join(
                args.pred_dir, name.replace(".h5", ".npz")))["prob"].astype(np.float32)
            if chans.shape[1:] != hi_shape:
                raise SystemExit(f"{name}: {chans.shape[1:]} != {hi_shape}")
            if not args.no_fov:
                fv = zoom(fov_mask(fr["image"]).astype(np.uint8), UP, order=0).astype(bool)
                fv = np.pad(fv, ((0, max(0, hi_shape[0] - fv.shape[0])),
                                 (0, max(0, hi_shape[1] - fv.shape[1]))),
                            constant_values=True)[:hi_shape[0], :hi_shape[1]]
                chans = chans * fv[None]
            thr_a = (pa or {}).get("prob_thr", args.prob_thr)
            if args.fuse == "max":
                fused = chans.max(axis=0)
            elif args.fuse == "sum":
                fused = chans.sum(axis=0)
            else:
                fused = 1.0 - np.prod(1.0 - chans, axis=0)
            mask = fused > thr_a

        n_cc = int(label(mask, structure=CONN8)[1])
        pls, _ = instance_a(mask, KAPPA_MAX, pa, channels=chans, prob=chans.max(axis=0))
        pmasks = oracle_instance_masks(pls, hi_shape, half_width=1.0, up=1.0)
        f1 = centerline_f1(pmasks, gt, tol=TOL, length_coverage=0.95,
                           precision_coverage=0.95)
        pair_by_gt = {p["gt"]: p for p in f1["pairs"]}
        frames.append({"name": name, "n_gt": len(gt), "n_pred": len(pls),
                       "mask_cc": n_cc, "fg_frac": float(mask.mean()),
                       "tp": f1["tp"], "fp": f1["fp"], "fn": f1["fn"], "f1": f1["f1"]})

        ptrees = [cKDTree(resample(p, ds=1.0)) for p in pls if len(p) > 1]
        gt_res = [resample(p, ds=1.0) for p in gt]
        alltree = [cKDTree(g) for g in gt_res]

        for gi, g in enumerate(gt_res):
            if len(g) < 3:
                continue
            pr = pair_by_gt.get(gi)
            is_tp = bool(pr and pr["tp"])
            # parallel zone: this GT within [3, 9] px of ANOTHER GT
            dmin = np.full(len(g), np.inf)
            for gj, tj in enumerate(alltree):
                if gj == gi or len(gt_res[gj]) < 2:
                    continue
                d, _ = tj.query(g, k=1)
                dmin = np.minimum(dmin, d)
            par_pts = g[(dmin >= PAR_LO) & (dmin <= PAR_HI)]
            ptree_par = cKDTree(par_pts) if len(par_pts) else None

            cov_frac, best, covered = 0.0, -1, np.zeros(len(g), bool)
            for pi, t in enumerate(ptrees):
                d, _ = t.query(g, k=1)
                c = d <= TOL
                if c.mean() > cov_frac:
                    cov_frac, best, covered = float(c.mean()), pi, c
            ang = tangent_deg(g)
            d2b = deg_to_boundary(ang)
            row = {"frame": name, "gt": gi, "n_gt_frame": len(gt),
                   "length": float(arclength(g)[-1]), "tp": is_tp,
                   "best_cov": cov_frac, "n_pred_touch": int(sum(
                       1 for t in ptrees if (t.query(g, k=1)[0] <= TOL).mean() > 0.10)),
                   "pair_rec": float(pr["recall"]) if pr else None,
                   "pair_prec": float(pr["precision"]) if pr else None,
                   "frac_par_zone": float(((dmin >= PAR_LO) & (dmin <= PAR_HI)).mean()),
                   "d2b_mean_covered": float(d2b[covered].mean()) if covered.any() else None}
            gts.append(row)
            if is_tp:
                continue

            # ---- breaks: every transition of the coverage indicator ----
            trans = np.where(np.diff(covered.astype(int)) != 0)[0]
            if cov_frac < 0.02:
                trans = np.array([], int)
                row["mode"] = "no_coverage"
            for t_i in trans:
                p = g[t_i]
                dc = float(ctree.query(p[None], k=1)[0][0]) if ctree is not None else np.inf
                ang_cr = np.nan
                if ctree is not None:
                    ang_cr = float(cr[int(ctree.query(p[None], k=1)[1][0])]["angle_deg"])
                dp = float(ptree_par.query(p[None], k=1)[0][0]) if ptree_par else np.inf
                rr = int(round(p[1])); cc = int(round(p[0]))
                r0, r1 = max(0, rr - 3), min(hi_shape[0], rr + 4)
                c0, c1 = max(0, cc - 3), min(hi_shape[1], cc + 4)
                breaks.append({
                    "frame": name, "gt": gi, "s": float(t_i),
                    "s_frac": float(t_i / max(1, len(g) - 1)),
                    "tip": bool(t_i < 15 or t_i > len(g) - 16),
                    "d_cross": dc, "cross_angle": ang_cr, "d_par": dp,
                    "mask_hole": bool(not mask[r0:r1, c0:c1].any()),
                    "deg_to_boundary": float(d2b[t_i]),
                })
    dt = time.time() - t0

    # ------------------------------- aggregate -------------------------------
    def frac(sel, tot):
        return float(sel / tot) if tot else float("nan")

    nb = len(breaks)
    out = {
        "config": {"data": args.data, "split": args.split, "masks": args.masks,
                   "pred_dir": args.pred_dir if args.masks == "model" else None,
                   "fuse": args.fuse, "params_a": args.params_a,
                   "near_px_eval_frame": args.near, "near2": args.near2,
                   "seconds": dt},
        "frames": {"n": len(frames),
                   "n_gt": sum(f["n_gt"] for f in frames),
                   "n_pred": sum(f["n_pred"] for f in frames),
                   "mask_cc_total": sum(f["mask_cc"] for f in frames),
                   "mask_cc_per_frame": float(np.mean([f["mask_cc"] for f in frames])),
                   "fg_frac": float(np.mean([f["fg_frac"] for f in frames])),
                   "tp": sum(f["tp"] for f in frames), "fp": sum(f["fp"] for f in frames),
                   "fn": sum(f["fn"] for f in frames),
                   "mean_f1": float(np.mean([f["f1"] for f in frames]))},
    }
    ng = len(gts)
    fails = [r for r in gts if not r["tp"]]
    out["gt"] = {
        "n": ng, "n_tp": sum(1 for r in gts if r["tp"]), "n_fail": len(fails),
        "mean_best_cov_fail": float(np.mean([r["best_cov"] for r in fails])) if fails else None,
        "median_best_cov_fail": float(np.median([r["best_cov"] for r in fails])) if fails else None,
    }
    # failure MODE, from the metric's own assigned pair
    modes = {"no_pred_pair": 0, "recall_only": 0, "precision_only": 0, "both": 0}
    for r in fails:
        if r["pair_rec"] is None:
            modes["no_pred_pair"] += 1
        else:
            lo_r, lo_p = r["pair_rec"] < 0.95, r["pair_prec"] < 0.95
            modes["both" if (lo_r and lo_p) else
                  ("recall_only" if lo_r else "precision_only")] += 1
    out["failure_mode"] = {**modes,
                           **{f"frac_{k}": frac(v, len(fails)) for k, v in modes.items()}}
    out["gt_best_cov_hist"] = {
        f"<{t}": int(sum(1 for r in fails if r["best_cov"] < t))
        for t in (0.02, 0.25, 0.5, 0.75, 0.90, 0.95, 1.01)}

    if nb:
        dcr = np.array([b["d_cross"] for b in breaks])
        dpr = np.array([b["d_par"] for b in breaks])
        mh = np.array([b["mask_hole"] for b in breaks])
        d2b_b = np.array([b["deg_to_boundary"] for b in breaks])
        tip = np.array([b["tip"] for b in breaks])
        for tag, R in (("near10", args.near), ("near15", args.near2)):
            c = dcr <= R
            p = (dpr <= R) & ~c
            n = ~c & ~p
            out[f"break_attribution_{tag}"] = {
                "n_breaks": nb,
                "crossing": frac(c.sum(), nb),
                "parallel_not_crossing": frac(p.sum(), nb),
                "neither": frac(n.sum(), nb),
                "neither_and_not_tip": frac((n & ~tip).sum(), nb),
                "n_crossing": int(c.sum()), "n_parallel": int(p.sum()),
                "n_neither": int(n.sum()),
                "mask_hole_rate_overall": float(mh.mean()),
                "mask_hole_rate_crossing": float(mh[c].mean()) if c.any() else None,
                "mask_hole_rate_parallel": float(mh[p].mean()) if p.any() else None,
                "mask_hole_rate_neither": float(mh[n].mean()) if n.any() else None,
            }
        ca = np.array([b["cross_angle"] for b in breaks], float)
        near_c = dcr <= args.near
        out["break_crossing_angle_split"] = {
            k: int(((ca >= lo) & (ca < hi) & near_c).sum())
            for k, (lo, hi) in {"shallow<30": (0, 30), "oblique30-60": (30, 60),
                                "steep>=60": (60, 91)}.items()}
        out["break_deg_to_bin_boundary"] = {
            "mean_at_break": float(d2b_b.mean()),
            "frac_break_within_5deg": float((d2b_b <= 5.0).mean()),
            "control_mean_on_covered": float(np.mean(
                [r["d2b_mean_covered"] for r in gts if r["d2b_mean_covered"] is not None])),
            "expected_frac_uniform": 1.0 / 3.0,
        }
        out["break_tip_frac"] = float(tip.mean())
    out["_breaks"] = breaks
    out["_gts"] = gts
    out["_frames"] = frames

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("_")}, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
