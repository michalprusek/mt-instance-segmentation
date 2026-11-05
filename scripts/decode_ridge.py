#!/usr/bin/env python3
"""Threshold-free RIDGE decoding of a probability map, screened against the global threshold.

Holm entry 14 of the 2026-09-02 campaign (docs/TODO.md §0.0-CAMPAIGN, declared 2026-09-04).

Why. The instancer needs a binary mask, and the mask is made by ONE global threshold. Measured on
roi303 VAL: at 0.98-0.99 the mask is broken wherever the ridge confidence dips (crossings, faint
stretches, whole low-SNR frames -- the per-frame optimal threshold spans 0.30-0.9995), and at a
low threshold two close parallels fuse into one band. Hysteresis / closing (null) grow CONNECTED
COMPONENTS, so they cannot make two ridges where the threshold left one. Non-maximum suppression
along the ridge normal (Steger 1998) can: a pixel is kept iff it is a local maximum of the map
across the ridge, at ANY confidence above a low existence level. Two ridges with a saddle
between them stay two ridges; a faint but locally maximal stretch stays a ridge.

What it does NOT fix: a ridge whose own maximum falls below the existence level (failing arms
have p_far_min p50 0.29), and junctions, where two ridge directions coexist and the Hessian
normal is undefined -- hence the UNION variant, which adds the skeleton of the confident mask so
junction neighbourhoods stay connected.

Variants (all fed to instancer A with the shipped v5H params, min_length untouched):
  B(t)       -- baseline: prob > t                                   (harness validation: must
                reproduce the eval_v5 sweep numbers exactly)
  N(t)       -- dilate3(NMS ridge, existence >= t)
  U(t)       -- dilate3(NMS ridge >= t  |  skeleton(prob > 0.98))
  W(t, r)    -- (prob > t) & dilate_r(NMS ridge >= t): the low-threshold mask restricted to a band
                of radius r around a ridge maximum, so saddles farther than r px from any maximum
                are cut while junction neighbourhoods (everything near a ridge) survive
  V(t, tr)   -- (prob > t) minus the VALLEY pixels (local minima across the map between two ridges
                >= tr within 2-4 px on both sides): the threshold mask with fused bands cut along
                a 1-px saddle line and nothing else changed
  H(t, tr)   -- hysteresis(0.98 seed, t grow) minus the same valleys: existence from a strong
                seed, extension through weak evidence, separation at saddles

Reads cached maps as `eval_v5.py --dump-pred` writes them (eval frame). Per-frame rows are saved
for every variant so a paired bootstrap is an offline re-read, never a re-run.

    PYTHONPATH=src:scripts python scripts/decode_ridge.py --data data/real/roi303_eval \
        --pred-dir preds/SPARSE35_roi_native --frames a,b,c --out runs/ridge/chunk.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, map_coordinates
from scipy.ndimage import label as cc_label
from skimage.morphology import skeletonize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from instance.instancer_a import instance_a  # noqa: E402
from instance.metrics import bundle_recovery, centerline_f1, junction_identity  # noqa: E402
from instance.oracle import oracle_instance_masks  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.gt_stats import count_crossings  # noqa: E402

UP, KAPPA_MAX = 1.5, 0.25


def _disk(r: int) -> np.ndarray:
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (yy ** 2 + xx ** 2) <= r * r


def ridge_nms(prob: np.ndarray, t_lo: float, sigma_s: float = 1.0, sigma_h: float = 1.5) -> np.ndarray:
    """Pixels that are local maxima of the (lightly smoothed) map ACROSS the ridge.

    The across-ridge direction is the eigenvector of the Hessian's most negative eigenvalue
    (Steger 1998). Ties (a saturated float16 plateau of 1.0) are kept, so a plateau yields a
    2-3 px ridge; the instancer skeletonises anyway. Returns a boolean [H, W] mask.
    """
    P = np.asarray(prob, np.float32)
    Ps = gaussian_filter(P, sigma_s) if sigma_s > 0 else P
    Hrr = gaussian_filter(P, sigma_h, order=(2, 0))
    Hcc = gaussian_filter(P, sigma_h, order=(0, 2))
    Hrc = gaussian_filter(P, sigma_h, order=(1, 1))
    tr = 0.5 * (Hrr + Hcc)
    dsc = np.sqrt(np.maximum((0.5 * (Hrr - Hcc)) ** 2 + Hrc ** 2, 0.0))
    lmin = tr - dsc                                    # most negative curvature (across ridge)
    # eigenvector of lmin: (Hrc, lmin - Hrr) in (row, col) order, or the axis-aligned fallback
    vr = Hrc.copy()
    vc = lmin - Hrr
    nrm = np.hypot(vr, vc)
    small = nrm < 1e-8
    vr[small] = np.where(Hrr[small] <= Hcc[small], 1.0, 0.0)
    vc[small] = np.where(Hrr[small] <= Hcc[small], 0.0, 1.0)
    nrm = np.hypot(vr, vc)
    vr, vc = vr / nrm, vc / nrm
    H, W = P.shape
    rr, cc = np.mgrid[0:H, 0:W].astype(np.float32)
    cand = (Ps >= t_lo) & (lmin < 0)
    out = np.zeros((H, W), bool)
    if not cand.any():
        return out
    r, c = np.where(cand)
    p0 = Ps[r, c]
    pp = map_coordinates(Ps, [rr[r, c] + vr[r, c], cc[r, c] + vc[r, c]], order=1, mode="nearest")
    pm = map_coordinates(Ps, [rr[r, c] - vr[r, c], cc[r, c] - vc[r, c]], order=1, mode="nearest")
    keep = (p0 >= pp) & (p0 >= pm)
    out[r[keep], c[keep]] = True
    return out


def valley_cut(prob: np.ndarray, t_ridge: float, sigma_s: float = 1.0, sigma_h: float = 1.5,
               reach: tuple = (2.0, 3.0, 4.0)) -> np.ndarray:
    """Pixels that sit in a SADDLE between two ridges: a local minimum of the map along the
    direction of maximum positive curvature, with a ridge (>= t_ridge) within ``reach`` px on
    BOTH sides. Removing them from a low-threshold mask cuts a fused band of two close parallels
    into two bands along a 1-px valley line while leaving everything else -- junction blobs, arm
    shape, spur-free contours -- exactly as the threshold made it. Returns a boolean [H, W] mask.
    """
    P = np.asarray(prob, np.float32)
    Ps = gaussian_filter(P, sigma_s) if sigma_s > 0 else P
    Hrr = gaussian_filter(P, sigma_h, order=(2, 0))
    Hcc = gaussian_filter(P, sigma_h, order=(0, 2))
    Hrc = gaussian_filter(P, sigma_h, order=(1, 1))
    tr = 0.5 * (Hrr + Hcc)
    dsc = np.sqrt(np.maximum((0.5 * (Hrr - Hcc)) ** 2 + Hrc ** 2, 0.0))
    lmax = tr + dsc                                    # most positive curvature (across a valley)
    vr = Hrc.copy()
    vc = lmax - Hrr
    nrm = np.hypot(vr, vc)
    small = nrm < 1e-8
    vr[small] = np.where(Hrr[small] >= Hcc[small], 1.0, 0.0)
    vc[small] = np.where(Hrr[small] >= Hcc[small], 0.0, 1.0)
    nrm = np.hypot(vr, vc)
    vr, vc = vr / nrm, vc / nrm
    H, W = P.shape
    rr, cc = np.mgrid[0:H, 0:W].astype(np.float32)
    cand = (lmax > 0) & (Ps < t_ridge)
    out = np.zeros((H, W), bool)
    if not cand.any():
        return out
    r, c = np.where(cand)
    p0 = Ps[r, c]
    pp = map_coordinates(Ps, [rr[r, c] + vr[r, c], cc[r, c] + vc[r, c]], order=1, mode="nearest")
    pm = map_coordinates(Ps, [rr[r, c] - vr[r, c], cc[r, c] - vc[r, c]], order=1, mode="nearest")
    ismin = (p0 <= pp) & (p0 <= pm)
    side_p = np.zeros(len(r), bool)
    side_m = np.zeros(len(r), bool)
    for d in reach:
        side_p |= map_coordinates(Ps, [rr[r, c] + d * vr[r, c], cc[r, c] + d * vc[r, c]],
                                  order=1, mode="nearest") >= t_ridge
        side_m |= map_coordinates(Ps, [rr[r, c] - d * vr[r, c], cc[r, c] - d * vc[r, c]],
                                  order=1, mode="nearest") >= t_ridge
    keep = ismin & side_p & side_m
    out[r[keep], c[keep]] = True
    return out


def hysteresis(prob: np.ndarray, hi: float, lo: float) -> np.ndarray:
    from scipy.ndimage import label as _label
    low = prob > lo
    lab, n = _label(low, structure=np.ones((3, 3), int))
    if n == 0:
        return low
    seeds = np.unique(lab[(prob > hi) & low])
    keep = np.zeros(n + 1, bool)
    keep[seeds[seeds > 0]] = True
    return keep[lab]


def prune_spurs(ridge: np.ndarray, min_len: int = 8) -> np.ndarray:
    """Remove ridge branches shorter than ``min_len`` px that end in a free tip (NMS spurs on the
    saturated plateau); iterated so a spur of a spur goes too. Loops and long branches survive."""
    from scipy.ndimage import convolve, label as cc_label
    r = ridge.copy()
    k = np.ones((3, 3), int); k[1, 1] = 0
    for _ in range(3):
        nb = convolve(r.astype(int), k, mode="constant")
        tips = r & (nb <= 1)
        if not tips.any():
            break
        # branch pixels = degree <= 2 pixels; label them, drop components that touch a tip and are short
        branch = r & (nb <= 2)
        lab, n = cc_label(branch, structure=np.ones((3, 3)))
        if n == 0:
            break
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        tip_lab = np.unique(lab[tips]); tip_lab = tip_lab[tip_lab > 0]
        drop = np.zeros(n + 1, bool); drop[tip_lab[sizes[tip_lab] < min_len]] = True
        if not drop.any():
            break
        r = r & ~drop[lab]
    return r


def ridge_hysteresis(prob: np.ndarray, lo: float, hi: float, t_valley: float = 0.9, min_spur: int = 8) -> np.ndarray:
    """Hysteresis ON A 1-PX RIDGE SET: the skeleton of the low-threshold mask with the VALLEY pixels
    between two ridges removed first, kept as a whole 8-connected component iff it holds a pixel
    >= hi. The closed 2-D hysteresis computed connectivity BEFORE any valley cut, so two parallels
    joined through a 0.35 valley were one component; here the valley is cut first, so a 0.9 dip
    inside a 0.999 filament is bridged while a parallel pair stays two components. (Plain NMS
    ridges were tried for this role and rejected: 8 205 branch points on the saturated plateau of
    roi_img_119 vs 65 for the skeleton.)"""
    from scipy.ndimage import label as cc_label
    low = (prob > lo) & ~valley_cut(prob, t_valley)
    ridge = prune_spurs(skeletonize(low), min_spur)
    lab, n = cc_label(ridge, structure=np.ones((3, 3)))
    if n == 0:
        return ridge
    keep = np.zeros(n + 1, bool)
    keep[np.unique(lab[ridge & (prob >= hi)])] = True
    keep[0] = False
    return keep[lab]


def build_masks(prob: np.ndarray, variants: list[str], hi_thr: float = 0.98,
                heat: np.ndarray | None = None) -> dict:
    """Variant name -> binary mask. Ridges are computed once per existence level.

    ``heat`` is the second channel of a ``--head heat`` dump (a Gaussian centerline heatmap); the
    G variants ridge-NMS THAT map and gate it with the existence map ``prob``:
      G(t, tx)  -- dilate3(NMS(heat >= t) & (prob > tx))
      GU(t, tx) -- dilate3((NMS(heat >= t) & (prob > tx)) | skeleton(prob > 0.98))
    A Gaussian peak has no saturated plateau, so the spurs that sank plain NMS on the binary map
    (789 vs 65 branch points on roi_img_119) should not recur -- that is the HEAT arm's claim.
    """
    out = {}
    ridges: dict[float, np.ndarray] = {}
    hridges: dict[float, np.ndarray] = {}
    skel_hi = None
    d3 = np.ones((3, 3), bool)

    def ridge(t):
        if t not in ridges:
            ridges[t] = ridge_nms(prob, t)
        return ridges[t]

    def hridge(t):
        if heat is None:
            raise ValueError("G variants need a 2-channel (heat) dump")
        if t not in hridges:
            hridges[t] = ridge_nms(heat, t)
        return hridges[t]

    for v in variants:
        kind, rest = v[0], v[1:].strip("()")
        parts = [float(x) for x in rest.split(",")] if (rest and not v[:2] in ("GU", "GP", "GQ", "RH", "RO")) else []
        if kind == "B":
            out[v] = prob > parts[0]
        elif kind == "N":
            out[v] = binary_dilation(ridge(parts[0]), structure=d3)
        elif kind == "U":
            if skel_hi is None:
                skel_hi = skeletonize(prob > hi_thr)
            out[v] = binary_dilation(ridge(parts[0]) | skel_hi, structure=d3)
        elif kind == "W":
            t, r = parts[0], int(parts[1])
            out[v] = (prob > t) & binary_dilation(ridge(t), structure=_disk(r))
        elif kind == "V":                      # low threshold, saddles between ridges cut out
            t, tr_ = parts[0], (parts[1] if len(parts) > 1 else max(0.9, parts[0]))
            out[v] = (prob > t) & ~valley_cut(prob, tr_)
        elif kind == "H":                      # hysteresis (existence) + valley cut (separation)
            t, tr_ = parts[0], (parts[1] if len(parts) > 1 else 0.98)
            out[v] = hysteresis(prob, hi_thr, t) & ~valley_cut(prob, tr_)
        elif v.startswith("RH"):                 # ridge-set hysteresis, union with the >= hi mask
            pr = [float(x) for x in v[2:].strip("()").split(",")]
            lo, hi, tv = pr[0], pr[1], (pr[2] if len(pr) > 2 else 0.9)
            out[v] = binary_dilation(ridge_hysteresis(prob, lo, hi, tv), structure=d3) | (prob > hi)
        elif v.startswith("RO"):                 # ridge-set hysteresis alone (no union)
            pr = [float(x) for x in v[2:].strip("()").split(",")]
            lo, hi, tv = pr[0], pr[1], (pr[2] if len(pr) > 2 else 0.9)
            out[v] = binary_dilation(ridge_hysteresis(prob, lo, hi, tv), structure=d3)
        elif v.startswith("GU"):
            t, tx = [float(x) for x in v[2:].strip("()").split(",")]
            if skel_hi is None:
                skel_hi = skeletonize(prob > hi_thr)
            out[v] = binary_dilation((hridge(t) & (prob > tx)) | skel_hi, structure=d3)
        elif v.startswith("GQ"):                 # GP + precision gate: ridge components with mean existence < q dropped
            pr = [float(x) for x in v[2:].strip("()").split(",")]
            t, tx, ml, q = pr[0], pr[1], int(pr[2]), pr[3]
            r = prune_spurs(skeletonize(hridge(t) & (prob > tx)), ml)
            lab, n = cc_label(r, structure=np.ones((3, 3)))
            if n:
                sums = np.bincount(lab.ravel(), weights=prob.ravel(), minlength=n + 1)
                cnt = np.bincount(lab.ravel(), minlength=n + 1).astype(float)
                keep = np.zeros(n + 1, bool); keep[1:] = (sums[1:] / np.maximum(cnt[1:], 1)) >= q
                r = keep[lab]
            out[v] = binary_dilation(r, structure=d3)
        elif v.startswith("GP"):                 # heat ridge, gated, SPUR-PRUNED (exploratory, 2026-09-06)
            pr = [float(x) for x in v[2:].strip("()").split(",")]
            t, tx, ml = pr[0], pr[1], (int(pr[2]) if len(pr) > 2 else 8)
            out[v] = binary_dilation(prune_spurs(skeletonize(hridge(t) & (prob > tx)), ml), structure=d3)
        elif kind == "G":
            t, tx = parts[0], parts[1]
            out[v] = binary_dilation(hridge(t) & (prob > tx), structure=d3)
        else:
            raise ValueError(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/roi303_eval")
    ap.add_argument("--split", default="val")
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--params", default="src/instance/params_a_model_v5H.json")
    ap.add_argument("--variants", default="B(0.98),B(0.99),N(0.3),N(0.5),N(0.7),U(0.3),U(0.5),W(0.3,2),W(0.5,2),W(0.5,3)")
    ap.add_argument("--frames", default="")
    ap.add_argument("--crossings", action="store_true", help="also junction identity + bundle recovery")
    ap.add_argument("--channel", type=int, default=0,
                    help="channel of the dump used as the probability map for B/N/U/W/V/H (0 = the "
                         "existence head; a 1-channel dump has only 0). Channel 1 of a 2-channel "
                         "dump is the heatmap, used by the G variants.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.split == "test":
        raise SystemExit("refusing to touch TEST")
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    # "W(0.5,2)" contains a comma: re-join tokens that were split inside parentheses
    fixed, buf = [], ""
    for tok in variants:
        buf = tok if not buf else buf + "," + tok
        if buf.count("(") == buf.count(")"):
            fixed.append(buf); buf = ""
    variants = fixed
    params = json.load(open(args.params))
    params.pop("kappa_max", None); params.pop("prob_thr", None)
    want = set(args.frames.split(",")) if args.frames else None
    if args.crossings:
        from run_oracle_eval import parallel_pairs

    rows = {v: [] for v in variants}
    for p in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        stem = os.path.basename(p)[:-3]
        if want and stem not in want:
            continue
        npz = os.path.join(args.pred_dir, stem + ".npz")
        if not os.path.exists(npz):
            continue
        fr = read_frame_h5(p)
        if str(fr["attrs"].get("split")) != args.split:
            continue
        shape = (int(fr["attrs"]["height"]), int(fr["attrs"]["width"]))
        hi = (int(round(shape[0] * UP)), int(round(shape[1] * UP)))
        gt = [np.asarray(q, float) * UP for q in fr["polylines"]]
        allch = np.load(npz)["prob"].astype(np.float32)
        prob = allch[min(args.channel, allch.shape[0] - 1)]
        heat = allch[1] if allch.shape[0] > 1 else None
        t0 = time.time()
        masks = build_masks(prob, variants, heat=heat)
        cr = count_crossings(gt, tol=2.0 * UP, endpoint_margin=8.0 * UP) if (args.crossings and len(gt) > 1) else []
        pairs = parallel_pairs(gt) if (args.crossings and len(gt) > 1) else []
        for v, m in masks.items():
            polys, _ = instance_a(m, KAPPA_MAX, params, channels=prob[None], prob=prob)
            im = oracle_instance_masks(polys, hi, half_width=1.0, up=1.0)
            r = centerline_f1(im, gt, tol=5.0, length_coverage=0.95, precision_coverage=0.95) if gt else \
                {"tp": 0, "fp": len(polys), "fn": 0, "f1": float("nan")}
            row = {"name": stem, "tp": int(r["tp"]), "fp": int(r["fp"]), "fn": int(r["fn"]),
                   "f1": float(r["f1"]), "n_gt": len(gt), "fg": float(m.mean()), "n_pred": len(polys)}
            if args.crossings and len(gt) > 1:
                ji = junction_identity(im, gt, cr)
                row["ji_n"] = ji["n_crossings"]; row["ji_pres"] = ji["n_preserved"]
                br = bundle_recovery(im, gt, pairs) if pairs else float("nan")
                row["bundle"] = None if br != br else float(br); row["n_pairs"] = len(pairs)
            rows[v].append(row)
        print(f"{stem}: nGT={len(gt)} " + " ".join(f"{v}={rows[v][-1]['f1']:.3f}" for v in variants)
              + f"  ({time.time() - t0:.0f}s)", flush=True)
    with open(args.out, "w") as fh:
        json.dump({"variants": variants, "pred_dir": args.pred_dir, "rows": rows}, fh, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
