#!/usr/bin/env python3
"""Phase 0 of the F1>=0.90 push: what does the pipeline look like with NO length filter,
and how much of the benchmark is unreachable because the ground truth is incomplete?

Two questions, one pass over the cached v5H predictions, no GPU and no training:

  0b  The no-post-processing baseline. ``min_length`` is a fitted constant that deletes 49 %
      of the HTW ground truth (measured 2026-08-27); the plan is to remove it. Removing it
      today converts every mask fragment into a false positive, and THAT number is the debt a
      new architecture has to pay off. Swept over prob_thr x min_length, per source, because
      a shared threshold across configurations measures calibration rather than quality
      (this project has been bitten by exactly that).

  0a  The benchmark ceiling. Predictions that the metric calls false positives are of two
      kinds: fragments of annotated microtubules, and detections sitting on real filaments
      the annotator never drew. Only the second kind is a ceiling -- it can never score
      correct. They are told apart by image contrast under the predicted centerline, against
      a NULL built by rigidly displacing the same curves within the same frame.

Everything runs in the 1.5x eval frame, matching ``run_oracle_eval.py``. Numbers are VAL
only; TEST carries an exposure budget and is not touched here.

    python scripts/phase0_nofilter_audit.py --pred-dir preds/v5H_combined_val \
        --data-dir data/real/mt34_htw_eval --split val --out phase0_val.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.ndimage import zoom
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from instance.instancer_a import instance_a  # noqa: E402
from instance.metrics import aggregate_benchmark, centerline_f1  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.fov import fov_mask  # noqa: E402

UP = 1.5
KAPPA_MAX = 0.25


def source_of(path: str, attrs: dict) -> str:
    """Which of the three benchmark blocks a frame belongs to."""
    if os.path.basename(path).startswith("htw"):
        return "htw"
    return "585" if str(attrs.get("source_task")) == "585" else "586"


def densify(poly: np.ndarray, step: float = 1.0) -> np.ndarray:
    """Resample a polyline to ~``step`` px spacing. Vertices are (x=col, y=row)."""
    if len(poly) < 2:
        return poly
    seg = np.sqrt(((poly[1:] - poly[:-1]) ** 2).sum(1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return poly
    t = np.arange(0.0, s[-1], step)
    return np.stack([np.interp(t, s, poly[:, 0]), np.interp(t, s, poly[:, 1])], axis=1)


def sample_contrast(img: np.ndarray, poly_native: np.ndarray, dark: bool) -> float:
    """Mean |filament - local background| under a centerline, in native image pixels.

    The background is taken from +-7 px along the curve normal, so the estimate is local and
    survives the uneven illumination these frames have.
    """
    q = densify(poly_native)
    if len(q) < 3:
        return float("nan")
    tang = np.gradient(q, axis=0)
    nrm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)

    H, W = img.shape

    def _sample(pts):
        # Test the ROUNDED index, not the float: 1119.6 is "inside" a 1120-wide frame but
        # rounds to 1120, which is out of bounds. Clipping instead would silently sample the
        # wrong pixel, which is worse than dropping the point.
        c = np.round(pts[:, 0]).astype(int)
        r = np.round(pts[:, 1]).astype(int)
        inside = (c >= 0) & (c < W) & (r >= 0) & (r < H)
        if inside.sum() < 3:
            return None
        return img[r[inside], c[inside]]

    core = _sample(q)
    if core is None:
        return float("nan")
    flanks = [_sample(q + nrm * o) for o in (-7.0, 7.0)]
    flanks = [f for f in flanks if f is not None]
    if not flanks:
        return float("nan")
    bg = float(np.mean([f.mean() for f in flanks]))
    val = float(core.mean()) - bg
    return -val if dark else val


def displaced_null(polys_native, shape, rng) -> list[np.ndarray]:
    """The same curves, each rigidly moved elsewhere in the same frame.

    This is the control: it carries the frame's own texture and illumination but is not
    aligned to anything real, so contrast above it is evidence of a filament.
    """
    H, W = shape
    out = []
    for p in polys_native:
        for _ in range(12):
            dx = rng.uniform(-0.45, 0.45) * W
            dy = rng.uniform(-0.45, 0.45) * H
            q = p + np.array([dx, dy])
            if (q[:, 0] >= 0).all() and (q[:, 0] < W).all() and \
               (q[:, 1] >= 0).all() and (q[:, 1] < H).all():
                out.append(q)
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True,
                    help="directory of per-frame .npz predictions (key 'prob')")
    ap.add_argument("--data-dir", default="data/real/mt34_htw_eval")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--params", default="src/instance/params_a_model_v5H.json")
    ap.add_argument("--prob-thr", type=float, nargs="+", default=[0.90, 0.95, 0.97, 0.99])
    ap.add_argument("--min-length", type=float, nargs="+", default=[44.74, 20.0, 3.0])
    ap.add_argument("--link-max-gap", type=float, nargs="+", default=[None],
                    help="sweep the gap-linking radius too; None keeps the params file value")
    ap.add_argument("--audit-config", type=float, nargs=2, metavar=("PROB_THR", "MIN_LENGTH"),
                    default=[0.99, 3.0],
                    help="the (prob_thr, min_length) whose false positives get the contrast audit")
    ap.add_argument("--limit", type=int, default=0, help="smoke-test on the first N frames")
    ap.add_argument("--out", default="phase0_audit.json")
    args = ap.parse_args()

    if args.split == "test":
        raise SystemExit("refusing to touch TEST: it carries a declared exposure budget")

    base = json.load(open(args.params))
    paths = sorted(glob.glob(os.path.join(args.data_dir, "*.h5")))
    frames = []
    for path in paths:
        fr = read_frame_h5(path)
        if str(fr["attrs"].get("split")) != args.split:
            continue
        frames.append((path, fr))
    if args.limit:
        frames = frames[:args.limit]
    if not frames:
        raise SystemExit(f"no {args.split} frames under {args.data_dir}")

    # ---- load every prediction once; the sweep then only re-thresholds ------------------
    loaded = []
    for path, fr in frames:
        name = os.path.basename(path).replace(".h5", ".npz")
        npz_path = os.path.join(args.pred_dir, name)
        if not os.path.isfile(npz_path):
            # A zero-GT frame is legitimately absent: eval_v5.py drops it unless --include-empty.
            # It contributes no true positives and is excluded from the macro mean anyway, so
            # skipping it is sound -- but say so, because a SILENT skip of a real frame would
            # change the denominator. Anything else stays fatal.
            if not [q for q in fr["polylines"] if len(q) >= 2]:
                print(f"  SKIP {name}: zero-GT frame, not in the prediction dump "
                      f"(excluded from the macro mean; its false positives are not counted)",
                      flush=True)
                continue
            raise SystemExit(f"missing prediction {npz_path} for a frame that HAS ground truth "
                             f"-- a silently skipped frame would change the denominator")
        prob = np.load(npz_path)["prob"].astype(np.float32)
        if prob.ndim == 3:
            prob = prob.max(axis=0)
        img = fr["image"]
        hi = (int(round(img.shape[0] * UP)), int(round(img.shape[1] * UP)))
        if prob.shape != hi:
            raise SystemExit(f"{name}: prediction {prob.shape} != eval frame {hi}")
        fv = zoom(fov_mask(img).astype(np.uint8), UP, order=0).astype(bool)
        fv = np.pad(fv, ((0, max(0, hi[0] - fv.shape[0])), (0, max(0, hi[1] - fv.shape[1]))),
                    constant_values=True)[:hi[0], :hi[1]]
        gt = [np.asarray(p, float) * UP for p in fr["polylines"] if len(p) >= 2]
        loaded.append(dict(path=path, name=name, img=img, prob=prob, fov=fv, gt=gt,
                           src=source_of(path, fr["attrs"])))
        print(f"  loaded {name} ({loaded[-1]['src']}, {len(gt)} GT)", flush=True)

    # ---- 0b: the sweep -----------------------------------------------------------------
    sweep = {}
    audit_thr, audit_ml = args.audit_config
    audit_rows = []
    for thr in args.prob_thr:
      for lg in args.link_max_gap:
        for ml in args.min_length:
            params = dict(base)
            params["min_length"] = ml
            params["prob_thr"] = thr
            if lg is not None:
                params["link_max_gap"] = lg
            per_src = defaultdict(list)
            for L in loaded:
                mask = (L["prob"] > thr) & L["fov"]
                # Match eval_v5.py exactly: v5H has a binary head, so `channels` is the single
                # probability plane. Calling instance_a differently from the shipped pipeline
                # would make this baseline incomparable to the recorded 0.3847.
                polys, masks = instance_a(mask, KAPPA_MAX, params=params,
                                          channels=L["prob"][None], prob=L["prob"])
                res = centerline_f1(masks, L["gt"], tol=5.0,
                                    length_coverage=0.95, precision_coverage=0.95)
                res["n_gt"] = len(L["gt"])          # aggregate_benchmark needs this
                per_src[L["src"]].append(res)

                if thr == audit_thr and ml == audit_ml:
                    matched = {p["pred"] for p in res["pairs"] if p["tp"]}
                    audit_rows.append((L, [polys[i] for i in range(len(polys))
                                           if i not in matched],
                                       [polys[i] for i in sorted(matched)]))
            key = f"thr{thr}_ml{ml}_lg{lg}"
            # mean_f1 is the headline: the recorded 0.3847 / 0.4091 are aggregate_benchmark's
            # macro mean over non-empty frames, NOT micro-F1. Reporting micro next to them
            # would compare two different harnesses.
            sweep[key] = {s: aggregate_benchmark(rows) for s, rows in per_src.items()}
            allrows = [r for rows in per_src.values() for r in rows]
            pool = aggregate_benchmark(allrows)
            sweep[key]["POOL"] = pool
            line = "  ".join(f"{s}={sweep[key][s]['mean_f1']:.3f}"
                             for s in ("585", "586", "htw") if s in per_src)
            print(f"thr={thr:<6} lg={str(lg):<6} ml={ml:<5} POOL mean_f1={pool['mean_f1']:.4f} "
                  f"(micro {pool['micro_f1']:.4f}, TP {pool['total_tp']} FP {pool['total_fp']} "
                  f"FN {pool['total_fn']})   {line}", flush=True)

    # ---- 0a: are the false positives real filaments? -----------------------------------
    rng = np.random.default_rng(20260827)
    contrast = defaultdict(lambda: defaultdict(list))
    for L, fp_polys, tp_polys in audit_rows:
        img = L["img"].astype(np.float64)
        gt_native = [g / UP for g in L["gt"]]
        # polarity from the ground truth: IRM flips between dark and bright filaments
        gt_c = [sample_contrast(img, g, dark=True) for g in gt_native]
        gt_c = [c for c in gt_c if np.isfinite(c)]
        dark = bool(np.median(gt_c) > 0) if gt_c else True
        # A false positive that lies ON an annotated microtubule is a FRAGMENT -- a failure of
        # assembly, and fixable. One that lies far from every annotation is a detection on a
        # filament the annotator never drew, which can never score correct and is therefore a
        # ceiling on the benchmark. Contrast alone cannot tell them apart: both sit on real
        # microtubule signal. Distance to the nearest ground-truth centerline can.
        gt_pts = np.concatenate([densify(g) for g in gt_native], axis=0) if gt_native else None
        tree = cKDTree(gt_pts) if gt_pts is not None and len(gt_pts) else None

        for tag, polys in (("gt", gt_native),
                           ("fp", [p / UP for p in fp_polys]),
                           ("tp", [p / UP for p in tp_polys])):
            for p in polys:
                c = sample_contrast(img, p, dark)
                if not np.isfinite(c):
                    continue
                contrast[L["src"]][tag].append(float(c))
                if tag == "fp" and tree is not None:
                    q = densify(p)
                    d = tree.query(q, k=1)[0]
                    # median distance along the FP, so a mostly-on-GT fragment counts as near
                    near = float(np.median(d)) <= 5.0
                    contrast[L["src"]]["fp_near" if near else "fp_far"].append(float(c))
                    contrast[L["src"]]["fp_dist"].append(float(np.median(d)))
        for p in displaced_null([q / UP for q in fp_polys], img.shape, rng):
            c = sample_contrast(img, p, dark)
            if np.isfinite(c):
                contrast[L["src"]]["null"].append(float(c))

    print()
    print(f"CONTRAST AUDIT at prob_thr={audit_thr}, min_length={audit_ml}  (VAL only)")
    print("A false positive NEAR a GT centerline (median distance <= 5 px) is a fragment of an")
    print("annotated microtubule -- an assembly failure. One FAR from every annotation, but")
    print("carrying real contrast, is a detection on an unannotated filament: a BENCHMARK CEILING.")
    print()
    print(f'{"src":5s} {"n FP":>6s} {"near(frag)":>11s} {"far":>6s} '
          f'{"far & real":>11s} {"CEILING F1":>11s} | '
          f'{"c(GT)":>9s} {"c(FP far)":>10s} {"c(null)":>9s} {"far/null":>9s}')
    ceiling = {}
    for s in ("585", "586", "htw"):
        d = contrast[s]
        if not d["fp"]:
            continue
        nul = np.array(d["null"]) if d["null"] else np.array([np.nan])
        cut = np.nanpercentile(nul, 95)
        near = np.array(d["fp_near"]) if d["fp_near"] else np.array([])
        far = np.array(d["fp_far"]) if d["fp_far"] else np.array([])
        far_real = int((far > cut).sum()) if len(far) else 0
        n_fp = len(d["fp"])
        # Upper bound on F1: the far-and-real detections can never be scored correct, so even
        # a perfect method keeps them as false positives against the full GT count.
        n_gt = sum(v for v in [len(d["gt"])])
        ceil_f1 = 2 * n_gt / max(2 * n_gt + far_real, 1)
        ceiling[s] = dict(n_fp=n_fp, n_near=len(near), n_far=len(far), n_far_real=far_real,
                          ceiling_f1=ceil_f1, n_gt=n_gt,
                          c_gt=float(np.median(d["gt"])) if d["gt"] else None,
                          c_far=float(np.median(far)) if len(far) else None,
                          c_null=float(np.nanmedian(nul)),
                          fp_dist_p50=float(np.median(d["fp_dist"])) if d["fp_dist"] else None)
        print(f'{s:5s} {n_fp:6d} {len(near):11d} {len(far):6d} {far_real:11d} {ceil_f1:11.3f} | '
              f'{np.median(d["gt"]) if d["gt"] else float("nan"):9.3f} '
              f'{np.median(far) if len(far) else float("nan"):10.3f} '
              f'{np.nanmedian(nul):9.3f} '
              f'{(np.median(far) if len(far) else np.nan) / max(abs(np.nanmedian(nul)), 1e-9):8.2f}x')

    json.dump(dict(sweep=sweep, ceiling=ceiling, split=args.split,
                   pred_dir=args.pred_dir, params=args.params,
                   audit_config=[audit_thr, audit_ml], n_frames=len(loaded)),
              open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
