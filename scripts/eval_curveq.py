#!/usr/bin/env python3
"""Score a CURVEQ model the way the project is judged, with NO instancer in the loop.

The point of an overlap-permitting emission is that it produces instances directly, so this script
takes the accepted queries, trims each to its own mask support, and hands the polylines straight to
`instance.metrics.centerline_f1`. Nothing is skeletonised, nothing is thresholded into a binary map,
and instancer A never runs — which is exactly what makes the comparison meaningful.

THREE COMPARATORS ARE PRODUCED IN THE SAME PASS, because a curve emission that cannot beat the
dense map sharing its own trunk has earned nothing:

  curveq   the query emission, decoded as above
  dense    the SAME model's retained binary head, through instancer A with the declared params --
           this is the honest baseline-equivalent, and it isolates what the queries added
  (the shipped SPARSE35 numbers are the external comparator and live in docs/TODO.md)

The QUERY-SIDE OVER-FIRING GATE is reported per frame and pooled: accepted curves and total emitted
centerline length per megapixel. The trainer's dense gate reads a foreground FRACTION, which a curve
emission does not have, so without this a runaway query model would look fine.

    PYTHONPATH=.:src:scripts python scripts/eval_curveq.py --ckpt runs/CQ/ep040.pth \\
        --data data/real/roi303_eval --split val --cov 0.80 --out cq_roi303.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from instance.curveq import CurveQNet, decode_frame, overfire_stats  # noqa: E402
from instance.instancer_a import instance_a  # noqa: E402
from instance.metrics import aggregate_benchmark, centerline_f1  # noqa: E402
from instance.oracle import oracle_instance_masks  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402

UP = 1.5      # the eval frame: every number in this project is quoted at 1.5x


def source_of(name: str) -> str:
    if name.startswith("htw_"):
        return "htw"
    if name.startswith("alice_"):
        return "alice"
    if name.startswith("training_img_"):
        return "586"
    return "roi303"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--cov", type=float, default=0.80,
                    help="the headline convention since 2026-09-08; 0.95 is the strict secondary")
    ap.add_argument("--tol", type=float, default=5.0)
    ap.add_argument("--end-tol", type=float, default=0.0)
    ap.add_argument("--infer-scale", type=float, default=1.0,
                    help="NATIVE by default: the network runs at 1x and the curves are scaled to "
                         "the 1.5x eval frame afterwards, which is the path every 2026-09 number "
                         "used (docs/TODO.md, the scale factor read of 2026-09-03)")
    ap.add_argument("--tile", type=int, default=0,
                    help="0 = WHOLE FRAME, no seams, no cross-tile merge -- the configuration in "
                         "which 'this emission needs no assembly step' is true as stated. A "
                         "positive value falls back to tiles and MERGES curves across seams, which "
                         "IS an assembly step and is reported as such in the output JSON.")
    ap.add_argument("--obj-thrs", default="0.05,0.08,0.10,0.15,0.20,0.30,0.40,0.50,0.60,0.70",
                    help="objectness grid for CURVEQ. BOTH arms are swept and each is reported at "
                         "its OWN best: the project's standing rule is that a shared threshold "
                         "measures calibration rather than quality and has already faked a 0.47 F1 "
                         "gap here. A winner on the edge of its grid is flagged UNRELIABLE. "
                         "GRID WIDENED DOWN TO 0.05 on 2026-09-14, declared before any result "
                         "existed: measured at run 5 epoch 8, objectness sits at the base rate "
                         "(p50 0.052 against 18 GT / 300 queries = 0.060) while still RANKING "
                         "usefully -- at 0.10 it emits 10 curves for 18 GT. A grid starting at "
                         "0.30 would have scored zero and called it a failure of the method when "
                         "it is a failure of calibration, which is the exact error the dense "
                         "comparator was nearly scored with.")
    ap.add_argument("--mask-thr", type=float, default=0.5)
    ap.add_argument("--min-support", type=float, default=0.5)
    ap.add_argument("--cq-queries", type=int, default=300)
    ap.add_argument("--cq-dim", type=int, default=128)
    ap.add_argument("--cq-layers", type=int, default=6)
    ap.add_argument("--dense-params", default="src/instance/params_a_derived.json",
                    help="the DERIVED min_length (15.0, step 0 of 2026-09-13), not the "
                         "fitted 44.74. Fixing that constant after a new method is "
                         "measured would credit +0.093 of free baseline gain to the "
                         "new method; it is fixed BEFORE, for both arms.")
    ap.add_argument("--dense-thrs", default="0.50,0.70,0.80,0.90,0.95,0.97,0.98,0.99",
                    help="probability grid for the DENSE comparator, swept for the same reason. "
                         "The default reaches down to 0.50 deliberately: this model's dense head "
                         "was measured to peak at 0.777, so a fixed 0.98 would score it ZERO and "
                         "hand CURVEQ a win it did not earn.")
    ap.add_argument("--no-dense", action="store_true",
                    help="skip the dense comparator (it costs an instancer run per frame)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import train_v5 as T

    sd = torch.load(args.ckpt, map_location="cpu")
    # READ the trunk's output width from the checkpoint instead of assuming it. A curveq trunk's
    # dense head is binary, but the arm launched 2026-09-13 ran before that was fixed in the
    # trainer and carries 7 output channels of which 6 never received gradient. Hard-coding either
    # number here would make one of the two unscoreable, and guessing is how a shape mismatch turns
    # into a silently half-loaded model.
    _w = [v for k, v in sd.items() if k.endswith("trunk.decoder.seg_layers.0.weight")]
    trunk_out = int(_w[0].shape[0]) if _w else 1
    if trunk_out != 1:
        print(f"note: this checkpoint's trunk has {trunk_out} output channels; only channel 0 was "
              f"supervised, the rest are dead weights in the final 1x1 conv and do not affect the "
              f"decoder feature the query head reads", flush=True)
    trunk = T.build_nnunet(trunk_out, in_ch=3)
    net = CurveQNet(trunk, CurveQNet.feat_dim_of(trunk), n_queries=args.cq_queries,
                    dim=args.cq_dim, n_layers=args.cq_layers)
    missing = net.load_state_dict(sd, strict=False)
    if getattr(missing, "missing_keys", None):
        raise SystemExit(f"{len(missing.missing_keys)} tensors missing from the checkpoint "
                         f"(e.g. {missing.missing_keys[:3]}) -- the head geometry does not match "
                         f"the flags; refusing to score a half-loaded model.")
    net = net.to(T.DEV).eval()

    dense_params = json.load(open(args.dense_params)) if not args.no_dense else None
    obj_grid = [float(v) for v in args.obj_thrs.split(",") if v.strip()]
    dense_grid = [float(v) for v in args.dense_thrs.split(",") if v.strip()]
    rows_cq: dict[float, dict] = {}      # obj_thr -> source -> rows
    rows_dn: dict[float, dict] = {}      # dense_thr -> source -> rows
    gate = []

    for path in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        name = os.path.basename(path)
        fr = read_frame_h5(path)
        if args.split != "all" and str(fr["attrs"].get("split", "")) != args.split:
            continue
        src = source_of(name)
        im01 = T.norm01(np.asarray(fr["image"], float))
        if abs(args.infer_scale - 1.0) > 1e-9:
            from scipy.ndimage import zoom
            im01 = zoom(im01, args.infer_scale, order=1)
        gt = [np.asarray(p, float) * UP for p in fr["polylines"]]
        hi = (int(round(fr["image"].shape[0] * UP)), int(round(fr["image"].shape[1] * UP)))

        k = UP / max(args.infer_scale, 1e-9)
        for ot in obj_grid:
            curves = decode_frame(net, im01, device=T.DEV, tile=args.tile, obj_thr=ot,
                                  mask_thr=args.mask_thr, min_support=args.min_support,
                                  ima_m=T.IMA_M, ima_s=T.IMA_S)
            curves = [c * k for c in curves]          # network ran at infer_scale; metric is at UP
            # ONE MASK PER CURVE, not one map: `centerline_f1` takes instance masks, and giving
            # each query its own array is what keeps the emission overlap-permitting all the way
            # to the metric. Flattening them into a single map would silently undo the design.
            cm = oracle_instance_masks(curves, hi, half_width=1.0, up=1.0) if curves else []
            r = centerline_f1(cm, gt, tol=args.tol, length_coverage=args.cov,
                              precision_coverage=args.cov, end_tol=args.end_tol)
            rows_cq.setdefault(ot, {}).setdefault(src, []).append(
                {"name": name, "n_gt": len(gt), **r})
            g = overfire_stats(curves, hi)
            g.update(name=name, n_gt=len(gt), obj_thr=ot)
            gate.append(g)

        if dense_params is not None:
            with torch.no_grad():
                ch = T.predict(net, im01, T.TILE_FOR["nnunet"])
            from scipy.ndimage import zoom as _z
            prob = _z(ch[0], k, order=1)
            prob = np.pad(prob, ((0, max(0, hi[0] - prob.shape[0])),
                                 (0, max(0, hi[1] - prob.shape[1]))))[:hi[0], :hi[1]]
            for dt in dense_grid:
                pls, _ = instance_a(prob > dt, 0.25, dense_params,
                                    channels=prob[None], prob=prob)
                rd = centerline_f1(pls, gt, tol=args.tol, length_coverage=args.cov,
                                   precision_coverage=args.cov, end_tol=args.end_tol)
                rows_dn.setdefault(dt, {}).setdefault(src, []).append(
                    {"name": name, "n_gt": len(gt), **rd})

    def sweep(rows):
        """threshold -> source -> aggregate, plus the best threshold per source and an edge flag."""
        agg = {f"{t:g}": {src: aggregate_benchmark(rr) for src, rr in by.items()}
               for t, by in rows.items()}
        best = {}
        if rows:
            grid = sorted(rows)
            srcs = {s for by in rows.values() for s in by}
            for src in srcs:
                cand = [(agg[f"{t:g}"][src]["mean_f1"], t) for t in grid
                        if src in agg[f"{t:g}"]]
                if not cand:
                    continue
                f1, t = max(cand)
                best[src] = {"thr": t, **agg[f"{t:g}"][src],
                             # the project's own rule: a winner on the edge of its grid means the
                             # optimum is outside it, so the number is not the arm's best and must
                             # not be quoted as one.
                             "edge_winner_UNRELIABLE": bool(t in (grid[0], grid[-1]))}
        return agg, best

    cq_agg, cq_best = sweep(rows_cq)
    dn_agg, dn_best = sweep(rows_dn)
    out = {"ckpt": args.ckpt, "data": args.data, "split": args.split, "cov": args.cov,
           "tol": args.tol, "end_tol": args.end_tol, "infer_scale": args.infer_scale,
           "tile": args.tile,
           "assembly_free": args.tile <= 0,
           "decode": {"obj_thrs": obj_grid, "mask_thr": args.mask_thr,
                      "min_support": args.min_support, "dense_thrs": dense_grid},
           "gate": gate,
           "curveq_sweep": cq_agg, "curveq": cq_best,
           "dense_sweep": dn_agg, "dense": dn_best}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)

    print(f"\n=== CURVEQ {'WHOLE-FRAME (assembly-free)' if args.tile <= 0 else 'TILED + MERGED '
          '(contains an assembly step)'} | cov {args.cov} tol {args.tol} end_tol {args.end_tol}")
    print("  EACH ARM AT ITS OWN BEST THRESHOLD (a shared threshold measures calibration, "
          "not quality)")
    for src in sorted(set(list(cq_best) + list(dn_best))):
        for tag, d in (("curveq", cq_best.get(src)), ("dense ", dn_best.get(src))):
            if not d:
                continue
            flag = "  <-- EDGE WINNER, UNRELIABLE" if d["edge_winner_UNRELIABLE"] else ""
            print(f"  {src:<8} {tag} thr {d['thr']:<6g} macro {d['mean_f1']:.4f}  "
                  f"micro {d['micro_f1']:.4f}  TP {d['total_tp']} FP {d['total_fp']} "
                  f"FN {d['total_fn']}{flag}")
        a, b = cq_best.get(src), dn_best.get(src)
        if a and b:
            print(f"  {src:<8} PRIMARY  d(macro) {a['mean_f1'] - b['mean_f1']:+.4f}   "
                  f"d(micro) {a['micro_f1'] - b['micro_f1']:+.4f}   "
                  f"(promotion needs both > +0.01)")
    if gate:
        for ot in obj_grid:
            gg = [g for g in gate if g["obj_thr"] == ot]
            if not gg:
                continue
            nc = np.array([g["n_curves"] for g in gg], float)
            ng = np.array([g["n_gt"] for g in gg], float)
            lp = np.array([g["len_per_Mpx"] for g in gg], float)
            ratio = np.median(nc) / max(np.median(ng), 1)
            print(f"  OVER-FIRING GATE @obj {ot:g}: curves/frame p50 {np.median(nc):.0f} "
                  f"(GT p50 {np.median(ng):.0f}, ratio {ratio:.2f}"
                  f"{'  <-- FALSIFIER: >3x GT' if ratio > 3 else ''}) | "
                  f"emitted px per Mpx p50 {np.median(lp):.0f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
