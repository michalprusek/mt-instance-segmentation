#!/usr/bin/env python3
"""Read 36a: does the multiplicity map fill the RIGHT holes, or just more of them?

Read 35 measured that 61 of 155 in-domain crossing failures are a mask that breaks the filament
at the crossing, and that the break is SHALLOW -- 84 % of them keep their weaker arm above 0.9.
A plain threshold drop therefore fills the same holes, and read 31 already measured that it buys
+0.093 junction identity for 0.003 of macro F1. So the multiplicity map earns nothing unless it
fills SELECTIVELY: the same identity for less added foreground, or more identity for the same.

Three masks per frame, one dump, one instancer, paired:

  base  -- prob > thr                                     (the published operating point)
  fill  -- (prob > thr) | (overlap > t_ov)                (the arm)
  drop  -- prob > thr' with thr' chosen PER FRAME so that the added foreground matches `fill`
           to within a pixel                              (the comparator that makes it a test)

Reports centerline-F1 under both conventions, junction identity from the GT crossings, and the
added foreground each variant spends, so the arm is charged for what it costs.

    PYTHONPATH=src:scripts python scripts/read_overlap_fill.py --data <dir> --pred-dir <2-ch dump> \
        --params src/instance/params_a.json --thr 0.99 --ov-thr 0.5 --out read36a.json
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

from instance.geometry import arclength, resample  # noqa: E402
from instance.instancer_a import instance_a  # noqa: E402
from instance.metrics import _covering_instances, _mask_trees, centerline_f1  # noqa: E402
from instance.oracle import oracle_instance_masks  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.gt_stats import count_crossings  # noqa: E402

UP, KAPPA_MAX = 1.5, 0.25


def matched_threshold(prob: np.ndarray, base_thr: float, target_px: int) -> float:
    """Largest threshold whose mask adds at least `target_px` pixels over the base mask.

    Bisection on the threshold rather than a grid: the comparator must spend the SAME
    foreground the fill spends, otherwise the two are read at different operating points --
    the confound that reversed arm 29.
    """
    if target_px <= 0:
        return base_thr
    lo, hi = 0.0, base_thr
    base_n = int((prob > base_thr).sum())
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if int((prob > mid).sum()) - base_n >= target_px:
            lo = mid
        else:
            hi = mid
    return lo


def junction_identity(masks, crossings, res, lens):
    """Share of GT crossings where ONE predicted instance covers both arms of both filaments.

    Deliberately identical to `diag_crossing_map.py`: arms sampled at +-15 px of arc length,
    a crossing skipped when either filament is too short to carry both samples, coverage tested
    with `_covering_instances` at tol 5. Any drift here would make read 36a incomparable to the
    read-35 numbers it is meant to move.
    """
    trees = _mask_trees(masks)
    ok = n = 0
    for c in crossings:
        good, usable = True, True
        for ik, sk in (("i", "s_i"), ("j", "s_j")):
            k, s0 = c[ik], c[sk]
            pts, L = res[k], lens[k]
            if s0 - 15 < 0 or s0 + 15 > L:
                usable = False
                break
            pb = pts[min(int(round(s0 - 15)), len(pts) - 1)]
            pa = pts[min(int(round(s0 + 15)), len(pts) - 1)]
            good &= bool(_covering_instances(trees, pb, 5.0)
                         & _covering_instances(trees, pa, 5.0))
        if usable:
            n += 1
            ok += int(good)
    return ok, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--thr", type=float, required=True)
    ap.add_argument("--ov-thr", type=float, default=0.5)
    ap.add_argument("--ov-channel", type=int, default=1)
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--oracle-overlap", action="store_true",
                    help="build the multiplicity map from the GT instead of the network: an "
                         "UPPER BOUND on what the decoder can buy, never a result. Lets the "
                         "decoder be read on a 1-channel dump, separating its contribution "
                         "from the map's quality.")
    ap.add_argument("--passthrough", action="store_true",
                    help="also read 36b: the shared-arc pass-through, on the BASELINE mask")
    ap.add_argument("--ov-frac", type=float, default=0.5)
    ap.add_argument("--ov-min-len", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    params = json.load(open(args.params))
    params.pop("kappa_max", None); params.pop("prob_thr", None)
    params["ov_thr"] = args.ov_thr
    params["ov_frac"] = args.ov_frac
    params["ov_min_len"] = args.ov_min_len
    from instance.instancer_a import default_params as _dp
    _diff = sorted(k for k, v in params.items() if k in _dp() and _dp()[k] != v)
    if _diff:
        print(f"WARNING: --params overrides the published instancer on: {', '.join(_diff)}")

    rows = []
    for p in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        stem = os.path.basename(p)[:-3]
        npz = os.path.join(args.pred_dir, stem + ".npz")
        if not os.path.exists(npz):
            continue
        fr = read_frame_h5(p)
        if str(fr["attrs"].get("split")) != args.split:
            continue
        allch = np.load(npz)["prob"].astype(np.float32)
        if not args.oracle_overlap and allch.shape[0] <= args.ov_channel:
            raise SystemExit(f"{npz} has {allch.shape[0]} channels; need an overlap channel "
                             f"at index {args.ov_channel} -- this dump is not from --head overlap")
        prob = allch[0]
        gt = [np.asarray(q, float) * UP for q in fr["polylines"]]
        if len(gt) < 1:
            continue
        shape_hi = prob.shape
        if args.oracle_overlap:
            # multiplicity from the GT: pixels covered by two or more filaments at the same
            # half-width the metric rasterises with. An UPPER BOUND, labelled as one.
            acc = np.zeros(shape_hi, np.int16)
            for gm in oracle_instance_masks(gt, shape_hi, half_width=1.0, up=1.0):
                acc += gm.astype(np.int16)
            ov = (acc >= 2).astype(np.float32)
        else:
            ov = allch[args.ov_channel]
        base = prob > args.thr
        fill = base | (ov > args.ov_thr)
        added = int(fill.sum() - base.sum())
        thr_d = matched_threshold(prob, args.thr, added)
        drop = prob > thr_d

        res = [resample(q, ds=1.0) for q in gt]
        lens = [float(arclength(q)[-1]) if len(q) > 1 else 0.0 for q in res]
        cr = count_crossings(gt, tol=2.0 * UP, endpoint_margin=8.0 * UP)

        row = {"frame": stem, "n_gt": len(gt), "added_px": added,
               "thr_drop": round(float(thr_d), 5),
               "base_px": int(base.sum()), "fill_px": int(fill.sum()),
               "drop_px": int(drop.sum())}
        variants = [("base", base, None), ("fill", fill, None), ("drop", drop, None)]
        if args.passthrough:
            # read 36b: same mask as the baseline, the map consumed only in the arc graph, so
            # the two reads are not confounded with each other.
            variants.append(("pass", base, ov))
            variants.append(("fill_pass", fill, ov))
        for tag, m, ovmap in variants:
            polys, _ = instance_a(m, KAPPA_MAX, params, channels=prob[None], prob=prob,
                                  overlap=ovmap)
            pm = oracle_instance_masks(polys, shape_hi, half_width=1.0, up=1.0)
            r95 = centerline_f1(pm, gt, tol=5.0, length_coverage=0.95, precision_coverage=0.95)
            r80 = centerline_f1(pm, gt, tol=5.0, length_coverage=0.80, precision_coverage=0.80)
            ok, n = junction_identity(pm, cr, res, lens)
            row[tag] = {"f1_95": r95["f1"], "tp95": r95["tp"], "fp95": r95["fp"], "fn95": r95["fn"],
                        "f1_80": r80["f1"], "tp80": r80["tp"], "fp80": r80["fp"], "fn80": r80["fn"],
                        "id_ok": ok, "id_n": n, "n_inst": len(polys)}
        rows.append(row)
        print(f"{stem}: added {added} px, thr_drop {thr_d:.4f}, "
              f"id base {row['base']['id_ok']}/{row['base']['id_n']} "
              f"fill {row['fill']['id_ok']} drop {row['drop']['id_ok']}", flush=True)
        if args.limit and len(rows) >= args.limit:
            break

    summ = {"frames": len(rows), "thr": args.thr, "ov_thr": args.ov_thr}
    tags = ["base", "fill", "drop"] + (["pass", "fill_pass"] if args.passthrough else [])
    for tag in tags:
        f95 = [r[tag]["f1_95"] for r in rows if r["n_gt"] > 0]
        f80 = [r[tag]["f1_80"] for r in rows if r["n_gt"] > 0]
        idk = sum(r[tag]["id_ok"] for r in rows); idn = sum(r[tag]["id_n"] for r in rows)
        tp = sum(r[tag]["tp80"] for r in rows); fp = sum(r[tag]["fp80"] for r in rows)
        fn = sum(r[tag]["fn80"] for r in rows)
        summ[tag] = {"macro_f1_95": round(float(np.mean(f95)), 4) if f95 else None,
                     "macro_f1_80": round(float(np.mean(f80)), 4) if f80 else None,
                     "micro_f1_80": round(2 * tp / max(2 * tp + fp + fn, 1), 4),
                     "junction_identity": round(idk / max(idn, 1), 4), "id_n": idn,
                     "fg_px": sum(r[f"{tag}_px"] for r in rows) if f"{tag}_px" in rows[0] else None}
    summ["added_px_total"] = sum(r["added_px"] for r in rows)
    # paired per-frame deltas: the arm against the foreground-matched threshold drop, which is
    # the comparison that decides whether the multiplicity map fills SELECTIVELY.
    for tag in [t for t in tags if t != "base"]:
        summ[f"{tag}_minus_base"] = {
            k: round(float(np.mean([r[tag][k] - r["base"][k] for r in rows])), 4)
            for k in ("f1_95", "f1_80")}
        summ[f"{tag}_minus_base"]["identity"] = (
            sum(r[tag]["id_ok"] for r in rows) - sum(r["base"]["id_ok"] for r in rows))
    summ["fill_minus_drop"] = {
        k: round(float(np.mean([r["fill"][k] - r["drop"][k] for r in rows])), 4)
        for k in ("f1_95", "f1_80")}
    summ["fill_minus_drop"]["identity"] = (
        sum(r["fill"]["id_ok"] for r in rows) - sum(r["drop"]["id_ok"] for r in rows))
    json.dump({"summary": summ, "rows": rows}, open(args.out, "w"))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
