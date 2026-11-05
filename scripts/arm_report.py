#!/usr/bin/env python3
"""Read a campaign arm against the baseline the way docs/TODO.md 0.0-CAMPAIGN pre-declares it.

Consumes the per-threshold sweep JSONs that ``eval_v5.py`` writes (each threshold carries its
``_frames`` rows), so nothing is re-run. For every arm file it reports, against the baseline:

  * each side's OWN best swept threshold (a shared threshold measures calibration, not quality);
  * paired per-frame bootstrap on the mean F1 (point, 95 % CI, two-sided p) on the frames the two
    files share, plus wins / losses;
  * micro F1 of both (promotion needs the arm's micro F1 not lower);
  * the MECHANISM CHECK for a calibration fix: the per-frame optimal-threshold spread on frames
    with >= 10 GT -- the 10th-90th percentile of the per-frame best threshold (as a grid index and
    as a value) and the oracle per-frame-threshold gain. A real calibration fix NARROWS the spread
    and SHRINKS the gain; an arm that only shifts the plateau leaves both alone.
  * Holm over the whole family: every arm's p plus the earlier entries' p-values passed with
    --extra-p, so a single arm is never judged as if it were the only test run.

If two arm names differ only by a trailing ``_native``, the scale gap |F1(native) - F1(1.5x)| is
printed as well (the SCALE arm's mechanism check).

    PYTHONPATH=src python scripts/arm_report.py \
        --baseline runs/scale/thr_SPARSE35_native_rep.json \
        --arm AUG_PHOTO_ep40_native=runs/scale/thr_AUG_PHOTO_ep40_native.json \
        --arm AUG_PHOTO_native=runs/scale/thr_AUG_PHOTO_native.json \
        --extra-p 1=0.41 --extra-p 2=0.07 ... --out runs/scale/report_AUG_PHOTO.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from instance.metrics import aggregate_benchmark, paired_bootstrap  # noqa: E402


def load(path: str) -> dict:
    """{thr(float): {name: row}} with n_gt filled in, plus the sorted grid."""
    d = json.load(open(path))["results"]
    out = {}
    for t, a in d.items():
        rows = {}
        for r in a["_frames"]:
            rr = dict(r)
            rr["n_gt"] = rr["tp"] + rr["fn"]
            rr["frame"] = rr["name"]
            rows[rr["name"]] = rr
        out[float(t)] = rows
    return out


def best_threshold(sw: dict, names: list[str]) -> float:
    """The threshold with the highest mean F1 on exactly these frames."""
    scores = {t: aggregate_benchmark([sw[t][n] for n in names])["mean_f1"] for t in sw}
    return max(scores, key=scores.get)


def spread(sw: dict, names: list[str], min_gt: int = 10) -> dict:
    """Per-frame optimal-threshold spread and oracle gain on frames with >= min_gt GT."""
    grid = sorted(sw)
    keep = [n for n in names if sw[grid[0]][n]["n_gt"] >= min_gt]
    if len(keep) < 5:
        return {"n": len(keep)}
    F = np.array([[sw[t][n]["f1"] for n in keep] for t in grid])       # (T, N)
    gb = int(F.mean(1).argmax())
    pf = F.argmax(0)
    oracle = float(F.max(0).mean())
    p10, p90 = np.percentile(pf, [10, 90])
    return {"n": len(keep), "global_thr": grid[gb], "global_mean": float(F[gb].mean()),
            "oracle_mean": oracle, "oracle_gain": oracle - float(F[gb].mean()),
            "best_idx_p10_p90": [float(p10), float(p90)],
            "best_idx_span": float(p90 - p10),
            "best_thr_p10_p90": [grid[int(round(p10))], grid[int(round(p90))]],
            "frac_at_global": float((pf == gb).mean())}


def holm(pvals: dict) -> dict:
    """Holm step-down adjusted p-values over the family."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, (m - i) * p)
        adj[k] = min(1.0, running)
    return adj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--baseline-name", default="SPARSE35")
    ap.add_argument("--arm", action="append", default=[], help="NAME=path.json (repeatable)")
    ap.add_argument("--extra-p", action="append", default=[],
                    help="ENTRY=p for earlier family members, e.g. 3=0.41 (repeatable)")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--min-gt", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = load(args.baseline)
    report, pvals = {"baseline": args.baseline, "arms": {}}, {}
    for k in args.extra_p:
        name, p = k.split("=")
        pvals[name] = float(p)

    print(f"baseline {args.baseline_name}: {args.baseline}")
    hdr = ("%-26s %6s %6s %7s %7s  %7s %6s [%6s,%6s] %7s %5s/%-5s  %6s %6s"
           % ("arm", "thr", "meanF1", "microF1", "fg", "delta", "base", "lo", "hi", "p", "win",
              "loss", "spread", "gain"))
    print(hdr)
    for spec in args.arm:
        name, path = spec.split("=", 1)
        sw = load(path)
        g0 = sorted(sw)[0]
        names = sorted(set(sw[g0]) & set(base[sorted(base)[0]]))
        tb, ta = best_threshold(base, names), best_threshold(sw, names)
        # An own threshold at the edge of the swept grid is not a read: the optimum may lie
        # beyond it (caught 2026-09-07 on SCALE_S2 ep40 native at the fast grid's floor 0.90).
        for who, thr, grid in (("arm", ta, sorted(sw)), ("baseline", tb, sorted(base))):
            if thr in (grid[0], grid[-1]):
                print(f"  WARNING {name}: {who} own threshold {thr:g} sits at the grid edge "
                      f"[{grid[0]:g}, {grid[-1]:g}] -- extend the sweep before reading this row")
        rows_b = [base[tb][n] for n in names]
        rows_a = [sw[ta][n] for n in names]
        pb = paired_bootstrap(rows_a, rows_b, stat="mean_f1", n_boot=args.boot,
                              stratum_key=None)
        agg_a, agg_b = aggregate_benchmark(rows_a), aggregate_benchmark(rows_b)
        da = np.array([r["f1"] for r in rows_a if r["n_gt"] > 0]) \
            - np.array([r["f1"] for r in rows_b if r["n_gt"] > 0])
        sp_a, sp_b = spread(sw, names, args.min_gt), spread(base, names, args.min_gt)
        fg = json.load(open(path))["results"][str(ta) if str(ta) in
                                                json.load(open(path))["results"]
                                                else f"{ta:g}"].get("fg", float("nan"))
        rec = {"path": path, "n_frames": len(names), "thr": ta, "baseline_thr": tb,
               "mean_f1": agg_a["mean_f1"], "baseline_mean_f1": agg_b["mean_f1"],
               "micro_f1": agg_a["micro_f1"], "baseline_micro_f1": agg_b["micro_f1"],
               "fg": fg, "delta": pb["point"] if "point" in pb else pb.get("diff"),
               "ci": [pb.get("lo", pb.get("ci_lo")), pb.get("hi", pb.get("ci_hi"))],
               "p": pb["p_two_sided"], "wins": int((da > 0).sum()), "losses": int((da < 0).sum()),
               "spread_arm": sp_a, "spread_baseline": sp_b}
        # declared secondary readings (2026-09-05): macro F1 over frames with >= 10 GT and over
        # the near-empty frames (1-9 GT), at the same own thresholds. The primary verdict is the
        # macro mean over all frames with >= 1 GT; these say where its number comes from.
        for key, lo, hi in (("dense", 10, 10 ** 9), ("sparse", 1, 10)):
            sel_a = [r for r in rows_a if lo <= r["n_gt"] < hi]
            sel_b = [r for r in rows_b if lo <= r["n_gt"] < hi]
            if sel_a:
                rec[f"macro_{key}"] = float(np.mean([r["f1"] for r in sel_a]))
                rec[f"baseline_macro_{key}"] = float(np.mean([r["f1"] for r in sel_b]))
                rec[f"n_{key}"] = len(sel_a)
        report["arms"][name] = rec
        pvals[name] = rec["p"]
        print("%-26s %6.4g %6.4f %7.4f %6.2f%%  %+7.4f %6.4f [%+6.3f,%+6.3f] %7.4f %5d/%-5d  %6.1f %+6.3f"
              % (name, ta, rec["mean_f1"], rec["micro_f1"], 100 * fg, rec["delta"],
                 rec["baseline_mean_f1"], rec["ci"][0], rec["ci"][1], rec["p"], rec["wins"],
                 rec["losses"], sp_a.get("best_idx_span", float("nan")),
                 sp_a.get("oracle_gain", float("nan"))))
        if "macro_dense" in rec:
            print("    secondary: macro>=10GT %.4f vs %.4f (%+.4f, n=%d) | macro 1-9GT %.4f vs %.4f (%+.4f, n=%d) | micro %+.4f"
                  % (rec["macro_dense"], rec["baseline_macro_dense"], rec["macro_dense"] - rec["baseline_macro_dense"], rec["n_dense"],
                     rec.get("macro_sparse", float("nan")), rec.get("baseline_macro_sparse", float("nan")),
                     rec.get("macro_sparse", float("nan")) - rec.get("baseline_macro_sparse", float("nan")), rec.get("n_sparse", 0),
                     rec["micro_f1"] - rec["baseline_micro_f1"]))
    print("baseline spread: idx span %.1f (thr %s), oracle gain %+.3f on n=%d"
          % (sp_b.get("best_idx_span", float("nan")), sp_b.get("best_thr_p10_p90"),
             sp_b.get("oracle_gain", float("nan")), sp_b.get("n", 0)))

    # scale gap for NAME / NAME_native pairs
    for name in list(report["arms"]):
        if name.endswith("_native") and name[:-7] in report["arms"]:
            gap = abs(report["arms"][name]["mean_f1"] - report["arms"][name[:-7]]["mean_f1"])
            report["arms"][name]["scale_gap"] = gap
            print(f"scale gap |F1(native) - F1(1.5x)| for {name[:-7]}: {gap:.4f}")

    adj = holm(pvals)
    report["holm"] = {k: {"p": pvals[k], "p_holm": adj[k]} for k in pvals}
    print("\nHolm over %d family members:" % len(pvals))
    for k in sorted(pvals, key=pvals.get):
        flag = "PROMOTE?" if (adj[k] < 0.05 and k in report["arms"]
                              and report["arms"][k]["delta"] > 0
                              and report["arms"][k]["micro_f1"] >= report["arms"][k]["baseline_micro_f1"]) else ""
        print("  %-26s p=%.4f  holm=%.4f  %s" % (k, pvals[k], adj[k], flag))
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
