#!/usr/bin/env python3
"""Summarise a D3 decode sweep: F1 vs the shipped operating point, with the gate decomposed.

The over-firing gate has TWO independent clauses (``mt_bench.fg_quality``): an over-firing
ceiling (fg <= 3x the in-domain synth reference) and a COLLAPSE floor (rec2 >= 0.90). A row
that fails needs to say which one, because "over-fires" and "misses a third of the ground
truth" are opposite diagnoses and the project has already misread one gate verdict once.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from instance.metrics import paired_bootstrap  # noqa: E402
from mt_bench.fg_quality import SYNTH_FG_REFERENCE  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--baseline", default="flat/0.4418")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    for path in args.reports:
        rep = json.load(open(path))
        S = rep["settings"]
        base = S.get(args.baseline)
        print(f"\n################ {os.path.basename(path)} "
              f"({len(S)} settings) ################")
        if rep.get("split_diag"):
            d = list(rep["split_diag"].values())
            print(f"  crossing-neighbourhood diagnostic over {len(d)} frames with crossings: "
                  f"px={sum(x['n_px'] for x in d)}  "
                  f"frac(max<thr)={np.mean([x['frac_max_below'] for x in d]):.3f}  "
                  f"frac(max<thr AND sum>=thr)="
                  f"{np.mean([x['frac_rescued_by_sum'] for x in d]):.3f}  "
                  f"mean max={np.mean([x['mean_max'] for x in d]):.3f} "
                  f"mean sum={np.mean([x['mean_sum'] for x in d]):.3f}")

        rows = []
        for name, e in S.items():
            p, agg = e["props"], e["pooled"]
            fgok = p.get("fg", np.nan) <= 3.0 * SYNTH_FG_REFERENCE
            recok = p.get("rec2", np.nan) >= 0.90
            d = {"name": name, "f1": agg["mean_f1"], "micro": agg["micro_f1"],
                 "tp": agg["total_tp"], "fp": agg["total_fp"], "fn": agg["total_fn"],
                 "fg": 100 * p.get("fg", np.nan), "rec2": p.get("rec2", np.nan),
                 "cc": p.get("cc_per_gt", np.nan), "gaps": p.get("gaps_per_mt", np.nan),
                 "q": e["quality"], "frag": e["frag"],
                 "gate": ("PASS" if (fgok and recok) else
                          ("FAIL:overfire" if not fgok else "FAIL:rec2")),
                 "by_task": {t: v["mean_f1"] for t, v in e["by_task"].items()}}
            if base is not None and name != args.baseline:
                bs = paired_bootstrap(e["frames"], base["frames"], stat="mean_f1",
                                      n_boot=args.n_boot, stratum_key="task",
                                      frame_key="name", seed=0)
                d.update(diff=bs["diff"], lo=bs["lo"], hi=bs["hi"], p=bs["p_two_sided"])
            rows.append(d)

        tasks = sorted({t for r in rows for t in r["by_task"]})
        hdr = (f"{'setting':30s} {'F1':>6s} " +
               " ".join(f"{t:>7s}" for t in tasks) +
               f" {'micro':>6s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'fg%':>5s} {'rec2':>5s} "
               f"{'cc/gt':>5s} {'gaps':>5s} {'dF1':>7s} {'95% CI':>16s} {'p':>5s}  gate")
        print(hdr)
        rows.sort(key=lambda r: -r["f1"])
        show = rows[:args.top] + [r for r in rows if r["name"] == args.baseline]
        seen = set()
        for r in show:
            if r["name"] in seen:
                continue
            seen.add(r["name"])
            ci = (f"[{r['lo']:+.3f},{r['hi']:+.3f}]" if "lo" in r else "".rjust(16))
            print(f"{r['name']:30s} {r['f1']:6.3f} " +
                  " ".join(f"{r['by_task'].get(t, float('nan')):7.3f}" for t in tasks) +
                  f" {r['micro']:6.3f} {r['tp']:4d} {r['fp']:4d} {r['fn']:4d} "
                  f"{r['fg']:5.2f} {r['rec2']:5.3f} {r['cc']:5.2f} {r['gaps']:5.2f} "
                  f"{r.get('diff', float('nan')):+7.3f} {ci:>16s} "
                  f"{r.get('p', float('nan')):5.3f}  {r['gate']}")
        # Does buying continuity buy F1? Five training-side interventions have already
        # improved connectivity with a null F1; this asks the same question of the decoder,
        # ACROSS settings (not the paired same-frame ranking test fg_quality documents).
        from scipy.stats import spearmanr
        f1s = np.array([r["f1"] for r in rows])
        for k in ("cc", "gaps", "q", "fg"):
            v = np.array([r[k] for r in rows], dtype=float)
            ok = np.isfinite(v) & np.isfinite(f1s)
            rho, pv = spearmanr(v[ok], f1s[ok])
            print(f"  spearman(F1, {k:5s}) over {ok.sum()} settings = {rho:+.3f} (p={pv:.3g})")
        best = max(rows, key=lambda r: -r["cc"])
        print(f"  most-continuous setting: {best['name']} cc/gt={best['cc']:.2f} "
              f"gaps={best['gaps']:.2f} F1={best['f1']:.3f} "
              f"(baseline cc/gt={rows[0]['cc'] if base is None else S[args.baseline]['props']['cc_per_gt']:.2f} "
              f"F1={base['pooled']['mean_f1']:.3f})" if base else "")
        n_pass = sum(r["gate"] == "PASS" for r in rows)
        print(f"  gate: {n_pass}/{len(rows)} PASS; "
              f"{sum(r['gate'] == 'FAIL:overfire' for r in rows)} fail the over-firing "
              f"ceiling, {sum(r['gate'] == 'FAIL:rec2' for r in rows)} fail the rec2 "
              f"collapse floor")
        print(f"  F1 range over all settings: {min(r['f1'] for r in rows):.3f} .. "
              f"{max(r['f1'] for r in rows):.3f}   (baseline "
              f"{base['pooled']['mean_f1']:.3f})" if base else "")


if __name__ == "__main__":
    main()
