#!/usr/bin/env python3
"""Pick each arm's own threshold from its sweep, then run the paired comparisons.

Two guards that this project has needed:

* **Grid edges.** A previous sweep stopped at 0.995, both arms landed on that edge, and extending
  the grid moved the optimum to 0.997 -- the "best" threshold was an artefact of where the grid
  ended. Any arm whose winner sits on the first or last grid point is reported as UNRELIABLE and
  the comparison that uses it is flagged, not silently run.
* **Equal tuning.** Every arm is picked from the SAME grid by the SAME rule. Comparing an arm at a
  tuned threshold against one at a historical threshold has already flipped a verdict here.

Selection is on the clean roi303 VAL (141 human-drawn frames, no model in its provenance).

    python scripts/pick_and_compare.py --runs runs --out roi_compare.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

ARMS = ["PHOTO_V6", "TIRF35", "SPARSE20", "SPARSE35", "SPARSE50"]

# (a, b, label) -- each answers one question, and the family is fixed BEFORE the numbers are seen
COMPARISONS = [
    ("TIRF35",   "SPARSE35",  "renderer_sparse_vs_continuous"),   # same fraction, renderer A/B
    ("PHOTO_V6", "SPARSE20",  "tirf20_vs_none"),
    ("PHOTO_V6", "SPARSE35",  "tirf35_vs_none"),
    ("PHOTO_V6", "SPARSE50",  "tirf50_vs_none"),
    ("SPARSE20", "SPARSE35",  "fraction_20_vs_35"),
    ("SPARSE35", "SPARSE50",  "fraction_35_vs_50"),
]


def read_sweep(path):
    """-> (best_thr, best_mean_f1, [(thr, mean_f1)...]) from an eval_v5 --out-json sweep.

    eval_v5 writes {"ckpt":..., "results": {"<thr>": {"mean_f1":..., ...}}}. The format is asserted
    rather than guessed at: a parser that silently returns nothing would surface only after hours of
    sweeping, and one that silently returns the WRONG field would surface never."""
    d = json.load(open(path))
    if not isinstance(d, dict) or "results" not in d:
        raise SystemExit(f"{path}: expected a dict with 'results', got keys={list(d)[:8]}")
    pts = []
    for k, v in d["results"].items():
        if "mean_f1" not in v:
            raise SystemExit(f"{path}: threshold {k} has no mean_f1 (keys={list(v)[:8]})")
        pts.append((float(k), float(v["mean_f1"])))
    if not pts:
        raise SystemExit(f"{path}: 'results' is empty -- the sweep produced nothing")
    pts.sort()
    best = max(pts, key=lambda t: t[1])
    return best[0], best[1], pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/disk2/prusek/mt_work/runs")
    ap.add_argument("--preds", default="/disk2/prusek/mt_work/preds")
    ap.add_argument("--data-dir", default="data/real/roi303_eval")
    ap.add_argument("--min-length", type=float, default=44.74)
    ap.add_argument("--python", default="/scratch/prusek/envs/nnunet/bin/python")
    ap.add_argument("--out", default="roi_compare.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    picked, edge = {}, {}
    print("PER-ARM THRESHOLD, chosen on roi303 VAL from one identical grid\n")
    print(f'{"arm":10s} {"best thr":>9s} {"mean-F1":>9s} {"grid":>6s}  edge?')
    for a in ARMS:
        p = os.path.join(args.runs, f"thr_{a}.json")
        if not os.path.isfile(p):
            print(f"{a:10s} {'-':>9s} {'-':>9s}   sweep not finished")
            continue
        thr, f1, pts = read_sweep(p)
        on_edge = thr in (pts[0][0], pts[-1][0])
        picked[a], edge[a] = (thr, f1), on_edge
        print(f'{a:10s} {thr:9.4f} {f1:9.4f} {len(pts):6d}  '
              f'{"UNRELIABLE - winner is on the grid edge" if on_edge else "ok"}')

    results = []
    print("\nPAIRED COMPARISONS (per-frame bootstrap, each arm at its own threshold)\n")
    for a, b, label in COMPARISONS:
        if a not in picked or b not in picked:
            print(f"  {label:34s} SKIPPED - a sweep is missing")
            continue
        flag = " [EDGE]" if edge[a] or edge[b] else ""
        cmd = [args.python, "scripts/paired_any.py",
               "--a", f"prob:{args.preds}/{a}_roi:{picked[a][0]}:{args.min_length}", "--a-label", a,
               "--b", f"prob:{args.preds}/{b}_roi:{picked[b][0]}:{args.min_length}", "--b-label", b,
               "--data-dir", args.data_dir, "--split", "val",
               "--out", os.path.join(args.runs, f"paired_{label}.json")]
        print(f"  {label:34s} {a} vs {b}{flag}")
        if args.dry_run:
            print("      " + " ".join(cmd))
            continue
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"      FAILED rc={r.returncode}: {r.stderr.strip()[-300:]}")
            continue
        for line in r.stdout.splitlines():
            if "=" in line and "95% CI" in line:
                print("      " + line.strip())
        results.append(dict(label=label, a=a, b=b, edge=flag != "",
                            json=os.path.join(args.runs, f"paired_{label}.json")))

    # Holm-Bonferroni over the whole family: a table of six paired tests contains "significant"
    # cells by chance, and dropping an uncomputable one would shrink the family in our favour.
    ps = []
    for r in results:
        try:
            ps.append((r["label"], float(json.load(open(r["json"]))["p"])))
        except Exception:
            ps.append((r["label"], float("nan")))
    import math
    valid = [(l, p) for l, p in ps if not math.isnan(p)]
    valid.sort(key=lambda t: t[1])
    m = len(ps)          # the FULL family, including any that failed to compute
    print(f"\nHOLM-BONFERRONI over the whole family of {m} tests\n")
    print(f'{"comparison":34s} {"raw p":>9s} {"Holm p":>9s}  verdict')
    run_max = 0.0
    for i, (l, p) in enumerate(valid):
        adj = min(1.0, max(run_max, (m - i) * p))
        run_max = adj
        print(f'{l:34s} {p:9.4f} {adj:9.4f}  {"SIGNIFICANT" if adj < 0.05 else "not significant"}')
    for l, p in ps:
        if math.isnan(p):
            print(f'{l:34s} {"-":>9s} {"-":>9s}  did not compute (still counted in the family)')

    json.dump(dict(picked={k: dict(thr=v[0], mean_f1=v[1], on_grid_edge=edge[k])
                           for k, v in picked.items()},
                   comparisons=results), open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
