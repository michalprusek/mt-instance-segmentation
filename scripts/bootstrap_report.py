#!/usr/bin/env python3
"""Re-derive bootstrap intervals from a saved eval report. Runs anywhere -- no data needed.

``run_oracle_eval.py`` stores its per-frame rows under ``_frames`` precisely so that every
interval can be recomputed offline. That matters for TEST: adding an interval to a published
number must never mean re-running the benchmark, because a second run under changed code is a
second TEST shot no matter how it is labelled.

Comparisons are PAIRED (same frame multiset for both methods) and STRATIFIED by source task,
since pooled MT-34 is a fixed 6 Alice + 11 new-22 design.

    python3 scripts/bootstrap_report.py data/enc_sensitivity_testset/TESTv2_oracle.json
    python3 scripts/bootstrap_report.py A.json B.json --label-a v2 --label-b v1
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from instance.metrics import bootstrap_ci, paired_bootstrap  # noqa: E402


def _frames(path: str) -> dict:
    report = json.load(open(path))
    if "_frames" not in report:
        raise SystemExit(
            f"{path} has no '_frames' block -- it predates per-frame dumping. Re-instrument "
            f"it with run_oracle_eval.py under BYTE-IDENTICAL frozen params.")
    return report["_frames"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", help="eval json with a '_frames' block")
    ap.add_argument("other", nargs="?", default=None,
                    help="second report, to compare the SAME method across runs (v1 vs v2)")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--stat", default="mean_f1")
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    a = _frames(args.report)
    print(f"=== {args.stat} | {args.n_boot} replicates | stratified by task | 95% CI ===\n")
    for m, rows in sorted(a.items()):
        ci = bootstrap_ci(rows, stat=args.stat, n_boot=args.n_boot,
                          stratum_key="task", seed=args.seed)
        print(f"  {m:10s} {ci['point']:.3f}  [{ci['lo']:.3f}, {ci['hi']:.3f}]  "
              f"(n={ci['n_frames']} frames)")

    def _cmp(rows_x, rows_y, label):
        r = paired_bootstrap(rows_x, rows_y, stat=args.stat, n_boot=args.n_boot,
                             stratum_key="task", frame_key="name", seed=args.seed)
        flag = "significant" if r["significant"] else "NOT separable"
        print(f"  {label}: {r['diff']:+.3f}  [{r['lo']:+.3f}, {r['hi']:+.3f}]  "
              f"p={r['p_two_sided']:.3f}  {flag}")

    if len(a) >= 2:
        print("\n--- paired differences WITHIN this report ---")
        for x, y in itertools.combinations(sorted(a), 2):
            _cmp(a[x], a[y], f"{x} - {y}")

    if args.other:
        b = _frames(args.other)
        print(f"\n--- paired differences ACROSS reports "
              f"({args.label_a} - {args.label_b}) ---")
        for m in sorted(set(a) & set(b)):
            _cmp(a[m], b[m], f"{m}: {args.label_a} - {args.label_b}")


if __name__ == "__main__":
    main()
