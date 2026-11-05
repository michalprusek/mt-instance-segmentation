#!/usr/bin/env python3
"""Report the head x generator factorial with every pairwise interval. RUNS ON TULEN.

Consumes the per-threshold JSONs that ``eval_v5.py`` writes (each already carries ``_frames``),
picks each arm's own best operating point, and prints every pairwise paired bootstrap. Nothing
is re-run: adding an interval to a number must never mean recomputing the number.

The arms, all through one trainer at one budget so head and generator stay separable:

    D  amodal head, original generator      G  binary head, original generator
    E  amodal head, SNR-corrected           F  binary head, SNR-corrected

    PYTHONPATH=src python scripts/factorial_report.py \
        D=/path/D.json E=/path/E.json F=/path/F.json G=/path/G.json \
        v4b=/tmp/v4b_fixed.json nnunet=/tmp/nnunet_val_fixed.json
"""
from __future__ import annotations

import itertools
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from instance.metrics import paired_bootstrap  # noqa: E402


def best_arm(path: str):
    """The arm's own optimum: highest mean_f1 over the thresholds swept for it."""
    d = json.load(open(path))["results"]
    k = max(d, key=lambda t: d[t]["mean_f1"])
    return {"thr": k, "f1": d[k]["mean_f1"], "rows": d[k]["_frames"],
            "tp": d[k]["total_tp"], "fp": d[k]["total_fp"], "fn": d[k]["total_fn"],
            "fg": d[k].get("fg", float("nan"))}


def main() -> None:
    arms = {}
    for a in sys.argv[1:]:
        if "=" not in a:
            raise SystemExit(f"expected NAME=path.json, got {a}")
        name, path = a.split("=", 1)
        if not os.path.exists(path):
            print(f"  (skipping {name}: {path} not found)")
            continue
        arms[name] = best_arm(path)
    if len(arms) < 2:
        raise SystemExit("need at least two arms")

    print("%-10s %-8s %-8s %5s %5s %5s %7s" % ("arm", "thr", "F1", "tp", "fp", "fn", "fg%"))
    for n, a in sorted(arms.items(), key=lambda kv: -kv[1]["f1"]):
        print("%-10s %-8s %.4f %5d %5d %5d %7.2f"
              % (n, a["thr"], a["f1"], a["tp"], a["fp"], a["fn"], 100 * a["fg"]))

    print("\npairwise paired bootstrap (n=%d frames, 10000 replicates):"
          % len(next(iter(arms.values()))["rows"]))
    for x, y in itertools.combinations(sorted(arms), 2):
        ra, rb = arms[x]["rows"], arms[y]["rows"]
        if [r["name"] for r in ra] != [r["name"] for r in rb]:
            print(f"  {x} vs {y}: FRAME SETS DIFFER -- not comparable")
            continue
        r = paired_bootstrap(ra, rb, stat="mean_f1", n_boot=10000)
        print("  %-8s - %-8s  %+.4f [%+.4f, %+.4f]  p=%.3f  %s"
              % (x, y, r["diff"], r["lo"], r["hi"], r["p_two_sided"],
                 "SIGNIFICANT" if r["significant"] else "ns"))

    if {"D", "E", "F", "G"} <= set(arms):
        print("\nfactorial main effects (mean over the other factor):")
        gen = 0.5 * ((arms["E"]["f1"] - arms["D"]["f1"]) + (arms["F"]["f1"] - arms["G"]["f1"]))
        head = 0.5 * ((arms["F"]["f1"] - arms["E"]["f1"]) + (arms["G"]["f1"] - arms["D"]["f1"]))
        inter = ((arms["F"]["f1"] - arms["E"]["f1"]) - (arms["G"]["f1"] - arms["D"]["f1"]))
        print("  generator (SNR-corrected - original): %+.4f" % gen)
        print("  head (binary - amodal):               %+.4f" % head)
        print("  interaction:                          %+.4f" % inter)
        print("  (point estimates only -- the intervals that matter are the pairwise ones)")


if __name__ == "__main__":
    main()
