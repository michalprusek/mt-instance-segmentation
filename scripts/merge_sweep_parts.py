#!/usr/bin/env python3
"""Merge per-threshold eval_v5.py JSONs (one per file, written by sweep_parallel.sh) into the single
sweep JSON eval_v5.py would have written. Works on a PARTIAL set of parts too, so a preliminary
read of the fast (high) thresholds is possible while the slow low thresholds are still running.

    python scripts/merge_sweep_parts.py OUT_JSON PARTS_DIR
"""
import glob, json, os, sys

out, tmp = sys.argv[1], sys.argv[2]
merged = None
for f in sorted(glob.glob(os.path.join(tmp, "thr_*.json"))):
    d = json.load(open(f))
    if merged is None:
        merged = {k: v for k, v in d.items() if k != "results"}; merged["results"] = {}
    merged["results"].update(d["results"])
if merged is None or not merged["results"]:
    sys.exit("merge_sweep_parts: no parts in " + tmp)
merged["results"] = dict(sorted(merged["results"].items(), key=lambda kv: float(kv[0])))
merged["n_thresholds"] = len(merged["results"])
json.dump(merged, open(out, "w"), indent=1)
best = max(merged["results"], key=lambda t: merged["results"][t]["mean_f1"])
print(f"{out}: {len(merged['results'])} thresholds, best {best} mean_f1 {merged['results'][best]['mean_f1']:.4f}")
