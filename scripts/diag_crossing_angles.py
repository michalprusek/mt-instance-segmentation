#!/usr/bin/env python3
"""D2/Q2b -- how many GT crossings are SHALLOWER than one 30-degree orientation bin?

Two filaments crossing at less than 30 degrees fall into the SAME K=6 bin, so the amodal
"overpass" representation cannot separate them by construction: at the shared pixel there is
one channel carrying both. This measures how much of the benchmark that costs, and pairs the
GT-side fraction with the instancer's measured junction identity on exactly those crossings
(read from the already-scored report JSONs -- nothing is re-run on TEST).

    PYTHONPATH=src /home/prusek/dinov3_env/bin/python scripts/diag_crossing_angles.py \
        --out data/enc_sensitivity_testset/diag_crossing_angles.json
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

from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.gt_stats import count_crossings  # noqa: E402

UP = 1.5


def angles_for(data: str, split: str) -> dict:
    per_task: dict[str, list] = {}
    n_frames = 0
    for path in sorted(glob.glob(os.path.join(data, "*.h5"))):
        fr = read_frame_h5(path)
        a = fr["attrs"]
        if split != "all" and str(a.get("split")) != split:
            continue
        n_frames += 1
        gt = [np.asarray(p, float) * UP for p in fr["polylines"]]
        cr = count_crossings(gt, tol=2.0 * UP, endpoint_margin=8.0 * UP)
        per_task.setdefault(str(a.get("source_task", "?")), []).extend(
            [c["angle_deg"] for c in cr])

    out = {"n_frames": n_frames, "by_task": {}}
    allang = []
    for t, v in per_task.items():
        allang += v
        out["by_task"][t] = summarise(np.array(v))
    out["pooled"] = summarise(np.array(allang))
    return out


def summarise(a: np.ndarray) -> dict:
    if not len(a):
        return {"n": 0}
    return {
        "n": int(len(a)),
        "frac_shallow_lt30": float((a < 30).mean()),
        "frac_oblique_30_60": float(((a >= 30) & (a < 60)).mean()),
        "frac_steep_ge60": float((a >= 60).mean()),
        "frac_lt15": float((a < 15).mean()),
        "frac_lt10": float((a < 10).mean()),
        "median_deg": float(np.median(a)),
        "n_shallow": int((a < 30).sum()),
    }


def from_report(path: str, keys) -> dict:
    """Pull the already-scored junction-identity-by-angle rows out of a report JSON."""
    if not os.path.exists(path):
        return {"missing": path}
    d = json.load(open(path))
    out = {}
    for k in keys:
        if k in d:
            out[k] = {"junction_by_angle": d[k].get("junction_by_angle"),
                      "n_by_angle": d[k].get("n_by_angle"),
                      "junction": d[k].get("junction")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/enc_sensitivity_testset/diag_crossing_angles.json")
    a = ap.parse_args()
    L = "data/enc_sensitivity_testset"
    res = {
        "mt34_val": angles_for("data/real/mt34_eval", "val"),
        "mt34_all": angles_for("data/real/mt34_eval", "all"),
        "synth_val": angles_for("data/synth_eval", "val"),
        "synth_all": angles_for("data/synth_eval", "all"),
        "shipped_junction_identity": {
            "mt34_TEST_oracle": from_report(f"{L}/TEST_oracle.json",
                                            ["a/585", "a/586", "pysoax/585", "pysoax/586"]),
            "mt34_TEST_model": from_report(f"{L}/TEST_model.json", ["a/585", "a/586"]),
            "synth_TEST_oracle": from_report(f"{L}/TEST_synth_oracle.json",
                                             ["a/synth", "pysoax/synth"]),
            "synth_TEST_model": from_report(f"{L}/TEST_synth_model.json", ["a/synth"]),
        },
    }
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
