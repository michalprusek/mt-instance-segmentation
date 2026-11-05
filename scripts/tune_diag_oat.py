#!/usr/bin/env python3
"""One-at-a-time knob sensitivity for instancer A. RUNS ON TULEN.

From the shipped `params_a_model_synthtuned.json`, each knob is swept over the SAME range
`scripts/tune_instancer.py:space_a` gives Optuna -- so the resulting table says what the tuner
would actually have to search, not what some arbitrary interval says. Every curve is forced
through the identical shipped baseline row (the shipped value is inserted into the grid and
deduplicated), so the curves are comparable.

Per-frame rows are stored for every configuration: the cliff analysis needs distinct-value
counts PER FRAME (averaging 25 staircases manufactures apparent smoothness) and the bootstrap
in `tune_diag_bootstrap.py` needs the paired per-frame scores.

    ~/dinov3_env/bin/python scripts/tune_diag_oat.py --data data/synth_link \
        --pred /home/prusek/mt_enc_exp/synth_link_pred --split val --n-points 21
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from tune_diag_common import load_frames, run_configs, save, shipped_params

#: (low, high) copied VERBATIM from tune_instancer.space_a. `min_arc_len` is not a search
#: dimension there (hard-wired to 3), so a plausible integer range is used and flagged.
RANGES = {
    "merge_radius": (2.0, 10.0),
    "window": (6.0, 36.0),
    "min_length": (6.0, 50.0),
    "w_theta": (0.2, 5.0),
    "w_kappa": (0.0, 40.0),
    "w_gap": (0.0, 0.2),
    "c_open": (0.3, 4.0),
    "gap_floor": (2.0, 12.0),
    "link_max_gap": (0.0, 40.0),
    "c_open_link": (0.3, 4.0),
    "bridge_thr": (0.0, 0.5),
    "bridge_max_len": (6.0, 50.0),
    "w_ori": (0.0, 6.0),
}
INT_GRIDS = {
    "smooth_size": list(range(3, 16, 2)),          # space_a: suggest_int(3, 15, step=2)
    "min_arc_len": list(range(1, 10)),             # NOT in space_a (fixed at 3)
}


def grid_for(knob: str, base: dict, n_points: int) -> list[float]:
    if knob in INT_GRIDS:
        return list(INT_GRIDS[knob])
    lo, hi = RANGES[knob]
    g = list(np.linspace(lo, hi, n_points))
    g.append(float(base[knob]))                    # every curve passes through the baseline
    return sorted(set(round(float(v), 6) for v in g))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--knobs", default="", help="comma list; default = all")
    ap.add_argument("--n-points", type=int, default=21)
    ap.add_argument("--n-jobs", type=int, default=24)
    ap.add_argument("--out", default="tune_diag_out/oat.json")
    a = ap.parse_args()

    base = shipped_params()
    frames = load_frames(a.data, a.pred, a.split, base["prob_thr"])
    print(f"{len(frames)} frames, prob_thr fixed at {base['prob_thr']:.4f}", flush=True)

    knobs = ([k.strip() for k in a.knobs.split(",") if k.strip()]
             or list(RANGES) + list(INT_GRIDS))
    configs, meta = [("BASE", dict(base))], {}
    for k in knobs:
        vals = grid_for(k, base, a.n_points)
        meta[k] = vals
        for i, v in enumerate(vals):
            p = dict(base)
            p[k] = int(v) if k in INT_GRIDS else float(v)
            configs.append((f"{k}#{i}", p))
    print(f"{len(configs)} configs x {len(frames)} frames", flush=True)

    t0 = time.time()
    res = run_configs(frames, configs, n_jobs=a.n_jobs)
    print(f"done in {time.time()-t0:.0f}s wall", flush=True)

    out = {"baseline": {k: v for k, v in base.items()},
           "base_f1": res["BASE"]["mean_f1"], "grids": meta,
           "n_frames": len(frames), "results": {}}
    for cid, r in res.items():
        out["results"][cid] = {"mean_f1": r["mean_f1"], "n_timeout": r["n_timeout"],
                               "f1": [x["f1"] for x in r["rows"]],
                               "tp": [x["tp"] for x in r["rows"]],
                               "fp": [x["fp"] for x in r["rows"]],
                               "frames": [x["frame"] for x in r["rows"]]}
    save(a.out, out)

    print(f"\nBASE mean_f1 = {res['BASE']['mean_f1']:.4f}")
    print(f"{'knob':>15} {'min':>8} {'max':>8} {'range':>8} {'argmax':>10} "
          f"{'distinct(mean)':>15} {'med distinct/frame':>19}")
    rank = []
    for k in knobs:
        vals = meta[k]
        f1 = [res[f"{k}#{i}"]["mean_f1"] for i in range(len(vals))]
        # per-frame f1 lives in the serialised view, not in run_configs' raw rows
        per = np.array([[r["f1"] for r in res[f"{k}#{i}"]["rows"]]
                        for i in range(len(vals))])       # (n_sweep_values, n_frames)
        dist_frame = [len(set(np.round(per[:, j], 9))) for j in range(per.shape[1])]
        rank.append((max(f1) - min(f1), k, min(f1), max(f1), vals[int(np.argmax(f1))],
                     len(set(np.round(f1, 9))), float(np.median(dist_frame)), len(vals)))
    for d, k, lo, hi, am, dm, df, n in sorted(rank, reverse=True):
        print(f"{k:>15} {lo:8.4f} {hi:8.4f} {d:8.4f} {am:10.4g} "
              f"{dm:>10}/{n:<4} {df:19.1f}")


if __name__ == "__main__":
    main()
