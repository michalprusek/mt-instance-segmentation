#!/usr/bin/env python3
"""2-D interaction grid for two instancer-A knobs. RUNS ON TULEN.

The question this answers is narrow and decides a tuner design: does the optimum of knob A
MOVE when knob B changes? If it does not, coordinate descent on this pair converges to the
joint optimum and a cheap sequential search suffices; if it does, the pair has to be searched
jointly (TPE, grid, CMA-ES).

    ~/dinov3_env/bin/python scripts/tune_diag_grid.py --data data/synth_link \
        --pred /home/prusek/mt_enc_exp/synth_link_pred --split val \
        --knob-a merge_radius --knob-b min_length --n 9
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from tune_diag_common import load_frames, run_configs, save, shipped_params
from tune_diag_oat import INT_GRIDS, RANGES


def axis(knob: str, n: int) -> list:
    if knob in INT_GRIDS:
        g = INT_GRIDS[knob]
        idx = np.unique(np.round(np.linspace(0, len(g) - 1, n)).astype(int))
        return [g[i] for i in idx]
    lo, hi = RANGES[knob]
    return [round(float(v), 6) for v in np.linspace(lo, hi, n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--knob-a", required=True)
    ap.add_argument("--knob-b", required=True)
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--n-jobs", type=int, default=24)
    ap.add_argument("--out", default="tune_diag_out/grid.json")
    a = ap.parse_args()

    base = shipped_params()
    frames = load_frames(a.data, a.pred, a.split, base["prob_thr"])
    ax, bx = axis(a.knob_a, a.n), axis(a.knob_b, a.n)
    configs = []
    for i, va in enumerate(ax):
        for j, vb in enumerate(bx):
            p = dict(base)
            p[a.knob_a] = int(va) if a.knob_a in INT_GRIDS else float(va)
            p[a.knob_b] = int(vb) if a.knob_b in INT_GRIDS else float(vb)
            configs.append((f"{i}_{j}", p))
    print(f"{len(configs)} configs x {len(frames)} frames", flush=True)

    t0 = time.time()
    res = run_configs(frames, configs, n_jobs=a.n_jobs)
    print(f"done in {time.time()-t0:.0f}s wall", flush=True)

    M = np.array([[res[f"{i}_{j}"]["mean_f1"] for j in range(len(bx))]
                  for i in range(len(ax))])
    save(a.out, {"knob_a": a.knob_a, "knob_b": a.knob_b, "axis_a": ax, "axis_b": bx,
                 "mean_f1": M.tolist(),
                 "per_frame": {cid: [r["f1"] for r in res[cid]["rows"]] for cid in res},
                 "frames": [r["frame"] for r in res["0_0"]["rows"]]})

    print(f"\nrows = {a.knob_a}, cols = {a.knob_b}")
    print(" " * 10 + "".join(f"{v:>8.3g}" for v in bx))
    for i, va in enumerate(ax):
        print(f"{va:>10.4g}" + "".join(f"{M[i, j]:8.4f}" for j in range(len(bx))))
    print(f"\nbest {M.max():.4f} at {a.knob_a}={ax[int(np.argmax(M) // len(bx))]:.4g}, "
          f"{a.knob_b}={bx[int(np.argmax(M) % len(bx))]:.4g}")
    print(f"\nargmax of {a.knob_a} at each level of {a.knob_b} (does the ridge move?):")
    for j, vb in enumerate(bx):
        i = int(np.argmax(M[:, j]))
        print(f"  {a.knob_b}={vb:>8.4g} -> {a.knob_a}={ax[i]:>8.4g}  f1={M[i, j]:.4f}")
    print(f"\nargmax of {a.knob_b} at each level of {a.knob_a}:")
    for i, va in enumerate(ax):
        j = int(np.argmax(M[i]))
        print(f"  {a.knob_a}={va:>8.4g} -> {a.knob_b}={bx[j]:>8.4g}  f1={M[i, j]:.4f}")


if __name__ == "__main__":
    main()
