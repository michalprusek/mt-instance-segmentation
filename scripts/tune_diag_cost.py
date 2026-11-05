#!/usr/bin/env python3
"""Step 0: baseline gate + per-frame cost of one instancer call. RUNS ON TULEN.

Scores the SHIPPED parameters on every frame of the split serially (so the timings are honest,
uncontended single-core numbers) and reports the sparsest and densest frame by MASK FOREGROUND
PIXELS -- graph size follows the mask, not the ground-truth count.

    ~/dinov3_env/bin/python scripts/tune_diag_cost.py \
        --data data/synth_link --pred /home/prusek/mt_enc_exp/synth_link_pred --split val
"""
from __future__ import annotations

import argparse
import time

from tune_diag_common import load_frames, save, score_frame, shipped_params


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--prob-thr", type=float, default=None)
    ap.add_argument("--out", default="tune_diag_out/cost.json")
    a = ap.parse_args()

    prm = shipped_params()
    thr = a.prob_thr if a.prob_thr is not None else prm["prob_thr"]
    t0 = time.time()
    frames = load_frames(a.data, a.pred, a.split, thr)
    print(f"loaded {len(frames)} frames in {time.time()-t0:.1f}s at prob_thr={thr:.4f}",
          flush=True)

    rows = []
    for f in frames:
        r = score_frame(f, prm)
        r["fg_px"] = f["fg_px"]
        rows.append(r)
        print(f"{r['frame']:>16} fg={r['fg_px']:>7} n_gt={r['n_gt']:>3} "
              f"f1={r['f1']:.4f} tp={r['tp']} fp={r['fp']} {r['secs']:.2f}s", flush=True)

    from instance.metrics import aggregate_benchmark
    agg = aggregate_benchmark(rows)
    order = sorted(rows, key=lambda r: r["fg_px"])
    print(f"\nBASELINE mean_f1={agg['mean_f1']:.4f}  micro_f1={agg.get('micro_f1')}")
    print(f"total {sum(r['secs'] for r in rows):.1f}s  median/frame "
          f"{sorted(r['secs'] for r in rows)[len(rows)//2]:.2f}s")
    print(f"sparsest {order[0]['frame']} fg={order[0]['fg_px']} {order[0]['secs']:.2f}s")
    print(f"densest  {order[-1]['frame']} fg={order[-1]['fg_px']} {order[-1]['secs']:.2f}s")
    save(a.out, {"prob_thr": thr, "params": prm, "agg": {k: float(v) for k, v in agg.items()
                                                          if isinstance(v, (int, float))},
                 "rows": rows})


if __name__ == "__main__":
    main()
