#!/usr/bin/env python3
"""Score explicit named parameter overrides on both splits. RUNS ON TULEN.

`tune_diag_compose.py` showed that stacking one-at-a-time optima does not transfer to held-out
frames while one particular subset does. This isolates single overrides so the surviving effect
can be attributed to a knob rather than to a bundle.

    ~/dinov3_env/bin/python scripts/tune_diag_verify.py --data data/synth_link \
        --pred /home/prusek/mt_enc_exp/synth_link_pred --set merge_radius=2.8 --set w_gap=0.0
"""
from __future__ import annotations

import argparse

from tune_diag_common import load_frames, run_configs, save, shipped_params


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--set", action="append", default=[],
                    help="knob=value; repeatable. Each becomes its OWN configuration.")
    ap.add_argument("--combo", action="append", default=[],
                    help="comma-joined knob=value list scored as one configuration")
    ap.add_argument("--n-jobs", type=int, default=24)
    ap.add_argument("--out", default="tune_diag_out/verify.json")
    a = ap.parse_args()

    base = shipped_params()
    configs = [("shipped", dict(base))]
    for s in a.set:
        k, v = s.split("=")
        p = dict(base)
        p[k] = int(v) if k in ("smooth_size", "min_arc_len") else float(v)
        configs.append((s, p))
    for c in a.combo:
        p = dict(base)
        for s in c.split(","):
            k, v = s.split("=")
            p[k] = int(v) if k in ("smooth_size", "min_arc_len") else float(v)
        configs.append((c, p))

    out = {"configs": {cid: p for cid, p in configs}, "splits": {}}
    for split in ("val", "test"):
        frames = load_frames(a.data, a.pred, split, base["prob_thr"])
        res = run_configs(frames, configs, n_jobs=a.n_jobs)
        print(f"\n--- {split} ({len(frames)} frames)")
        for cid, _ in configs:
            r = res[cid]
            print(f"  {cid:>34} mean_f1={r['mean_f1']:.4f} "
                  f"tp={sum(x['tp'] for x in r['rows']):4d} "
                  f"fp={sum(x['fp'] for x in r['rows']):4d}")
        out["splits"][split] = {cid: {"mean_f1": res[cid]["mean_f1"],
                                      "f1": [x["f1"] for x in res[cid]["rows"]],
                                      "frames": [x["frame"] for x in res[cid]["rows"]]}
                                for cid, _ in configs}
    save(a.out, out)


if __name__ == "__main__":
    main()
