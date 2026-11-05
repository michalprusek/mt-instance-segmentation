#!/usr/bin/env python3
"""Does composing the one-at-a-time optima reproduce a joint optimum? RUNS ON TULEN.

If the knobs were separable, setting every knob to its own OAT argmax simultaneously would land
at least as high as any single-knob move. Doing it and measuring is the cheapest possible test
of whether coordinate descent is enough, and it extends the 2-knob grids to all 15 at once.

Every argmax is selected on the SAME frames the composed configuration is then scored on, so
the val number is in-sample and optimistic by construction. The held-out split is therefore
scored too, and that is the number to quote.

    ~/dinov3_env/bin/python scripts/tune_diag_compose.py --oat tune_diag_out/oat21.json \
        --data data/synth_link --pred /home/prusek/mt_enc_exp/synth_link_pred
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from tune_diag_common import load_frames, run_configs, save, shipped_params
from tune_diag_oat import INT_GRIDS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oat", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--n-jobs", type=int, default=24)
    ap.add_argument("--out", default="tune_diag_out/compose.json")
    a = ap.parse_args()
    with open(a.oat) as fh:
        d = json.load(fh)
    base, R = shipped_params(), d["results"]

    ranked = []
    for k, vals in d["grids"].items():
        f1 = np.array([R[f"{k}#{i}"]["mean_f1"] for i in range(len(vals))])
        ranked.append((float(f1.max() - f1.min()), k, vals[int(np.argmax(f1))],
                       float(f1.max())))
    ranked.sort(reverse=True)

    def compose(n: int) -> dict:
        p = dict(base)
        for _, k, am, _ in ranked[:n]:
            p[k] = int(am) if k in INT_GRIDS else float(am)
        return p

    configs = [("shipped", dict(base))]
    for n in (1, 2, 3, 5, 8, len(ranked)):
        configs.append((f"argmax_top{n}", compose(n)))

    out = {"ranked": [{"knob": k, "spread": s, "argmax": am, "f1_at_argmax": f}
                      for s, k, am, f in ranked], "splits": {}}
    for split in ("val", "test"):
        frames = load_frames(a.data, a.pred, split, base["prob_thr"])
        res = run_configs(frames, configs, n_jobs=a.n_jobs)
        print(f"\n--- {split} ({len(frames)} frames) "
              f"{'IN-SAMPLE, argmaxes chosen here' if split == 'val' else 'HELD OUT'}")
        for cid, _ in configs:
            r = res[cid]
            print(f"  {cid:>14} mean_f1={r['mean_f1']:.4f} "
                  f"tp={sum(x['tp'] for x in r['rows']):4d} "
                  f"fp={sum(x['fp'] for x in r['rows']):4d}")
        out["splits"][split] = {cid: {"mean_f1": res[cid]["mean_f1"],
                                      "f1": [x["f1"] for x in res[cid]["rows"]],
                                      "tp": [x["tp"] for x in res[cid]["rows"]],
                                      "fp": [x["fp"] for x in res[cid]["rows"]],
                                      "frames": [x["frame"] for x in res[cid]["rows"]]}
                                for cid, _ in configs}
    out["configs"] = {cid: p for cid, p in configs}
    save(a.out, out)


if __name__ == "__main__":
    main()
