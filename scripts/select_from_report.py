#!/usr/bin/env python3
"""Re-run checkpoint selection from a finished ``train_gated.json``. RUNS ON TULEN.

Two uses:

* the training run in progress on 2026-08-11 loaded ``select_checkpoint`` **before** the
  ``min_frames`` partial-collapse guard was added, so its own choice must be re-derived rather
  than trusted;
* more generally, selection is a decision over recorded metrics and should never require a
  retrain to revisit. Every epoch's full battery is in the report.

It also prints the counterfactual selections -- last epoch, best coverage F1 -- so the claim
"the gate picks a different checkpoint than the metric we used to use" is checkable rather
than assumed.

    cd /home/prusek/mt_enc_exp/mt34_work
    SEG_MODE=ori PYTHONPATH=src ~/dinov3_env/bin/python scripts/select_from_report.py \
        --report data/enc_sensitivity_testset/train_gated.json \
        --out /home/prusek/mt_enc_exp/dino_seg_ori_gated.pth
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))
sys.path.insert(0, "/home/prusek/mt_enc_exp/scripts")

os.environ.setdefault("SEG_MODE", "ori")
os.environ.setdefault("SEG_BACKBONE", "dinov2")
os.environ.setdefault("SEG_INPUT", "raw")

from mt_bench.fg_quality import (passes_overfiring_gate, quality_score,  # noqa: E402
                                 select_checkpoint)


def cov_f1(h: dict) -> float:
    p, r = h.get("prec2", 0.0), h.get("rec2", 0.0)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="data/enc_sensitivity_testset/train_gated.json")
    ap.add_argument("--out", default=None, help="write the winning FULL state_dict here")
    ap.add_argument("--synth-fg", type=float, default=None)
    ap.add_argument("--min-frames", type=int, default=None,
                    help="required VAL frame count; defaults to the max seen in the report")
    args = ap.parse_args()

    report = json.load(open(args.report))
    history = report["history"]
    synth_fg = args.synth_fg if args.synth_fg is not None else report.get("synth_fg", 0.016)
    min_frames = args.min_frames or max(h.get("n_frames", 0) for h in history)

    print(f"{'epoch':>6s} {'frames':>7s} {'contin.':>8s} {'covF1':>7s} {'fg%':>6s} "
          f"{'fg_e%':>6s} {'rec2':>6s} {'gate':>6s}")
    for h in history:
        full = h.get("n_frames", 0) >= min_frames
        gate = passes_overfiring_gate(h, synth_fg) and full
        print(f"{h['epoch']:6d} {h.get('n_frames', 0):7d} {quality_score(h):8.3f} "
              f"{cov_f1(h):7.3f} {100 * h.get('fg', float('nan')):6.2f} "
              f"{100 * h.get('fg_empty', float('nan')):6.2f} "
              f"{h.get('rec2', float('nan')):6.3f} "
              f"{'PASS' if gate else ('short' if not full else 'FAIL'):>6s}")

    best = select_checkpoint(history, synth_fg=synth_fg, min_frames=min_frames)
    if best is None:
        print("\nNO checkpoint passed the gate -- nothing selected.")
        raise SystemExit(2)

    chosen, last = history[best], history[-1]
    by_cov = max(history, key=cov_f1)
    print("\n=== selection ===")
    for label, h in (("fg_quality gate", chosen), ("'last epoch'", last),
                     ("'best coverage F1'", by_cov)):
        print(f"  {label:20s} -> epoch {h['epoch']:3d}  "
              f"continuity {quality_score(h):.3f}  covF1 {cov_f1(h):.3f}")
    if chosen["epoch"] == last["epoch"] == by_cov["epoch"]:
        print("  (all three agree -- the gate changed nothing on this run)")

    if args.out:
        import torch

        from dino_seg import DinoSeg
        model = DinoSeg().to("cuda")
        # The frozen backbone is identical at every epoch, so loading the winner's trainable
        # params reconstructs the full model exactly.
        model.load_state_dict(torch.load(chosen["ckpt"], map_location="cuda"), strict=False)
        torch.save(model.state_dict(), args.out)
        print(f"wrote {args.out} (epoch {chosen['epoch']})")


if __name__ == "__main__":
    main()
