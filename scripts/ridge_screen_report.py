#!/usr/bin/env python3
"""Merge the chunked `decode_ridge.py` outputs and read the screen the pre-declared way.

Per variant: mean-F1 (macro over frames with GT), micro-F1, TP/FP/FN, junction identity and
bundle recovery pooled over frames, and a paired per-frame bootstrap of mean-F1 against the
baseline variant (10 000 resamples, frame = unit). Frames excluded by the roi303 repair can be
dropped with --exclude so old and new numbers refer to one frame set.

    python scripts/ridge_screen_report.py --inputs runs/ridge/out_chunk_*.json --baseline "B(0.98)"
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--baseline", default="B(0.98)")
    ap.add_argument("--exclude", default="", help="comma list of frame stems to drop")
    ap.add_argument("--min-gt", type=int, default=1)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    files = []
    for pat in args.inputs:
        files += glob.glob(pat)
    rows = {}
    for f in sorted(files):
        d = json.load(open(f))
        for v, rs in d["rows"].items():
            rows.setdefault(v, []).extend(rs)
    excl = set(x for x in args.exclude.split(",") if x)
    variants = list(rows)
    names = sorted(set(r["name"] for r in rows[variants[0]]) - excl)
    by = {v: {r["name"]: r for r in rows[v]} for v in variants}
    names = [n for n in names if all(n in by[v] for v in variants)
             and by[variants[0]][n]["n_gt"] >= args.min_gt]
    print(f"{len(names)} frames, {len(variants)} variants, baseline {args.baseline}")
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(names), (args.n_boot, len(names)))
    base = np.array([by[args.baseline][n]["f1"] for n in names])
    summary = {}
    hdr = f"{'variant':12s} {'meanF1':>7s} {'micro':>7s} {'TP':>6s} {'FP':>6s} {'FN':>6s} {'jid':>6s} {'bundle':>7s} {'fg%':>5s} {'dMean':>7s} {'CI95':>17s} {'p':>6s} {'wins':>5s}"
    print(hdr)
    for v in variants:
        f1 = np.array([by[v][n]["f1"] for n in names])
        tp = sum(by[v][n]["tp"] for n in names); fp = sum(by[v][n]["fp"] for n in names)
        fn = sum(by[v][n]["fn"] for n in names)
        micro = 2 * tp / max(1, 2 * tp + fp + fn)
        jn = sum(by[v][n].get("ji_n", 0) for n in names); jp = sum(by[v][n].get("ji_pres", 0) for n in names)
        bs = [by[v][n]["bundle"] for n in names if by[v][n].get("bundle") is not None]
        npairs = [by[v][n].get("n_pairs", 0) for n in names if by[v][n].get("bundle") is not None]
        bundle = float(np.average(bs, weights=npairs)) if bs and sum(npairs) else float("nan")
        fg = 100 * float(np.mean([by[v][n]["fg"] for n in names]))
        d = f1 - base
        boots = d[idx].mean(1)
        ci = np.percentile(boots, [2.5, 97.5])
        p = float(2 * min((boots <= 0).mean(), (boots >= 0).mean())) if v != args.baseline else 1.0
        wins = int((d > 0).sum())
        summary[v] = dict(mean_f1=float(f1.mean()), micro=micro, tp=tp, fp=fp, fn=fn,
                          junction_identity=(jp / jn if jn else float("nan")), bundle=bundle,
                          fg=fg, delta=float(d.mean()), ci=[float(ci[0]), float(ci[1])], p=p,
                          wins=wins)
        print(f"{v:12s} {f1.mean():7.4f} {micro:7.4f} {tp:6d} {fp:6d} {fn:6d} {jp / max(jn, 1):6.3f} {bundle:7.3f} {fg:5.2f} {d.mean():+7.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] {p:6.3f} {wins:5d}")
    if args.out:
        json.dump({"n_frames": len(names), "baseline": args.baseline, "summary": summary,
                   "frames": names}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
