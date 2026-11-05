#!/usr/bin/env python3
"""Tabulate diag_ceiling_union.py outputs: one row per (band half-width, region, delete-far,
channels), pooled micro over the block and over frames with a name prefix (e.g. htw_).

    python scripts/ceil_table.py --dir runs/step2 --block hb --prefix htw_
"""
import argparse
import glob
import json
import os


def micro(fr):
    tp = sum(f["tp"] for f in fr); fp = sum(f["fp"] for f in fr); fn = sum(f["fn"] for f in fr)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r, tp, fp, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--block", required=True)
    ap.add_argument("--prefix", default=None)
    args = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(os.path.join(args.dir, f"ceil_{args.block}_*.json"))):
        d = json.load(open(f))
        r = d["result"]; fr = r["_frames"]
        sub = [x for x in fr if x["name"].startswith(args.prefix)] if args.prefix else fr
        m = micro(fr); ms = micro(sub)
        rows.append((float(d.get("band_hw", 1.0)), d["region"], float(d.get("delete_far", 0.0)), int(d.get("paint_channels", 0)),
                     m, ms, os.path.basename(f)))
    rows.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
    sub_lab = f"{args.prefix}only" if args.prefix else "-"
    print(f"{'hw':>4} {'region':<8} {'delfar':>6} {'K':>2} | {'block micro':>11} {'P':>6} {'R':>6} {'tp/fp/fn':>16} | {sub_lab:>9} {'P':>6} {'R':>6}   file")
    for hw, reg, df, K, m, ms, f in rows:
        print(f"{hw:>4} {reg:<8} {df:>6} {K:>2} | {m[0]:>11.4f} {m[1]:>6.3f} {m[2]:>6.3f} {m[3]:>5}/{m[4]:>4}/{m[5]:>4} | "
              f"{ms[0]:>9.4f} {ms[1]:>6.3f} {ms[2]:>6.3f}   {f}")


if __name__ == "__main__":
    main()
