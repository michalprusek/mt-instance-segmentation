#!/usr/bin/env python3
"""Tabulate diag_fp_anatomy.py logs side by side (baseline / ceiling residual / oracle) and print
residual - oracle: the classes of the model-owned merging term. Optional frame-name prefix
restricts to one source (e.g. htw_).

    python scripts/anat_table.py --dir runs/step2 --block hb --prefix htw_
"""
import argparse
import collections
import os
import re

KEYS_FP = ["off_other", "off_border", "overshoot", "frag_overshoot", "fragment", "gap_bridge", "junction"]
KEYS_FN = ["covered_split", "partial", "missed"]
PAT = re.compile(r"nGT=(\d+) tp=(\d+) fp=(\d+) fn=(\d+) \| FP (.*?) \| FN (.*?) \| gaps")


def tot(path, prefix):
    fp, fn, T = collections.Counter(), collections.Counter(), collections.Counter()
    for line in open(path):
        if prefix and not line.startswith(prefix):
            continue
        m = PAT.search(line)
        if not m:
            continue
        T["tp"] += int(m[2]); T["fp"] += int(m[3]); T["fn"] += int(m[4])
        for k, v in re.findall(r"(\w+)=(\d+)", m[5]):
            fp[k] += int(v)
        for k, v in re.findall(r"(\w+)=(\d+)", m[6]):
            fn[k] += int(v)
    return T, fp, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--block", required=True, help="hb or roi303")
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--base-tag", default="SPARSE35_der")
    args = ap.parse_args()
    rows = {}
    for lab, tag in (("baseline", args.base_tag), ("resid(band+del)", "ceilresid"), ("oracle", "oracle")):
        p = os.path.join(args.dir, f"fp_{args.block}_{tag}.log")
        if os.path.exists(p):
            rows[lab] = tot(p, args.prefix)
    hdr = f"{'':<16}{'tp':>6}{'fp':>6}{'fn':>6} | " + "".join(f"{k[:10]:>11}" for k in KEYS_FP) + " | " + "".join(f"{k[:10]:>11}" for k in KEYS_FN)
    print(f"== {args.block} {'prefix ' + args.prefix if args.prefix else 'all frames'}")
    print(hdr)
    for lab, (T, fp, fn) in rows.items():
        print(f"{lab:<16}{T['tp']:>6}{T['fp']:>6}{T['fn']:>6} | " + "".join(f"{fp[x]:>11}" for x in KEYS_FP) + " | " + "".join(f"{fn[x]:>11}" for x in KEYS_FN))
    if "resid(band+del)" in rows and "oracle" in rows:
        (T1, f1, n1), (T0, f0, n0) = rows["resid(band+del)"], rows["oracle"]
        print(f"{'resid - oracle':<16}{T1['tp'] - T0['tp']:>6}{T1['fp'] - T0['fp']:>6}{T1['fn'] - T0['fn']:>6} | "
              + "".join(f"{f1[x] - f0[x]:>11}" for x in KEYS_FP) + " | " + "".join(f"{n1[x] - n0[x]:>11}" for x in KEYS_FN))


if __name__ == "__main__":
    main()
