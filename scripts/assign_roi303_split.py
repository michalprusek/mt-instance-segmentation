#!/usr/bin/env python3
"""Declare the VAL/TEST split of the ROI-303 block, once, before anything is measured on it.

The block is 281 frames of human-drawn annotation with no model anywhere in its provenance. That
makes it the only clean measuring instrument the project has, and it is worth more as a matched
pair of splits than as one large test set: without a clean VAL every development decision keeps
being made on MT-34, whose 585 half is 69 % model-drawn and inflated by a measured +0.18 to +0.30.

The split is STRATIFIED rather than random or alternating. The frames come from several
acquisition sessions, visible as distinct image shapes, and they differ enormously in crowding
(3 to 300 curves per frame). An unstratified draw can put most of one session, or most of the
dense frames, on one side, and then the two halves are not comparable and neither is any
before/after measured across them.

Strata are crowding terciles; image shape is spread by ordering within each. Frames alternate
VAL/TEST with a counter that carries over between strata, so odd remainders cannot accumulate

Once written, this is fixed. Re-running with a different seed after seeing any result would be
choosing the split that flatters the answer.

    python scripts/assign_roi303_split.py --data-dir data/real/roi303_eval --write
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/real/roi303_eval")
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--write", action="store_true", help="write the split into the h5 attrs")
    ap.add_argument("--declare", default="docs/roi303_split.json")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.data_dir, "*.h5")))
    if not paths:
        raise SystemExit(f"no frames under {args.data_dir}")

    info = []
    for p in paths:
        with h5py.File(p, "r") as f:
            info.append(dict(name=os.path.basename(p),
                             shape=tuple(int(v) for v in f["image"].shape),
                             n=int(len(f["polylines"]))))
    n_all = np.array([i["n"] for i in info])
    q1, q2 = np.percentile(n_all, [33.3, 66.7])

    def stratum(i):
        return 0 if i["n"] <= q1 else (1 if i["n"] <= q2 else 2)

    groups = defaultdict(list)
    for i in info:
        groups[stratum(i)].append(i)

    # Alternate with a counter that CARRIES OVER between strata. Stratifying on exact image shape
    # was too fine -- most shapes occur once, and every odd remainder then landed on the same
    # side, giving 155/126. Crowding is the stratum that has to balance; shape is spread by
    # ordering within it, and the carry-over stops the remainders accumulating one way.
    rng = np.random.default_rng(args.seed)
    split, k = {}, 0
    for t in sorted(groups):
        g = sorted(groups[t], key=lambda i: (i["shape"], i["name"]))
        for j in rng.permutation(len(g)):
            split[g[j]["name"]] = "val" if k % 2 == 0 else "test"
            k += 1

    val = [i for i in info if split[i["name"]] == "val"]
    tst = [i for i in info if split[i["name"]] == "test"]
    print(f"{len(info)} frames -> {len(val)} val / {len(tst)} test   "
          f"(seed {args.seed}, stratified by crowding tercile, shape spread within)\n")
    print(f'{"":10s} {"frames":>7s} {"polylines":>10s} {"per frame p25/50/75":>22s}')
    for lab, g in (("VAL", val), ("TEST", tst)):
        a = np.array([i["n"] for i in g])
        print(f'{lab:10s} {len(g):7d} {int(a.sum()):10d} '
              f'{np.percentile(a,25):6.0f}/{np.percentile(a,50):5.0f}/{np.percentile(a,75):6.0f}')
    print(f'\n{"image shape":>16s} {"val":>5s} {"test":>5s}')
    shapes = sorted({i["shape"] for i in info})
    for s in shapes:
        v = sum(1 for i in val if i["shape"] == s)
        t = sum(1 for i in tst if i["shape"] == s)
        print(f'{str(s):>16s} {v:5d} {t:5d}')

    if args.write:
        for p in paths:
            with h5py.File(p, "r+") as f:
                f.attrs["split"] = split[os.path.basename(p)]
        os.makedirs(os.path.dirname(args.declare) or ".", exist_ok=True)
        json.dump(dict(seed=args.seed, n_val=len(val), n_test=len(tst),
                       method="stratified by crowding tercile, alternating with carry-over across strata "
                              "in a seeded shuffle; image shape spread by ordering within",
                       declared="2026-08-28, before any model was scored on this block",
                       split=split),
                  open(args.declare, "w"), indent=1, sort_keys=True)
        print(f"\nwritten into the h5 attrs and declared in {args.declare}")
    else:
        print("\n(dry run -- pass --write to apply)")


if __name__ == "__main__":
    main()
