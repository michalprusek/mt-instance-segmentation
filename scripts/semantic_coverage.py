#!/usr/bin/env python3
"""Semantic coverage and precision of a probability map, BEFORE any instancing.

The instance score of a new foreground cannot be read while the instancer still carries parameters
tuned for a different one -- that is the shared-operating-point trap one level up from the threshold,
and it has manufactured phantom gaps in this project twice. This measures the two arms where they
are directly comparable: how much of the ground truth the foreground covers, and how much of the
foreground lies on ground truth. No instancer, no assembly, no matching.

    python scripts/semantic_coverage.py --pred-dir preds/EXEMPLAR_roi --label EXEMPLAR
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.spatial import cKDTree

TOL = 5.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--frames", default="/disk2/prusek/mt_work/.exemplar_frames",
                    help="the exported eval-frame cache; its masks are the ground truth")
    ap.add_argument("--manifest", default="",
                    help="scored-frame list; defaults to <pred-dir>/manifest.json, else all")
    ap.add_argument("--thrs", default="0.3,0.5,0.7,0.9,0.97,0.99")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    man = args.manifest or os.path.join(args.pred_dir, "manifest.json")
    if os.path.isfile(man):
        names = json.load(open(man))["scored"]
    else:
        names = json.load(open(os.path.join(args.frames, "frames.json")))["frames"]
    names = [n for n in names if os.path.isfile(os.path.join(args.pred_dir, n))]
    if not names:
        raise SystemExit(f"no predictions in {args.pred_dir} for the scored frames")

    print(f"{args.label or os.path.basename(args.pred_dir)}   {len(names)} frames\n")
    print(f'{"thr":>7s} {"GT covered":>11s} {"pred on GT":>11s} {"pred fg %":>10s} {"gt fg %":>9s}')
    for thr in [float(t) for t in args.thrs.split(",")]:
        cov, prec, pfg, gfg = [], [], [], []
        for n in names:
            gt = np.load(os.path.join(args.frames, n))["mask"]
            z = np.load(os.path.join(args.pred_dir, n))["prob"].astype(np.float32)
            if z.ndim == 3:
                z = z.max(0)
            m = z > thr
            gr, gc = np.nonzero(gt)
            pr, pc = np.nonzero(m)
            pfg.append(float(m.mean()))
            gfg.append(float(gt.mean()))
            if not len(gr):
                continue
            if not len(pr):
                cov.append(0.0)
                prec.append(0.0)
                continue
            tp = cKDTree(np.stack([pr, pc], 1).astype(float))
            tg = cKDTree(np.stack([gr, gc], 1).astype(float))
            cov.append(float((tp.query(np.stack([gr, gc], 1).astype(float), k=1)[0] <= TOL).mean()))
            prec.append(float((tg.query(np.stack([pr, pc], 1).astype(float), k=1)[0] <= TOL).mean()))
        print(f'{thr:7.3f} {100*np.mean(cov):10.1f}% {100*np.mean(prec):10.1f}% '
              f'{100*np.mean(pfg):9.2f}% {100*np.mean(gfg):8.2f}%')
    print("\n  GT covered = fraction of ground-truth pixels within 5 px of ANY foreground pixel")
    print("  pred on GT = fraction of foreground pixels within 5 px of ANY ground-truth pixel")


if __name__ == "__main__":
    main()
