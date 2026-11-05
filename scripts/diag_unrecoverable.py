#!/usr/bin/env python3
"""What do the microtubules the model NEVER finds actually look like?

Everything post-hoc is closed and the only route left is generator contrast realism. This names
what the generator would have to produce. A GT curve is UNRECOVERABLE if it never reaches the
headline metric's 0.80 coverage at ANY threshold from 0.50 to 0.99 — no operating point, no
decoder and no repair can rescue it, so it is a pure evidence deficit.

For each curve it measures what the IMAGE shows, independently of the model:
  * local contrast |mean(on-curve) - mean(background ring)| in percentile-normalised units,
    polarity-agnostic because IRM filaments flip sign;
  * local noise from the background ring, so contrast/noise is a per-curve SNR;
  * length and the model's own best coverage.
and reports the SNR distribution of the unrecoverable set against the recoverable one, so the
answer is "the generator must reach down to SNR x", not "make it more realistic".

    PYTHONPATH=src:scripts python scripts/diag_unrecoverable.py --data data/real/roi303_eval \
        --pred-dir <dump> --out unrecoverable.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import map_coordinates

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("src", "scripts"):
    sys.path.insert(0, os.path.join(ROOT, sub))

from instance.geometry import resample  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402

UP = 1.5


def curve_snr(img01, pts, off=6.0):
    """Polarity-agnostic local contrast over local noise, from the image alone."""
    d = np.gradient(pts, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    on = map_coordinates(img01, [pts[:, 1], pts[:, 0]], order=1, mode="nearest")
    ring = []
    for s in (-1, 1):
        q = pts + s * off * n
        ring.append(map_coordinates(img01, [q[:, 1], q[:, 0]], order=1, mode="nearest"))
    bg = np.concatenate(ring)
    sigma = float(np.std(bg))
    return float(abs(np.mean(on) - np.mean(bg))), sigma, (abs(np.mean(on) - np.mean(bg)) / (sigma + 1e-9))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--thrs", default="0.5,0.7,0.9,0.96,0.99")
    ap.add_argument("--need", type=float, default=0.80)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    thrs = [float(t) for t in args.thrs.split(",")]
    rows = []
    for p in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        stem = os.path.basename(p)[:-3]
        npz = os.path.join(args.pred_dir, stem + ".npz")
        if not os.path.exists(npz):
            continue
        fr = read_frame_h5(p)
        if str(fr["attrs"].get("split")) != args.split:
            continue
        prob = np.load(npz)["prob"].astype(np.float32)[0]
        img = np.asarray(fr["image"], float)
        lo, hi = np.percentile(img, [1, 99])
        img01 = np.clip((img - lo) / (hi - lo + 1e-9), 0, 1)
        from scipy.ndimage import zoom
        img_e = zoom(img01, UP, order=1)
        for q in fr["polylines"]:
            pts = resample(np.asarray(q, float) * UP, ds=1.0)
            if len(pts) < 12:
                continue
            v = map_coordinates(prob, [pts[:, 1], pts[:, 0]], order=1, mode="nearest")
            best = max(float((v >= t).mean()) for t in thrs)
            c, s, snr = curve_snr(img_e, pts)
            rows.append({"frame": stem, "len": len(pts), "best_cov": best,
                         "contrast": c, "sigma": s, "snr": snr})
        print(stem, flush=True)

    R = rows
    bad = [r for r in R if r["best_cov"] < args.need]
    good = [r for r in R if r["best_cov"] >= args.need]
    def q(v, k):
        return {str(x): round(float(np.percentile([r[k] for r in v], x)), 4) for x in (10, 25, 50, 75, 90)}
    summ = {"curves": len(R), "need": args.need, "thrs": thrs,
            "unrecoverable_n": len(bad), "unrecoverable_share": round(len(bad) / max(len(R), 1), 4),
            "UNRECOVERABLE": {"snr": q(bad, "snr"), "contrast": q(bad, "contrast"),
                              "sigma": q(bad, "sigma"), "len": q(bad, "len")},
            "RECOVERABLE": {"snr": q(good, "snr"), "contrast": q(good, "contrast"),
                            "sigma": q(good, "sigma"), "len": q(good, "len")}}
    if bad and good:
        summ["snr_ratio_median"] = round(np.median([r["snr"] for r in good])
                                         / max(np.median([r["snr"] for r in bad]), 1e-9), 2)
    json.dump(summ, open(args.out, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
