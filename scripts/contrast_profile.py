#!/usr/bin/env python3
"""Does the generator produce the WEAK SEGMENTS the model is blind to?

The measurement that motivates this: within a microtubule the model does detect, 15.8 % of its
length falls below the operating threshold, and at those places the local contrast is **0.43x**
what it is where the model succeeds. Two post-processing fixes were ruled out first -- 69 % of
gap pixels sit under p = 0.05 (so hysteresis cannot reach them) and the evidence is not split
across orientation bins (sum 0.0200 vs max 0.0165). The model is genuinely blind there.

The hypothesis this tests is that it is blind because it was never shown such segments: if the
generator's along-filament intensity variation is milder than the real one, faint stretches do
not exist in training and cannot be learned. That is a generator fix, which is where this
project's evidence says effort belongs.

The statistic is deliberately WITHIN-filament and self-normalised. Absolute contrast varies
with exposure, illumination NA and microtubule height, none of which is the question; what
matters is how much the contrast varies ALONG one filament, expressed as a fraction of that
filament's own median. A real and a synthetic frame can then be compared without calibrating
anything.

    PYTHONPATH=src python scripts/contrast_profile.py \
        --real data/real/mt34_eval --synth data/synth_eval
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from instance.geometry import arclength, resample  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402

#: Background scale for the residual. Larger than a microtubule's width by an order of
#: magnitude, so the residual keeps the filament and removes illumination shading.
BG_SIGMA = 12.0
MIN_LEN_PX = 40.0


def profiles(data_dir: str, split: str | None, limit: int | None = None):
    """Per-filament contrast profiles, each normalised by its own median."""
    out = []
    paths = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
    for path in paths:
        fr = read_frame_h5(path)
        a = fr["attrs"]
        if split and str(a.get("split")) != split:
            continue
        if not fr["polylines"]:
            continue
        img = np.asarray(fr["image"], dtype=float)
        resid = np.abs(img - gaussian_filter(img, BG_SIGMA))
        H, W = resid.shape
        for pl in fr["polylines"]:
            p = np.asarray(pl, dtype=float)
            if len(p) < 2 or arclength(p)[-1] < MIN_LEN_PX:
                continue
            q = resample(p, ds=1.0)
            r = np.clip(np.rint(q[:, 1]).astype(int), 0, H - 1)
            c = np.clip(np.rint(q[:, 0]).astype(int), 0, W - 1)
            v = resid[r, c]
            med = float(np.median(v))
            if med <= 1e-9:
                continue
            out.append(v / med)                     # self-normalised profile
        if limit and len(out) >= limit:
            break
    return out


def describe(name: str, profs) -> dict:
    if not profs:
        return {}
    flat = np.concatenate(profs)
    # The quantity that matters: how much of a filament sits FAINT relative to its own body.
    frac = {t: float(np.mean([np.mean(p < t) for p in profs])) for t in (0.7, 0.5, 0.43, 0.3)}
    iqr = float(np.mean([np.percentile(p, 75) - np.percentile(p, 25) for p in profs]))
    print(f"\n=== {name} ===  {len(profs)} filaments, {len(flat)} centerline px")
    print(f"  within-filament spread (mean IQR of self-normalised contrast): {iqr:.3f}")
    print(f"  fraction of a filament's length below a given share of its own median:")
    for t, f in frac.items():
        print(f"    < {t:.2f} x median : {100 * f:5.1f} %")
    return {"iqr": iqr, **{f"frac_{t}": v for t, v in frac.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="data/real/mt34_eval")
    ap.add_argument("--synth", default="data/synth_eval")
    ap.add_argument("--split", default=None)
    args = ap.parse_args()

    r = describe("REAL", profiles(args.real, args.split))
    s = describe("SYNTHETIC", profiles(args.synth, args.split))
    if not (r and s):
        return
    print("\n=== verdict ===")
    print(f"  within-filament spread  real {r['iqr']:.3f}  vs  synth {s['iqr']:.3f}  "
          f"(synth / real = {s['iqr'] / max(r['iqr'], 1e-9):.2f})")
    key = "frac_0.43"
    print(f"  length below 0.43x its own median: real {100 * r[key]:.1f} %  "
          f"vs synth {100 * s[key]:.1f} %")
    if s[key] < 0.6 * r[key]:
        print("\n  => The generator produces filaments that are too UNIFORM along their length.")
        print("     Faint stretches of the kind the model fails on are under-represented in")
        print("     training, which is a generator fix, not a decoder or a threshold fix.")
    elif s[key] > 1.6 * r[key]:
        print("\n  => The generator is FAINTER along the filament than reality; the blindness is")
        print("     not explained by missing weak segments in training.")
    else:
        print("\n  => The distributions are comparable: the training set DOES contain such")
        print("     segments, so the blindness is not a missing-data problem and the fix lies")
        print("     elsewhere (supervision, capacity, or the crossing appearance).")


if __name__ == "__main__":
    main()
