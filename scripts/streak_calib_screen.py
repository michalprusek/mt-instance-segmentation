#!/usr/bin/env python3
"""Arm 41's generator check, run BEFORE any training: does a streak rate reach the real band?

Pre-declared as a VOID condition, not a result: if no rate brings the render's off-filament ridge
share onto the real distribution (median 0.0062, q75 0.0193, measured on 48 roi303 frames), the
arm is void rather than negative -- the intervention would not have been applied.

Renders N frames per candidate rate with the shipped calibration plus `streak_rate_range`, and
measures the same statistic `diag_background_richness.py` measures on real frames, on OFF-FILAMENT
pixels only, so the two are directly comparable.

    PYTHONPATH=src:scripts:synth python scripts/streak_calib_screen.py --calib <file> --n 24
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("src", "scripts", "synth"):
    sys.path.insert(0, os.path.join(ROOT, sub))

from diag_background_richness import norm, ridgeness  # noqa: E402
from mt_generator import GenConfig, generate_frame, rasterize_mask  # noqa: E402

REAL = {"median": 0.00624, "q25": 0.00409, "q75": 0.01931}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--backgrounds", required=True)
    ap.add_argument("--rates", default="0,4,10,20,40,80")
    ap.add_argument("--ranges", default="",
                    help="explicit lo:hi rate bands, comma-separated, e.g. 5:400,20:300. The real "
                         "ridge share spans q25 0.0041 to q75 0.0193 (a 4.7x spread) while any "
                         "FIXED rate produces ~1.3x, so matching the distribution needs a WIDE "
                         "per-frame rate band, not a tuned scalar.")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--exclude", type=float, default=10.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cal = json.load(open(args.calib))
    over = cal.get("gen_overrides", cal if isinstance(cal, dict) else {})
    bgs = sorted(glob.glob(os.path.join(args.backgrounds, "*.npy")))
    if not bgs:
        bgs = [p for p in sorted(glob.glob(os.path.join(args.backgrounds, "*"))) if not p.endswith(".json")]
    print(f"{len(bgs)} backgrounds", flush=True)

    cands = []
    for r in args.rates.split(","):
        if not r.strip():
            continue
        v = float(r)
        cands.append((f"{v:g}", v, (max(v * 0.4, 0.0), v * 1.6) if v > 0 else None))
    for spec in [x for x in args.ranges.split(",") if x.strip()]:
        lo, hi = (float(v) for v in spec.split(":"))
        cands.append((f"band {lo:g}-{hi:g}", hi, (lo, hi)))

    out = {}
    for name, rate, band in cands:
        kw = {k: v for k, v in over.items() if k in GenConfig.__dataclass_fields__}
        kw["streak_rate_range"] = band
        kw["streak_rate"] = rate
        cfg = GenConfig(**kw)
        shares = []
        for i in range(args.n):
            rng = np.random.default_rng(70000 + i)
            bg = np.load(bgs[i % len(bgs)]) if bgs[i % len(bgs)].endswith(".npy") else None
            if bg is None:
                import imageio.v3 as iio
                bg = np.asarray(iio.imread(bgs[i % len(bgs)]), float)
            raw, instances, _ = generate_frame(bg.astype(float), rng, cfg, modality="irm")
            img = norm(np.asarray(raw, float))
            core = np.zeros(img.shape, bool)
            for ins in instances:
                cl = ins["centerline"] if isinstance(ins, dict) else ins
                core |= rasterize_mask(np.asarray(cl, float), img.shape, half_width=1.0)
            m = distance_transform_edt(~core) > args.exclude
            if m.sum() < 1000:
                continue
            shares.append(float((ridgeness(img) & m).sum() / m.sum()))
        if not shares:
            continue
        v = np.array(shares)
        out[name] = {k: round(float(np.percentile(v, q)), 5)
                     for k, q in (("q25", 25), ("median", 50), ("q75", 75))}
        out[name]["n"] = len(shares)
        out[name]["spread"] = round(out[name]["q75"] / max(out[name]["q25"], 1e-9), 2)
        print(f"{name:>14s}: {out[name]}", flush=True)

    out["REAL_TARGET"] = REAL
    best = min((k for k in out if k not in ("REAL_TARGET",)),
               key=lambda k: abs(out[k]["median"] - REAL["median"]), default=None)
    out["closest_rate"] = best
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
