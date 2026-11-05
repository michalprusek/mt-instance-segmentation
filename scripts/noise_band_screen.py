#!/usr/bin/env python3
"""Arm 42's generator check: does the render ever produce a QUIET frame?

Measured 2026-09-09 on off-filament pixels: the real corpus spans high-pass std q25 0.0183 to
q75 0.0756 (a 4.1x spread) while a render spans 0.0551 to 0.0821 (1.5x). The synthetic
distribution is TRUNCATED FROM BELOW -- it covers roughly the real upper half and never produces
the quiet frames the real corpus is full of. `noise_span_rel_range` is (0.3, 0.3), a FIXED photon
scale, so the amplitude has no per-frame spread at all.

A model trained only on noisy frames learns to fire on weak evidence; on a quiet real frame it
then fires on quiet structure -- which is the ordering failure read 38b localised to the 0.5-0.97
band. This screens candidate bands for `noise_span_rel_range` BEFORE any training, and is a VOID
condition, not a result: if no band brackets the real distribution, the arm is not applied.

    PYTHONPATH=src:scripts:synth python scripts/noise_band_screen.py --calib <file> \
        --backgrounds <dir> --bands 0.3:0.3,0.1:0.6,0.05:0.8 --n 14 --out noise_band.json
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

from diag_background_richness import norm, ridgeness, stats  # noqa: E402
from mt_generator import GenConfig, generate_frame, rasterize_mask  # noqa: E402

# CORRECTED 2026-09-09 after trap 6. The old targets were measured on real frames normalised over
# their OWN 1024-2048 px area while renders were normalised over 768 px, so they compared different
# percentile supports. Re-measured on the SAME real pixels at the render's 768 support:
#   hp_std       q25 0.0620  q50 0.1077  q75 0.1264   (was 0.0183 / 0.0285 / 0.0756)
#   ridge_share  q25 0.0036  q50 0.0143  q75 0.0289   (was 0.0041 / 0.0062 / 0.0193)
# The sign of the noise mismatch REVERSES: the render (0.0657) is 1.6x QUIETER than real, not
# noisier, so arm 42 (which made it quieter still) was pointed the wrong way and its recall
# falsifier duly fired. The correct direction has never been run.
REAL = {"hp_std": {"q25": 0.06204, "q50": 0.10773, "q75": 0.12636},
        "ridge_share": {"q25": 0.00364, "q50": 0.01431, "q75": 0.02894}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--backgrounds", required=True)
    ap.add_argument("--bands", default="0.3:0.3,0.15:0.5,0.08:0.7,0.05:1.0")
    ap.add_argument("--n", type=int, default=14)
    ap.add_argument("--exclude", type=float, default=10.0)
    ap.add_argument("--field", default="gain", choices=["gain", "read", "span_rel"],
                    help="which generator range to sweep; 'gain' is poisson_gain_range, where a "
                         "HIGHER gain means more photons and a QUIETER frame")
    ap.add_argument("--extra", default="", help="k=v generator overrides applied to every band")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cal = json.load(open(args.calib))
    over = cal.get("gen_overrides", {})
    extra = {}
    for kv in [x for x in args.extra.split(",") if x.strip()]:
        k, v = kv.split("=")
        extra[k] = json.loads(v)
    bgs = [p for p in sorted(glob.glob(os.path.join(args.backgrounds, "*"))) if not p.endswith(".json")]
    import imageio.v3 as iio

    out = {}
    for spec in args.bands.split(","):
        lo, hi = (float(v) for v in spec.split(":"))
        kw = {k: v for k, v in over.items() if k in GenConfig.__dataclass_fields__}
        kw.update({k: v for k, v in extra.items() if k in GenConfig.__dataclass_fields__})
        # `noise_span_rel_range` is inert unless noise_ref == "median", and the shipped
        # calibration uses "span" -- the screen caught that before a training run was spent on it.
        # The live levers in the span path are the photon gain (higher = QUIETER) and read noise.
        if args.field == "gain":
            kw["poisson_gain_range"] = (lo, hi)
        elif args.field == "read":
            kw["read_noise_range"] = (lo, hi)
        else:
            kw["noise_span_rel_range"] = (lo, hi)
        cfg = GenConfig(**kw)
        hp, rs = [], []
        for i in range(args.n):
            rng = np.random.default_rng(80000 + i)
            bg = np.asarray(iio.imread(bgs[i % len(bgs)]), float)
            raw, instances, _ = generate_frame(bg, rng, cfg, modality="irm")
            img = norm(np.asarray(raw, float))
            core = np.zeros(img.shape, bool)
            for ins in instances:
                core |= rasterize_mask(np.asarray(ins["centerline"], float), img.shape, half_width=1.0)
            m = distance_transform_edt(~core) > args.exclude
            s = stats(img, m)
            if s:
                hp.append(s["hp_std"]); rs.append(s["ridge_share"])
        if not hp:
            continue
        row = {}
        for name, v in (("hp_std", np.array(hp)), ("ridge_share", np.array(rs))):
            row[name] = {k: round(float(np.percentile(v, q)), 5)
                         for k, q in (("q25", 25), ("q50", 50), ("q75", 75))}
            row[name]["spread"] = round(row[name]["q75"] / max(row[name]["q25"], 1e-9), 2)
        row["n"] = len(hp)
        # does this band reach DOWN to the real quiet frames? that is the whole point of the arm
        row["reaches_real_q25"] = bool(row["hp_std"]["q25"] <= REAL["hp_std"]["q25"] * 1.25)
        out[f"{args.field} {lo:g}-{hi:g}"] = row
        print(f"{args.field} {lo:g}-{hi:g}: hp_std {row['hp_std']}  "
              f"ridge {row['ridge_share']['q50']}", flush=True)

    out["REAL_TARGET"] = REAL
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
