#!/usr/bin/env python3
"""Turn the co-registered TIRF frames into an evaluation block, and measure how far its ground
truth can actually be trusted.

`real_eval_alice_tirf` holds 12 TIRF frames with 231 polylines whose h5 attribute says
`gt_source: coregistered_irm_job557` -- the annotation was drawn on the IRM channel of the same
acquisition and carried across. `real_eval_alice_job557` is the same 12 fields in IRM with the
same 231 curves, so the two form a paired cross-modality set, which no published dataset offers.

🔴 THE DEFECT THAT HAS TO BE STATED, because it is inherent and cannot be fixed here:
IRM sees whatever lies against the coverslip; TIRF sees only labelled tubulin within the
evanescent field. A microtubule present in one channel need not be present in the other, and the
transfer assumes it is. So a polyline can sit on nothing in TIRF (unlabelled, or lifted out of the
evanescent field), and a real TIRF filament can be missing from the annotation. On top of that the
transfer inherits any registration error, and the underlying IRM annotation is task-557 work whose
provenance is the same human-corrected model output as the rest of the Alice block.

This script therefore does not just convert -- it MEASURES that defect, per polyline, against a
displaced-curve null in the same frame. Curves carrying no TIRF signal are flagged in the h5, and
their DUTY CYCLE is stored alongside, so any score can be reported with and without them.

The statistic is the fraction of the curve's length that is bright -- NOT the median contrast along
it. The median is structurally blind to exactly the signal this data has: a real TIRF microtubule at
a low labelling ratio is a chain of puncta with a duty cycle around 0.3-0.5, so more than half the
samples sit in a dark gap and the median lands in the null. That mistake threw away visible
filaments, which is an error in the scorer's favour -- a curve wrongly flagged don't-care is a
filament the model is no longer asked to find.

    python scripts/build_tirf_eval.py --src <real_eval_alice_tirf/h5> --out data/real/tirf12_eval
"""
from __future__ import annotations

import argparse
import glob
import os

import h5py
import numpy as np


def densify(p, step=1.0):
    d = np.sqrt(((p[1:] - p[:-1]) ** 2).sum(1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 0:
        return p
    t = np.arange(0.0, s[-1], step)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)


def profile(img, poly, half=7.0):
    """Signed core-minus-flank intensity at every sample along a curve, at 1 px spacing.

    TIRF filaments are emissive, so a real one is POSITIVE where a fluorophore sits and ~0 in the
    gaps between them."""
    q = densify(poly)
    if len(q) < 5:
        return None
    t = np.gradient(q, axis=0)
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    H, W = img.shape

    def take(p):
        c = np.round(p[:, 0]).astype(int)
        r = np.round(p[:, 1]).astype(int)
        ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
        v = np.full(len(p), np.nan)
        v[ok] = img[r[ok], c[ok]]
        return v

    return take(q) - np.nanmean(np.stack([take(q + n * half), take(q - n * half)]), axis=0)


# The along-curve profile is sampled every 1 px while the PSF spreads a point source over 1-2 px,
# so neighbouring samples are not independent. Counting them as if they were would inflate every
# significance below. Taking every 3rd sample is a decorrelation step read off the optics, not a
# constant fitted to any result.
DECORR_PX = 3


def bright_count(img, poly, sigma, k=3.0):
    """(bright samples, independent samples) along a curve, above k sigma of the frame's noise.

    The statistic is how much of the curve is bright -- NOT the median contrast along it. The median
    is blind to exactly this data: a punctate filament at duty 0.3-0.5 has most of its samples in a
    gap, so its median sits in the null while the filament is plainly visible."""
    v = profile(img, poly)
    if v is None:
        return 0, 0
    v = v[np.isfinite(v)][::DECORR_PX]
    if len(v) < 8:
        return 0, 0
    return int((v > k * sigma).sum()), int(len(v))


def displaced_nulls(polys, H, W, rng, min_dist=8.0, tries=40):
    """Copies of each curve moved elsewhere in the same frame and AWAY from every annotation.

    The distance rule is the point: these fields are dense with long filaments, and a displaced
    copy that lands on another one inflates the noise floor until real but faint filaments read as
    empty. That error under-counted the signal by a third before it was caught."""
    from scipy.spatial import cKDTree
    anno = cKDTree(np.concatenate([densify(q, 2.0) for q in polys], axis=0))
    out = []
    for q in polys:
        for _ in range(tries):
            z = q + np.array([rng.uniform(-.4, .4) * W, rng.uniform(-.4, .4) * H])
            if not ((z[:, 0] >= 0).all() and (z[:, 0] < W).all()
                    and (z[:, 1] >= 0).all() and (z[:, 1] < H).all()):
                continue
            if anno.query(densify(z, 2.0), k=1)[0].min() > min_dist:
                out.append(z)
                break
    return out


def frame_noise_sigma(img, polys, seed=20260829):
    """Spread of the core-minus-flank statistic where there is no filament, in this frame's units.

    Everything downstream is expressed in multiples of this, so exposure and gain divide out and a
    bright frame is not mistaken for a well-labelled one."""
    H, W = img.shape
    rng = np.random.default_rng(seed)
    prof = [profile(img, z) for z in displaced_nulls(polys, H, W, rng)]
    prof = [v[np.isfinite(v)] for v in prof if v is not None]
    if not prof:
        return np.nan
    return float(np.std(np.concatenate(prof)))


def visible_mask(img, poly, sigma, k=3.0):
    """Per-sample bright/dark along one curve, at 1 px spacing, against the frame's own noise."""
    v = profile(img, poly)
    if v is None:
        return None
    return np.isfinite(v) & (v > k * sigma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory of the TIRF h5 frames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"],
                    help="12 frames is a development-sized block, so val by default")
    ap.add_argument("--k-noise", type=float, default=3.0,
                    help="a sample counts as bright above this many sigma of the off-filament noise")
    ap.add_argument("--fdr", type=float, default=0.05,
                    help="Benjamini-Hochberg false-discovery rate over the curves of one frame")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(20260829)
    tot = kept = 0
    per_frame = []

    for p in sorted(glob.glob(os.path.join(args.src, "*.h5"))):
        with h5py.File(p, "r") as f:
            img = f["image"][()].astype(np.float64)
            polys = [f["polylines"][k][()] for k in sorted(f["polylines"])]
            attrs = dict(f.attrs)
        H, W = img.shape
        polys = [np.asarray(q, float) for q in polys if len(q) >= 2]
        if not polys:
            continue

        # Null: the same curves displaced within the same frame. A displaced copy must also land
        # AWAY FROM EVERY ANNOTATION -- these fields are dense with long filaments, and a copy
        # that happens to fall on another one inflates the null until real but faint filaments
        # are scored as empty. That error rejected two thirds of the curves on the noisy frames
        # before it was caught by looking at the overlay.
        nulls = displaced_nulls(polys, H, W, rng)
        # noise floor from the raw samples off the filaments
        nprof = [profile(img, z) for z in nulls]
        nprof = [v[np.isfinite(v)] for v in nprof if v is not None]
        if not nprof:
            continue
        sigma = float(np.std(np.concatenate(nprof)))

        # p0: how often the FRAME ITSELF produces a bright sample where there is no filament.
        nb = [bright_count(img, z, sigma, args.k_noise) for z in nulls]
        n_hit = sum(b for b, m in nb)
        n_tot = sum(m for b, m in nb)
        if n_tot < 50:
            continue
        p0 = max(n_hit / n_tot, 1.0 / n_tot)

        # A curve carries signal when it is bright MORE OFTEN THAN THE BACKGROUND IS, by more than
        # chance. A bare percentile of the null cannot do this job: on a frame with no signal the
        # null duties are all zero, its 95th percentile is zero, and every curve with a single
        # bright pixel passes -- the test has no power floor and flags noise. A binomial tail has
        # one, and it also scales correctly with curve length, which a percentile ignores.
        from scipy.stats import binom
        cnt = np.array([bright_count(img, q, sigma, args.k_noise) for q in polys])
        pvals = np.array([binom.sf(b - 1, m, p0) if m >= 8 else 1.0 for b, m in cnt])
        # Benjamini-Hochberg across the curves of this frame: one test per curve, and a frame with
        # 38 curves would otherwise collect false positives by count alone.
        order = np.argsort(pvals)
        thr = args.fdr * (np.arange(1, len(pvals) + 1) / len(pvals))
        passed = pvals[order] <= thr
        cutoff = np.max(np.flatnonzero(passed)) if passed.any() else -1
        has = np.zeros(len(polys), bool)
        if cutoff >= 0:
            has[order[:cutoff + 1]] = True
        signal = np.array([b / m if m else 0.0 for b, m in cnt])      # duty, for the record
        tot += len(polys)
        kept += int(has.sum())
        per_frame.append((os.path.basename(p), len(polys), int(has.sum()),
                          float(np.nanmedian(signal)), p0))

        name = "tirf_" + os.path.basename(p).replace("_f00_tirf.h5", "").replace(".h5", "") + ".h5"
        with h5py.File(os.path.join(args.out, name), "w") as f:
            f.create_dataset("image", data=img.astype(np.float32), compression="gzip")
            g = f.create_group("polylines")
            for k, q in enumerate(polys):
                g.create_dataset(f"pl_{k:04d}", data=q)
            f.attrs.update(height=H, width=W, n_polylines=len(polys),
                           n_manual=len(polys), reviewed=True, source_task="tirf12",
                           split=args.split, modality="tirf",
                           gt_source=str(attrs.get("gt_source", "coregistered_irm")),
                           original_name=str(attrs.get("original_name", "")),
                           sources=",".join(["manual"] * len(polys)),
                           # per-curve flag: does this polyline sit on TIRF signal at all?
                           has_tirf_signal=has.astype(np.uint8),
                           # the duty cycle itself, not only the binary flag: a curve with signal on
                           # 70 % of its length systematically fails the metric's 95 % coverage
                           # criterion even when the model segments the visible extent perfectly, so
                           # the cutoff belongs at scoring time, not baked in here
                           tirf_duty=np.nan_to_num(signal),
                           tirf_duty_background=float(p0),
                           n_with_tirf_signal=int(has.sum()))

    print(f"{len(per_frame)} TIRF frames -> {args.out}   (split={args.split})\n")
    print(f'{"frame":46s} {"curves":>7s} {"with TIRF signal":>17s} {"median duty":>13s} {"bg rate":>9s}')
    for nm, n, k, med, p0 in per_frame:
        print(f'{nm[:46]:46s} {n:7d} {k:10d} ({100*k/max(n,1):3.0f} %) {med:13.2f} {p0:9.3f}')
    print(f'\nTOTAL {tot} curves, {kept} carry TIRF signal ({100*kept/max(tot,1):.1f} %)')
    print("\nThe rest were annotated on the co-registered IRM channel and have no counterpart in")
    print("TIRF -- unlabelled tubulin, or lifted out of the evanescent field. They are kept but")
    print("flagged (`has_tirf_signal`), so a score can be reported both ways. Reporting only the")
    print("all-curves number would penalise a method for not finding what is not there.")


if __name__ == "__main__":
    main()
