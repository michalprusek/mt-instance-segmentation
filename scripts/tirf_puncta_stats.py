#!/usr/bin/env python3
"""Measure how a TIRF microtubule is actually broken into puncta, to anchor the generator.

The generator renders a TIRF filament as a CONTINUOUS bright line. The real frames are not that:
at 500x dilution of labelled tubulin the fluorophores sit sparsely along the lattice, so a TIRF
microtubule is a chain of discrete puncta with dark gaps between them (visible at native resolution
in `data/enc_sensitivity_testset/tirf_zoom.png`). IRM images MASS and is continuous; TIRF images
LABELS and is not. That is a difference in the SHAPE of the signal, not just its polarity, and a
model trained only on continuous TIRF has never seen the gaps it must bridge.

This measures, along every annotated curve that carries signal:
  * duty cycle  -- the fraction of arclength that is above the local noise floor
  * ON  run lengths -- how long a punctum reads along the filament
  * OFF run lengths -- how long the dark gaps are
  * peak-to-peak variability of the punctum amplitudes

The numbers ANCHOR the sparse end of a wide labelling-density prior. They are deliberately NOT
fitted to: this corpus is one lab's labelling ratio, and other labs label denser (see the
biased-corpus rule in CLAUDE.md). The generator samples across continuous->sparse; these numbers
only say that the sparse end must reach this far.

    python scripts/tirf_puncta_stats.py --src <real_eval_alice_tirf/h5>
"""
from __future__ import annotations

import argparse
import glob
import os

import h5py
import numpy as np
from scipy.spatial import cKDTree


def densify(p, step=1.0):
    d = np.sqrt(((p[1:] - p[:-1]) ** 2).sum(1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 0:
        return p
    t = np.arange(0.0, s[-1], step)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)


def profile(img, poly, half=7.0):
    """Background-subtracted intensity along a curve: core minus the mean of the two flanks."""
    q = densify(poly)
    if len(q) < 5:
        return None, None
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

    return q, take(q) - np.nanmean(np.stack([take(q + n * half), take(q - n * half)]), axis=0)


def runs(mask):
    """Lengths of the True runs and of the False runs, in samples (= px, the curve is densified 1 px)."""
    on, off, cur, val = [], [], 0, None
    for m in mask:
        if val is None or m == val:
            cur += 1
        else:
            (on if val else off).append(cur)
            cur = 1
        val = m
    if val is not None:
        (on if val else off).append(cur)
    return np.array(on), np.array(off)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--pattern", default="atp_ch4",
                    help="only frames whose name contains this; the dim NOatp frames carry no "
                         "measurable signal so their 'gaps' would be pure noise")
    ap.add_argument("--exclude", default="NOatp")
    ap.add_argument("--k-noise", type=float, default=3.0,
                    help="a sample is ON above k times the off-filament noise sigma")
    args = ap.parse_args()

    duty, on_all, off_all, amp_cv, per_frame = [], [], [], [], []
    for p in sorted(glob.glob(os.path.join(args.src, "*.h5"))):
        nm = os.path.basename(p)
        if args.pattern not in nm or (args.exclude and args.exclude in nm):
            continue
        with h5py.File(p, "r") as f:
            img = f["image"][()].astype(np.float64)
            polys = [np.asarray(f["polylines"][k][()], float) for k in sorted(f["polylines"])]
        polys = [q for q in polys if len(q) >= 2]
        if not polys:
            continue

        # noise floor: the same statistic on curves displaced AWAY from every annotation
        anno = cKDTree(np.concatenate([densify(q, 2.0) for q in polys], axis=0))
        H, W = img.shape
        rng = np.random.default_rng(7)
        nvals = []
        for q in polys:
            for _ in range(40):
                z = q + np.array([rng.uniform(-.4, .4) * W, rng.uniform(-.4, .4) * H])
                if not ((z[:, 0] >= 0).all() and (z[:, 0] < W).all()
                        and (z[:, 1] >= 0).all() and (z[:, 1] < H).all()):
                    continue
                if anno.query(densify(z, 2.0), k=1)[0].min() <= 8.0:
                    continue
                _, v = profile(img, z)
                if v is not None:
                    nvals.append(v[np.isfinite(v)])
                break
        if not nvals:
            continue
        nv = np.concatenate(nvals)
        sigma = float(np.std(nv))
        cut = args.k_noise * sigma

        fd, fon, foff = [], [], []
        for q in polys:
            _, v = profile(img, q)
            if v is None:
                continue
            v = v[np.isfinite(v)]
            if len(v) < 30:
                continue
            m = v > cut
            if m.mean() < 0.05:          # a curve with no signal at all says nothing about gaps
                continue
            a, b = runs(m)
            fd.append(m.mean())
            fon.append(a)
            foff.append(b[1:-1] if len(b) > 2 else b)   # drop the runs that touch the curve ends
            pk = v[m]
            if len(pk) > 5:
                amp_cv.append(float(np.std(pk) / max(np.mean(pk), 1e-9)))
        if not fd:
            continue
        duty += fd
        on_all.append(np.concatenate(fon))
        off_all.append(np.concatenate([x for x in foff if len(x)]) if any(len(x) for x in foff)
                       else np.array([]))
        per_frame.append((nm, len(fd), float(np.median(fd)), sigma))

    on = np.concatenate(on_all) if on_all else np.array([])
    off = np.concatenate([x for x in off_all if len(x)]) if off_all else np.array([])
    duty = np.array(duty)

    print(f"PUNCTA ALONG REAL TIRF MICROTUBULES  ({len(per_frame)} frames, {len(duty)} curves, "
          f"ON = above {args.k_noise:.0f} sigma of the off-filament noise)\n")
    print(f'{"frame":46s} {"curves":>7s} {"median duty":>12s} {"noise sigma":>12s}')
    for nm, k, d, s in per_frame:
        print(f'{nm[:46]:46s} {k:7d} {d:12.2f} {s:12.1f}')

    if len(duty):
        print(f'\nduty cycle (fraction of arclength that is bright)')
        print(f'    p10/25/50/75/90 = ' + "/".join(f"{np.percentile(duty,q):.2f}"
                                                   for q in (10, 25, 50, 75, 90)))
    if len(on):
        print(f'\nON  run length (one punctum), px   p25/50/75/90 = ' +
              "/".join(f"{np.percentile(on,q):.0f}" for q in (25, 50, 75, 90)))
    if len(off):
        print(f'OFF run length (the gap),     px   p25/50/75/90 = ' +
              "/".join(f"{np.percentile(off,q):.0f}" for q in (25, 50, 75, 90)))
    if amp_cv:
        print(f'\npunctum amplitude CV (std/mean along one filament)  p25/50/75 = ' +
              "/".join(f"{np.percentile(amp_cv,q):.2f}" for q in (25, 50, 75)))
    print("\nThese anchor the SPARSE end of the generator's labelling-density prior. They are not")
    print("fitted to -- another lab's labelling ratio gives a different duty cycle, and the prior")
    print("must span continuous (duty 1.0) through to at least this sparse.")


if __name__ == "__main__":
    main()
