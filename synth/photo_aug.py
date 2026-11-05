"""Photometric augmentation of a synthetic frame AFTER rendering, BEFORE the network sees it.

Why this exists (measured 2026-09-04, roi303 VAL, `scripts/diag_frame_calibration.py` and the
per-frame threshold sweep): the network's confidence along a real microtubule is a property of
the FRAME, not of the filament. The per-frame optimal decode threshold of SPARSE35 spans 0.30 to
0.9995 and a per-frame oracle threshold would add +0.048 mean-F1; in the densest frames 30 % of
the annotated centerline sits under p < 0.5. The real corpus is 23 acquisitions on several
microscopes, stored with anything from 16 to 36 000 distinct grey levels between the 1st and
99th percentile, at SNR 1.9-9, with either polarity. The generator renders one photometric
regime per frame (contrast, INA, noise gain, PSF) and the trainer applies only a [1, 99]
percentile stretch and a polarity flip -- nothing that varies the transfer curve, the
quantisation or the illumination the way real exports do.

Every operation here is a nuisance transform the annotation is invariant to, so the target is
untouched. The image is re-stretched with the SAME [1, 99] rule the evaluation applies to a real
frame, so the network's input distribution is defined by one rule at train and test time.

`strength = 0` is a strict no-op that draws NOTHING from the generator, so an unflagged run stays
bit-identical to every number measured before this module existed.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, zoom


def _stretch(im, lo_p=1.0, hi_p=99.0):
    lo, hi = np.percentile(im, [lo_p, hi_p])
    return np.clip((im - lo) / (hi - lo + 1e-6), 0.0, 1.0)


def low_frequency_field(shape, rng, cells=(3, 7)):
    """Smooth random field in [-1, 1] from a coarse grid, for illumination gradients / vignetting."""
    n = int(rng.integers(cells[0], cells[1] + 1))
    g = rng.normal(0.0, 1.0, (n, n))
    f = zoom(g, (shape[0] / n, shape[1] / n), order=3)
    f = f[:shape[0], :shape[1]]
    if f.shape != tuple(shape):
        f = np.pad(f, ((0, shape[0] - f.shape[0]), (0, shape[1] - f.shape[1])), mode="edge")
    return f / (np.abs(f).max() + 1e-8)


def photometric_augment(im: np.ndarray, rng: np.random.Generator, strength: float = 1.0) -> np.ndarray:
    """``im`` in [0, 1] (already percentile-stretched). Returns a float32 image in [0, 1].

    Draw order is fixed so a given (seed, strength) is reproducible. Each transform fires with its
    own probability; ``strength`` scales the amplitude ranges (1.0 = the declared recipe).
    """
    if strength <= 0:
        return im
    s = float(strength)
    out = np.asarray(im, np.float32).copy()
    H, W = out.shape

    # 1. low-frequency illumination: multiplicative gain + additive offset (vignetting, uneven
    #    interference background, a bright field-stop rim bleeding in)
    if rng.random() < 0.5:
        f = low_frequency_field((H, W), rng).astype(np.float32)
        a_mul = rng.uniform(0.05, 0.30) * s
        a_add = rng.uniform(0.0, 0.15) * s
        out = out * (1.0 + a_mul * f) + a_add * low_frequency_field((H, W), rng).astype(np.float32)

    # 2. gamma (display LUT / detector non-linearity)
    if rng.random() < 0.5:
        g = float(np.exp(rng.uniform(np.log(0.6), np.log(1.6)) * s))
        out = np.clip(out, 0.0, 1.0) ** g

    # 3. contrast and brightness
    if rng.random() < 0.5:
        c = 1.0 + rng.uniform(-0.4, 0.4) * s
        b = rng.uniform(-0.15, 0.15) * s
        out = (out - 0.5) * c + 0.5 + b

    # 4. blur (defocus / a coarser PSF than rendered)
    if rng.random() < 0.3:
        out = gaussian_filter(out, rng.uniform(0.3, 1.0) * s)

    # 5. additive Gaussian noise (read noise / a noisier camera than rendered)
    if rng.random() < 0.5:
        out = out + rng.normal(0.0, rng.uniform(0.005, 0.05) * s, out.shape).astype(np.float32)

    # 6. export path: a clipped percentile stretch followed by quantisation to L grey levels.
    #    The real block carries between 16 and ~36 000 levels across its [1, 99] span.
    if rng.random() < 0.5:
        pl = rng.uniform(0.0, 3.0)
        ph = rng.uniform(97.0, 100.0)
        out = _stretch(out, pl, ph)
        L = int(round(2 ** rng.uniform(4, 8)))            # 16 .. 256 levels
        out = np.round(out * (L - 1)) / (L - 1)

    # the evaluation's own normalisation, so train and test inputs obey one rule
    return _stretch(out).astype(np.float32)
