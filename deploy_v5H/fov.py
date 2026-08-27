"""Field-of-view masking for MT-34.

The task-586 frames were acquired through an OCTAGONAL field stop: a central imaged region
surrounded by a hard transition into a saturated surround. Alice frames are full-frame and
have no such boundary.

That boundary is a strong, elongated intensity edge, and the v4b foreground fires on it: on
VAL frames the detections that lie away from any ground-truth centerline carry 2.06x the
residual contrast of an annotated microtubule and 3.65x the background, and rendering them
shows them lying on the octagon's corner wedges. The annotators, correctly, ignored it. The
generator composites synthetic microtubules onto real EMPTY IRM backgrounds, which apparently
carry no field stop, so the model never had reason to learn that a field boundary is not a
filament.

Masking the field of view is the standard remedy (every retinal-vessel benchmark ships an FOV
mask); it excludes the region where the instrument has no data rather than suppressing
inconvenient predictions. Frames without a field stop -- Alice -- come back fully valid, so
the same code path is safe for the whole benchmark.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import (binary_erosion, binary_fill_holes, label,
                           uniform_filter)


def fov_mask(image: np.ndarray, margin: float = 1.5, erode_px: int = 4,
             smooth: int = 5) -> np.ndarray:
    """Boolean mask of the imaged field.

    A pixel is inside when its (locally smoothed) value falls within the intensity band of
    the frame's central region, widened by ``margin`` times that band's width. The component
    containing the frame centre is kept, holes are filled, and the result is eroded by
    ``erode_px`` to drop the transition ring, whose gradient is what the segmenter fires on.

    ``erode_px=4`` is a deliberate compromise, measured on MT-34: a 10 px erosion removes 40%
    of the spurious prediction mass but deletes up to 6% of the ground-truth vertices on the
    worst frame -- microtubules that would then be scored as false negatives through no fault
    of the model. At 4 px the worst frame keeps 96% of its GT while ~16% of the spurious mass
    goes. The field stop is therefore only PART of the model-mask gap, and the erosion knob
    cannot close the rest without destroying ground truth.
    """
    img = np.asarray(image, dtype=float)
    if img.ndim != 2:
        img = img.squeeze()
    h, w = img.shape

    core = img[h // 4:3 * h // 4, w // 4:3 * w // 4]
    lo, hi = np.percentile(core, [0.5, 99.5])
    span = max(hi - lo, 1e-9)

    sm = uniform_filter(img, size=max(smooth, 1))
    inside = (sm >= lo - margin * span) & (sm <= hi + margin * span)
    inside = binary_fill_holes(inside)

    lab, n = label(inside)
    if n == 0:
        return np.ones_like(img, dtype=bool)
    # The imaged field is the component containing the frame centre; falling back to the
    # largest component would pick the surround on a frame that is mostly stop.
    centre = lab[h // 2, w // 2]
    keep = (lab == centre) if centre > 0 else (lab == (np.bincount(lab.ravel())[1:].argmax() + 1))

    if erode_px > 0:
        # Erode only against the FIELD STOP, never against the frame edge: Alice frames have
        # no stop and their microtubules run right up to the image border, so a plain erosion
        # would delete real ground truth there.
        keep = binary_erosion(keep, structure=np.ones((3, 3), dtype=bool),
                              iterations=int(erode_px), border_value=1)
    return keep if keep.any() else np.ones_like(img, dtype=bool)


def fov_fraction(image: np.ndarray, **kw) -> float:
    """Fraction of the frame that is inside the field of view."""
    return float(fov_mask(image, **kw).mean())
