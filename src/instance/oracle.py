"""Oracle (ground-truth-derived) masks and orientation channels.

These stand in for a perfect semantic model, so the instancer can be developed and tuned
without the segmenter's errors in the loop. The orientation channels reproduce the amodal
K=6 "overpass" representation of ``dino_seg_ori_v4b.pth``: each microtubule is painted into
the channel of its LOCAL tangent bin, so a crossing writes into two different channels and
the two filaments stay separable at the shared pixel.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation

from instance.geometry import resample, segment_angles

# Sub-pixel step used when stamping polylines, so the rasterised centerline has no gaps
# even on diagonal runs.
_STAMP_DS = 0.4


def _footprint(half_width: float) -> np.ndarray:
    r = int(np.ceil(half_width))
    if r < 1:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (yy ** 2 + xx ** 2) <= half_width ** 2 + 1e-9


def _upscaled_shape(shape: tuple[int, int], up: float) -> tuple[int, int]:
    return int(round(shape[0] * up)), int(round(shape[1] * up))


def _stamp_centerline(points: np.ndarray, out: np.ndarray, up: float) -> np.ndarray:
    """Rasterise one polyline's centerline pixels into ``out`` (modified in place)."""
    h, w = out.shape
    pts = resample(np.asarray(points, dtype=float) * up, ds=_STAMP_DS)
    if len(pts) == 0:
        return out
    cc = np.clip(np.rint(pts[:, 0]).astype(int), 0, w - 1)
    rr = np.clip(np.rint(pts[:, 1]).astype(int), 0, h - 1)
    out[rr, cc] = True
    return out


def oracle_mask(polylines, shape: tuple[int, int], half_width: float = 1.0,
                up: float = 1.5) -> np.ndarray:
    """Union foreground mask in the UPSCALED frame.

    ``shape`` is the native ``(H, W)``; the returned mask is ``(H*up, W*up)``.
    ``half_width=1.0`` reproduces the ``mask_hw=1.0`` training convention (a 3-px band).
    """
    hi = np.zeros(_upscaled_shape(shape, up), dtype=bool)
    for p in polylines:
        _stamp_centerline(p, hi, up)
    return binary_dilation(hi, structure=_footprint(half_width))


def oracle_instance_masks(polylines, shape: tuple[int, int], half_width: float = 1.0,
                          up: float = 1.5) -> list[np.ndarray]:
    """One mask per GT polyline, in the upscaled frame."""
    out = []
    fp = _footprint(half_width)
    for p in polylines:
        m = np.zeros(_upscaled_shape(shape, up), dtype=bool)
        _stamp_centerline(p, m, up)
        out.append(binary_dilation(m, structure=fp))
    return out


def oracle_ori_channels(polylines, shape: tuple[int, int], K: int = 6,
                        half_width: float = 1.0, up: float = 1.5) -> np.ndarray:
    """Amodal orientation-keyed channels, ``(K, H*up, W*up)`` float32 in {0, 1}.

    Bin ``b`` covers tangent directions ``[b*180/K, (b+1)*180/K)`` degrees (mod 180, since a
    filament has no head or tail). A pixel shared by two filaments of different orientation
    is set in two channels -- that is what keeps the crossing resolvable.
    """
    hi_shape = _upscaled_shape(shape, up)
    chans = np.zeros((K, *hi_shape), dtype=bool)
    h, w = hi_shape
    width_deg = 180.0 / K

    for p in polylines:
        pts = resample(np.asarray(p, dtype=float) * up, ds=_STAMP_DS)
        if len(pts) < 2:
            continue
        ang = np.rad2deg(segment_angles(pts)) % 180.0
        # Give every sample the angle of the segment it starts, and the last sample the
        # angle of the segment that ends at it.
        ang = np.concatenate([ang, ang[-1:]])
        bins = np.clip((ang // width_deg).astype(int), 0, K - 1)
        cc = np.clip(np.rint(pts[:, 0]).astype(int), 0, w - 1)
        rr = np.clip(np.rint(pts[:, 1]).astype(int), 0, h - 1)
        for b in range(K):
            sel = bins == b
            if sel.any():
                chans[b, rr[sel], cc[sel]] = True

    fp = _footprint(half_width)
    return np.stack([binary_dilation(chans[b], structure=fp)
                     for b in range(K)]).astype(np.float32)
