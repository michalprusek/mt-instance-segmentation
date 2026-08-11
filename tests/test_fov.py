import glob
import os

import numpy as np
import pytest

from mt_bench.cvat_import import read_frame_h5
from mt_bench.fov import fov_fraction, fov_mask

DATA = "data/real/mt34_eval"


def test_synthetic_field_stop_is_excluded():
    # Interior noise around 100, saturated surround at 255 outside a disc.
    rng = np.random.default_rng(0)
    img = np.full((200, 200), 255.0)
    yy, xx = np.mgrid[:200, :200]
    disc = (yy - 100) ** 2 + (xx - 100) ** 2 < 70 ** 2
    img[disc] = 100.0 + rng.normal(0, 3, disc.sum())
    m = fov_mask(img, erode_px=5)
    assert m[100, 100] and not m[5, 5]
    assert 0.2 < m.mean() < 0.45      # ~pi*65^2/200^2 = 0.33 after erosion


def test_frame_without_a_field_stop_stays_fully_valid():
    rng = np.random.default_rng(1)
    img = 100.0 + rng.normal(0, 3, (200, 200))
    assert fov_fraction(img, erode_px=5) > 0.9


@pytest.mark.skipif(not glob.glob(os.path.join(DATA, "*.h5")),
                    reason="MT-34 not built")
def test_every_gt_vertex_lies_inside_the_field_of_view():
    """The strongest available check: the annotators only drew inside the imaged field, so a
    correct mask must contain every ground-truth vertex on every frame."""
    worst = 1.0
    for path in sorted(glob.glob(os.path.join(DATA, "*.h5"))):
        fr = read_frame_h5(path)
        if not fr["polylines"]:
            continue
        m = fov_mask(fr["image"])
        h, w = m.shape
        pts = np.concatenate([np.asarray(p, float) for p in fr["polylines"]])
        rr = np.clip(np.rint(pts[:, 1]).astype(int), 0, h - 1)
        cc = np.clip(np.rint(pts[:, 0]).astype(int), 0, w - 1)
        frac = float(m[rr, cc].mean())
        worst = min(worst, frac)
        assert frac > 0.95, f"{os.path.basename(path)}: only {frac:.3f} of GT inside FOV"
    assert worst > 0.95
