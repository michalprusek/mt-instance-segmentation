"""Read 36b: a shared arc the multiplicity map marks may be traversed by both filaments.

The default path must stay bit-identical: `overlap=None` is what every published number was
produced with, so any drift there silently invalidates the whole campaign rather than failing.
"""
import numpy as np

from instance.instancer_a import instance_a


def _x_with_shared_stretch(h=120, w=200, half=1, shared=60):
    """Two filaments that run together for `shared` px in the middle, then separate.

    That is the geometry read 35 found behind 3-arm junctions: the skeleton fuses the pair into
    one arc with a junction at each end, and the arc graph can traverse it only once.
    """
    m = np.zeros((h, w), bool)
    x0, x1 = (w - shared) // 2, (w + shared) // 2
    yc = h // 2

    def draw(y_from, y_to, xa, xb):
        for x in range(xa, xb):
            y = int(round(y_from + (y_to - y_from) * (x - xa) / max(xb - xa - 1, 1)))
            m[max(y - half, 0):y + half + 1, x] = True

    draw(yc - 30, yc, 5, x0)          # A comes in from above
    draw(yc + 30, yc, 5, x0)          # B comes in from below
    m[yc - half:yc + half + 1, x0:x1] = True          # the shared stretch
    draw(yc, yc + 30, x1, w - 5)      # A leaves downward
    draw(yc, yc - 30, x1, w - 5)      # B leaves upward
    return m


def test_overlap_none_is_bit_identical():
    m = _x_with_shared_stretch()
    a, _ = instance_a(m, 0.25)
    b, _ = instance_a(m, 0.25, overlap=None)
    assert len(a) == len(b)
    for p, q in zip(a, b):
        assert np.array_equal(p, q)


def test_zero_overlap_map_changes_nothing():
    m = _x_with_shared_stretch()
    base, _ = instance_a(m, 0.25)
    off, _ = instance_a(m, 0.25, overlap=np.zeros(m.shape, np.float32))
    assert len(base) == len(off)
    for p, q in zip(base, off):
        assert np.array_equal(p, q)


def _spans_the_frame(poly, w):
    return poly[:, 0].min() < 10 and poly[:, 0].max() > w - 10


def test_shared_arc_admits_a_second_traversal():
    """Without the map one filament passes and the other is left as two stubs; with it, both pass.

    This is read 35's measured failure in miniature: the graph gives each of the two 3-arm
    junctions only one pair to make, so the second filament never gets through the shared arc.
    """
    m = _x_with_shared_stretch()
    ov = np.zeros(m.shape, np.float32)
    ov[m] = 1.0                                    # everything shared: the strongest signal
    base, _ = instance_a(m, 0.25)
    with_ov, _ = instance_a(m, 0.25, overlap=ov)

    assert sum(_spans_the_frame(p, m.shape[1]) for p in base) == 1
    assert len(with_ov) == 2
    assert all(_spans_the_frame(p, m.shape[1]) for p in with_ov)


def test_a_realistic_thin_map_marks_only_a_fraction_of_the_arc():
    """The exact map is centerline-thin while the fused ridge is long -- measured at 12.3 %.

    The first version of this suite marked the WHOLE mask as shared, which is not a map any
    network or generator produces, and that hid a criterion (`ov_frac` 0.5) which a real map can
    never satisfy: 1 junction-to-junction arc in 88 reached it. The criterion must fire on an arc
    that merely PASSES THROUGH a marked region.
    """
    m = _x_with_shared_stretch()
    ov = np.zeros(m.shape, np.float32)
    h, w = m.shape
    ov[h // 2 - 2:h // 2 + 3, w // 2 - 4:w // 2 + 5] = 1.0     # a small blob, as a real map gives
    base, _ = instance_a(m, 0.25)
    strict, _ = instance_a(m, 0.25, overlap=ov, params={"ov_frac": 0.5})
    loose, _ = instance_a(m, 0.25, overlap=ov, params={"ov_frac": 0.02})
    assert len(strict) == len(base)          # the strict criterion cannot fire on a thin map
    assert len(loose) < len(base)            # the loose one joins the stubs into whole filaments
    assert all(_spans_the_frame(p, w) for p in loose)


def test_arc_overlap_fraction_is_measured_on_the_arc():
    from instance.instancer_a import _arc_overlap_frac
    ov = np.zeros((10, 10), np.float32)
    ov[5, 2:6] = 1.0
    arc = np.array([[float(x), 5.0] for x in range(2, 10)])   # (x, y), half marked
    assert abs(_arc_overlap_frac(arc, ov, 0.5) - 0.5) < 1e-9
    assert _arc_overlap_frac(arc, ov, 1.5) == 0.0
