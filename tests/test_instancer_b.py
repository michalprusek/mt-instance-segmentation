import numpy as np

from instance.geometry import total_length
from instance.instancer_b import instance_b
from instance.lifted import refine_theta
from instance.oracle import oracle_ori_channels

KMAX = 0.25


def test_refine_theta_preserves_the_peak_orientation():
    ch = np.zeros((6, 4, 4), dtype=np.float32)
    ch[1] = 1.0                                   # bin 1 = 30..60 deg, centre 45
    out = refine_theta(ch, K_out=18)
    peak_deg = (out[:, 2, 2].argmax() + 0.5) * (180.0 / 18)
    assert 25.0 < peak_deg < 65.0


def test_refine_theta_is_a_noop_for_matching_K():
    ch = np.random.default_rng(0).random((6, 3, 3)).astype(np.float32)
    np.testing.assert_allclose(refine_theta(ch, K_out=6), ch)


def test_single_line_gives_one_instance():
    line = np.array([[10.0, 50.0], [90.0, 50.0]])
    ch = oracle_ori_channels([line], (100, 100), K=6, up=1.0)
    pls, _ = instance_b(ch, KMAX)
    assert len(pls) == 1 and total_length(pls[0]) > 60.0


def test_perpendicular_crossing_gives_two_instances():
    h = np.array([[10.0, 50.0], [90.0, 50.0]])
    v = np.array([[50.0, 10.0], [50.0, 90.0]])
    ch = oracle_ori_channels([h, v], (100, 100), K=6, up=1.0)
    pls, _ = instance_b(ch, KMAX)
    assert len(pls) == 2


def test_removing_one_theta_slice_leaves_the_pixel_for_the_other_mt():
    # Both crossing MTs must be recovered at FULL length -- the amodal win over instancer A,
    # which has to give the shared pixel to one of them.
    h = np.array([[10.0, 50.0], [90.0, 50.0]])
    v = np.array([[50.0, 10.0], [50.0, 90.0]])
    ch = oracle_ori_channels([h, v], (100, 100), K=6, up=1.0)
    pls, _ = instance_b(ch, KMAX)
    lengths = sorted(total_length(p) for p in pls)
    assert lengths[0] > 65.0, "the second MT must survive the crossing at full length"


def test_wavy_microtubule_stays_ONE_instance_across_theta_bins():
    # A sine sweeping its tangent through many bins -- the failure mode that sank the
    # earlier PER-BIN approach (F1 0.11). The joint graph must not shatter it.
    x = np.linspace(5.0, 145.0, 400)
    y = 75.0 + 30.0 * np.sin(2 * np.pi * x / 90.0)
    ch = oracle_ori_channels([np.stack([x, y], axis=1)], (150, 150), K=6, up=1.0)
    pls, _ = instance_b(ch, KMAX)
    assert len(pls) == 1, f"wavy MT must not fragment across bins; got {len(pls)}"


def test_close_parallels_stay_two_instances():
    a = np.array([[10.0, 50.0], [90.0, 50.0]])
    b = np.array([[10.0, 56.0], [90.0, 56.0]])
    ch = oracle_ori_channels([a, b], (100, 100), K=6, up=1.0)
    pls, _ = instance_b(ch, KMAX)
    assert len(pls) == 2
