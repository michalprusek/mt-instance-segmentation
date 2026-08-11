import numpy as np

from instance.geometry import resample, total_length
from instance.instancer_a import DEFAULTS, KAPPA_DS, instance_a
from instance.metrics import max_curvature
from instance.oracle import oracle_mask

KMAX = 0.25   # just above the largest curvature in 957 human-annotated MTs (0.239 rad/px)


def _p(**over):
    """Pristine defaults plus explicit overrides.

    NEVER let these tests fall through to ``instance_a``'s ``default_params()``: that reads
    ``params_a.json``, so a tuning run silently redefines what the unit tests are testing. A
    tuned ``link_max_gap`` did exactly that once.
    """
    return {**DEFAULTS, **over}


def _mask(polylines, shape=(100, 100)):
    return oracle_mask(polylines, shape, half_width=1.0, up=1.0)


def test_two_isolated_lines_give_two_instances():
    a = np.array([[10.0, 20.0], [90.0, 20.0]])
    b = np.array([[10.0, 70.0], [90.0, 70.0]])
    pls, masks = instance_a(_mask([a, b]), KMAX, _p())
    assert len(pls) == 2 and len(masks) == 2


def test_perpendicular_crossing_gives_two_instances_not_four():
    h = np.array([[10.0, 50.0], [90.0, 50.0]])
    v = np.array([[50.0, 10.0], [50.0, 90.0]])
    pls, _ = instance_a(_mask([h, v]), KMAX, _p())
    assert len(pls) == 2, f"crossing must not fragment; got {len(pls)}"
    assert all(total_length(p) > 60.0 for p in pls), "each instance must span the whole line"


def test_shallow_crossing_gives_two_instances():
    ang = np.deg2rad(15.0)
    c = np.array([50.0, 50.0])
    h = np.array([c - [40, 0], c + [40, 0]])
    d = np.array([c - [40 * np.cos(ang), 40 * np.sin(ang)],
                  c + [40 * np.cos(ang), 40 * np.sin(ang)]])
    pls, _ = instance_a(_mask([h, d]), KMAX, _p())
    assert len(pls) == 2


def test_output_polylines_respect_the_curvature_bound():
    h = np.array([[10.0, 50.0], [90.0, 50.0]])
    v = np.array([[50.0, 10.0], [50.0, 90.0]])
    pls, _ = instance_a(_mask([h, v]), KMAX, _p())
    assert max_curvature(pls, ds=KAPPA_DS) <= KMAX + 1e-6


def test_close_parallels_stay_two_instances():
    a = np.array([[10.0, 50.0], [90.0, 50.0]])
    b = np.array([[10.0, 55.0], [90.0, 55.0]])
    pls, _ = instance_a(_mask([a, b]), KMAX, _p())
    assert len(pls) == 2


def test_t_junction_keeps_the_through_line_whole():
    through = np.array([[10.0, 50.0], [90.0, 50.0]])
    stem = np.array([[50.0, 50.0], [50.0, 92.0]])
    pls, _ = instance_a(_mask([through, stem]), KMAX, _p())
    lengths = sorted(total_length(p) for p in pls)
    assert len(pls) == 2
    assert lengths[-1] > 70.0, "the through line must not be cut at the T"


def test_curved_microtubule_is_one_instance():
    t = np.linspace(0.0, 1.4, 300)
    R = 60.0
    curve = np.stack([20 + R * np.sin(t), 20 + R * (1 - np.cos(t))], axis=1)
    pls, _ = instance_a(_mask([curve]), KMAX, _p())
    assert len(pls) == 1
    assert total_length(pls[0]) > 0.75 * total_length(resample(curve, ds=1.0))


def test_enforce_curvature_straightens_a_kink_but_keeps_a_real_hairpin():
    from instance.instancer_a import enforce_curvature
    kink = resample(np.array([[0.0, 0.0], [40.0, 0.0], [40.0, 40.0]]), ds=2.0)
    fixed = enforce_curvature(kink, KMAX, ds=2.0)
    assert max_curvature([fixed], ds=KAPPA_DS) <= KMAX + 1e-6

    # A hairpin of radius 12 px has curvature 1/12 = 0.083 rad/px, well under KMAX,
    # so it must pass through essentially untouched.
    th = np.linspace(-np.pi / 2, np.pi / 2, 200)
    hairpin = np.concatenate([
        np.stack([np.linspace(60, 12, 60), np.full(60, 8.0)], axis=1),
        np.stack([12 + 12 * np.cos(th + np.pi), 20 + 12 * np.sin(th + np.pi)], axis=1),
        np.stack([np.linspace(12, 60, 60), np.full(60, 32.0)], axis=1),
    ])
    hp = resample(hairpin, ds=2.0)
    kept = enforce_curvature(hp, KMAX, ds=2.0)
    assert abs(total_length(kept) - total_length(hp)) < 0.05 * total_length(hp)


def test_gap_linking_rejoins_a_microtubule_broken_by_a_foreground_hole():
    # The measured failure on real data: v4b's foreground drops out 1.56 times per real
    # microtubule. Without linking, each dropout is a permanent break.
    full = np.array([[10.0, 50.0], [90.0, 50.0]])
    broken = _mask([np.array([[10.0, 50.0], [44.0, 50.0]]),
                    np.array([[56.0, 50.0], [90.0, 50.0]])])
    off, _ = instance_a(broken, KMAX, _p(link_max_gap=0.0))
    on, _ = instance_a(broken, KMAX, _p(link_max_gap=20.0))
    assert len(off) == 2, "control: without linking the hole splits the microtubule"
    assert len(on) == 1, "with linking the two halves become one instance"
    assert total_length(on[0]) > 0.9 * total_length(full)


def test_a_sideways_link_costs_more_than_a_straight_one_and_c_open_can_veto_it():
    # Two offset microtubules whose facing ends are 19 px apart: joining them turns 51 deg up
    # and 51 deg back, which is kappa = 0.093 rad/px -- BELOW the 0.25 bound, because real
    # microtubules do bend that hard (max observed 0.239). So the hard constraint neither can
    # nor should reject this; the COST must. That distinction is the thing worth testing.
    from instance.matching import ArmEnd, pair_cost

    def arm(deg, pos):
        return ArmEnd(0, "end", np.deg2rad(deg), 0.0, np.array(pos, dtype=float))

    straight = pair_cost(arm(180, (44.0, 45.0)), arm(0, (56.0, 45.0)), None)
    sideways = pair_cost(arm(180, (44.0, 45.0)), arm(0, (56.0, 60.0)), None)
    assert sideways[1] < KMAX, "physics permits this bend; only the cost may reject it"
    assert sideways[0] > straight[0] + 1.0

    a = np.array([[10.0, 45.0], [44.0, 45.0]])
    b = np.array([[56.0, 60.0], [90.0, 60.0]])
    m = _mask([a, b])
    lax, _ = instance_a(m, KMAX, _p(link_max_gap=25.0, c_open_link=1.2))
    strict, _ = instance_a(m, KMAX, _p(link_max_gap=25.0, c_open_link=0.9))
    assert len(lax) == 1 and len(strict) == 2

    # A genuinely collinear continuation across the same distance survives the strict setting.
    c = np.array([[10.0, 45.0], [44.0, 45.0]])
    d = np.array([[56.0, 45.0], [90.0, 45.0]])
    joined, _ = instance_a(_mask([c, d]), KMAX, _p(link_max_gap=25.0, c_open_link=0.9))
    assert len(joined) == 1


def test_bridge_evidence_blocks_a_link_over_empty_background():
    broken = _mask([np.array([[10.0, 50.0], [44.0, 50.0]]),
                    np.array([[56.0, 50.0], [90.0, 50.0]])])
    empty = np.zeros_like(broken, dtype=np.float32)      # no image support anywhere
    linked, _ = instance_a(broken, KMAX, _p(link_max_gap=20.0, bridge_thr=0.15),
                           prob=empty)
    assert len(linked) == 2, "a link needs image evidence along the bridge"
    support = np.zeros_like(broken, dtype=np.float32)
    support[48:53, :] = 0.4                              # faint trace across the hole
    ok, _ = instance_a(broken, KMAX, _p(link_max_gap=20.0, bridge_thr=0.15),
                       prob=support)
    assert len(ok) == 1


def test_orientation_term_is_inert_without_channels():
    h = np.array([[10.0, 50.0], [90.0, 50.0]])
    v = np.array([[50.0, 10.0], [50.0, 90.0]])
    m = _mask([h, v])
    a, _ = instance_a(m, KMAX, _p(w_ori=0.0))
    b, _ = instance_a(m, KMAX, _p(w_ori=3.0))           # no channels supplied
    assert len(a) == len(b) == 2


def test_orientation_channels_are_accepted_and_keep_a_crossing_split():
    from instance.oracle import oracle_ori_channels
    h = np.array([[10.0, 50.0], [90.0, 50.0]])
    v = np.array([[50.0, 10.0], [50.0, 90.0]])
    ch = oracle_ori_channels([h, v], (100, 100), K=6, half_width=1.0, up=1.0)
    pls, _ = instance_a(ch.max(axis=0) > 0.5, KMAX, _p(w_ori=3.0), channels=ch)
    assert len(pls) == 2
    assert all(total_length(p) > 60.0 for p in pls)
