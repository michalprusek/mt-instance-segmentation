import numpy as np

from instance.matching import ArmEnd, match_junction


def _arm(i, which, deg, kappa=0.0, pos=(0.0, 0.0)):
    return ArmEnd(arc_idx=i, which=which, theta=np.deg2rad(deg), kappa=kappa,
                  pos=np.array(pos, dtype=float))


def _pairs(arms, **kw):
    return {tuple(sorted(p)) for p in match_junction(arms, **kw)}


def test_perpendicular_x_pairs_opposite_arms():
    # Outgoing headings of the four arms of a + shaped crossing.
    arms = [_arm(0, "end", 0), _arm(1, "start", 180), _arm(2, "end", 90), _arm(3, "start", 270)]
    assert _pairs(arms, kappa_max=0.5, gap_len=4.0) == {(0, 1), (2, 3)}


def test_shallow_crossing_still_pairs_the_collinear_arms():
    # 20-degree crossing: arms at 0/180 and 20/200. The wrong pairing is a pair of
    # shallow Vs, which the hard curvature bound cannot reject -- only the cost ranking can.
    arms = [_arm(0, "end", 0), _arm(1, "start", 180), _arm(2, "end", 20), _arm(3, "start", 200)]
    assert _pairs(arms, kappa_max=0.5, gap_len=4.0) == {(0, 1), (2, 3)}


def test_hairpin_pairing_is_rejected_in_favour_of_the_through_path():
    arms = [_arm(0, "end", 0), _arm(1, "start", 180), _arm(2, "end", 20), _arm(3, "start", 200)]
    p = _pairs(arms, kappa_max=0.5, gap_len=4.0)
    assert (0, 2) not in p and (1, 3) not in p


def test_kappa_max_forbids_a_sharp_join():
    # Two arms meeting at 90 degrees over a 4 px gap -> 0.39 rad/px.
    arms = [_arm(0, "end", 0), _arm(1, "start", 270)]
    assert match_junction(arms, kappa_max=0.05, gap_len=4.0) == []
    assert len(match_junction(arms, kappa_max=0.5, gap_len=4.0)) == 1


def test_t_junction_leaves_the_stem_open():
    arms = [_arm(0, "end", 0), _arm(1, "start", 180), _arm(2, "end", 90)]
    pairs = match_junction(arms, kappa_max=0.5, gap_len=4.0)
    assert len(pairs) == 1 and tuple(sorted(pairs[0])) == (0, 1)


def test_curvature_continuity_breaks_a_tie():
    # Both candidates are collinear, so the angle term cannot decide. Measured OUTWARD from
    # the junction the two arms of one smooth curve have OPPOSITE signed curvature, so the
    # right partner for kappa=+0.05 is kappa=-0.05, not kappa=+0.30.
    arms = [_arm(0, "end", 0, kappa=0.05),
            _arm(1, "start", 180, kappa=-0.05),
            _arm(2, "start", 180, kappa=0.30)]
    pairs = match_junction(arms, kappa_max=0.5, w_kappa=10.0, gap_len=4.0)
    assert len(pairs) == 1 and tuple(sorted(pairs[0])) == (0, 1)


def test_c_open_can_veto_a_bad_pair_entirely():
    # A 60-degree join is inside kappa_max but expensive; a small c_open prefers two open ends.
    arms = [_arm(0, "end", 0), _arm(1, "start", 240)]
    assert match_junction(arms, kappa_max=0.9, gap_len=4.0, c_open=0.2) == []
    assert len(match_junction(arms, kappa_max=0.9, gap_len=4.0, c_open=2.0)) == 1


def test_six_arms_of_two_crossings_pair_into_three_through_paths():
    arms = [_arm(0, "end", 0), _arm(1, "start", 180),
            _arm(2, "end", 60), _arm(3, "start", 240),
            _arm(4, "end", 120), _arm(5, "start", 300)]
    assert _pairs(arms, kappa_max=0.9, gap_len=4.0) == {(0, 1), (2, 3), (4, 5)}


def _at(i, deg, pos, kappa=0.0, ori=None):
    return ArmEnd(arc_idx=i, which="end", theta=np.deg2rad(deg), kappa=kappa,
                  pos=np.array(pos, dtype=float), ori=ori)


def test_parallel_bundle_is_not_joined_across_the_gap():
    # Two microtubules 4 px apart, both horizontal, meeting at a ladder-like junction.
    # The WRONG pairing joins the left arm of one to the right arm of the other. Under a
    # direct |theta_a + pi - theta_b| cost that wrong join scores EXACTLY zero -- both arms
    # are collinear -- which is why bundles merged. Charging the turn via the gap direction
    # makes the sideways jog visible.
    arms = [_at(0, 180, (-3.0, 0.0)), _at(1, 0, (3.0, 0.0)),      # MT 1
            _at(2, 180, (-3.0, 4.0)), _at(3, 0, (3.0, 4.0))]      # MT 2
    pairs = {tuple(sorted(p)) for p in
             match_junction(arms, kappa_max=0.25, gap_len=None, gap_floor=2.0)}
    assert pairs == {(0, 1), (2, 3)}, f"bundle merged across the gap: {pairs}"


def test_a_pure_lateral_step_is_forbidden_outright():
    # Nothing in front, everything sideways: the join is a 180-degree double turn.
    arms = [_at(0, 180, (0.0, 0.0)), _at(1, 0, (0.0, 4.0))]
    assert match_junction(arms, kappa_max=0.25, gap_len=None, gap_floor=2.0) == []


def test_collinear_continuation_across_a_hole_is_linked():
    # A genuine foreground dropout: the two arms line up and the bridge points along both.
    arms = [_at(0, 180, (0.0, 0.0)), _at(1, 0, (10.0, 0.0))]
    assert len(match_junction(arms, kappa_max=0.25, gap_len=None, gap_floor=2.0)) == 1


def test_orientation_profiles_separate_a_steep_crossing():
    # Four arms of a perpendicular X. Geometry alone already pairs them; the orientation
    # channels must agree rather than fight it, and must penalise the cross-pairing.
    h = np.array([1.0, 0, 0, 0, 0, 0])
    v = np.array([0, 0, 0, 1.0, 0, 0])
    same = [_at(0, 0, (3.0, 0.0), ori=h), _at(1, 180, (-3.0, 0.0), ori=h),
            _at(2, 90, (0.0, 3.0), ori=v), _at(3, 270, (0.0, -3.0), ori=v)]
    pairs = {tuple(sorted(p)) for p in
             match_junction(same, kappa_max=0.5, gap_len=None, gap_floor=2.0, w_ori=2.0)}
    assert pairs == {(0, 1), (2, 3)}

    from instance.matching import pair_cost
    c_same, _ = pair_cost(same[0], same[1], None, w_ori=2.0, gap_floor=2.0)
    c_cross, _ = pair_cost(same[0], same[2], None, w_ori=2.0, gap_floor=2.0)
    assert c_cross > c_same


def test_bridge_evidence_reads_the_probability_map():
    from instance.matching import bridge_evidence
    prob = np.zeros((40, 40), dtype=np.float32)
    prob[20, :] = 0.4                      # a faint horizontal trace at row 20
    a = _at(0, 180, (5.0, 20.0))
    b = _at(1, 0, (35.0, 20.0))
    assert bridge_evidence(prob, a, b) > 0.35
    c = _at(2, 0, (35.0, 5.0))             # bridge over empty background
    assert bridge_evidence(prob, a, c) < 0.1
