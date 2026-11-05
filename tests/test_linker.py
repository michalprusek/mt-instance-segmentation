"""Tests for the learned arm-pairing cost."""
import numpy as np
import pytest

from instance.linker import FEATURE_NAMES, N_FEATURES, Linker, edge_features
from instance.matching import ArmEnd, match_junction


def _arm(arc_idx, which, theta, kappa, pos, arc_len=50.0, ori=None):
    return ArmEnd(arc_idx=arc_idx, which=which, theta=theta, kappa=kappa,
                  pos=np.asarray(pos, float), ori=ori, arc_len=arc_len)


def _identity_linker(n_features=N_FEATURES, hidden=4):
    """A Linker whose weights are fixed, so score() is a deterministic function of features."""
    rng = np.random.default_rng(0)
    return Linker({
        "W0": rng.normal(size=(n_features, hidden)).astype(np.float32),
        "b0": np.zeros(hidden, np.float32),
        "W1": rng.normal(size=(hidden, hidden)).astype(np.float32),
        "b1": np.zeros(hidden, np.float32),
        "W2": rng.normal(size=(hidden, 1)).astype(np.float32),
        "b2": np.zeros(1, np.float32),
        "mu": np.zeros(n_features, np.float32),
        "sd": np.ones(n_features, np.float32),
        "n_features": np.array([n_features]),
    })


def test_feature_vector_shape_and_names():
    a = _arm(0, "end", 0.0, 0.0, (10.0, 10.0))
    b = _arm(1, "start", np.pi, 0.0, (30.0, 10.0))
    f = edge_features(a, b)
    assert f.shape == (N_FEATURES,)
    assert len(FEATURE_NAMES) == N_FEATURES
    assert np.isfinite(f).all()


def test_features_are_symmetric_in_the_two_arms():
    """The junction enumerates arms in an arbitrary order; the cost must not depend on it."""
    a = _arm(0, "end", 0.2, 0.01, (10.0, 10.0), arc_len=40.0)
    b = _arm(1, "start", np.pi - 0.1, -0.02, (35.0, 12.0), arc_len=80.0)
    np.testing.assert_allclose(edge_features(a, b), edge_features(b, a), atol=1e-6)


def test_collinear_pair_scores_a_smaller_turn_than_a_bent_one():
    # theta points from the junction INTO the arc's body, so a straight through-path along +x
    # has the left arm pointing back (-x, theta=pi) and the right arm pointing on (+x, theta=0).
    a = _arm(0, "end", np.pi, 0.0, (0.0, 0.0))
    straight = _arm(1, "start", 0.0, 0.0, (20.0, 0.0))
    bent = _arm(1, "start", np.pi / 2, 0.0, (20.0, 0.0))
    i_turn = FEATURE_NAMES.index("total_turn")
    assert edge_features(a, straight)[i_turn] == pytest.approx(0.0, abs=1e-6)
    assert edge_features(a, straight)[i_turn] < edge_features(a, bent)[i_turn]


def test_parallel_filaments_are_penalised_by_the_displacement_term():
    """Two parallel microtubules a few px apart look collinear to the DIRECT turn and must not
    to the displacement-aware one -- this is the bundle-merge failure the split turn exists
    for."""
    a = _arm(0, "end", np.pi, 0.0, (0.0, 0.0))
    b = _arm(1, "start", 0.0, 0.0, (0.0, 6.0))       # alongside, not ahead
    f = edge_features(a, b)
    assert f[FEATURE_NAMES.index("direct_turn")] == pytest.approx(0.0, abs=1e-6)
    assert f[FEATURE_NAMES.index("total_turn")] > 1.0


def test_bridge_features_read_the_probability_map():
    prob = np.zeros((40, 40), np.float32)
    prob[20, :] = 0.9                                # a filament along row 20
    a = _arm(0, "end", np.pi, 0.0, (5.0, 20.0))
    b = _arm(1, "start", 0.0, 0.0, (30.0, 20.0))
    on = edge_features(a, b, prob=prob)
    off = edge_features(_arm(0, "end", np.pi, 0.0, (5.0, 5.0)),
                        _arm(1, "start", 0.0, 0.0, (30.0, 5.0)), prob=prob)
    assert on[FEATURE_NAMES.index("bridge_mean")] > 0.8
    assert off[FEATURE_NAMES.index("bridge_mean")] == pytest.approx(0.0, abs=1e-6)


def test_bridge_min_separates_a_dropout_from_a_true_gap():
    """Mean cannot tell 'weak everywhere' from 'strong at the ends, empty between'."""
    dip = np.full((40, 40), 0.4, np.float32)
    hole = np.zeros((40, 40), np.float32)
    hole[:, :10] = 1.0
    hole[:, 30:] = 1.0
    a = _arm(0, "end", np.pi, 0.0, (5.0, 20.0))
    b = _arm(1, "start", 0.0, 0.0, (35.0, 20.0))
    i_min = FEATURE_NAMES.index("bridge_min")
    assert edge_features(a, b, prob=dip)[i_min] > edge_features(a, b, prob=hole)[i_min]


def test_linker_score_is_a_probability():
    m = _identity_linker()
    f = np.stack([edge_features(_arm(0, "end", np.pi, 0.0, (0.0, 0.0)),
                                _arm(1, "start", 0.0, 0.0, (float(d), 0.0)))
                  for d in (5, 10, 20)])
    s = m.score(f)
    assert s.shape == (3,)
    assert ((s >= 0) & (s <= 1)).all()


def test_checkpoint_with_wrong_feature_count_is_rejected():
    """A silent feature re-ordering would train one thing and serve another."""
    with pytest.raises(ValueError, match="feature"):
        _identity_linker(n_features=N_FEATURES - 1)


def test_scorer_overrides_the_geometric_cost_in_match_junction():
    """Same four arms, opposite scorers, opposite pairings -- the hook really is in the loop."""
    arms = [_arm(0, "end", np.pi, 0.0, (0.0, 0.0)),
            _arm(1, "start", 0.0, 0.0, (20.0, 0.0)),
            _arm(2, "end", -np.pi / 2, 0.0, (10.0, -10.0)),
            _arm(3, "start", np.pi / 2, 0.0, (10.0, 10.0))]

    def prefer_02(arms_, i, j, n):
        return 0.9 if {i, j} == {0, 2} else 0.1

    pairs = match_junction(arms, kappa_max=10.0, scorer=prefer_02, p_open=0.5)
    assert (0, 2) in pairs


def test_kappa_bound_still_forbids_impossible_pairs_under_a_learned_scorer():
    """A learned score must not be able to buy its way through a kink."""
    # Both bodies extend the SAME way from tips 2 px apart: continuing one into the other
    # requires a 180-degree turn inside 2 px.
    arms = [_arm(0, "end", np.pi, 0.0, (0.0, 0.0)),
            _arm(1, "start", np.pi, 0.0, (2.0, 0.0))]

    def always_yes(arms_, i, j, n):
        return 1.0

    assert match_junction(arms, kappa_max=0.25, scorer=always_yes, p_open=0.0) == []


def test_geometric_behaviour_is_unchanged_when_no_scorer_is_given():
    arms = [_arm(0, "end", np.pi, 0.0, (0.0, 0.0)),
            _arm(1, "start", 0.0, 0.0, (20.0, 0.0))]
    assert match_junction(arms, kappa_max=10.0) == [(0, 1)]
