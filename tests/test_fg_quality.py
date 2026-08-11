import numpy as np

from mt_bench.fg_quality import (CONTINUITY_KEYS, REFERENCE, foreground_quality,
                                 mean_properties, passes_overfiring_gate, quality_score,
                                 ranking_accuracy, select_checkpoint)


def _line_mask(shape, row, c0, c1, half=1):
    m = np.zeros(shape, dtype=bool)
    m[row - half:row + half + 1, c0:c1] = True
    return m


def _line_gt(row, c0, c1):
    """GT polyline in (x=col, y=row) -- the transpose of the mask's indexing."""
    return np.array([[float(c0), float(row)], [float(c1 - 1), float(row)]])


def test_a_perfect_foreground_scores_one_component_and_no_gaps():
    shape = (100, 200)
    mask = _line_mask(shape, 50, 20, 180)
    props = foreground_quality(mask, [_line_gt(50, 20, 180)])
    assert props["cc_per_gt"] == 1.0
    assert props["gaps_per_mt"] == 0.0
    assert props["rec2"] > 0.99


def test_coordinate_convention_is_x_col_y_row():
    """GT is (x=col, y=row); a transposed read would score a matching mask as a miss.

    The mask is a HORIZONTAL line at row 50 spanning columns 20-180. Read correctly, coverage
    is complete. Read as (row, col), the GT would be a vertical line at column 50 -- almost
    entirely off the mask -- so rec2 would collapse. This has been a silent bug twice in this
    project, hence a dedicated asymmetric case (the shape is non-square on purpose, so a
    transpose cannot even be indexed by accident).
    """
    shape = (100, 200)
    mask = _line_mask(shape, 50, 20, 180)
    good = foreground_quality(mask, [_line_gt(50, 20, 180)])
    swapped = foreground_quality(mask, [_line_gt(50, 20, 180)[:, ::-1]])
    assert good["rec2"] > 0.99
    assert swapped["rec2"] < 0.2


def test_a_gap_in_the_foreground_is_counted():
    shape = (100, 200)
    broken = _line_mask(shape, 50, 20, 90) | _line_mask(shape, 50, 110, 180)
    props = foreground_quality(broken, [_line_gt(50, 20, 180)])
    assert props["gaps_per_mt"] == 1.0
    assert props["cc_per_gt"] == 2.0
    assert props["rec2"] < 0.95, "the dropout must cost recall too"


def test_quality_score_is_lower_for_the_more_continuous_foreground():
    intact = {"cc_per_gt": 1.0, "endp_per_kpx": 4.0, "gaps_per_mt": 0.0}
    shattered = {"cc_per_gt": 8.0, "endp_per_kpx": 30.0, "gaps_per_mt": 2.0}
    assert quality_score(intact) < quality_score(shattered)
    # 1.0 reads as "v4b-level" by construction
    assert abs(quality_score(REFERENCE) - 1.0) < 1e-9


def test_quality_score_is_nan_when_nothing_is_measurable():
    assert np.isnan(quality_score({"cc_per_gt": float("nan")}))


def test_the_gate_rejects_an_over_firing_foreground_that_scores_well():
    """The trap this module exists to block: dilation IMPROVES every continuity metric."""
    flooded = {"cc_per_gt": 0.4, "endp_per_kpx": 1.0, "gaps_per_mt": 0.0,
               "fg": 0.20, "rec2": 1.0}
    sane = {"cc_per_gt": 2.7, "endp_per_kpx": 6.7, "gaps_per_mt": 0.37,
            "fg": 0.032, "rec2": 0.99}
    assert quality_score(flooded) < quality_score(sane), "precondition: it does score better"
    assert not passes_overfiring_gate(flooded)
    assert passes_overfiring_gate(sane)
    assert select_checkpoint([flooded, sane]) == 1


def test_the_gate_rejects_a_collapsed_foreground_too():
    """A nearly empty mask also has few components and -- since a microtubule it misses
    entirely contributes no dropouts -- few gaps."""
    collapsed = {"cc_per_gt": 0.2, "endp_per_kpx": 2.0, "gaps_per_mt": 0.05,
                 "fg": 0.001, "rec2": 0.11}
    assert quality_score(collapsed) < 1.0
    assert not passes_overfiring_gate(collapsed)


def test_a_partially_collapsed_checkpoint_cannot_win_on_a_subset_of_frames():
    """Scored on only the 3 frames it fired on, a mostly-dead checkpoint looks excellent.

    This is the hole `min_frames` closes: empty masks drop out of the average entirely, so the
    survivors -- easy frames -- set the score, and every other guard (rec2, fg) is computed on
    those same easy frames and passes.
    """
    partial = {"cc_per_gt": 1.0, "endp_per_kpx": 3.0, "gaps_per_mt": 0.0,
               "fg": 0.010, "rec2": 0.99, "n_frames": 3}
    healthy = {"cc_per_gt": 2.7, "endp_per_kpx": 6.7, "gaps_per_mt": 0.37,
               "fg": 0.032, "rec2": 0.99, "n_frames": 16}
    assert quality_score(partial) < quality_score(healthy), "precondition: it scores better"
    assert passes_overfiring_gate(partial), "precondition: every value-based guard passes"
    assert select_checkpoint([partial, healthy]) == 0, "unguarded, the collapse wins"
    assert select_checkpoint([partial, healthy], min_frames=16) == 1


def test_select_checkpoint_returns_none_rather_than_the_least_bad_bad_one():
    bad = [{"cc_per_gt": 0.4, "endp_per_kpx": 1.0, "gaps_per_mt": 0.0,
            "fg": 0.30, "rec2": 1.0},
           {"cc_per_gt": 0.5, "endp_per_kpx": 1.2, "gaps_per_mt": 0.0,
            "fg": 0.25, "rec2": 1.0}]
    assert select_checkpoint(bad) is None


def test_mean_properties_skips_empty_frames_and_nans():
    agg = mean_properties([{"cc_per_gt": 2.0, "gaps_per_mt": float("nan")},
                           {"cc_per_gt": 4.0, "gaps_per_mt": 1.0},
                           None])
    assert agg["cc_per_gt"] == 3.0
    assert agg["gaps_per_mt"] == 1.0
    assert agg["n_frames"] == 2


def test_foreground_quality_returns_none_for_an_empty_mask():
    assert foreground_quality(np.zeros((50, 50), bool), [_line_gt(25, 5, 45)]) is None
    assert foreground_quality(_line_mask((50, 50), 25, 5, 45), []) is None


def test_ranking_accuracy_is_direction_aware():
    """A property that ranks perfectly one way ranks at 0 the other way -- so reading the
    number without its direction turns a chance-level control into an apparent signal."""
    rows = [{"frame": "f1", "model": "good", "inst_f1": 0.8, "cc_per_gt": 2.0},
            {"frame": "f1", "model": "bad", "inst_f1": 0.3, "cc_per_gt": 9.0},
            {"frame": "f2", "model": "good", "inst_f1": 0.7, "cc_per_gt": 1.5},
            {"frame": "f2", "model": "bad", "inst_f1": 0.2, "cc_per_gt": 7.0}]
    assert ranking_accuracy(rows, "cc_per_gt") == 1.0
    assert ranking_accuracy(rows, "cc_per_gt", lower_is_better=False) == 0.0


def test_ranking_accuracy_skips_ties_and_cross_frame_pairs():
    """Only pairs that are (same frame, different model, no tie) may be scored.

    Here a-b ties on F1 and a-c ties on the property, so both are dropped; the b-c pair is
    the only comparison left and the property gets it right. If either tie leaked in, the
    accuracy would fall to 0.5 or 0.33.
    """
    rows = [{"frame": "f1", "model": "a", "inst_f1": 0.5, "cc_per_gt": 2.0},
            {"frame": "f1", "model": "b", "inst_f1": 0.5, "cc_per_gt": 9.0},
            {"frame": "f1", "model": "c", "inst_f1": 0.9, "cc_per_gt": 2.0}]
    assert ranking_accuracy(rows, "cc_per_gt") == 1.0

    # A wrong-way pair in a SECOND frame must not be silently paired across frames.
    rows += [{"frame": "f2", "model": "b", "inst_f1": 0.9, "cc_per_gt": 9.0},
             {"frame": "f2", "model": "c", "inst_f1": 0.1, "cc_per_gt": 2.0}]
    assert ranking_accuracy(rows, "cc_per_gt") == 0.5, "one hit (f1), one miss (f2)"


def test_continuity_keys_all_have_a_reference_scale():
    assert set(CONTINUITY_KEYS) == set(REFERENCE)
