import numpy as np

from instance.metrics import (bundle_recovery, centerline_f1, fragmentation,
                              junction_identity, max_curvature)
from instance.oracle import oracle_instance_masks


def _m(polyline, shape=(60, 60)):
    return oracle_instance_masks([polyline], shape, half_width=1.0, up=1.0)[0]


def test_centerline_f1_perfect_prediction_scores_one():
    gt = [np.array([[5.0, 20.0], [55.0, 20.0]])]
    masks = oracle_instance_masks(gt, (60, 60), half_width=1.0, up=1.0)
    r = centerline_f1(masks, gt, tol=5.0, length_coverage=0.95, precision_coverage=0.95)
    assert r["tp"] == 1 and r["fp"] == 0 and r["fn"] == 0


def test_centerline_f1_rejects_a_fragment_under_strict_coverage():
    gt = [np.array([[5.0, 20.0], [55.0, 20.0]])]
    half = oracle_instance_masks([np.array([[5.0, 20.0], [28.0, 20.0]])], (60, 60), up=1.0)
    r = centerline_f1(half, gt, tol=5.0, length_coverage=0.95, precision_coverage=0.95)
    assert r["tp"] == 0 and r["fn"] == 1


def test_fragmentation_counts_a_split_microtubule_as_two():
    gt = [np.array([[5.0, 20.0], [55.0, 20.0]])]
    left = _m(np.array([[5.0, 20.0], [28.0, 20.0]]))
    right = _m(np.array([[32.0, 20.0], [55.0, 20.0]]))
    assert fragmentation([left, right], gt, tol=5.0) == 2.0


def test_fragmentation_of_a_perfect_prediction_is_one():
    gt = [np.array([[5.0, 20.0], [55.0, 20.0]])]
    assert fragmentation([_m(gt[0])], gt, tol=5.0) == 1.0


def _cross_setup():
    h = np.array([[5.0, 30.0], [55.0, 30.0]])
    v = np.array([[30.0, 5.0], [30.0, 55.0]])
    crossings = [{"i": 0, "j": 1, "x": 30.0, "y": 30.0,
                  "s_i": 25.0, "s_j": 25.0, "angle_deg": 90.0}]
    return h, v, crossings


def test_junction_identity_zero_when_both_mts_are_cut_at_the_crossing():
    h, v, crossings = _cross_setup()
    frags = [np.array([[5.0, 30.0], [27.0, 30.0]]), np.array([[33.0, 30.0], [55.0, 30.0]]),
             np.array([[30.0, 5.0], [30.0, 27.0]]), np.array([[30.0, 33.0], [30.0, 55.0]])]
    masks = [_m(f) for f in frags]
    r = junction_identity(masks, [h, v], crossings, tol=5.0)
    assert r["n_crossings"] == 1 and r["n_preserved"] == 0 and r["rate"] == 0.0


def test_junction_identity_one_when_both_pass_through_intact():
    h, v, crossings = _cross_setup()
    masks = [_m(h), _m(v)]
    r = junction_identity(masks, [h, v], crossings, tol=5.0)
    assert r["n_preserved"] == 1 and r["rate"] == 1.0


def test_junction_identity_zero_when_only_one_mt_survives():
    h, v, crossings = _cross_setup()
    masks = [_m(h), _m(np.array([[30.0, 5.0], [30.0, 27.0]])),
             _m(np.array([[30.0, 33.0], [30.0, 55.0]]))]
    r = junction_identity(masks, [h, v], crossings, tol=5.0)
    assert r["n_preserved"] == 0


def test_bundle_recovery_is_one_when_parallels_get_separate_instances():
    a = np.array([[5.0, 30.0], [55.0, 30.0]])
    b = np.array([[5.0, 34.0], [55.0, 34.0]])
    assert bundle_recovery([_m(a), _m(b)], [a, b], [(0, 1)], tol=1.5) == 1.0


def test_bundle_recovery_is_zero_when_parallels_are_merged():
    a = np.array([[5.0, 30.0], [55.0, 30.0]])
    b = np.array([[5.0, 34.0], [55.0, 34.0]])
    merged = _m(a) | _m(b)
    # A second, far-away instance exists so the Hungarian assignment HAS a distinct
    # partner available -- the pair must still fail, on coverage rather than on distinctness.
    far = _m(np.array([[5.0, 5.0], [55.0, 5.0]]))
    assert bundle_recovery([merged, far], [a, b], [(0, 1)], tol=1.5) == 0.0


def test_bundle_recovery_needs_tol_below_half_the_gap():
    # Both predictions sit on filament A and NOTHING is on filament B -- the bundle was
    # missed entirely. At a loose tol each prediction still "covers" B, so the metric is
    # fooled into reporting a recovered bundle; below half the gap it is not.
    a = np.array([[5.0, 30.0], [55.0, 30.0]])
    b = np.array([[5.0, 34.0], [55.0, 34.0]])
    both_on_a = [_m(a), _m(np.array([[5.0, 31.0], [55.0, 31.0]]))]
    assert bundle_recovery(both_on_a, [a, b], [(0, 1)], tol=3.0) == 1.0    # fooled
    assert bundle_recovery(both_on_a, [a, b], [(0, 1)], tol=1.5) == 0.0    # not fooled


def test_max_curvature_flags_a_kinked_prediction():
    kinked = np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 20.0]])
    assert max_curvature([kinked], ds=1.0) > 1.0
    straight = np.array([[0.0, 0.0], [40.0, 0.0]])
    assert max_curvature([straight], ds=1.0) < 1e-6


def test_coverage_f1_of_a_perfect_prediction_is_near_one():
    """Guards the (row, col) vs (x, y) mix-up in scripts/semantic_compare.py, which reports
    ~0.02 instead of ~0.95 and looks like a catastrophic model failure rather than a bug."""
    import sys
    sys.path.insert(0, "scripts")
    from semantic_compare import coverage_f1
    from instance.oracle import oracle_mask
    from instance.geometry import resample
    pl = np.array([[5.0, 20.0], [55.0, 20.0]])
    mask = oracle_mask([pl], (60, 60), half_width=1.0, up=1.0)
    pts = resample(pl, 1.0)
    gt_rc = np.stack([pts[:, 1], pts[:, 0]], axis=1)      # (row, col)
    rec, prec = coverage_f1(mask, gt_rc, tol=2.0)
    assert rec > 0.95 and prec > 0.95


# ---- end tolerance (declared secondary metric variant, 2026-09-05) -------------------------------
def _line_mask(x0, x1, y=50, shape=(100, 500)):
    m = np.zeros(shape, bool)
    m[y, x0:x1] = True
    return m


def _gt_line(x0, x1, y=50):
    return np.array([[x0, y], [x1 - 1, y]], float)   # (x, y) vertices


def test_end_tol_default_is_bit_identical():
    from instance.metrics import centerline_f1
    gt = [_gt_line(100, 300)]
    pred = [_line_mask(100, 320)]                          # 20 px overshoot: precision 200/220 < 0.95
    a = centerline_f1(pred, gt, tol=5.0, length_coverage=0.95, precision_coverage=0.95)
    b = centerline_f1(pred, gt, tol=5.0, length_coverage=0.95, precision_coverage=0.95, end_tol=0.0)
    assert a["tp"] == b["tp"] == 0 and a["pairs"][0]["precision"] == b["pairs"][0]["precision"]


def test_end_tol_credits_small_overshoot_and_short_stop():
    from instance.metrics import centerline_f1
    gt = [_gt_line(100, 300)]
    over = [_line_mask(100, 320)]                          # 20 px past the annotation's end
    short = [_line_mask(115, 285)]                         # stops 15 px short at both ends: recall 0.85
    for pred in (over, short):
        strict = centerline_f1(pred, gt, tol=5.0, length_coverage=0.95, precision_coverage=0.95)
        tol25 = centerline_f1(pred, gt, tol=5.0, length_coverage=0.95, precision_coverage=0.95, end_tol=25.0)
        assert strict["tp"] == 0 and tol25["tp"] == 1, (strict["pairs"], tol25["pairs"])


def test_end_tol_does_not_credit_long_overshoot_or_gap_bridge():
    from instance.metrics import centerline_f1
    gt = [_gt_line(100, 300)]
    far = [_line_mask(100, 360)]                           # 60 px past the end: core still 0.86 on GT
    r = centerline_f1(far, gt, tol=5.0, length_coverage=0.95, precision_coverage=0.95, end_tol=25.0)
    assert r["tp"] == 0 and r["fp"] == 1
    pieces = [_gt_line(100, 200), _gt_line(260, 360)]      # one filament annotated as two pieces
    bridge = [_line_mask(100, 360)]
    r = centerline_f1(bridge, pieces, tol=5.0, length_coverage=0.95, precision_coverage=0.95, end_tol=25.0)
    assert r["tp"] == 0 and r["fp"] == 1 and r["fn"] == 2


def test_end_tol_short_curve_keeps_full_set():
    from instance.metrics import _core_points
    pts = np.stack([np.full(30, 50.0), np.arange(30, dtype=float)], 1)
    ends = np.array([[50.0, 0.0], [50.0, 29.0]], float)
    assert len(_core_points(pts, ends, 25.0)) == 30        # 30 px curve, 25 px tolerance: nothing left -> no tolerance
    assert len(_core_points(pts, ends, 5.0)) == 18
