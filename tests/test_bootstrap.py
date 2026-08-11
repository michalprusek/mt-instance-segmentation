import numpy as np
import pytest

from instance.metrics import aggregate_benchmark, bootstrap_ci, paired_bootstrap


def _row(frame, task, f1, tp=10, fp=1, fn=1, n_gt=10):
    return {"frame": frame, "task": task, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "n_gt": n_gt}


def _pair(n=17, delta=0.08, sigma=0.08, seed=0):
    """Two methods on the same frames, b uniformly worse by ``delta``.

    ``sigma`` is the BETWEEN-FRAME spread, which is the quantity pairing removes: frames
    differ far more from each other than the two methods differ on any one frame.
    """
    rng = np.random.default_rng(seed)
    a, b = [], []
    for i in range(n):
        task = "585" if i < 6 else "586"
        f1 = float(np.clip(rng.normal(0.85, sigma), 0, 1))
        a.append(_row(f"f{i}", task, f1))
        b.append(_row(f"f{i}", task, max(0.0, f1 - delta)))
    return a, b


def test_ci_brackets_the_point_estimate():
    rows = [_row(f"f{i}", "585" if i < 6 else "586", 0.5 + 0.02 * i) for i in range(17)]
    ci = bootstrap_ci(rows, n_boot=400, seed=1)
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert ci["point"] == pytest.approx(aggregate_benchmark(rows)["mean_f1"])


def test_a_consistent_gap_is_significant_but_a_zero_gap_is_not():
    a, b = _pair(delta=0.08)
    res = paired_bootstrap(a, b, n_boot=800, seed=2)
    assert res["diff"] == pytest.approx(0.08, abs=1e-6)
    assert res["significant"] and res["lo"] > 0

    same = paired_bootstrap(a, [dict(r) for r in a], n_boot=800, seed=2)
    assert same["diff"] == 0.0
    assert not same["significant"]


def test_pairing_beats_two_marginal_intervals_at_n_17():
    """The reason the paired form exists.

    At n = 17 with realistic between-frame spread, the two marginal intervals overlap almost
    completely -- reading them side by side would report "no difference" -- even though every
    single frame favours the same method by the same amount. Pairing removes the shared
    frame-difficulty variance and the difference is resolved cleanly.
    """
    a, b = _pair(delta=0.03, sigma=0.15)
    ca, cb = bootstrap_ci(a, n_boot=800, seed=3), bootstrap_ci(b, n_boot=800, seed=3)
    assert ca["lo"] < cb["hi"], "precondition: the marginal intervals do overlap"
    assert paired_bootstrap(a, b, n_boot=800, seed=3)["significant"]


def test_empty_gt_frames_follow_the_benchmark_policy_inside_every_replicate():
    """A zero-GT frame must never enter the macro mean, not even after resampling."""
    rows = [_row("f0", "585", 1.0, tp=0, fp=7, fn=0, n_gt=0)] + \
           [_row(f"f{i}", "585", 0.4) for i in range(1, 8)]
    ci = bootstrap_ci(rows, n_boot=400, seed=4)
    assert ci["hi"] <= 0.4 + 1e-9, "the free 1.0 from the empty frame leaked into the mean"
    micro = bootstrap_ci(rows, stat="micro_f1", n_boot=200, seed=4)
    assert micro["point"] < aggregate_benchmark(rows[1:])["micro_f1"], \
        "the empty frame's false positives must still count in micro"


def test_stratification_holds_the_design_fixed():
    """Unstratified resampling can draw an all-585 replicate; stratified cannot.

    Made visible by giving the two tasks disjoint F1 ranges: any replicate that keeps the
    6 + 11 design has a mean strictly between them, while an unstratified draw can land
    outside that band.
    """
    rows = [_row(f"a{i}", "585", 0.9) for i in range(6)] + \
           [_row(f"b{i}", "586", 0.3) for i in range(11)]
    strat = bootstrap_ci(rows, n_boot=600, seed=5)
    assert strat["lo"] == pytest.approx(strat["hi"]), \
        "with constant per-task values the stratified mean has no variance left"
    assert strat["point"] == pytest.approx((6 * 0.9 + 11 * 0.3) / 17)

    loose = bootstrap_ci(rows, n_boot=600, seed=5, stratum_key=None)
    assert loose["hi"] > strat["hi"] + 0.05, "unstratified adds composition variance"


def test_mismatched_frames_raise_rather_than_compare_silently():
    a, b = _pair(n=8)
    with pytest.raises(ValueError):
        paired_bootstrap(a, b[:-1], n_boot=10)
    shuffled = [dict(r) for r in b]
    shuffled[0]["frame"] = "not_a_frame"
    with pytest.raises(ValueError):
        paired_bootstrap(a, shuffled, n_boot=10)


def test_rows_are_realigned_by_frame_not_by_position():
    a, b = _pair(n=9, delta=0.05)
    res_ordered = paired_bootstrap(a, b, n_boot=400, seed=6)
    res_shuffled = paired_bootstrap(a, list(reversed(b)), n_boot=400, seed=6)
    assert res_ordered["lo"] == pytest.approx(res_shuffled["lo"])
    assert res_ordered["hi"] == pytest.approx(res_shuffled["hi"])
