import numpy as np
import pytest

from mt_generator import GenConfig, generate_frame
from mt_sequence import (MotionConfig, arclength, extend_path, generate_sequence,
                         resample_window)

SHAPE = (320, 320)


def _bg(seed=7):
    return np.random.default_rng(seed).normal(0.5, 0.02, SHAPE).clip(0, 1)


def _cfg(regime=None):
    cfg = GenConfig()
    if regime is not None:
        cfg.regime_gliding_prob = 1.0 if regime == "gliding" else 0.0
        cfg.regime_dynamic_prob = 1.0 if regime == "dynamic" else 0.0
    return cfg


def _lengths(frame):
    return {i["inst_id"]: float(np.sum(np.linalg.norm(
        np.diff(np.asarray(i["centerline"]), axis=0), axis=1))) for i in frame}


def test_one_frame_is_byte_identical_to_the_still_generator():
    """The still path and the video path must not be able to drift apart.

    The deployed hub already carries a comment about two copies of a microtubule pipeline that
    "silently drifted apart"; this pins that the sequence generator IS the still generator when
    asked for one frame, rather than a parallel implementation that merely agrees today.
    """
    bg, cfg = _bg(), _cfg()
    img_a, inst_a, _ = generate_frame(bg, np.random.default_rng(3), cfg)
    imgs_b, inst_b, meta = generate_sequence(bg, np.random.default_rng(3), cfg, n_frames=1)
    assert np.array_equal(img_a, imgs_b[0])
    assert len(inst_a) == len(inst_b[0])
    assert meta["n_frames"] == 1
    for a, b in zip(inst_a, inst_b[0]):
        assert np.array_equal(a["centerline"], b["centerline"])


def test_identities_are_stable_and_unique_within_a_frame():
    imgs, per, _ = generate_sequence(_bg(), np.random.default_rng(11), _cfg("static"),
                                     n_frames=4)
    assert len(imgs) == len(per) == 4
    for frame in per:
        ids = [i["inst_id"] for i in frame]
        assert len(ids) == len(set(ids)), "an id appeared twice in one frame"
    first, last = {i["inst_id"] for i in per[0]}, {i["inst_id"] for i in per[-1]}
    assert last <= first, "ids may vanish (death) but must never be invented mid-sequence"
    assert len(last) > 0


def test_static_filaments_keep_their_length_and_barely_move():
    _, per, _ = generate_sequence(_bg(), np.random.default_rng(5), _cfg("static"), n_frames=4)
    la, lb = _lengths(per[0]), _lengths(per[-1])
    shared = set(la) & set(lb)
    assert shared
    assert max(abs(la[i] - lb[i]) for i in shared) < 1e-6, "static length must not change"


def test_gliding_conserves_length_and_advances_the_tip_by_the_sampled_speed():
    """The defining property of a gliding filament: it moves ALONG its own contour.

    So its length is conserved exactly, and the leading tip advances by the sampled speed. A
    naive implementation that translated the whole filament sideways would pass a
    "did it move?" check and fail both of these.
    """
    _, per, meta = generate_sequence(_bg(), np.random.default_rng(5), _cfg("gliding"),
                                     n_frames=5)
    la, lb = _lengths(per[0]), _lengths(per[-1])
    shared = set(la) & set(lb)
    assert shared
    # Conservation is exact in ARCLENGTH (pinned by the test below); what is measured here is a
    # polyline resampled to a fixed point count, and where those points land relative to the
    # curvature shifts as the window slides. That wobble is ~2e-5 relative -- a discretisation
    # artefact, not motion. Asserting bitwise equality here would be asserting the wrong thing.
    assert max(abs(la[i] - lb[i]) / la[i] for i in shared) < 1e-3, \
        "gliding must conserve length"

    speeds = meta["speeds"]
    a = {i["inst_id"]: np.asarray(i["centerline"]) for i in per[0]}
    b = {i["inst_id"]: np.asarray(i["centerline"]) for i in per[1]}
    moved, expected = [], []
    for i in set(a) & set(b):
        moved.append(float(np.linalg.norm(b[i][-1] - a[i][-1])))
        expected.append(abs(speeds[i]))
    assert np.corrcoef(moved, expected)[0, 1] > 0.9
    assert 0.7 < np.median(np.array(moved) / np.array(expected)) < 1.3


def test_gliding_conserves_the_arclength_window_exactly():
    """The mechanism behind the test above, checked where it is exact.

    A gliding filament is a fixed-width window sliding along a path, so ``s1 - s0`` is invariant
    by construction. Pinning it here separates "the physics is right" from "the polyline
    resampling wobbles", which are different claims with different tolerances.
    """
    from mt_generator import sample_scene
    from mt_sequence import init_tracks, step_tracks
    cfg, rng = _cfg("gliding"), np.random.default_rng(5)
    instances, cond = sample_scene(SHAPE, rng, cfg)
    tracks = init_tracks(instances, "gliding", rng, cfg, MotionConfig(), 5)
    before = [t.s1 - t.s0 for t in tracks]
    for _ in range(4):
        step_tracks(tracks, "gliding", rng, MotionConfig())
    after = [t.s1 - t.s0 for t in tracks]
    assert max(abs(a - b) for a, b in zip(before, after)) < 1e-9


def test_dynamic_instability_changes_length_but_leaves_the_body_anchored():
    _, per, _ = generate_sequence(_bg(), np.random.default_rng(5), _cfg("dynamic"), n_frames=5)
    la, lb = _lengths(per[0]), _lengths(per[-1])
    shared = set(la) & set(lb)
    assert shared
    assert max(abs(la[i] - lb[i]) for i in shared) > 1.0, "tips must actually move"

    # The body stays put: a midpoint of frame 0 is still on the frame-1 curve, up to drift.
    a = {i["inst_id"]: np.asarray(i["centerline"]) for i in per[0]}
    b = {i["inst_id"]: np.asarray(i["centerline"]) for i in per[1]}
    off = [float(np.min(np.linalg.norm(b[i] - a[i][len(a[i]) // 2], axis=1)))
           for i in set(a) & set(b)]
    assert np.median(off) < 3.0, "the anchored body drifted sideways"


def test_stage_drift_is_shared_by_the_whole_field_not_per_filament():
    """Drift is a microscope property, so every filament must move by the SAME vector.

    This is the confound that matters for a motility assay: a tracker that reports stage drift
    as gliding velocity is wrong in exactly the way the measurement cares about, and it can only
    be caught if the generator produces drift that is common-mode.
    """
    mcfg = MotionConfig(static_jitter_px=0.0, drift_px_std=2.0, drift_correlated=True)
    _, per, _ = generate_sequence(_bg(), np.random.default_rng(5), _cfg("static"),
                                  n_frames=3, mcfg=mcfg)
    a = {i["inst_id"]: np.asarray(i["centerline"]) for i in per[0]}
    b = {i["inst_id"]: np.asarray(i["centerline"]) for i in per[1]}
    shared = sorted(set(a) & set(b))
    assert len(shared) >= 3
    shifts = np.array([(b[i] - a[i]).mean(axis=0) for i in shared])
    assert np.allclose(shifts, shifts[0], atol=1e-6), "drift must be common-mode"
    assert np.linalg.norm(shifts[0]) > 1.0, "precondition: the drift is actually applied"


def test_appearance_is_replayed_across_frames_while_noise_is_not():
    """Same field, same dirt, different noise -- the premise temporal fusion rests on.

    If appearance were re-drawn per frame, consecutive frames would differ globally and a
    temporal model would learn nothing useful; if noise were replayed too, the frames would be
    identical and fusion could not repair anything.
    """
    mcfg = MotionConfig(static_jitter_px=0.0, drift_px_std=0.0)
    imgs, per, _ = generate_sequence(_bg(), np.random.default_rng(5), _cfg("static"),
                                     n_frames=3, mcfg=mcfg)
    # Geometry is frozen by the config above, so any difference between frames is noise alone.
    for a, b in zip(per[0], per[1]):
        assert np.array_equal(a["centerline"], b["centerline"])
    d01 = np.abs(imgs[0] - imgs[1])
    assert d01.max() > 0, "frames are identical: the noise stream did not advance"
    # ...and that difference must be small and unstructured, not a global appearance change.
    assert abs(float(imgs[0].mean() - imgs[1].mean())) < 0.02 * float(imgs[0].std() + 1e-9) \
        or abs(float(imgs[0].mean() - imgs[1].mean())) < 1e-3


def test_resample_window_returns_a_fixed_point_count():
    path = np.stack([np.linspace(0, 100, 200), np.zeros(200)], axis=1)
    s = arclength(path)
    for n in (8, 50, 137):
        assert len(resample_window(path, s, 10.0, 60.0, n)) == n
    # a degenerate window must not crash or return a single point
    assert len(resample_window(path, s, 30.0, 30.0, 16)) == 16


def test_extend_path_lengthens_without_moving_the_original():
    rng = np.random.default_rng(0)
    cfg = GenConfig()
    path = np.stack([np.linspace(0, 100, 120), np.zeros(120)], axis=1)
    out = extend_path(path, rng, cfg, extra_len=40.0, where="head")
    assert arclength(out)[-1] > arclength(path)[-1] + 30.0
    assert len(out) > len(path)
    # the head extension must continue from the end, not teleport
    assert np.linalg.norm(out[len(path) - 1] - path[-1]) < 5.0


def test_extending_by_nothing_is_a_no_op():
    cfg = GenConfig()
    path = np.stack([np.linspace(0, 50, 60), np.zeros(60)], axis=1)
    assert extend_path(path, np.random.default_rng(0), cfg, 0.0) is path


@pytest.mark.parametrize("regime", ["static", "gliding", "dynamic"])
def test_every_regime_produces_renderable_frames(regime):
    imgs, per, meta = generate_sequence(_bg(), np.random.default_rng(13), _cfg(regime),
                                        n_frames=3)
    assert meta["regime"] == regime
    assert len(imgs) == 3 and all(im.shape == SHAPE for im in imgs)
    assert all(len(f) > 0 for f in per), "a frame came out empty"
    assert all(np.isfinite(im).all() for im in imgs)
