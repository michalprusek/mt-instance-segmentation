"""Contract tests for the deployment package's tracking entry point, ``deploy_v5H/track.py``.

The two checks that validated this tool when it shipped -- agreement with the evaluation code
path on a real sequence, and a scale mutation -- both need a GPU and a 560 MB checkpoint, so
neither can run here. What CAN run here is everything between the model and the files on disk,
which is where the defects actually were: ordering, the 1.5x round trip, and the map from
detections to tracks. ``track.py`` imports torch only inside ``Segmenter.__init__``, so stubbing
four names from ``predict`` makes ``main()`` run in about a tenth of a second on injected
polylines.

What is deliberately NOT tested here, so it does not get re-litigated: numerical agreement with
the evaluation path (it measures fp32-vs-bf16 and a one-pixel tiling stride difference, which
stubs erase) and anything requiring real detections.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

_DEPLOY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deploy_v5H")
if _DEPLOY not in sys.path:
    sys.path.insert(0, _DEPLOY)

import track  # noqa: E402
from instance import tracker as tracker_mod  # noqa: E402

UP = track.UP
RAW = 40                      # tiny frames: the model is stubbed, only the geometry matters


class _FakeSegmenter:
    device = "cpu"

    def __init__(self, *_a, **_k):
        pass

    def channels(self, img01):
        return np.zeros((1,) + tuple(img01.shape), dtype=np.float32)


def _line(x0, y0, x1, y1, n=25):
    """A polyline in the 1.5x working frame -- the frame the instancer hands back."""
    return np.stack([np.linspace(x0, x1, n), np.linspace(y0, y1, n)], axis=1)


def _drive(tmp_path, monkeypatch, per_frame, names=None, extra_argv=(), params=None,
           loader=None, no_fov=True, seen_calls=None):
    """Run ``track.main()`` with the model stubbed. ``per_frame[k]`` is frame k's polylines."""
    seen_calls = seen_calls if seen_calls is not None else []
    names = names or [f"f_{i}.tif" for i in range(len(per_frame))]
    in_dir = tmp_path / "frames"
    in_dir.mkdir(exist_ok=True)
    for n in names:
        (in_dir / n).write_bytes(b"")
    pfile = tmp_path / "params.json"
    pfile.write_text(json.dumps(params or {"prob_thr": 0.5}))
    out = tmp_path / "out"

    seq = iter(per_frame)
    monkeypatch.setattr(track, "Segmenter", _FakeSegmenter)
    monkeypatch.setattr(track, "load_image", loader or (lambda p: np.zeros((RAW, RAW))))
    # A sentinel rather than None: None is exactly what --no-fov produces, so a stub returning
    # None makes the masked path -- the one every accuracy figure was measured with -- invisible.
    monkeypatch.setattr(track, "field_of_view",
                        lambda raw, hi: np.ones(hi, dtype=bool))
    monkeypatch.setattr(track, "instance_polylines",
                        lambda ch, params, thr, fov: (seen_calls.append(
                            {"shape": ch.shape, "params": params, "thr": thr, "fov": fov}),
                            next(seq))[1])
    argv = ["track.py", "--input", str(in_dir), "--out-dir", str(out), "--params", str(pfile)]
    if no_fov:
        argv.append("--no-fov")
    monkeypatch.setattr(sys, "argv", [*argv, *extra_argv])
    track.main()

    tracks = json.loads((out / "tracks.json").read_text())
    frames = [json.loads((out / f"{os.path.splitext(n)[0]}.json").read_text())
              for n in tracks["frames"]]
    return tracks, frames


# --------------------------------------------------------------------------------------------
# The invariant that was broken: the two output artifacts must describe the same detections.
# --------------------------------------------------------------------------------------------

def test_every_detection_appears_exactly_once_even_when_two_are_identical(tmp_path, monkeypatch):
    """Two value-equal polylines in one frame must not collapse onto the same index.

    The original lookup matched by identity OR value per candidate, so a value-equal detection
    at a lower index won over the correct one: one detection vanished from the per-frame file
    and the other was written out under a different track's id -- which the colour-by-track
    overlay then draws as fact. Nothing raised; the two JSON artifacts simply disagreed.
    """
    a, b = _line(2, 2, 30, 2), _line(2, 20, 30, 20)
    dup = _line(2, 20, 30, 20)                 # distinct object, element-wise equal to b
    per_frame = [[a, b, dup], [a.copy(), b.copy(), dup.copy()]]

    tracks, frames = _drive(tmp_path, monkeypatch, per_frame)

    assert sum(f["n_instances"] for f in frames) == 6, \
        "per-frame files lost a detection"
    assert sum(len(t["frames"]) for t in tracks["tracks"]) == 6, \
        "tracks.json and the injected detections disagree"
    for f in frames:
        assert len(set(f["track_ids"])) == len(f["track_ids"]), \
            "two detections in one frame claim the same track"


def test_a_polyline_that_matches_no_detection_raises(tmp_path, monkeypatch):
    """The map is an invariant, not a guess: if it cannot be recovered, refuse to write.

    Silently skipping is what produced inconsistent artifacts before, so ``assign_owners``
    raises. Mutating a track's polyline breaks the invariant the way a future refactor would.
    """
    frames = [[_line(2, 2, 30, 2)], [_line(2, 3, 30, 3)]]

    class _Track:
        track_id, frames, polylines = 0, [0], [_line(9, 9, 19, 19)]
        length = 1

    with pytest.raises(SystemExit, match="matches no unclaimed detection"):
        track.assign_owners([_Track()], frames)


# --------------------------------------------------------------------------------------------
# The 1.5x contract -- the one thing that fails silently rather than loudly.
# --------------------------------------------------------------------------------------------

def test_output_coordinates_are_native_and_association_runs_in_the_working_frame(
        tmp_path, monkeypatch):
    """Association happens at 1.5x; only the output is divided by it.

    Tracking at native scale runs perfectly well and merely loosens every gate by 1.5x, so
    nothing would raise -- it is the failure this test exists for. The captured polylines prove
    the tracker saw working-frame coordinates; the JSON proves the caller sees native ones.
    """
    seen = {}
    real = tracker_mod.track_sequence

    def spy(frames, params=None):
        seen["frames"], seen["params"] = frames, params
        return real(frames, params)

    monkeypatch.setattr(tracker_mod, "track_sequence", spy)
    per_frame = [[_line(20, 30, 44, 30)], [_line(20, 30, 44, 30)]]
    tracks, frames = _drive(tmp_path, monkeypatch, per_frame)

    assert seen["frames"][0][0][0] == pytest.approx([20.0, 30.0]), \
        "the tracker was not given working-frame coordinates"
    assert tracks["tracks"][0]["polylines"][0][0] == pytest.approx([20 / UP, 30 / UP], abs=0.01)
    assert frames[0]["polylines"][0][0] == pytest.approx([20 / UP, 30 / UP], abs=0.01)
    assert tracks["shape"] == [RAW, RAW], "shape must be the input frame's, not the 1.5x one"


def test_distance_gates_are_converted_from_input_pixels_and_the_fraction_is_not(
        tmp_path, monkeypatch):
    """--max-shift and --overlap-tol are input pixels; --min-overlap is a unitless fraction."""
    seen = {}
    real = tracker_mod.track_sequence
    monkeypatch.setattr(tracker_mod, "track_sequence",
                        lambda f, p=None: (seen.update(params=p), real(f, p))[1])
    per_frame = [[_line(2, 2, 30, 2)], [_line(2, 2, 30, 2)]]
    _drive(tmp_path, monkeypatch, per_frame,
           extra_argv=["--max-shift", "16", "--overlap-tol", "3", "--min-overlap", "0.4"])
    assert seen["params"] == {"max_shift": 24.0, "overlap_tol": 4.5, "min_overlap": 0.4}


def test_unset_gates_pass_the_fitted_defaults_through_untouched(tmp_path, monkeypatch):
    """The default invocation must be the configuration every published number was measured
    with -- so nothing may be injected when no flag is given."""
    seen = {}
    real = tracker_mod.track_sequence
    monkeypatch.setattr(tracker_mod, "track_sequence",
                        lambda f, p=None: (seen.update(params=p), real(f, p))[1])
    per_frame = [[_line(2, 2, 30, 2)], [_line(2, 2, 30, 2)]]
    _drive(tmp_path, monkeypatch, per_frame)
    assert seen["params"] is None


def test_overlay_is_drawn_in_the_working_frame(tmp_path, monkeypatch):
    """A 'consistency' refactor that divided the polylines in place would misregister the
    overlay by 1.5x against its background -- and the picture would still look plausible."""
    seen = {}
    monkeypatch.setattr(track, "save_track_overlay",
                        lambda path, img, polys, ids: seen.update(img=img, polys=polys))
    per_frame = [[_line(20, 30, 44, 30)], [_line(20, 30, 44, 30)]]
    _drive(tmp_path, monkeypatch, per_frame, extra_argv=["--overlay"])
    assert seen["img"].shape == (int(RAW * UP), int(RAW * UP))
    assert np.asarray(seen["polys"][0])[0] == pytest.approx([20.0, 30.0])


# --------------------------------------------------------------------------------------------
# Frame order IS the time axis.
# --------------------------------------------------------------------------------------------

def test_frames_are_ordered_naturally_not_lexically(tmp_path, monkeypatch):
    per_frame = [[_line(2, 2, 30, 2)] for _ in range(3)]
    tracks, frames = _drive(tmp_path, monkeypatch, per_frame,
                            names=["f_1.tif", "f_2.tif", "f_10.tif"])
    assert tracks["frames"] == ["f_1.tif", "f_2.tif", "f_10.tif"]
    assert [f["frame_index"] for f in frames] == [0, 1, 2]


def test_an_uppercase_extension_is_not_silently_dropped(tmp_path):
    """glob is case-sensitive even on a case-insensitive filesystem. A skipped frame doubles
    the step across the gap, breaks links that should have been made, and therefore reports
    MORE fragmentation than the data has -- which is this project's headline measurement."""
    d = tmp_path / "frames"
    d.mkdir()
    for n in ["f_1.tif", "f_2.TIF", "f_3.Tiff"]:
        (d / n).write_bytes(b"")
    paths, notes = track.order_frames(str(d))
    # .tif and .tiff are ONE format: splitting them would drop half of this folder.
    assert [os.path.basename(p) for p in paths] == ["f_1.tif", "f_2.TIF", "f_3.Tiff"]
    assert not notes


def test_mixed_extensions_are_not_interleaved_as_extra_time_points(tmp_path):
    """Running predict.py --overlay into your own frame folder produces exactly this."""
    d = tmp_path / "frames"
    d.mkdir()
    for n in ["f_1.tif", "f_1.png", "f_2.tif", "f_2.png"]:
        (d / n).write_bytes(b"")
    paths, notes = track.order_frames(str(d))
    assert [os.path.basename(p) for p in paths] == ["f_1.tif", "f_2.tif"]
    assert notes and "IGNORED" in notes[0] and "--ext" in notes[0]


def test_ambiguous_numbering_is_refused_rather_than_left_to_the_filesystem(tmp_path):
    """f2.tif and f02.tif sort identically, so the time axis would differ between machines."""
    d = tmp_path / "frames"
    d.mkdir()
    for n in ["f2.tif", "f02.tif"]:
        (d / n).write_bytes(b"")
    with pytest.raises(SystemExit, match="sort identically"):
        track.order_frames(str(d))


def test_colliding_stems_are_refused(tmp_path):
    """Both would write the same <stem>.json and the second would overwrite the first."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "f_1.tif").write_bytes(b"")
    (tmp_path / "b" / "f_1.tif").write_bytes(b"")
    with pytest.raises(SystemExit, match="share the stem"):
        track.order_frames(str(tmp_path / "*" / "*.tif"))


# --------------------------------------------------------------------------------------------
# Refusals: a gate that can never be satisfied must not look like a successful run.
# --------------------------------------------------------------------------------------------

def test_min_overlap_given_in_pixels_is_refused(tmp_path, monkeypatch):
    """A fraction of 5 can never be met, so every link is rejected and the tool would
    otherwise report a normal-looking run in which nothing was associated."""
    per_frame = [[_line(2, 2, 30, 2)], [_line(2, 2, 30, 2)]]
    with pytest.raises(SystemExit, match="FRACTION"):
        _drive(tmp_path, monkeypatch, per_frame, extra_argv=["--min-overlap", "5"])


def test_a_single_frame_is_refused_and_points_at_predict(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="predict.py"):
        _drive(tmp_path, monkeypatch, [[_line(2, 2, 30, 2)]], names=["f_1.tif"])


def test_min_frames_also_filters_the_per_frame_files(tmp_path, monkeypatch):
    """Documented, not accidental: --min-frames drops a track's detections from the per-frame
    files too, so those report fewer instances than predict.py would for the same frame."""
    stays = _line(2, 2, 30, 2)
    flotsam = _line(2, 34, 30, 34)
    per_frame = [[stays, flotsam], [stays.copy()]]
    tracks, frames = _drive(tmp_path, monkeypatch, per_frame, extra_argv=["--min-frames", "2"])
    assert tracks["n_tracks"] == 1
    assert frames[0]["n_instances"] == 1, "the single-frame detection should be filtered out"
    assert frames[0]["track_ids"] == [tracks["tracks"][0]["track_id"]]


def test_no_velocity_is_emitted_anywhere(tmp_path, monkeypatch):
    """Velocity is not measurable from this foreground (protocol 18a) and a field in a JSON
    file will be used, so the decision is enforced rather than documented."""
    per_frame = [[_line(2, 2, 30, 2)], [_line(4, 2, 32, 2)]]
    tracks, frames = _drive(tmp_path, monkeypatch, per_frame)
    blob = json.dumps(tracks) + json.dumps(frames)
    assert "veloc" not in blob.lower() and "speed" not in blob.lower()


# --------------------------------------------------------------------------------------------
# Regressions introduced by the first fix pass, each reproduced before being fixed.
# --------------------------------------------------------------------------------------------

def test_the_most_numerous_format_wins_not_a_fixed_preference(tmp_path):
    """Ten PNG frames beside two TIFF reference exports must not become a two-frame series.

    Preferring TIFF by declaration order tracked `calibration.tif` and `maxproj.tif` and
    ignored every real frame -- with exit 0 and a plausible tracks.json.
    """
    d = tmp_path / "frames"
    d.mkdir()
    for i in range(10):
        (d / f"frame_{i:02d}.png").write_bytes(b"")
    for n in ["maxproj.tif", "calibration.tif"]:
        (d / n).write_bytes(b"")
    paths, notes = track.order_frames(str(d))
    assert len(paths) == 10 and all(p.endswith(".png") for p in paths)
    assert notes and "IGNORED 2 .tif" in notes[0]


def test_ext_overrides_the_majority(tmp_path):
    d = tmp_path / "frames"
    d.mkdir()
    for i in range(10):
        (d / f"frame_{i:02d}.png").write_bytes(b"")
    for i in range(2):
        (d / f"other_{i}.tif").write_bytes(b"")
    paths, _ = track.order_frames(str(d), "tif")
    assert [os.path.basename(p) for p in paths] == ["other_0.tif", "other_1.tif"]
    with pytest.raises(SystemExit, match="matched no files"):
        track.order_frames(str(d), "jpg")


def test_a_directory_is_not_a_frame(tmp_path):
    """A subdirectory named like an image must not become a time point.

    Deliberately given the SAME extension as the real frames: with a different one the
    majority rule would hide the missing check, which is how the first version of this test
    passed while the filter it was meant to pin was absent.
    """
    d = tmp_path / "frames"
    d.mkdir()
    for i in range(3):
        (d / f"f{i}.tif").write_bytes(b"")
    (d / "stack.tif").mkdir()
    paths, _ = track.order_frames(str(d))
    assert [os.path.basename(p) for p in paths] == ["f0.tif", "f1.tif", "f2.tif"]


def test_glob_metacharacters_in_the_directory_name_are_escaped(tmp_path):
    """`Pos[1]` is an ordinary microscope export name; unescaped it globs to nothing and the
    tool reports 'no frames found' about a folder full of frames."""
    d = tmp_path / "Pos[1]"
    d.mkdir()
    for i in range(3):
        (d / f"f{i}.tif").write_bytes(b"")
    paths, _ = track.order_frames(str(d))
    assert len(paths) == 3


def test_a_glob_input_gets_the_same_guarantees_as_a_directory(tmp_path):
    """`--input 'frames/*'` is documented, and used to bypass case-insensitivity, format
    grouping and the ignored-format note entirely."""
    d = tmp_path / "frames"
    d.mkdir()
    for n in ["f_1.tif", "f_2.TIF", "ov_1.png", "ov_2.png", "notes.txt"]:
        (d / n).write_bytes(b"")
    paths, notes = track.order_frames(str(d / "*"))
    assert [os.path.basename(p) for p in paths] == ["f_1.tif", "f_2.TIF"]
    assert notes and "IGNORED 2 .png" in notes[0]


def test_min_frames_removing_every_track_is_not_reported_as_a_failure_to_associate(
        tmp_path, monkeypatch, capsys):
    """Diagnosing from the filtered tracks made this print a warning in which every clause was
    false: association had worked and the user's own --min-frames was the cause."""
    a = _line(2, 2, 30, 2)
    per_frame = [[a], [a.copy()]]
    _drive(tmp_path, monkeypatch, per_frame, extra_argv=["--min-frames", "3"])
    out = capsys.readouterr().out
    assert "nothing was associated" not in out
    assert "--min-frames 3 removed every track" in out


def test_a_real_association_failure_is_still_reported(tmp_path, monkeypatch, capsys):
    far = [[_line(2, 2, 30, 2)], [_line(2, 200, 30, 200)]]
    _drive(tmp_path, monkeypatch, far)
    out = capsys.readouterr().out
    assert "nothing was associated" in out


def test_tracks_json_is_written_before_the_per_frame_map_is_recovered(tmp_path, monkeypatch):
    """A tripped invariant must not discard a whole GPU run for an artifact already computed."""
    def _boom(tracks, per_frame):
        raise SystemExit("internal error: simulated")

    monkeypatch.setattr(track, "assign_owners", _boom)
    per_frame = [[_line(2, 2, 30, 2)], [_line(2, 2, 30, 2)]]
    with pytest.raises(SystemExit, match="simulated"):
        _drive(tmp_path, monkeypatch, per_frame)
    assert (tmp_path / "out" / "tracks.json").exists(), "the run's main artifact was lost"


def test_non_finite_gate_values_are_refused(tmp_path, monkeypatch):
    """`nan <= 0` is False, so nan passed the bounds check and then disabled its gate."""
    per_frame = [[_line(2, 2, 30, 2)], [_line(2, 2, 30, 2)]]
    for flag in ("--max-shift", "--overlap-tol", "--min-overlap", "--prob-thr"):
        with pytest.raises(SystemExit, match="finite"):
            _drive(tmp_path, monkeypatch, per_frame, extra_argv=[flag, "nan"])


def test_prob_thr_out_of_range_is_refused(tmp_path, monkeypatch):
    per_frame = [[_line(2, 2, 30, 2)], [_line(2, 2, 30, 2)]]
    with pytest.raises(SystemExit, match=r"\[0, 1\]"):
        _drive(tmp_path, monkeypatch, per_frame, extra_argv=["--prob-thr", "5"])


# --------------------------------------------------------------------------------------------
# Guards the first round of tests left unpinned -- each one killed a surviving mutant.
# --------------------------------------------------------------------------------------------

def test_the_field_stop_mask_is_applied_unless_no_fov_is_given(tmp_path, monkeypatch):
    """Every accuracy figure for this model was measured WITH the mask, and the first round of
    tests all passed --no-fov, so the default path had no coverage at all."""
    per_frame = [[_line(2, 2, 30, 2)], [_line(2, 2, 30, 2)]]
    calls = []
    _drive(tmp_path, monkeypatch, per_frame, no_fov=False, seen_calls=calls)
    assert calls and calls[0]["fov"] is not None, "the field-stop mask never reached instancing"

    calls = []
    _drive(tmp_path, monkeypatch, per_frame, no_fov=True, seen_calls=calls)
    assert calls and calls[0]["fov"] is None


def test_prob_thr_overrides_the_params_file(tmp_path, monkeypatch):
    """The same precedence bug that made a threshold sweep silently flat in the eval harness."""
    per_frame = [[_line(2, 2, 30, 2)], [_line(2, 2, 30, 2)]]
    calls = []
    _drive(tmp_path, monkeypatch, per_frame, params={"prob_thr": 0.42}, seen_calls=calls)
    assert calls[0]["thr"] == pytest.approx(0.42)

    calls = []
    _drive(tmp_path, monkeypatch, per_frame, params={"prob_thr": 0.42},
           extra_argv=["--prob-thr", "0.8"], seen_calls=calls)
    assert calls[0]["thr"] == pytest.approx(0.8)


def test_frames_of_different_sizes_are_refused(tmp_path, monkeypatch):
    """Association compares coordinates between frames, so a mixed-size folder would give a
    confident answer that means nothing."""
    sizes = {"f_0.tif": (40, 40), "f_1.tif": (40, 60)}
    per_frame = [[_line(2, 2, 30, 2)], [_line(2, 2, 30, 2)]]
    with pytest.raises(SystemExit, match="frame sizes differ"):
        _drive(tmp_path, monkeypatch, per_frame,
               loader=lambda p: np.zeros(sizes[os.path.basename(p)]))


def test_the_model_sees_the_frame_upscaled_by_UP(tmp_path, monkeypatch):
    """The 1.5x working frame starts at the model input; a dropped zoom would shift every
    coordinate the tracker and the instancer ever see."""
    per_frame = [[_line(2, 2, 30, 2)], [_line(2, 2, 30, 2)]]
    calls = []
    _drive(tmp_path, monkeypatch, per_frame, seen_calls=calls)
    assert calls[0]["shape"][1:] == (int(RAW * UP), int(RAW * UP))


def test_the_src_instance_package_is_the_one_under_test(tmp_path):
    """This module puts deploy_v5H on sys.path, which exposes the generic names `track`,
    `predict`, `fov`, `model` and -- in a full deployment package -- `instance`. The suite
    currently passes only because an earlier test file caches `instance` from src/ first.
    """
    import instance
    assert os.path.join("src", "instance") in instance.__file__, \
        f"tests are running against {instance.__file__}, not src/instance"
