"""ARM 65: the clip generator. Frame t must be the still recipe exactly; neighbours must replay t's appearance and noise levels (docs/TODO.md 2026-11-04 08:10)."""
import dataclasses
import json
import os
import sys

import numpy as np
import pytest

sys.path[:0] = [os.path.join(os.path.dirname(__file__), "..", "synth"), os.path.join(os.path.dirname(__file__), "..", "scripts")]
from mt_clip import ClipConfig, clip_triplet, freeze_noise, generate_clip, move_image
from mt_generator import GenConfig, generate_frame, render_irm, render_tirf, sample_scene

CALIB = os.environ.get("MT_CLIP_CALIB", "calib_V16_debris.json")
NOISELESS = dict(noise_alt_prob=0.0, poisson_gain_range=(1e7, 1e7), read_noise_range=(0.0, 0.0), noise_corr_frac_range=(0.0, 0.0), noise_impulse_frac_range=(0.0, 0.0))
STILL = ClipConfig(jitter_px=(0.0, 0.0), shift_scale_px=(0.0, 0.0))


def v16():
    """The recipe's config, built as the trainer builds it."""
    from gen_train import build_cfg
    d = json.load(open(CALIB))
    cfg = build_cfg(d["best_params"], mask_hw=1.0)
    return dataclasses.replace(cfg, **d["gen_overrides"])


def bg(seed=0, n=400):
    r = np.random.default_rng(seed)
    return np.clip(0.5 + 0.02 * r.normal(size=(n, n)), 0.01, None).astype(np.float32)


@pytest.mark.parametrize("modality", ["irm", "tirf"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_frame_t_and_the_stream_are_the_still_recipe(modality, seed):
    cfg = v16(); b = bg()
    r0, r1 = np.random.default_rng(seed), np.random.default_rng(seed)
    a, ia, _ = generate_frame(b, r0, cfg, modality=modality)
    imgs, per, meta = generate_clip(b, r1, cfg, 1, 1, modality)
    assert np.array_equal(a, imgs[1])
    assert len(ia) == len(per[1]) and all(np.array_equal(x["centerline"], y["centerline"]) for x, y in zip(ia, per[1]))
    assert r0.bit_generator.state == r1.bit_generator.state          # crop and flips of the item do not move
    assert not np.array_equal(imgs[0], imgs[1]) and not np.array_equal(imgs[2], imgs[1])
    assert len(per[0]) == len(per[1]) == len(per[2])                 # nobody is dropped: the appearance replay depends on it


def test_a_single_item_is_the_still_recipe():
    cfg = v16(); b = bg()
    r0, r1 = np.random.default_rng(9), np.random.default_rng(9)
    a, _, _ = generate_frame(b, r0, cfg)
    c, _, m = clip_triplet(b, r1, cfg, p_single=1.0)
    assert np.array_equal(a, c) and m["neighbours"] is None and r0.bit_generator.state == r1.bit_generator.state


def test_noise_regime_and_levels_are_frozen_and_the_realisation_is_not():
    cfg = v16(); b = bg(); seen = set()
    for seed in range(12):
        rng = np.random.default_rng(seed)
        inst, cond = sample_scene(b.shape, rng, cfg)
        _, m = render_irm(inst, cond, b, rng, cfg)
        seen.add(bool(m["noise_alt"]))
        fr = freeze_noise(cfg, m, "irm"); out = []
        for k in (100, 101):
            app = np.random.default_rng(seed + 1000)
            im, n = render_irm(inst, cond, b, app, fr, noise_rng=np.random.default_rng(k)); out.append(im)
            assert "noise_alt" not in n
            for key in ("poisson_gain", "read_noise", "noise_span_rel", "noise_corr_frac", "noise_corr_sigma", "noise_impulse_frac"):
                assert (key in m) == (key in n) and (key not in m or n[key] == m[key]), key
        assert not np.array_equal(out[0], out[1])
    assert seen == {True, False}                                      # both NMIX regimes were exercised


def test_tirf_levels_are_frozen():
    cfg = v16(); b = bg(); rng = np.random.default_rng(5)
    inst, cond = sample_scene(b.shape, rng, cfg)
    _, m = render_tirf(inst, cond, b, rng, cfg)
    fr = freeze_noise(cfg, m, "tirf")
    _, n = render_tirf(inst, cond, b, np.random.default_rng(6), fr, noise_rng=np.random.default_rng(7))
    assert n["poisson_gain"] == m["poisson_gain"] and n["read_noise"] == m["read_noise"]


def test_static_neighbours_replay_the_appearance_exactly():
    cfg = dataclasses.replace(v16(), **NOISELESS); b = bg()
    imgs, _, m = generate_clip(b, np.random.default_rng(11), cfg, 2, 2, ccfg=STILL, regime="static")
    for k in (0, 1, 3, 4):
        assert np.abs(imgs[k] - imgs[2]).max() < 2e-3, k
    assert m["motion"] == "static"


def test_the_whole_frame_moves_with_its_truth():
    cfg = dataclasses.replace(v16(), **NOISELESS); b = bg()
    cc = ClipConfig(jitter_px=(0.0, 0.0), shift_scale_px=(4.0, 4.0))
    moved = 0
    for seed in range(6):
        imgs, per, _ = generate_clip(b, np.random.default_rng(seed), cfg, 0, 1, ccfg=cc, regime="static")
        dx, dy = np.rint(per[1][0]["centerline"][0] - per[0][0]["centerline"][0]).astype(int)
        m = 8
        assert np.abs(imgs[1] - move_image(imgs[0], (dy, dx)))[m:-m, m:-m].max() < 2e-3
        moved += int(dx != 0 or dy != 0)
    assert moved >= 3


def test_gliding_moves_along_the_filament_and_stuck_ones_stay():
    cfg = v16(); b = bg()
    cc = ClipConfig(jitter_px=(0.0, 0.0), shift_scale_px=(0.0, 0.0), glide_speed_px=(6.0, 6.0), stuck_frac=(0.3, 0.3), glide_scatter=0.0)
    _, per, _ = generate_clip(b, np.random.default_rng(3), cfg, 1, 1, ccfg=cc, regime="gliding")
    moved, stayed = [], 0
    for a, c in zip(per[1], per[2]):
        if len(a["centerline"]) < 30: continue
        d = np.linalg.norm(a["centerline"] - c["centerline"], axis=1).mean()
        if d == 0: stayed += 1
        else: moved.append(d)
    assert stayed > 0 and len(moved) > 0
    assert 3.0 < np.median(moved) < 9.0                               # about the 6 px asked for


def test_dynamic_tips_change_length_and_nobody_vanishes():
    cfg = v16(); b = bg()
    _, per, _ = generate_clip(b, np.random.default_rng(4), cfg, 1, 1, ccfg=STILL, regime="dynamic")
    length = lambda c: np.linalg.norm(np.diff(c, axis=0), axis=1).sum()
    diff = [abs(length(a["centerline"]) - length(c["centerline"])) for a, c in zip(per[1], per[2])]
    assert len(per[0]) == len(per[1]) == len(per[2]) and np.median(diff) > 0.3
