"""The TIRF training option must change the rendered IMAGE, not just be plumbed through.

A previous parameter reached the sampler and was never read, and two GPUs then compared a
configuration with itself for three hours. These tests therefore compare rendered pixels and the
statistics that distinguish the two modalities, never a signature or a flag.

What actually separates TIRF from IRM here:
  * IRM is multiplicative on a BRIGHT field, filaments dark at contact, with an interference halo.
  * TIRF is additive on a DARK field, filaments always bright, no halo, shot-noise limited.
"""
import numpy as np
import pytest

from synth.mt_generator import GenConfig, generate_frame

SHAPE = (256, 256)


def _bg(seed=0):
    """A plausible IRM background: bright, gently shaded, mildly noisy."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    shade = 1.0 + 0.05 * (x / SHAPE[1]) - 0.03 * (y / SHAPE[0])
    return (200.0 * shade + rng.normal(0, 2.0, SHAPE)).astype(np.float64)


def _render(modality, seed=3):
    cfg = GenConfig(n_mt_range=(6, 10))
    return generate_frame(_bg(), np.random.default_rng(seed), cfg, modality=modality)


def _filament_polarity(img, instances):
    """Median (filament minus frame background) over the rendered instances."""
    bg = float(np.median(img))
    vals = []
    for ins in instances:
        m = ins["mask"]
        if m.any():
            vals.append(float(np.median(img[m])) - bg)
    return float(np.median(vals)) if vals else float("nan")


def test_the_two_modalities_render_different_images():
    a, ia, ma = _render("irm")
    b, ib, mb = _render("tirf")
    assert ma["modality"] == "irm" and mb["modality"] == "tirf"
    assert a.shape == b.shape
    # same morphology seed, so the geometry matches and only the appearance may differ
    assert len(ia) == len(ib)
    assert not np.allclose(a, b), "the TIRF renderer produced the IRM image"


def test_tirf_filaments_are_bright_on_a_dark_field():
    """The defining appearance difference, measured on pixels rather than asserted."""
    a, ia, _ = _render("irm")
    b, ib, _ = _render("tirf")
    assert _filament_polarity(a, ia) < 0, "IRM filaments should be darker than the field"
    assert _filament_polarity(b, ib) > 0, "TIRF filaments should be brighter than the field"


def test_tirf_filaments_are_ALWAYS_bright_while_irm_polarity_varies():
    """The deeper difference, and the reason TIRF is a second modality rather than a recolouring.

    Fluorescence is emissive -- `render_tirf` clips the signal field at zero -- so a TIRF filament
    is bright in every frame. IRM contrast is -cos(2*k*h) in the height above the coverslip, so its
    filaments are dark at contact and bright once lifted, and the sign genuinely varies frame to
    frame. A renderer that merely inverted the image would fail this.
    """
    irm, tirf = [], []
    for seed in range(12):
        a, ia, _ = _render("irm", seed=seed)
        b, ib, _ = _render("tirf", seed=seed)
        irm.append(_filament_polarity(a, ia))
        tirf.append(_filament_polarity(b, ib))
    irm = np.array([v for v in irm if np.isfinite(v)])
    tirf = np.array([v for v in tirf if np.isfinite(v)])
    assert len(irm) >= 8 and len(tirf) >= 8
    assert (tirf > 0).all(), f"TIRF must never render a dark filament, got {tirf}"
    assert (irm < 0).mean() < 1.0, "IRM polarity should vary across frames, not be constant"


def test_instances_carry_the_modality_polarity():
    _, ia, _ = _render("irm")
    _, ib, _ = _render("tirf")
    assert all(i["polarity_base"] == -1.0 for i in ia)
    assert all(i["polarity_base"] == +1.0 for i in ib)


def test_dataset_mix_reaches_both_modalities():
    """Simulate the dataset's own choice rule and confirm both branches are actually taken.

    Reproduces the one line in OnlineDS rather than importing the trainer, which needs the
    training host's paths. If that rule is ever changed, this test does not protect it -- the
    guard that does is the pixel comparison above.
    """
    rng = np.random.default_rng(0)
    picks = ["tirf" if rng.random() < 0.35 else "irm" for _ in range(400)]
    assert 0.25 < picks.count("tirf") / len(picks) < 0.45
    picks0 = ["tirf" if rng.random() < 0.0 else "irm" for _ in range(200)]
    assert set(picks0) == {"irm"}
