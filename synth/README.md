# Synthetic microtubule generator (`synth/`)

Renders synthetic microtubules onto **real empty-field IRM backgrounds** (`data/real/irm_backgrounds_v2/`)
with free ground truth (per-instance centerline polylines + masks). Built for the no-annotation
setting; inspired by the BIOCEV `synthmt_irm` generator, reimplemented parametrically.

## Files
- `mt_generator.py` — `GenConfig` (all params) + `generate_frame(background, rng, cfg) -> (image, instances)`.
- `render_samples.py` — driver: renders sample frames + zoom + GT overlay (run on tulen over IRM_backgrounds_v2).

## Frame-level conditions (important)
Properties that are **acquisition/buffer conditions** are sampled ONCE per frame and applied to ALL
microtubules (per-MT only phase + small degree jitter); if a condition is off, NO MT shows it.
`sample_frame_cond()` sets: **waviness** (a frame is wavy or not — if wavy every MT is wavy in the
same regime, if not none are), **contrast level**, **width/PSF**, **halo/interference**,
**detachment activity**, **inversion**. Per-MT: geometry, length, position, base curvature, jitters.

## What it models
- **Morphology (stiff):** microtubules are nearly straight with gentle long-range curvature —
  straight baseline + heavily-smoothed lateral deviation (bounded to `curve_frac_range` of length)
  + gentle arc + **frame-level** amplitude-modulated waviness. Optional **parallel bundles**.
- **IRM appearance & polarity:** parametric cross-section = signed Gaussian core + opposite-sign
  **interference halo** (`halo_ratio`). Microtubules are **DARK** (attached to the coverslip);
  occasional **LOCAL bright segments** where the filament **detaches from the glass**
  (`detach_prob`, smooth Gaussian transitions). Whole **frames are sometimes inverted incl.
  background** (`frame_invert_prob`) → models inverted-contrast IRM acquisitions (MT bright on
  inverted bg). `generate_frame` returns `(image, instances, meta)` with `meta["inverted"]`.
- **Optics & sensor:** Gaussian PSF on the signal, **multiplicative** composite `img = bg·(1+contrast)`,
  Poisson shot + Gaussian read + correlated mid-frequency texture noise.
- **Crossings:** additive signed contrast (overlapping interference).

## Run (on tulen — see [[always-use-tulen]])
`/home/prusek/dinov3_env/bin/python render_samples.py --out <dir>` (needs numpy, scipy, PIL).

## Calibration loop
`calibrate.py` (on tulen) — label-free Optuna/TPE tuning of `GenConfig` knobs so synthetic frames
match the REAL corpus distribution in **DINOv2 (blocks 11/15/17) background-subtracted-residual**
patch features via linear **MMD**. Real target = `morphology_reference_frames/irm` (320 frames);
backgrounds from `IRM_backgrounds_v2`. Knob ranges seeded from BIOCEV `synthmt_irm` priors. Outputs
`calib/best.json` + calibrated sample renders. See `docs/LABELFREE_FOREGROUND_ENCODING.md`.

## Status / next
✓ Realistic stiff morphology + IRM polarity flips + interference halo + real-background composite + free GT.
✓ Frame-level conditions (waviness etc.). ✓ Optuna calibration loop (`calibrate.py`).
TODO: (1) the morphology/appearance/**density** distributions are NOT yet calibrated to real —
that's the DINOv2 foreground-aware calibration loop (`docs/ENCODER_SENSITIVITY_EXPERIMENT.md`,
`docs/SYNTH_CALIBRATION_RESEARCH.md`); (2) some filaments render too high-contrast vs real — tune
`contrast_range`; (3) parametric profile could be refined from background-subtracted real signal;
(4) add IRM interference spots / Newton rings.
