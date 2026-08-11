# Label-free foreground-aware encoding for optimizing MT generator parameters

How to encode IRM images so the metric respects the microtubule foreground (not the background)
and lets us **optimize MT generator parameters with NO annotations**. Combines a rigorous tulen
experiment (realistic generator) with a literature review. 2026-06-28.

## The problem (recap)
Global feature distances (FID / mean-pooled features) are **foreground-blind**: swapping the
background moves the embedding more than adding microtubules (`ENCODER_SENSITIVITY_EXPERIMENT.md`).
We need a label-free objective that is sensitive to MT parameters so an optimizer can recover them.

## Experiment (realistic generator, DINOv2 ViT-L/14 patch tokens, on real backgrounds)
**Two-part test on 106 controlled frames + a distributional recovery run (M=24 frames/value):**

### A. Single-frame strategy comparison — does background subtraction help? (R = dMT/dBG)
Exploiting the **given background** via an estimated-background residual `r = I/lowpass(I) − 1`:

| strategy (label-free) | R=dMT/dBG | note |
|---|---|---|
| RAW global-pool (FID-style) | 0.85 | foreground-blind (R<1) |
| RAW top-5% patch | 1.07 | partial |
| **RESID global-pool** | **1.65** | bg-subtraction collapses dBG → foreground-aware |
| RESID mask-weighted | 1.27 | |

→ **Subtracting the given/estimated background makes even a global metric foreground-aware**
(R 0.85 → 1.65). Single-frame distances are still noisy for parameter recovery (instance-level
geometry randomness), so the calibration objective must be **distributional (MMD over many frames)**.

### B. Distributional MMD parameter recovery — the key proof
For each parameter value, render M=24 frames (random real backgrounds + seeds), embed with DINOv2
patch tokens, pool, and compute **linear MMD to a target set** with known params (`mmd_recovery.py`).
The objective is **minimized at the true value** for both axes and all feature spaces:

| axis | truth | argmin (raw) | argmin (residual) | residual sharper? |
|---|---|---|---|---|
| density | 16 | **16** | **16** | yes (~2.4× deeper well) |
| contrast | 1.0 | **1.0** | **1.0** | yes (~2× deeper) |

Curves are convex with a clear minimum at the truth (`data/enc_sensitivity_testset/labelfree_recovery.png`).
**Parameters are recoverable label-free.** Residual (background-subtracted) encoding gives a
**sharper minimum** (better SNR for the optimizer); raw DINOv2 also recovers the truth.

### C. Which DINOv2 layer best reacts to MT changes? (block sweep, `layer_sweep.py`)
R = dMT/dBG per ViT-L/24 block, raw vs residual input:

| block | 1 | 4 | **5** | 9 | **11** | **13** | 15 | **17** | 21 | 23(last) |
|---|---|---|---|---|---|---|---|---|---|---|
| R (residual) | 2.18 | 1.61 | 1.69 | 2.29 | **2.71** | 2.70 | 2.46 | 2.23 | 1.55 | 1.66 |
| R (raw) | 0.70 | 0.92 | 0.99 | 1.10 | 1.15 | 1.19 | 1.01 | 1.04 | 0.84 | 0.85 |
| density-monotonicity | 0.84 | 0.76 | 0.80 | 0.72 | 0.72 | 0.72 | 0.80 | **0.92** | 0.92 | 0.88 |

(`data/enc_sensitivity_testset/dinov2_layer_sweep.png`). Findings:
- **Residual beats raw at EVERY layer** — background subtraction is the dominant lever, layer-independent.
- **Peak MT sensitivity is at MID blocks 11–13 (R≈2.7), NOT SynthMT's layer 5 (R≈1.7) and NOT the
  last layer (R≈1.66).** SynthMT's layer-5 choice is sub-optimal for our IRM setup.
- **Best monotonicity (smooth optimization signal) at blocks 17–21 (0.92).** There's a trade-off:
  blocks 11–13 maximize raw discrimination, blocks 15–17 give high R **and** good monotonicity →
  **sweet spot for an optimizer ≈ block 15–17** (or concatenate 11–17). Use a MID-to-late block,
  not the last embedding.

### D. Can MMD recover a property's VARIANCE? NO (with linear MMD) — `variance_recovery.py`
Tested whether the objective recovers the **std** (spread) of a property, not just its mean: build a
target set with known per-MT std, sweep the generator's std knob (mean fixed), check argmin.

| property | truth std | argmin | MMD range | verdict |
|---|---|---|---|---|
| width std | 0.4 | 0.7 (wrong) | 0.0009–0.007 (flat, noisy) | ✗ not recoverable |
| contrast std | 0.5 | 0.3 (wrong) | 0.001–0.013 (flat, noisy) | ✗ not recoverable |

Contrast with MEAN recovery (clean convex well at truth, MMD range 0.005–0.09). **Root cause:** the
objective uses **LINEAR MMD = ‖mean(features_synth) − mean(features_real)‖²**, which by construction
compares only the FIRST moment of the feature distribution → blind to a generative property's spread.
**Implication:** calibrate the **MEAN** of each property (works), keep each property's **std at a
fixed domain prior** (BIOCEV-seeded), and do NOT add std knobs — they would be unidentifiable and
just add noise/confounding. If variance-matching is ever needed, switch to an **RBF-kernel MMD /
covariance term** (captures higher moments) and re-test — but defer; the gate is the downstream test.
(`data/enc_sensitivity_testset/variance_recovery_test.png`.)

Generator fixes applied alongside (step 1): per-MT width is now a truncated **Normal(mean,std)** with
a lower mean (sub-diffraction, fixes too-thick MTs); **irregular bundles** (variable gap, per-member
angular divergence, staggered ends — fixes too-regular parallels); **dirt specks/interference spots**.

## Literature (cited research)
- **SynthMT** (Koddenbrock et al., PLOS Comput. Biol. 2026, `10.1371/journal.pcbi.1013901`,
  code `github.com/ml-lab-htw/SynthMT`) — the DIRECT precedent: tunes a synthetic-IRM-MT generator
  against real **unlabeled** frames by **maximizing cosine similarity of DINOv2 *5th-layer* features**,
  optimized by **TPE (Optuna), ~1000 trials**, **no background subtraction / no masking** — and gets
  near-human downstream skeleton-IoU. ⇒ The objective is already validated; residual/masking is an
  *enhancement*, not a prerequisite.
- **Use MMD/CMMD, not FID:** FID is sample-size-dependent (rankings flip), resize/JPEG-fragile, needs
  >20k samples (Chong&Forsyth CVPR’20; clean-FID Parmar’22; CMMD Jayasumana CVPR’24). Localized/patch
  MMD is provably more sensitive to spatially-local signal; style-loss = quadratic-kernel MMD.
- **Encoder:** DINOv2 ViT-L/14 with **register tokens** (suppress background artifact tokens that
  concentrate on low-info background patches, Darcet ICLR’24), **mid/intermediate patch tokens**
  (layer ~5, as SynthMT) — not the CLS/global embedding.
- **Background models** (we have the empty-field library): temporal/spatial median is the IRM
  field-standard (Mahamdeh PMC6858481); black+white top-hat / Frangi tubeness give a label-free
  foreground weight; add a small offset before subtraction; avoid aggressive rolling-ball (halo).
- **Cautions:** perceptual metrics are adversarially exploitable → optimizer can reward-hack a
  single cosine distance (use harder-to-game patch-MMD, a held-out 2nd metric, param priors).
  Generator knobs (density/contrast/width/noise) are partly **non-identifiable** (a ridge of
  solutions) → regularize + validate per-axis. Residual encoding may push DINOv2 OOD — validate.
  Feature distance is the SEARCH signal; the ACCEPTANCE test is downstream (does a model trained on
  the synth over-fire on the 12 Alice frames? arXiv:2502.17160 warns feature distance ≠ downstream).

## The calibration loop (built & working — `synth/calibrate.py`)
Label-free Optuna/TPE tuning of `GenConfig` knobs so synthetic frames match the **real corpus**
distribution (320 frames, `morphology_reference_frames/irm`) in **DINOv2 blocks 11/15/17
background-subtracted-residual** patch features via linear **MMD**. 13 knobs (density, contrast,
width, curvature, halo, length, waviness frame-prob+amp, detachment frame-prob+prob, inversion,
PSF, texture), ranges seeded from BIOCEV priors; M=16 synth frames/trial; 150 trials.

**Result:** MMD **0.057 → 0.032 (44 % lower)**, clean TPE convergence
(`data/enc_sensitivity_testset/calibration/`). Recovered params are physically sensible: density
center **14** (real ≈16), contrast **1.6 %**, waviness in **~48 %** of frames, inversion **~51 %**,
sharp PSF. Calibrated renders look IRM-plausible side-by-side with real (`compare_calibrated_vs_real.png`).
Generator knobs that are acquisition CONDITIONS (waviness, contrast, width, halo, detachment,
inversion) are sampled per-FRAME and applied to ALL microtubules (per the frame-condition design).

Remaining visible gaps (next): real frames have more **dirt specks / interference spots** and
**hairpin/U-turn** morphology not yet modelled; the real set is heterogeneous (some near-empty
frames). The objective is the SEARCH signal — acceptance is still the downstream **over-firing
check on the 12 Alice frames**.

## ✅ Downstream acceptance test — PASSED (2026-06-30)
The real gate (per CLAUDE.md): does a foreground trained on the synth over-fire on the 12 Alice
frames? Trained a **U-Net** (binary MT-vs-background) on **600 CALIBRATED synthetic frames** (free GT,
no real labels), predicted on the 12 real Alice frames:
- predicted foreground **2.76%** on real vs **3.44%** in-domain synth → **over-firing ratio 0.8× = CLEAN**
  (historically synth-trained foregrounds over-fired ~6× and broke every downstream step).
- Visually the predicted foreground traces real MTs cleanly — including **hairpins/U-turns that were
  NOT in the synthetic morphology** (generalizes) — with only a few dirt-speck false positives.

**The calibrated synthetic data transfers to real.** Scripts: `gen_train.py`, `train_unet.py`,
`predict_alice.py`; overlays in `data/enc_sensitivity_testset/alice_overfiring/`.

## ✅✅ Strict centerline-F1 — beats the synth-only bar (2026-06-30)
Added the INSTANCE step (`instance_eval.py`): skeletonize the U-Net foreground → skeleton graph →
**collinear arm-pairing at junctions (ORION-lite)** → trace instances → strict centerline-F1
(tol=5, length/precision coverage 0.95, 1.5× upscale) vs the GT polylines, using BIOCEV's exact
`centerline_f1.py` scorer.

| method | strict F1 |
|---|---|
| **ours (synthetic-only, ZERO real annotations)** | **0.501** (mean) / 0.478 (micro) |
| TARDIS zero-shot (honest synth-only target) | 0.326 |
| ORION (best tuned heuristic) | 0.519 |
| v19 (production, real self-train) | 0.696 |

TP=140, FP=215, FN=91 (recall 0.61, precision 0.39). **Decisively beats TARDIS 0.326 and essentially
matches ORION 0.519 — with no real labels anywhere in the pipeline.** Key tracer fixes: prune short
skeleton spurs + drop sub-`MIN_SKEL` fragments (raised F1 0.11 → 0.50 by killing fragment FPs).
Headroom: the over-segmented dense/wavy frames (precision-limited) — better junction pairing /
gap-bridging, or TARDIS-DIST as the instance head, should push past ORION.

## Recommended approach (reconciles experiment + literature)
1. **Encode** with DINOv2 ViT-L/14 (register checkpoint), **mid-to-late patch tokens — empirically
   block ~15–17 (or concat 11–17), NOT SynthMT's layer 5 and NOT the last layer** (see §C), ~518px.
2. **Foreground-aware input:** exploit the given background — compute the **background-subtracted
   residual** (estimated via median/top-hat/low-pass of the real-background library); it sharpens
   the objective ~2×. Keep a RAW-feature variant too (the SynthMT route that provably works) and
   let the downstream check pick.
3. **Objective = distributional MMD** (patch-token / foreground-weighted, RBF or linear kernel)
   over many synthetic frames vs the real set — NOT single-frame distance, NOT FID.
4. **Optimizer:** TPE/Optuna (or CMA-ES) on the `GenConfig` knobs, per real condition/density regime.
5. **Guards:** patch-MMD (not a single cosine), a held-out non-optimized metric, priors/bounds on
   each knob (non-identifiability), and **final acceptance = no over-firing on the 12 Alice frames**.

Scripts: `scratchpad/enc_exp/{gen_test_realistic,encode_strategies,analyze_strategies,mmd_recovery}.py`.
Data on tulen `/home/prusek/mt_enc_exp/realtest/`.
