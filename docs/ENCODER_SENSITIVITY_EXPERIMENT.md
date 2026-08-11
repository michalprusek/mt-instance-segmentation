# Which encoder's features react to microtubule changes? (empirical)

Empirical test (2026-06-27, run on tulen A100) of the claim that **global feature distances
(FID/MMD-style) are foreground-blind** to thin microtubules, and of **which image encoder is
most sensitive to MT changes on a fixed background** — i.e. best suited to drive a label-free
synthetic-generator calibration loop.

## Method
- **Controlled test set:** 6 real empty-field IRM backgrounds (microscopes A/B/C, from
  `irm_backgrounds_v2`). On each FIXED background we render synthetic MTs and vary ONE factor at
  a time (density 0/5/15/30/60, contrast, width, polarity) → a pure FOREGROUND change. We also
  paste one fixed-geometry MT field onto different backgrounds → a background-swap baseline.
  90 images total. (Scripts: `scratchpad/enc_exp/make_testset.py`, `encode_grids.py`, `analyze_grids.py`.)
- **Encoders** (patch-token grids, L2-normalized per token): DINOv2 ViT-L/14, DINOv3 ViT-L/16,
  DINOv3-ConvNeXt-L, ConvNeXtV2-L, CLIP ViT-L/14, Inception-v3 (the classic **FID** encoder),
  **C-RADIOv4-h**, SAM2-Hiera-L (SAM-family proxy; SAM3 needs transformers≥5, absent on tulen).
- **Readouts** (3 ways to turn patch grids into a distance):
  - `GLOBAL` = distance between **mean-pooled** embeddings — what FID/MMD actually use.
  - `PATCH`  = mean over patches of **per-patch** cosine distance.
  - `TOP5`   = mean of the **top-5% per-patch** distances — foreground-focused.
- **Metric:** `R = dMT / dBG` = (embedding move when 15 MTs APPEAR on a fixed background) ÷
  (move when the background is SWAPPED, no MT). **R < 1 ⇒ background-dominated (foreground-blind).**

## Results (ranked by TOP5_R, the best achievable foreground signal)

| encoder | tokens | GLOBAL_R | PATCH_R | **TOP5_R** | density-monotonic |
|---|---|---|---|---|---|
| **DINOv2 ViT-L/14** (518px) | 1369 | 0.28 | 0.32 | **0.79** | 1.00 |
| CLIP ViT-L/14 (224) | 256 | 0.46 | 0.49 | 0.77 | 0.98 |
| DINOv2 ViT-L/14 (768) | 2916 | 0.27 | 0.27 | 0.72 | 1.00 |
| DINOv2 ViT-L/14 (1024) | 5329 | 0.25 | 0.22 | 0.68 | 1.00 |
| Inception-v3 (FID) (299) | 64 | 0.57 | 0.54 | 0.67 | 1.00 |
| ConvNeXtV2-L (384) | 144 | 0.52 | 0.33 | 0.61 | 1.00 |
| SAM2-Hiera-L (1024) | 4096 | 0.69 | 0.37 | 0.54 | 0.98 |
| **C-RADIOv4-h** (512) | 1024 | 0.26 | 0.16 | 0.39 | 0.92 |
| DINOv3-ConvNeXt-L (512) | 256 | 0.38 | 0.18 | 0.31 | 0.87 |
| DINOv3 ViT-L/16 (512) | 1024 | 0.12 | 0.11 | 0.31 | 1.00 |
| DINOv3 ViT-L/16 (768) | 2304 | 0.18 | 0.11 | 0.27 | 0.98 |
| DINOv3 ViT-L/16 (1024) | 4096 | 0.14 | 0.09 | 0.24 | 1.00 |

## MT-segmentation-model encoder (TARDIS FNet, real-MT-trained)
We also tested the encoder of **TARDIS FNet** (`fnet_attn_32/microtubules_tirf`, the real-trained
MT foreground CNN that works on real data) — bottleneck features (T=1024, D=512), images fed as-is
(dark MT) and inverted (TARDIS was trained on TIRF = bright MT).

| variant | GLOBAL_R | PATCH_R | TOP5_R | **TOP5_dMT** | dens-mono |
|---|---|---|---|---|---|
| **TARDIS FNet (as-is)** | 0.31 | 0.30 | 0.68 | **0.70 (highest of ALL)** | 0.82 |
| TARDIS FNet (inverted) | 0.02 | 0.07 | 0.26 | 0.26 | 0.97 |

- The MT-trained encoder has the **strongest ABSOLUTE response to MTs** (TOP5_dMT 0.70 — it fires
  hardest on microtubules, exactly as hoped), beating DINOv2's 0.61.
- But its **ratio R=0.68 < DINOv2's 0.79**: it ALSO reacts strongly to BACKGROUND changes (it was
  trained on **TIRF**; our IRM backgrounds from 3 unseen microscopes shift it), and its density
  response is **less monotonic** (0.82). Feeding dark MT **as-is beats inverting** (inverting
  collapses it, GLOBAL_R 0.02).
- Implication: a foreground model trained **on our IRM domain** (not TIRF) would likely give the
  best foreground/background ratio — i.e. once we have an in-domain MT model, its features become a
  strong calibration metric (a self-improving loop). For now, DINOv2 is the most reliable choice;
  TARDIS is the most MT-reactive but domain-confounded.

## Conclusions
1. **FID/MMD global pooling is foreground-blind — confirmed.** `GLOBAL_R < 1` for **every**
   encoder (0.12–0.69): swapping the background moves the global embedding MORE than 15 MTs
   appearing. The exact failure mode predicted in `SYNTH_CALIBRATION_RESEARCH.md`.
2. **The readout matters as much as the encoder.** Going from global pooling → top-5% patch
   readout roughly **doubles** sensitivity for the good encoders (DINOv2 0.28 → 0.79). A
   foreground-focused (patch / top-k) distance is essential; don't calibrate on global FID.
3. **Best encoder for MT sensitivity: DINOv2 ViT-L/14** (TOP5_R 0.79), CLIP close behind.
   This independently corroborates **SynthMT**, the published MT-IRM calibration precedent,
   which used **DINOv2**.
4. **DINOv3 is consistently the WORST** — both ViT and ConvNeXt variants, at every resolution
   (768/1024 do NOT rescue it: 0.31→0.27→0.24). Its features are genuinely insensitive to thin,
   low-contrast lines (DINOv3's Gram-anchored, high-level/smooth features suppress exactly the
   structure we need). **Counterintuitive but robust — do NOT use DINOv3 for this.**
5. **C-RADIOv4 (0.39) and SAM2-Hiera (0.54) are mediocre** — better than DINOv3, well below DINOv2.
6. Even the best (DINOv2 + TOP5, R=0.79) stays **< 1**: adding 15 MTs still moves the signal a
   bit less than a full background swap. ⇒ The calibration objective must ALSO control the
   background — **composite synthetic MTs onto the real backgrounds** (which we do) so background
   is held ~fixed and the residual signal is the foreground.

## Practical recommendation for the calibration metric
Drive the analysis-by-synthesis loop with **DINOv2 ViT-L/14 patch tokens at ~518px**, comparing
**foreground-weighted / top-k patch distributions** (MMD or top-k patch cosine), on **synthetic-
on-real-background composites**. Avoid DINOv3 and avoid global-pooled FID/KID. Optionally weight
patches by the background-subtracted MT mask to further suppress background.

## Caveats
- Synthetic (not real) MTs, additive IRM-like contrast (~6% default), `dMT` measured at the
  density-15 point; sensitivity ranks should hold but absolute R values are setup-specific.
- "Sensitivity to controlled MT change" is the right proxy for a calibration *gradient*, but the
  ultimate test remains downstream real centerline-F1 + the over-firing check on the 12 Alice frames.
- SAM2 is a SAM-family stand-in; **real SAM3** needs transformers≥5 (only on the local Mac, which
  is excluded from compute) — can be added in a dedicated tulen venv if wanted.
