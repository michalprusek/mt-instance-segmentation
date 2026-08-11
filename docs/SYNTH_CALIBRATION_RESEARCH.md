# Calibrating a synthetic MT generator to real data WITHOUT annotations

Deep-research synthesis (2026-06-27). Question: how to tune a synthetic generator's MT
**appearance**, **morphology**, and **count/density** so synth-trained segmentation transfers
to real label-free microscopy (IRM/TIRF), given ~520 unlabeled real images + 428 real
empty-field IRM backgrounds and **zero MT annotations**. Confidence tags reflect adversarial
verification (3-vote) + source quality.

## TL;DR — the validated recipe
**Analysis-by-synthesis on top of your real backgrounds.** Render synthetic MTs onto real
empty-field backgrounds, then **tune generator parameters by minimizing a distribution-matching
objective between synthetic and real images in a foundation-model feature space** (DINO/CNN),
optimized with a black-box optimizer (Optuna-TPE / CMA-ES). Estimate the priors you can
(appearance from background-subtracted real signal; morphology from unsupervised curvilinear
detectors) and let the optimizer fit the rest. Close any residual appearance gap with
**label-preserving unpaired translation** (SimGAN/CUT). **Density is the dangerous unknown** —
no method in the literature; engineer and validate it explicitly because it is *the* documented
cause of over-firing.

## ⭐ Near-identical precedent — read this first
**SynthMT** — "Synthetic data enables human-grade microtubule analysis with foundation models
for segmentation", PLOS Comput. Biol. (`10.1371/journal.pcbi.1013901`). Synthetic microtubules
tuned on **real IRM frames of in-vitro MTs without human annotations**; the calibration loop
optimizes generator parameters with a **tree-structured Parzen estimator (TPE/Optuna)** to
**maximize cosine similarity between DINOv2 (layer-5) embeddings of real vs synthetic images** —
a label-free analysis-by-synthesis loop using foundation-model features as the domain-gap
objective. This is essentially the blueprint for our task. *(Surfaced in search+fetch; its
specific numeric claims were outside the adversarially-verified top-25, so read the paper
directly — but it is the strongest single lead.)*

## 1. Calibration strategy — distribution matching, not task-driven (HIGH)
- **Meta-Sim** (Kar et al., ICCV 2019, `arXiv:1904.11621`) tunes a procedural generator by
  minimizing the gap between rendered and real outputs via **MMD on pretrained InceptionV3
  features** — needs only **unlabeled** real images. ✅ usable.
- **Learning to Simulate** (Ruiz et al., ICLR 2019, `arXiv:1810.02513`) and Meta-Sim's stronger
  meta-objective optimize a **REINFORCE/task reward computed on a real *labeled* validation
  set** → ❌ **off the table** for us (zero annotations).
- **Practical:** drive Optuna/CMA-ES on generator params against an MMD-style / DINO-feature
  distribution objective (as SynthMT does). Reserve any tiny labeled set only for final eval.

## 2. Appearance — model the IRM physics, fit from background-subtracted signal (HIGH)
- **IRM two-reflection interference model** (Mahamdeh & Howard, JoVE 2019, `PMC6858481`):
  contrast = interference of glass/water reflection (I1) and water/MT reflection (I2). Because
  optical path depends on filament height, **contrast flips dark↔bright (and intermediate)
  every ~100–113 nm of MT height** (λ/4n ≈ 600/(4·1.33)). → The generator **must model both
  dark and bright polarity (and along-filament variation at crossings/bundles)**, not a single
  fixed polarity. First-order Fresnel; MTs are sub-diffraction (~25 nm) so **also PSF-convolve**
  and treat MT as a weak phase/scatter object.
- **Label-free signal isolation** (same source, §4.4): estimate background as the **pixelwise
  median over a frame stack** (preserves illumination + stationary dirt, removes moving MTs),
  then subtract. → You already hold **428 real empty-field backgrounds** (per microscope/iris/
  exposure): use them to (a) background-subtract the ~520 real images and measure real
  cross-sectional intensity profiles, contrast-sign distribution, and SNR; (b) composite
  synthetic MTs onto real backgrounds with a matched residual.
- **Residual gap closer — label-preserving unpaired translation (HIGH):**
  - **SimGAN** (Shrivastava et al., CVPR 2017 best paper, `arXiv:1612.07828`): adversarial
    refiner + **self-regularization** loss (minimal per-pixel change) so **annotations survive**;
    SOTA gaze with zero real labels.
  - **CUT** (Park et al.) for synth→real (Imbusch et al., IEEE CASE 2022, `arXiv:2203.09454`):
    **reuse the renderer's GT unchanged** on refined images; segmentation **mIoU 0.701→0.763
    (~99% of real-trained 0.770)**. ⚠️ **Caveat for thin filaments:** full-res CUT/CycleGAN
    **deforms/hallucinates thin structures** — they went **patch-based** and re-tuned PatchNCE
    per dataset. **Verify centerline GT validity after translation.**
  - **Ui2i** (bioRxiv `2025.05.26.656226`, MEDIUM — single preprint): content-preserving
    CycleGAN recipe — U-Net generators w/ skip connections, ~spectral normalization, channel+
    spatial attention; closest "translate appearance, keep labels" recipe (validated on nuclei,
    not filaments).

## 3. Morphology — estimate from unlabeled real images, then match (HIGH)
Unsupervised curvilinear detectors give length / width / curvature / orientation / spatial
layout / connectivity distributions to fit the generator against:
- **SOAX** (Xu & Vavylonis, Sci Rep 2015, `srep09081`): Stretching Open Active Contours seeded
  at intensity ridges → centerlines, junctions, lengths in 2D/3D; **ground-truth-free parameter
  selection** via its F-function sweep.
- **Basu/Rohde** (J. Microsc. 2014, `PMC5890959` — cite the **2014 jmi.12209**, *not* the
  retracted 2013 jmi.12018): filter detection + constrained reverse-diffusion centerlines, **no
  manual seeds**; outputs curvature/orientation/connectivity/length distributions.
- **CT-FIRE** (curvelet + FIRE) and **OrientationJ** (gradient structure tensor): per-fiber
  length/width/angle/curvature, and orientation/isotropy fields — both unsupervised.
- **Persistence length** from curvature variance (Biophys. J. 2022, `PMC9199094`): valid in
  principle but **noise-limited for very stiff MTs** (Lp ~300–3000 µm ≫ frame). → treat MTs as
  **near-rigid / high-Lp prior** and **match empirical curvature/orientation distributions
  directly** rather than fitting Lp.
- ⚠️ All are **SNR-dependent** and validated on confocal/AFM → low-contrast IRM (even after
  background subtraction) is harder; FiberApp is **semi-automated** (manual tracing = labeling).
  A claim that FiberApp auto-traces *any* modality unlabeled was **refuted (0-3)**.

## 4. Count / density — THE GAP, must engineer (LOW evidence, HIGH importance)
- **No verified method** surfaced for label-free per-frame filament count / foreground fraction
  / crossing-parallel statistics — yet **density mismatch is the documented driver of
  over-firing**. The literature is silent on the single factor most likely to break transfer.
- Supporting signal: matching **molecular crowding/diversity** is a *primary* transfer lever for
  cellular segmentation (bioRxiv `2025.02.07.637194`), not an afterthought; crowd-counting
  domain adaptation shows synth-trained density models get feature maps "**excessively
  activated**" on real → over-count (FADA/SDA, `arXiv:1912.03672`) — the same over-firing.
- **Engineer it:** estimate **foreground fraction** from background-subtracted real images;
  estimate **filament count + crossing/parallel rates** with the same unsupervised detectors
  (SOAX); make per-frame MT count & foreground fraction **explicit generator parameters** and
  **match their real distributions**; validate that synth-trained foreground does **not**
  over-fire on the 12 Alice frames (the existing in-house validation).

## 5. Label-free metric to drive the loop
- Prefer **MMD/CORAL on DINO/foundation-model features** (SynthMT uses DINOv2 cosine; Meta-Sim
  uses MMD on Inception). Naive image-space **FID/KID** is weaker.
- **SADGE** (`arXiv:2605.22467`, MEDIUM — recent preprint, n=15): fusing **appearance (DINOv3)
  + geometric structure (MASt3R)** predicts synth→real transfer better than appearance-only or
  geometry-only (authors report r≈0.88). Principle is sound (fused appearance+structure beats
  image-only); **the metric→real-F1 correlation is not MT-validated — measure it on our task.**

## 6. Precedent that actually transferred without real labels
- **SynthMT** (PLOS CB, MT/IRM) — §⭐ above. Direct.
- **sOCT vasculature** (`arXiv:2407.01419`): segmentation CNN trained **entirely on synthetic**
  (label synthesis + label-to-image), **no manual annotations**, reached human-level precision
  on 5 real acquisitions — a curvilinear synth→real success.
- **TARDIS FNet** (bioRxiv `2024.12.19.629196`): the foreground that transfers is trained on a
  large **real annotated** set (382 imgs, 71,747 objects) — **confirms the in-house finding**
  that real-trained foreground works and synth-only is the hard part; all the more reason to
  squeeze the synth→real appearance/density gap.

## Concrete prioritized plan
1. **Background-subtract** the 520 real images using the 428 empty-field backgrounds (per
   microscope/iris/exposure, median model) → isolate real MT signal.
2. **Measure real priors** from the residual: cross-sectional intensity profiles + **contrast-
   polarity distribution** (appearance); length/curvature/orientation via SOAX/CT-FIRE
   (morphology); **foreground fraction + filament count + crossing/parallel rates** (density).
3. **Build the generator**: parametric MTs with the **IRM two-reflection appearance** (both
   polarities + PSF blur), composited onto **real backgrounds**; expose appearance/morphology/
   **density** as tunable parameters seeded from step 2.
4. **Calibrate by analysis-by-synthesis**: Optuna-TPE / CMA-ES minimizing **MMD (or DINOv2
   cosine, SynthMT-style)** between synthetic and real feature distributions.
5. **Validate label-free**: feature-distribution gap + **the over-firing check on the 12 Alice
   frames** (does a foreground trained on this synth produce a clean, non-over-firing
   prediction?). Reserve any tiny labeled set only here.
6. **If a residual appearance gap remains**: add **patch-based CUT or SimGAN** refinement with
   self-regularization, then **re-verify centerline GT survives** the translation.

## Caveats / what's NOT established
- **Density** has zero verified method (biggest risk).
- SADGE & Ui2i are single recent **non-peer-reviewed preprints** (treat numbers as "authors
  report"). Meta-Sim, L2S, SimGAN, CUT, SOAX, Basu/Rohde, IRM model, curvature/Lp are
  peer-reviewed.
- Appearance-transfer precedents are RGB/gaze/objects/nuclei, **not thin IRM filaments** →
  relevance by analogy; thin-structure deformation/hallucination is real → verify GT.
- IRM model is first-order (ignores sub-diffraction PSF / weak-phase scattering — add PSF).
- Persistence-length-from-curvature is noise-limited for stiff MTs → use near-rigid prior.

### Sources (verified primary unless noted)
SynthMT PLOS CB pcbi.1013901 · Meta-Sim arXiv:1904.11621 · Learning to Simulate arXiv:1810.02513
· IRM model PMC6858481 · SimGAN arXiv:1612.07828 · CUT-synth2real arXiv:2203.09454 · Ui2i
bioRxiv 2025.05.26.656226 *(preprint)* · SOAX srep09081 · Basu/Rohde PMC5890959 (2014 jmi.12209)
· persistence/curvature PMC9199094 · SADGE arXiv:2605.22467 *(preprint)* · sOCT vasculature
arXiv:2407.01419 · TARDIS bioRxiv 2024.12.19.629196 · crowding bioRxiv 2025.02.07.637194 ·
crowd-counting FADA arXiv:1912.03672 · CT-FIRE/OrientationJ (tooling).
