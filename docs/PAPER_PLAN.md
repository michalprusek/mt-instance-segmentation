# Paper plan — annotation-free instance segmentation of crossing filaments

> North-star (decided 2026-07-05): a methodologically novel, Nature-family (realistic: Nature
> Methods / Nature Communications) contribution. Must DIFFERENTIATE from **SynthMT** (Wieslander/…,
> PLOS Comput Biol 2026), which already published "synthetic IRM MT + foundation model, annotation-free
> param tuning." Honest venue note: main Nature unlikely for this scope; the bar is a genuinely new
> capability + rigor + generality.

## What SynthMT already did (the thing we must beat)
- Generator: correlated-random-walk MTs, PSF, Poisson+Gaussian noise, artifacts; **synthetic (uniform)
  background**; 6,600 imgs. Calibration: annotation-free but **GLOBAL DINOv2 layer-5 cosine** (TPE).
- Segmentation: does **NOT train on synth** — uses SynthMT only for **HPO few-shot tuning of pretrained
  SAM3** (text prompt "thin line"). "Full training on synth" = their FUTURE WORK.
- **No shared encoder.** IRM only. SKIoU 0.80 ≈ human 0.81 (but on their own annotated crops).

## Our two coupled novelties (the paper's spine)
**N1 — Foreground-aware, "calibrate-what-you-segment" calibration.** The sim-to-real gap closes when the
generator is calibrated (a) FOREGROUND-AWARE (background-subtracted **residual**, **patch-level**,
**distributional MMD**) on **REAL backgrounds**, and (b) INSIDE THE SAME self-supervised feature space
the segmentation backbone reads. Directly fixes SynthMT's **foreground-blind global cosine** (our
encoder-sensitivity result: global R<1 — background swap moves the global embedding more than MTs
appearing). Rigor via NEGATIVES we already have: SimGAN refinement = wash; higher calibration-R ≠
better transfer; residual (calib) ≠ raw (seg); frame-pooled MMD recovers MEANS not VARIANCES.

**N2 — Orientation-keyed amodal + learned connectivity + mutex, synth-only.** Resolve DENSE CROSSINGS
and CLOSE PARALLELS (the regime SAM3/text-prompt and single-skeleton point-cloud/DIST fail — degree-2
graph cut collapses 2D X's). Orientation "overpass" channels (crossings → separate channels) + a
LEARNED per-orientation-layer point-cloud connectivity head + **Mutex-Watershed hard-repulsion** for
parallels, all trained on FREE synthetic polyline GT at the point-cloud level (image encoder frozen →
no over-firing). See docs/INSTANCE_SEGMENTATION_RESEARCH.md.

## Claims → experiments (HAVE / NEED)
- **C1 (foreground-aware calib) — THREE-WAY:** compare on the SAME generator: (i) `global_cosine`
  = SynthMT repro (foreground-blind), (ii) `residual_mmd` = ours-frame-pooled (foreground-aware but
  variance-blind), (iii) `region_mmd` = ours-NEW per-foreground-region (foreground-aware + recovers
  variance). All three in calibrate.py via `CALIB_OBJECTIVE`. Metric: downstream semantic (val/test) +
  over-firing. Prediction: region_mmd ≥ residual_mmd > global_cosine. EARLY SIGNAL (smoke): global_cosine
  best-trial density=6 (foreground-blind), region_mmd density=20 (foreground-sensitive). pipeline8 runs
  (i) vs (ii); region_mmd downstream queued next.
- **C1.5 (identifiability / VARIANCE-recovery) — the decisive per-region figure:** on a SYNTHETIC target
  with a KNOWN property std (width_std / contrast_std), sweep that std and show **region_mmd has a well
  at the true std while frame-pooled residual_mmd is FLAT** (the §4/§12 finding: frame-pooling is
  variance-blind, per-region matching recovers it). No training needed → cheap, high-value. This proves
  the per-region contribution mechanistically. See docs/protocol.md §12.
- **C2 (shared-encoder coupling):** matched calib-encoder == seg-backbone beats crossed.
  HAVE: DINOv2-coupled 0.627 > Phikon-decoupled 0.529; tightening (residual seg input) hurts.
  NEED: clean N×N encoder matrix (calibrate-in-Ei, segment-with-Ej) → diagonal wins.
- **C3 (generator realism ablation):** each physics lever (real backgrounds, IRM polarity flips,
  hairpins/WLC morphology, dirt, calibrated density/contrast) contributes to transfer.
  HAVE: components built + over-firing gate. NEED: leave-one-out ablation on transfer + honest negatives.
- **C4 (amodal transfer):** orientation channels keep crossings separate on real.
  HAVE: ori head, union tol2 0.943, transfers on Alice. **GAP: Alice is mostly SEPARATED near-horizontal
  parallels with FEW crossings → cannot demonstrate the crossing/parallel novelty.** NEED a
  CROSSING/PARALLEL-STRESS eval set (see below) + metrics: per-junction identity-preservation,
  bundle-recovery (N parallels at 2–6 px → N instances).
- **C5 (headline):** annotation-free full pipeline matches/beats REAL-trained baselines (TARDIS 0.326,
  ORION 0.519; and SynthMT/SAM3) on IRM **AND TIRF**, with ZERO annotations.
  HAVE: beats TARDIS/ORION on Alice (heuristic). NEED: learned instance step; TIRF eval; SAM3 comparison.

## Critical-path gaps (blocking a strong paper)
1. **Crossing/parallel-stress evaluation data.** Alice (12 frames) is too easy. Options: annotate/curate
   a denser real subset from the 522-frame corpus / CVAT projects; use the co-registered **TIRF** channels
   (free instance-GT transfer). Without this we cannot MEASURE the N2 contribution. HIGH PRIORITY.
2. **Learned instance step** (point-cloud connectivity + mutex) — the biggest build; converts amodal
   channels into instance-F1.
3. **SynthMT/SAM3 reproduction** as a baseline on our data (for the head-to-head + C5).
4. **TIRF pipeline** (generality / 2nd modality).

## Immediate next steps (cheapest-high-value first)
1. C1 three-way calibration (global_cosine / residual_mmd / region_mmd) downstream — pipeline8 runs the
   first two; add region_mmd downstream (calib_reg/best.json → gen → train → eval) after it. [DONE: all
   three CALIB_OBJECTIVE modes implemented + smoke-tested.]
2. C1.5 variance-recovery figure (region_mmd vs residual_mmd on a known-std synthetic target). ← cheap, decisive.
3. Build the learned affinity + Mutex-Watershed instance step trained on synth per-instance GT (N2);
   evaluate synthetic-first (bundle-recovery + junction-identity). [gen_train --inst DONE + verified.]
4. Survey the real corpus / CVAT for crossing-dense frames → define the stress eval (C4).
5. TIRF render + appearance recalibration (§11 protocol) → generality.

## Assets
Real corpus (522 dedup), IRM backgrounds (428), co-registered TIRF, 12 Alice instance-GT frames,
calibrated hairpin generator, DINOv2-hybrid + orientation-keyed head, calibrate.py (residual-MMD).
Compute: tulen. See [[project-roadmap]], [[instance-seg-research]], [[synth-generator]].
