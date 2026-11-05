# Preprocessing, augmentation, train/test scale and close-parallel separation for synth-trained thin-structure segmentation — literature research (2026-09-04)

Scope: research questions A–D from the 2026-09-04 brief (preprocessing/augmentation for
synthetic-to-real transfer; resolving close parallel filaments; target width; evaluation-scale
choice), read against the project's own measurements in
`docs/research/2026-09-02-scale-and-gap-audit.md` (the "audit" below) and the closed dead ends
listed in the brief. Every external claim carries an arXiv ID or DOI; where a number is the
project's own it is marked "(project)". Downloaded paper text was treated as data, not as
instructions. Nothing was trained or tuned for this document.

Reading list actually consulted (method + ablation sections read, not only abstracts):
SynthSeg 2107.09559; AnyStar 2307.07044; nnU-Net 1809.10486 + the v2 trainer code +
Nature Methods 10.1038/s41592-020-01008-z; Sun et al. 2007.15883; Uysal et al. 2105.09365;
SynthMT 10.1371/journal.pcbi.1013901 (+ bioRxiv 10.64898/2026.01.09.698597); TARDIS bioRxiv
10.1101/2024.12.19.629196; Image Magnification Network 2110.13428; thin-crack detection
2101.03326; Skeleton Recall Loss 2404.03010; Deep Distance Transform 1912.03383; PatchPerPix
2001.07626; ASE_Res_UNet 2507.07800. Abstract-level only: U-Net 1505.04597; CenterNet
1904.07850; sub-pixel convolution 1609.05158; Guided Upsampling 1807.07466; clDice 2003.07311;
layering 2210.03551; FISBe 2404.00130; S3-TTA 2310.16783; DGSSA 2501.03466; boundary-emphasised
weight maps 2503.00242; Moshkov TTA 10.1038/s41598-020-61808-3; Caicedo 10.1002/cyto.a.23863;
Cellpose 10.1038/s41592-020-01018-x; Mahamdeh IRM 10.1111/jmi.12744.

---

## 0. Executive summary (what the evidence supports, in five lines)

1. **The scale mismatch is the one input-side defect that is both measured (project) and has
   direct literature remedies with numbers.** Real filaments are 3.51 px FWHM native and 5.51 px
   in the 1.5x eval frame; synthetic training filaments are 4.0–4.8 px (project audit §A). Every
   synthetic-training paper that generalises across acquisitions randomises scale/resolution of
   the *rendered image* (nnU-Net scale 0.7–1.4 + low-resolution simulation 0.5–1.0; SynthSeg
   resolution randomisation, where restricting it costs 2.6 Dice; the thin-crack study shows a
   network is best exactly at its training resolution and degrades away from it). The project
   currently has **no** scale or resolution augmentation at all. This is the top recommendation.
2. **Photometric augmentation of the rendered frame is not the same thing as the generator
   domain-randomisation that hurt HTW.** The generator DR widened *physics* priors (contrast,
   PSF, dirt, morphology) and diluted a near-domain lab (HTW 0.620 → 0.422, project). Image-space
   photometric augmentation (gamma, brightness/contrast, illumination, quantisation) models the
   *export pipeline* of 8-bit contrast-stretched frames, which the generator does not model at all.
   The strongest cross-camera number in the thin-structure literature is from exactly this kind of
   augmentation: random gamma on retinal vessels lifted cross-camera F1 from 0.377 to 0.756
   (2007.15883). Recommended, but gated on HTW and roi303 VAL with a pre-declared no-regression
   criterion, because the DR episode shows this benchmark can only see dilution costs.
3. **Close parallels below ~3 px native gap are not geometrically resolvable in the image**
   (two Gaussian ridges of FWHM 3.5 px are unimodal below a 3.0 px separation; the audit's image
   dip in the 4.5–6 eval-px bin is only 0.26). Above that, the network already separates them
   at p ≥ 0.5 in 86 % of stretches but loses a ridge at the 0.98 threshold in 57 % (audit §B.1).
   So the parallel problem splits: **(i) 3–8 px gaps are a decoder + target-sharpness problem**,
   addressable with a sharper (Gaussian-heatmap) target, a 2x sub-pixel output head and a ridge
   non-maximum-suppression decoder instead of a global 0.98 threshold; **(ii) < 3 px gaps are a
   multiplicity problem** that only the double-contrast signal can solve, via a count channel
   supervised by the free synthetic GT.
4. **No quantitative "label width vs FWHM" ablation exists in the thin-structure literature that
   I could find**; the closest evidence is indirect (Skeleton Recall Loss dilates the skeleton by
   radius 2 for loss stability; the thin-crack study keeps 1 px labels and a 3 px evaluation
   dilation "the smallest that still separates two close cracks"; IMN found direct skeleton
   targets poor and recommends segment-then-skeletonise). The project's own 1 px null is
   consistent with that. The recommendation is not "thinner" but "softer and finer-sampled":
   a Gaussian ridge target whose σ is chosen from the separation you need.
5. **Upsampling the input before the CNN helps thin structures in the literature only when the
   training scale is matched to it** (IMN gains +0.6 Dice / +1.9 skeleton-recall from an
   up-sampling encoder trained end-to-end; SynthSeg always resamples inputs to the training
   resolution; the crack study's train/test resolution matrix is diagonal-dominant). Upsampling
   the *output* (sub-pixel head, labels rasterised finer than the input) is supported by SynthSeg
   and the crack study. For this project: keep the 1.5x eval frame for the instancer and metric,
   but stop feeding the network an image scale it never trained on — either train with scale
   augmentation that covers 1.5x, or infer at native and resample probabilities (already +0.02
   mean-F1 / +0.06 micro-F1, p = 0.23, project).

---

## 1. Ranked recommendations for THIS project

Columns: reported gain in the source (never assume it transfers); expected mechanism here;
cost; risk. Sorted by (evidence strength x expected effect) / cost.

| # | Change | Source(s) | Reported gain in source | Expected mechanism here | Cost | Risk |
|---|---|---|---|---|---|---|
| 1 | **Isotropic scale augmentation of the rendered synthetic frame**, s ~ LogU(0.75, 1.5), applied to image + polylines *before* rasterising the target (target width stays fixed in output pixels), interpolation order drawn from {1, 3}; plus nnU-Net's low-resolution simulation (downsample x0.6–1.0 nearest, upsample cubic, p = 0.2) | nnU-Net v2 defaults (scale 0.7–1.4, p = 0.2; SimulateLowResolution 0.5–1.0, p = 0.25) — trainer code + 10.1038/s41592-020-01008-z; SynthSeg 2107.09559 §3.1.4 (blur → downsample → upsample back to HR; ablation: restricting resolution/contrast costs 1.4–2.6 Dice); thin-crack 2101.03326 Fig. 4 (best F1 at matched train/test resolution) | SynthSeg: 1.4–2.6 Dice lost when the randomisation is narrowed; crack: a network trained at n× is best at test m = n | Synthetic FWHM 4.0–4.8 px x [0.75, 1.5] = 3.0–7.2 px, covering real native (3.5) and real eval (5.5); the bilinear zoom's noise smoothing is reproduced because the *noisy* rendered frame is zoomed. Should recover at least the +0.02/+0.06 measured for native inference (project) and make F1 flat in inference scale | 3 h code (scale polylines + image; rasterise after), 1 GPU-day retrain | Low. Possible slight loss of in-domain sharpness; gate: F1(1.0x) − F1(1.5x) on roi303 VAL must shrink to < 0.01 |
| 2 | **Photometric augmentation of the rendered frame** (gamma with retained stats, brightness, contrast, mild blur, mild noise, smooth illumination field, 8-bit quantisation, percentile-jitter stretch) with the parameter ranges in §3 | nnU-Net v2 defaults (gamma 0.7–1.5 p = 0.3 + inverted p = 0.1, brightness x0.75–1.25 p = 0.15, contrast 0.75–1.25 p = 0.15, blur σ 0.5–1.0 p = 0.2, noise var 0–0.1 p = 0.1); Sun 2007.15883 (gamma γ ∈ [0.33, 3]); SynthSeg 2107.09559 (bias field ablation −3.7 Dice); AnyStar 2307.07044 (bias fields, gamma, sharpening, Gaussian/Poisson/speckle noise) | Sun: cross-camera CHASE-DB1 F1 0.377 → 0.756, STARE 0.670 → 0.806, in-domain DRIVE 0.813 → 0.817; SynthSeg −3.7 Dice without bias field | Real frames are 8-bit contrast-stretched exports from 23 acquisitions with unknown gain/gamma; the generator renders physics but not the export pipeline. Expected: mostly cross-lab (HTW) and calibration robustness, +0.01–0.03; little in-domain gain | 4 h + 1 GPU-day (same retrain as #1) | Medium: the same dilution that hurt HTW under generator-DR is possible; Sun shows additive Gaussian noise can *hurt* cross-domain (STARE F1 0.670 → 0.627). Use mild ranges; gate on HTW |
| 3 | **Ridge non-maximum-suppression decoder with peak acceptance** instead of global threshold 0.98–0.99 (0.98 per the brief, 0.99 in the audit's shipped decode) → skeletonise: keep pixels that are local maxima of p along the normal (structure tensor / Hessian direction), link into curves, accept a curve if its median peak ≥ τ_peak (calibrated on synth) | Steger 1998 ridge detector 10.1109/34.659930 (sub-pixel, unbiased line extraction via NMS along the normal); project audit §B.1 | Audit: in the 4.5–6 eval-px gap bin two ridges exist at p ≥ 0.5 in 86 % of stretches vs 43 % at 0.99; ridge max of the pair 0.964 < 0.98 | Recovers pairs lost *by the threshold, not by fusion* (audit line 3). Not hysteresis (which was null): hysteresis grows connected components; NMS finds two maxima with a dip between them at any level | 1–2 days (decoder + regression tests on the (x, y)-vs-(row, col) convention) | Medium: the p > 0.5 band on isolated filaments is 4.1 px wide; NMS on a plateau can emit double ridges → require a dip ≥ δ between accepted twin ridges and tune δ on synth only |
| 4 | **2x sub-pixel heatmap head with a Gaussian centerline target and a multiplicity channel** (the parallel-separation proposal of §4) | Sub-pixel conv 1609.05158; SynthSeg 2107.09559 (LR input → crisp HR labels); thin-crack 2101.03326 (inputs downsampled, labels kept at full resolution: same accuracy at 1/3 input resolution); CenterNet 1904.07850 (Gaussian heatmap + penalty-reduced focal loss); DDT 1912.03383 (distance target: +13 % DSC on the thinnest structure); U-Net weight map 1505.04597 | Crack: F1 at test resolution m = 2, 3 equal to m = 1 with super-resolution labels; DDT +13 % DSC (pancreatic duct) | Target FWHM 2.1 px native keeps a ≥ 50 % dip between two bumps down to 2.1 px gap in the target (§4.1), below the ~3 px physical limit; the multiplicity channel reads the double contrast where geometry cannot. Aim: bundle recovery 0.232 → ≥ 0.45 on MT-34 VAL (oracle 0.717, project) | 3–5 days (head, rasteriser at 2x, loss, decoder, tests) + 1–2 GPU-days | Medium-high: new head + new decoder + new metric dependence; GT contains 26 coincident/duplicate polyline runs (audit §B.5) that will confound the multiplicity evaluation |
| 5 | **Per-frame pixel-size normalisation at inference**: measure the frame's isolated-filament FWHM with the audit's estimator and rescale so FWHM_in ≈ FWHM_train (4.5 px), then resample probabilities to the 1.5x frame | Cellpose canonical diameter (30 px cyto / 17 px nuclei; rescale by 30/diam) 10.1038/s41592-020-01018-x; TARDIS pixel-size normalisation (micrographs to 4 or 8 Å, tomograms 15 Å; works 0.55–56 Å only with normalisation) bioRxiv 10.1101/2024.12.19.629196 | TARDIS: cross-pixel-size use "requires appropriate pixel-size normalization"; Cellpose: whole design rests on it | The 23 acquisitions have varying pixel size; a single global 1.5x cannot be right for all. Either #1 (make the net scale-tolerant) or #5 (normalise the input) — #5 alone is the cheaper diagnostic | 1 day (estimator exists in `scripts/diag_scale_and_gap.py`) | Medium: estimator noise per frame (needs ≥ ~50 isolated samples); frames with only bundles bias the width up |
| 6 | **Percentile-jitter normalisation at train time** (lo ~ U(0, 2.5), hi ~ U(97.5, 100)) instead of a fixed [1, 99] stretch; keep the fixed [1, 99] at test | nnU-Net (per-image z-score + contrast/brightness aug); AnyStar 2307.07044 (histogram shifting in the baseline aug set) | No isolated number | Real exports were stretched by unknown software with unknown clipping; the network should not key on the exact tail placement. Cheap insurance | 1 h | Low |
| 7 | **Skeleton Recall auxiliary loss on the tubed skeleton** (skeleton dilated with a diamond of radius 2), weight 0.1–1.0 | Kirchhoff et al. 2404.03010 (Table 2) | DRIVE: β0 error 57.0 → 38.75, clDice 80.26 → 80.83, Dice 80.87 → 80.99; Roads Dice 78.99 → 79.25 | The 586 block is coverage-limited (memory): fewer breaks lifts coverage. Cheap CPU-side loss compatible with nnU-Net training | 2 h | Low; effect on Dice-type metrics is tiny in the source — the gain is topological |
| 8 | **8-fold flip/rot90 test-time ensembling** (diagnostic only) | Moshkov et al. 10.1038/s41598-020-61808-3; nnU-Net mirroring TTA 1809.10486 §2.4 | DSB 2018 score 0.633 → 0.644 | Training already uses rot90 + flips, so the ensemble mostly measures calibration noise at the 0.98 threshold. Memory says prefer retraining over inference tricks: use it to *diagnose*, not to ship | 0.5 h | Low; if TTA gains > 0.02 the threshold is over-tight and #3 is the real fix |
| 9 | **Multi-scale test ensembling** (1.0x and 1.5x probability maps averaged in the eval frame) | S3-TTA 2310.16783 (scale/style-selected TTA: +3.4 % on cell segmentation) | +3.4 % (cells), +1.3 % (lung) | Would paper over the scale mismatch that #1 fixes at the root; report only as a comparator | 2 h | Low |
| 10 | **Sharpening / cutout augmentations** | AnyStar 2307.07044 (sharpening, cutout in the augmentation sequence) | Not isolated | Cheap; sharpening mimics deconvolved exports | 1 h | Low, low value |

Items explicitly *not* on the list because the evidence is against them or the project has
closed them: CLAHE/local normalisation, additive Gaussian noise at nnU-Net's default variance,
elastic deformation, wider generator randomisation, a thinner binary target, pos_weight,
hysteresis, DINOv2-hybrid decoders, fine-tuning on real. See §5.

---

## 2. Evidence, by research question

### 2A. Preprocessing and augmentation for synthetic-to-real transfer of thin structures

#### 2A.1 nnU-Net's own recipe (the reference point for a ResEnc-M model)

nnU-Net's 2018 description lists "random rotations, random scaling, random elastic
deformations, gamma correction augmentation and mirroring", applied on the fly with the
`batchgenerators` framework, and adds test-time mirroring over all valid axes (1809.10486,
§2.3–2.4). The authors state there that the augmentation parameters "were not properly validated"
and defer ablations to future work; the Nature Methods version (10.1038/s41592-020-01008-z)
fixes the recipe as a dataset-independent "rule". The nnU-Net v2 trainer code
(`nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py`, `get_training_transforms`) gives the
current defaults, in order:

| Transform | Parameters (v2 code) | Probability |
|---|---|---|
| Spatial: rotation | 2D: ±180° for square-ish patches, ±15° only if `max(patch_size)/min(patch_size) > 1.5` (verified in the v2 code, `configure_rotation_dummyDA_mirroring_and_inital_patch_size`); 3D: ±30° isotropic, ±180° in-plane for anisotropic ("dummy 2D") patches | p_rot = 0.2 |
| Spatial: scaling | (0.7, 1.4), isotropic | p_scale = 0.2 |
| Elastic deformation | **not present in v2** (was in the 2018 list) | — |
| Gaussian noise | variance (0, 0.1) on z-scored input | 0.1 |
| Gaussian blur | σ (0.5, 1.0) px, independent per channel, p_per_channel 0.5 | 0.2 |
| Multiplicative brightness | x(0.75, 1.25) | 0.15 |
| Contrast | (0.75, 1.25) about the mean, stats preserved | 0.15 |
| Simulate low resolution | downsample scale (0.5, 1) nearest, upsample cubic, p_per_channel 0.5 | 0.25 |
| Gamma, inverted image | γ (0.7, 1.5), retain_stats | 0.1 |
| Gamma | γ (0.7, 1.5), retain_stats | 0.3 |
| Mirroring | all axes | 1 |

Normalisation: per-image z-score for non-CT modalities; CT uses dataset-level 0.5–99.5
percentile clipping then z-score (10.1038/s41592-020-01008-z). Two things matter for this project:
(i) elastic deformation was *dropped* between the 2018 list and the v2 code — there is no
published ablation, but the maintained default does without it; (ii) the recipe contains **three
different scale/resolution perturbations** (scaling, blur, low-resolution simulation), i.e. the
framework treats pixel-size and PSF mismatch as the first-order nuisance. The project's training
has none of the three.

Caveat on transfer of the noise setting: nnU-Net's noise variance (0, 0.1) is defined on
z-scored inputs (σ up to 0.32 in units of the image standard deviation). On a [0, 1]
percentile-stretched IRM frame whose filament contrast is a few percent of the range, that would
bury the filaments; §3 scales it down.

#### 2A.2 SynthSeg: what a resolution/contrast randomisation ablation looks like (2107.09559)

SynthSeg trains only on synthetic images generated on the fly from label maps with "fully
randomised" contrast and resolution, and its ablation (§5.2, Fig. 9) is the cleanest quantitative
statement available on *which* randomisations matter:

- Narrowing the generated distribution to a specific resolution (SynthSeg-R) or resolution +
  contrast (SynthSeg-RC), even when retrained per target domain, costs **1.4 and 2.6 Dice** on
  average relative to the fully randomised model.
- Slightly tighter intensity (GMM) priors: no change; further tightening "at the risk of
  excluding intensities encountered in real scans": **−2.1 Dice**.
- Removing the bias field (smooth multiplicative illumination): **−3.7 Dice**.
- Removing spatial deformation: **−4.3 Dice** (anatomy label maps; not transferable to a
  generator whose morphology is already randomised — see §5).
- Resolution mechanism (§3.1.4, eqs. 20–26): blur with a Gaussian whose σ is set from the
  target slice thickness (σ_thick = 2α·log(10)·(2π)^−1 · r_thick / r_HR, α ~ U(0.95, 1.05)),
  **downsample** to the random low resolution r_spac ~ U(1 mm, 9 mm), then **upsample back to
  1 mm "such that the CNN is trained to produce crisp HR segmentations, regardless of the
  simulated resolution"**. Hyperparameters (Table S1): scaling 0.8–1.2, rotation ±20°, shear
  ±0.015, bias-field amplitude 0.6, gamma σ_γ² = 0.4, b_res = 9 mm.

The transferable lesson is the *shape* of the resolution augmentation — blur → downsample →
upsample to the fixed working grid, labels untouched — not the ranges (MRI slice thickness is a
much larger effect than the 1.15–1.37x width mismatch here).

#### 2A.3 AnyStar: domain randomisation for microscopy instances (2307.07044)

AnyStar generates blob-like instances with GMM intensities modulated by Perlin noise and an
augmentation sequence A(·) of: affine (translation, rotation, scale, shear), random bias fields,
k-space spikes, Gibbs ringing, sharpening, gamma, cutout, axis-wise Gaussian blur (partial
voluming), spatial deformation, blank regions, and Gaussian / Poisson / speckle noise on non-zero
regions (§3, Fig. 3). Ablations vary the *intensity prior*, not individual augmentations:
the bright-foreground-only prior fails on a dark-foreground dataset (NucMM-M), the mixed prior
(AS-Mix) is consistently strong (§4.1). One number-free but relevant finding: AnyStar is "more
stable against blur as opposed to domain-specific fully-supervised networks trained with blur
augmentation" (Fig. 4C) — randomising the *generator's* PSF is more robust than adding blur
augmentation to real-trained networks. For this project the generator already randomises PSF
σ (0.49–0.92 px, project); AnyStar's list is the source for adding sharpening and illumination
fields on top. AnyStar itself notes that real-microscopy baselines drop the MRI-specific
transforms (bias, spikes, Gibbs, cutout) and keep "gamma adjustments, blurring, histogram
shifting, axis flips, 90-degree rotations" (App. B.2) — i.e. the photometric core is the same
across the field.

#### 2A.4 Retinal-vessel augmentation studies (thin structures, cross-camera numbers)

Sun et al. 2007.15883 is the only thin-structure paper found with a per-augmentation
cross-domain table. U-Net (ResNet-50 encoder), trained on DRIVE only, step-scaling 0.75–1.25:

| Train-time augmentation | DRIVE F1 (same camera) | STARE F1 (other camera + pathology) | CHASE-DB1 F1 (other camera, smaller FOV) |
|---|---|---|---|
| flips only (BS) | 0.8126 | 0.6699 | 0.3768 |
| + additive Gaussian noise σ = 20/255 (RGN) | 0.8131 | **0.6273** | 0.5279 |
| + HSV gamma [0.25, 4] (SVGC) | 0.8167 | 0.6769 | 0.4841 |
| + channel-wise random gamma γ ∈ [0.33, 3] (CWRGC) | 0.8174 | 0.8056 | 0.7563 |
| + vessel-region intensity augmentation (CWRVA) | 0.8205 | 0.7652 | 0.6385 |
| + both | 0.8209 | 0.8082 | 0.7565 |

Three points for this project: (a) the in-domain gain from photometric augmentation is ~0.005 F1
while the cross-camera gain is 0.14–0.38 — expect the effect on HTW, not on roi303/MT-34;
(b) additive Gaussian noise *hurt* on STARE, so noise augmentation is not free; (c) the authors
report that CLAHE-type preprocessing gives "limited improvement yet require additional inference
time" (§1) — an argument against local normalisation. Uysal et al. 2105.09365 reach AUC 0.9855 on
DRIVE with plain U-Net and a large augmentation set (rotation, flips, zoom-out, random cropping,
shifting, shearing, elastic, grid/optical distortion, white noise, gamma, histogram equalisation,
dropout, sharpening, blur, contrast), explicitly adding "shifted, zoomed and cropped images" to fix
thin-vessel misses; no per-augmentation ablation. DGSSA 2501.03466 (abstract) combines synthetic
vessel *structure* generation with PixMix-style random photometric augmentation for domain
generalisation across DRIVE/CHASE/HRF/STARE. Wang et al. 2019 ("Data augmentation is more
important than model architectures for retinal vessel segmentation", 10.1145/3348416.3348425,
cited via 2105.09365) make the same point at the title level.

#### 2A.5 SynthMT (PLOS Comput Biol 2026, 10.1371/journal.pcbi.1013901; bioRxiv 10.64898/2026.01.09.698597)

What it actually shows, read carefully because the project cites it often:

- Generator: polylines with stochastic curvature → binary mask convolved with a PSF,
  I(x) = A·(M_S ∗ PSF)(x) + B, then additive Gaussian noise, Poisson shot noise, multiplicative
  speckle, spatially correlated background, vignetting, blur, distractor spots. Parameters tuned
  by TPE (1000 iterations per target) maximising DINOv2 layer-5 cosine similarity to 10 real
  reference crops per target domain; 6600 images of 512 x 512.
- **No neural network is trained on the synthetic frames.** The transfer evidence is
  hyper-parameter optimisation of pre-trained foundation models (SAM3 with text prompt) on 10
  synthetic images. Real preprocessing for all methods: histogram normalisation + background
  subtraction, "no additional processing". There is **no augmentation ablation** and no
  component-wise realism ablation; the one realism statement is that tuning on the simplified
  binary DRIFT images "leads to a substantial performance degradation" relative to SynthMT.
- Numbers (bioRxiv tables): zero-shot SAM3 SKIoU 0.51 → after HPO on 10 SynthMT images 0.68 on
  synthetic; on 66 real IRM images SAM3Text mean SKIoU **0.74 ± 0.18 vs human 0.80 ± 0.14**,
  count deviation 1.17 instances, length deviation 0.3 µm. TARDIS 0.27 (both), StarDist 0.05–0.09.
  (The PLOS HTML summary quotes 0.91 vs 0.89 for a different aggregation; use the tables.)
- Not stated: pixel size, magnification, filament FWHM in pixels, or any train/test scale
  handling. SynthMT therefore provides a *calibration* precedent (already in the project's
  memory), not a preprocessing/augmentation recipe. Do not cite it for "what was necessary for
  transfer" of a trained network.

#### 2A.6 Other synthetic-trained filament segmenters

- **TARDIS** (bioRxiv 10.1101/2024.12.19.629196): per-image z-score then scaling to [−1, 1];
  **pixel-size normalisation** before the CNN (micrographs ≤ 10 Å → 4 Å, > 10 Å → 8 Å; tomograms
  15 Å); 256 x 256 patches; the semantic CNN is trained on real data (augmentations not itemised
  in the Methods), the DIST instance head on 50 % simulated point clouds with point jitter and
  10 % random noise points / point drops. Cross-pixel-size use from 0.55 to 56 Å is claimed
  *conditional on* the normalisation. The relevant precedent: a filament pipeline that works
  across microscopes normalises pixel size at inference rather than relying on scale augmentation.
- **Cellpose** (10.1038/s41592-020-01018-x): trains on images rescaled so objects have a canonical
  diameter (30 px cyto, 17 px nuclei) and rescales test images by 30/diameter using a size
  model; the same "normalise scale at inference" pattern.
- **ASE_Res_UNet** (2507.07800): U-Net variant trained on Cytosim + ConfocalGN synthetic
  fluorescence microtubules; augmentation is only "random rotations, horizontal and vertical
  flips, and slight colour jittering" (§2.5); transfer to real is evaluated by *retraining on
  real* (19 images) rather than zero-shot, so it carries no zero-shot augmentation evidence.
  MTFlow (2601.14841) and the Pix2Pix filament generator (2512.10334) likewise do not report
  augmentation ablations.

#### 2A.7 Test-time augmentation for thin structures

- Moshkov et al. (10.1038/s41598-020-61808-3): flip/rotation TTA with proper merging
  "outperformed single instance predictions at each test case"; DSB 2018 best score 0.633 → 0.644
  (nuclei, U-Net and Mask R-CNN). No thin-structure-specific number.
- nnU-Net uses mirroring TTA by default (1809.10486 §2.4); no ablation number given.
- S3-TTA (2310.16783): selecting test scale and style per image by a transformation-consistency
  metric improves cell segmentation by 3.4 % and lung by 1.3 % over prior TTA; the interesting
  part for this project is the premise: naive multi-scale TTA "produce[s] undesirable results" for
  many augmented versions, so scale TTA must be selective.
- No paper found that reports TTA gains for centerline-F1-type metrics on filaments. Expect
  ≤ 0.01 here (rot90/flips are already in training), and treat any larger gain as a symptom of
  threshold miscalibration (§1 item 8).

### 2B. Resolving close parallel thin structures

#### 2B.1 What is physically resolvable (derivation, project numbers)

Model two parallel filaments as two equal Gaussian ridges of standard deviation σ_img separated
by g; in the image their intensities *add*. The sum of two equal-weight, equal-variance Gaussians
is unimodal iff g ≤ 2σ_img and bimodal otherwise (classical result: Helguero 1904; Eisenberger
1964, DOI 10.1080/00401706.1964.10490199).
With the real native FWHM of 3.51 px (audit §A), σ_img = 3.51/2.355 = 1.49 px, so:

- g ≤ 3.0 px native (≤ 4.5 eval px): **unimodal — no dip exists in the image**; the pair is
  visible only as one wider, ~2x-contrast ridge.
- g = 4 px native (6 eval): midpoint/peak ratio 2·exp(−g²/(8σ²)) / (1 + exp(−g²/(2σ²))) ≈ 0.79,
  a 21 % dip.
- g = 5 px native (7.5 eval): ≈ 0.51, a 49 % dip.

This matches the audit's measured image dips (0.26 in the 4.5–6 eval-px bin, 0.43 at 6–7.5,
0.65 at 7.5–9, §B.1) and explains why the network's ridge confidence collapses exactly in the
4.5–6 px bin: the target asks for two 3 px lines where the input has one ridge. Consequence:
**separation by any foreground map is only possible above ~3 px native gap; below it only the
double contrast carries the information**, and the target must be designed for each regime
separately (§4).

#### 2B.2 Targets that keep two bumps separable

- **Gaussian centerline heatmaps** (CenterNet 1904.07850 for keypoints; used throughout
  keypoint/centerline detection) — a target H(x) = max_k exp(−d_k²/2σ_t²) with a
  penalty-reduced focal loss. Two bumps in the *target* stay separable iff g > 2σ_t; with
  σ_t = 0.9 px native, down to 1.8 px. This is strictly sharper than the current 3 px box target,
  which is unimodal for g < 3 px plus anti-aliasing.
- **Distance-transform regression** (DDT 1912.03383): predict the distance to the structure
  surface, quantised into K bins as a classification, in a multi-task net with the mask; "the ridge
  of the distance map is the skeleton"; +13 % DSC on the thinnest structure (pancreatic duct) and
  the explicit motivation that a skeleton/non-skeleton pixel classifier "is hard for the classifier
  to distinguish voxels on the skeleton from those immediately next to it, as they have similar
  features but different labels" (§1). That last sentence is the cleanest published argument
  against hard 1-px (or 3-px) line targets and for graded targets. Distance-transform regression
  for spatially aware segmentation more generally: 1909.01671.
- **U-Net separation weight map** (1505.04597): w(x) = w_c(x) + w_0·exp(−(d_1(x)+d_2(x))²/2σ²)
  with w_0 = 10 and σ ≈ 5 px (constants as stated in the paper's §3; abstract only was re-read
  this session), where d_1, d_2 are distances to the nearest and second-nearest instance
  borders; it up-weights the *gap* pixels between touching instances. The same map can
  be used as a loss weight for a heatmap target (§4.2).
- **Boundary/gap channels as explicit classes**: the 2018 Data Science Bowl analysis
  (10.1038/s41592-019-0612-7) and Caicedo et al. (10.1002/cyto.a.23863) formulate nuclei as
  three classes (interior / boundary / background) so that touching instances are split by the
  boundary class; boundary-emphasised weight maps for distal airways (2503.00242) report better
  topology metrics than centerline-based weighting on small branches. For filaments the analogue is
  a "between-parallels" class (§4.1, channel G).
- **Layering** (2210.03551): distribute crowded or overlapping objects into layers so that
  connected components within a layer are separate instances; competitive on C. elegans overlaps.
  For two parallels this means predicting each filament into a different output layer keyed by
  some ordering (e.g. left/right of the local normal); not needed if the heatmap + multiplicity
  route works, but it is the natural amodal representation if it does not.
- **PatchPerPix** (2001.07626, already in `docs/superpowers/specs/2026-08-27-patchperpix-
  instancer-design.md`): dense local shape-patch prediction handles "overlap (including
  crossovers)"; its stated limitation is that overlapping regions must be smaller than the patch
  size and only occlusions within the patch-graph neighbourhood can be bridged — fine for
  crossings, marginal for long parallel bundles (hundreds of px, audit §B.5). FISBe (2404.00130)
  is the public benchmark for long-range thin filament instances with overlaps.

#### 2B.3 Exploiting "two filaments are darker": multiplicity / count channels

No paper was found that regresses a per-pixel *filament count* from intensity multiplicity in
IRM or TIRF. The nearest precedents are density-map counting of overlapping cells (2203.15691;
He et al. 2011.03683; 1903.00388), where a Gaussian-sum density is regressed and integrated to
count. IRM contrast for a single microtubule is a few percent of background at a reported SNR
comparable to DIC/fluorescence (Mahamdeh et al. 2018, 10.1111/jmi.12744); the project's own
observation that overlapping parallels appear with roughly double contrast is what makes a count
channel learnable, and the generator can render it with free GT (it already models "irregular
bundles", memory). §4.1 defines the channel. The evaluation caveat is severe: 13 of the 26
near-coincident GT runs are exact duplicates with identical vertex counts (audit §B.5), so the
multiplicity channel must be scored only on the physically verified bundles, otherwise it will
"recover" annotation errors.

#### 2B.4 Super-resolution / upsampled prediction heads

- **Sub-pixel convolution / PixelShuffle** (1609.05158): features stay at input resolution, the
  last layer learns r² upscaling filters and rearranges them into an r x r larger output; it is
  the standard way to predict at 2x the input grid at almost no cost.
- **Guided upsampling** (GUM, 1807.07466): learnable, image-guided offsets for the decoder's
  upsampling so that semantic boundaries follow the high-resolution guidance; plug-in to any
  encoder–decoder.
- **Scale-aware super-resolution segmentation** (2305.19063): a dual-branch net that
  super-resolves the image and the mask simultaneously from low-resolution medical inputs, with
  feature-affinity coupling; consistent gains on small lesions (abstract-level).
- **Image Magnification Network** (2110.13428): U-Net with an *up-sampling encoder* and a
  down-sampling decoder for 1-px-wide OCTA capillaries; OCTA-SS Dice 0.8958 (U-Net) → 0.9019,
  skeleton recall 0.7550 → 0.7740, CAL 0.9038 → 0.9075 (Table 1). The authors attribute the gain
  to detail preservation before any pooling.
- **Labels finer than inputs** (2101.03326): training on images downsampled n× then upsampled
  back, with labels kept at full resolution ("super-human labels"), gives "about the same
  accuracy" at 1/3 of the annotation resolution; Fig. 4 shows F1 curves for every train/test
  resolution pair n ∈ [1, 8], m ∈ [1, 12].

Together: a 2x sub-pixel head trained on GT rasterised at 2x is supported by direct precedent
(SynthSeg's crisp-HR-from-LR, the crack study's fine labels on coarse inputs) and costs far less
than upsampling the encoder (IMN, 4x activations everywhere).

### 2C. Target width relative to the structure's FWHM

Direct evidence (a table of break rate or separation vs label width for one network) was **not
found** in the vessel/neuron/road/crack literature searched. What exists:

| Source | Target definition | Relation to structure width | Outcome reported |
|---|---|---|---|
| Thin-crack 2101.03326 §3.4 | 1 px human labels; evaluation dilates the *ground truth* by a disk of radius 3 px, "the smallest that achieves these effects, to make it possible to distinguish two isolated cracks that are close to each other" | cracks have sub-pixel widths; tolerance ≈ 3x the structure width | F1 unchanged down to 1/3 input resolution |
| Skeleton Recall 2404.03010 §3 | recall computed on the skeleton dilated with a diamond of radius 2 (5 px tube) "to make it tubular, thereby enlarging the effective area for loss computation ... enhances the stability of the loss" | vessels/roads/cracks of a few px | DRIVE β0 error 57.0 → 38.75; Dice +0.1 |
| IMN 2110.13428 §4.3 | direct training on 1-px skeleton labels (ROSE) vs segment-then-skeletonise | capillaries ~1 px | direct skeleton training "not suitable"; recommends pixel-level segmentation then skeletonise |
| DDT 1912.03383 §1 | distance-to-surface regression instead of skeleton label | tubular, 1–10 vx radius | +13 % DSC on the thinnest class; explicit argument that skeleton-vs-neighbour pixels have "similar features but different labels" |
| U-Net 1505.04597 | full masks + separation weights σ ≈ 5 px, w_0 = 10 | cells | ISBI 2012 best warping error at the time |
| Project (memory) | 1 px target: null; 3 px target: current | FWHM 4.0–4.8 px synth / 3.5 px real native / 5.5 px eval | predicted p > 0.99 band 2.08 px, p > 0.5 band 4.14 px in the eval frame |

Reading across: (i) hard binary line targets much thinner than the PSF create label noise at the
edge (DDT's argument; IMN's negative result; the project's 1 px null); (ii) targets much wider
than the structure merge close pairs (the 3 px box target is unimodal below 3 px gap, §2B.1);
(iii) graded targets (distance or Gaussian) avoid both, and their effective width is a free
parameter to set from the separation requirement, not from the FWHM.

**Does the target scale with the image under scale augmentation? Decision: no — fixed in output
pixels.** The literature offers both conventions: nnU-Net's `SpatialTransform` scales image and
mask *jointly* (a width mask is a property of the object, so it must scale), whereas SynthSeg
keeps HR labels while the image is blurred/downsampled/upsampled (§3.1.4) and the thin-crack
study keeps labels "as is" while inputs are downsampled (2101.03326 §4.1). This project's target
is a *centerline rasterisation*, not a width mask: its 3 px (or Gaussian σ_t) is a localisation
tolerance chosen relative to the fixed 5 px evaluation tolerance in a fixed frame, and the
polylines themselves are exact at any scale. So the SynthSeg/crack convention applies: scale the
image and the polylines, rasterise the target afterwards at fixed pixel width. The alternative
(nnU-Net-style joint scaling, target width 2.25–4.5 px across the [0.75, 1.5] range) would make the
network's ridge width track the image FWHM and reintroduce the box-merging of §2B.1 at the wide
end; it was considered and rejected. §3.3 follows the fixed-width decision.

### 2D. Evaluation-scale choice: upsample the input or the output?

Evidence that bears on the project's 1.5x convention:

- **Train/test scale must match** (2101.03326, Fig. 4): for every test resolution m the best
  model is the one trained at n = m (one exception, 4x best at m = 3); the envelope over training
  resolutions is flat for m = 1–3 but any single model degrades away from its training scale.
  The project runs a model trained at 1.0x on inputs at 1.5x — off-diagonal by construction.
- **Resample the input to the training grid, always** (SynthSeg §3.1.4; TARDIS; Cellpose): the
  three systems that work across resolutions all bring the input to the training scale, either by
  fixed resampling (SynthSeg to 1 mm, TARDIS to 4/8/15 Å) or by object-size normalisation
  (Cellpose 30 px).
- **Upsampling the input helps thin structures only as an *architecture trained that way***
  (IMN: +0.6 Dice, +1.9 skeleton recall with an up-sampling encoder). Nobody reports a gain from
  bilinear-upsampling the test input into a network trained at native scale — which is what the
  project does — and the project's own paired measurement is that undoing it (native inference,
  resample probabilities) gains +0.02 mean-F1 / +0.06 micro-F1 (p = 0.23, n = 141; underpowered,
  direction consistent with the literature).
- **Upsampling the output is supported** when the labels are finer than the input (SynthSeg
  crisp-HR targets; crack super-human labels; PixelShuffle heads). It also has a mechanistic
  advantage for the audit's finding that the p > 0.99 ridge is only 2.08 px wide: at 2x output the
  same ridge is 4 px wide and the skeletoniser's discretisation error halves.
- Interaction with training scale: with scale augmentation covering [0.75, 1.5], the network is
  on-diagonal at both 1.0x and 1.5x, so the choice becomes an *instancer* convention rather than a
  network one. The audit's noise-scale caveat (bilinear zoom smooths the pixel noise the network
  trained on) is resolved by the same augmentation because the noisy rendered frame is zoomed.

Bottom line for D: keep the 1.5x frame as the instancer/metric frame (nothing in the literature
argues against a fixed working grid; SynthSeg does the same), but make the network's input scale
either (a) covered by training (recommendation 1) or (b) normalised per frame (recommendation 5),
and, if the 2x head is adopted, produce the eval frame by resampling the 2x output by 0.75 rather
than by zooming the input.

---

## 3. Proposed normalisation + augmentation recipe and scale strategy

All augmentations act on the **rendered synthetic frame** (image already composited on a real
background with Poisson + read noise), never on the generator's physics priors. Probabilities and
ranges are per sample; ranges are given at the *native* training scale. Order matters: geometric
first (so the target is rasterised after scaling), then intensity, then noise, then quantisation.

### 3.1 Geometric

| Step | Definition | Range / p | Source | Note |
|---|---|---|---|---|
| G1 rot90 + flips | as now | p = 1 | nnU-Net mirroring; 2101.03326 uses 360° rotation + flips | keep |
| G2 isotropic scale | s ~ LogUniform(0.75, 1.5); image resampled with order ∈ {1, 3} drawn at random; polylines multiplied by s; crop 512 after scaling | p = 0.6 | nnU-Net (0.7, 1.4) p = 0.2; SynthSeg (0.8, 1.2) + resolution DR; audit: real/synth width ratio 0.73–1.37 | Synthetic FWHM 4.0–4.8 → 3.0–7.2 px, brackets real native 3.5 and real eval 5.5 (p95 6.1 → 9.2 covers the real-eval p95 9.64) |
| G3 low-resolution simulation | downsample by f ~ U(0.6, 1.0) with nearest, upsample cubic to the original grid | p = 0.2 | nnU-Net SimulateLowResolution (0.5, 1) p = 0.25; SynthSeg eqs. 25–26 | models coarser pixel sizes *and* the 1.5x bilinear zoom's smoothing without changing the target |
| G4 small anisotropic scale | per-axis factors ~ U(0.92, 1.08) | p = 0.15 | SynthSeg shear ±0.015; AnyStar affine incl. shear | cheap; covers non-square pixels / residual astigmatism; keep κ_max derivation valid (affine with ≤ 8 % anisotropy changes curvature by ≤ 8 %) |
| G5 arbitrary rotation | θ ~ U(−180°, 180°), bicubic | p = 0.2 | nnU-Net 2D ±180° p = 0.2 | optional; rot90 already covers isotropy, the extra value is the interpolation blur |
| — elastic deformation | **none** | — | nnU-Net v2 has none; SynthSeg's −4.3 Dice is for *label-map* anatomy, not rendered frames | would alter local curvature and break the physics-derived κ_max = 0.25 rad/px |

Target rasterisation happens **after** G1–G5, at fixed pixel width (3 px now; §4 for the
heatmap), so the target never scales with the image.

### 3.2 Intensity (after geometric, on the float frame)

| Step | Definition | Range / p | Source | Note |
|---|---|---|---|---|
| I1 percentile-jitter stretch | lo ~ U(0, 2.5), hi ~ U(97.5, 100); I ← clip((I − P_lo)/(P_hi − P_lo), 0, 1) | p = 1 (jitter p = 0.5, else exactly [1, 99]) | nnU-Net z-score + contrast aug; AnyStar histogram shifting | test-time stays exactly [1, 99]; keep the project's "stretch before the bright surround" ordering |
| I2 polarity inversion | I ← 1 − I | p = 0.5 | project (already) | keep |
| I3 gamma | I ← I^γ, γ ~ LogU(0.7, 1.5), **retain mean/std** (nnU-Net `retain_stats`) | p = 0.3 | nnU-Net; Sun 2007.15883 (γ ∈ [0.33, 3] gave the cross-camera gain; use nnU-Net's narrower band first) | with I2, "inverted gamma" is already covered |
| I4 brightness / contrast | I ← a·I + b with a ~ U(0.75, 1.25), b ~ U(−0.1, 0.1); contrast about the mean c ~ U(0.75, 1.25) | p = 0.15 each | nnU-Net | clip to [0, 1] after |
| I5 smooth illumination field | I ← I·(1 + A·B(x)), B = zero-mean Gaussian random field with σ ~ U(100, 300) px normalised to max|B| = 1, A ~ U(0, 0.15) | p = 0.3 | SynthSeg bias field (ablation −3.7 Dice); AnyStar bias fields | real empty-field backgrounds already carry vignetting; this adds the export-time flat-field mismatch |
| I6 Gaussian blur | σ ~ U(0.3, 0.8) px | p = 0.15 | nnU-Net σ (0.5, 1.0) p = 0.2, reduced | σ = 1 px alone widens a 3.5 px FWHM to 4.2 px; with G2/G3 present keep blur mild |
| I7 unsharp mask | I ← I + α·(I − G_1px ∗ I), α ~ U(0.2, 0.6) | p = 0.1 | AnyStar sharpening | some exports are deconvolved/sharpened |
| I8 additive Gaussian noise | σ_n ~ U(0, 0.05) on the [0, 1] frame | p = 0.15 | nnU-Net var (0, 0.1) on z-scored input, **scaled down**; Sun: σ = 20/255 hurt cross-domain | generator already models Poisson + read noise; this is for re-quantised/re-scaled exports only |
| I9 quantisation | round to 8 bit always; posterise to 6–7 bit | p = 1 / p = 0.1 | none found for thin structures (RandAugment-style posterise is generic) | real inputs are 8-bit exports; matches train to test exactly |
| I10 cutout | 1–3 squares of 16–48 px replaced by local mean + noise | p = 0.05 | AnyStar cutout; Uysal dropout | mimics blank/bleached regions; low value |

Not included: CLAHE / local contrast normalisation (would flatten the double-contrast cue and
the interference polarity flips; Sun reports "limited improvement" for such preprocessing);
JPEG artefacts (the project's exports are 8-bit PNG/TIFF; add only if a lab delivers JPEG);
style transfer / SimGAN (project: wash).

### 3.3 Train/test scale strategy (decision)

Three options were weighed:

| Option | What changes | Evidence | Verdict |
|---|---|---|---|
| S1 **train with G2 + G3, keep 1.5x inference and the 1.5x instancer/metric frame** | training only | nnU-Net/SynthSeg/crack: on-diagonal at any scale in the covered range | **Adopt first.** Cheapest, no eval-pipeline change, directly testable: the F1 gap between 1.0x and 1.5x inference on roi303 VAL should collapse from +0.02/+0.06 to < 0.01 |
| S2 native inference + probability resampling to 1.5x | inference only | project: +0.02 mean-F1 / +0.06 micro-F1, p = 0.23 | Keep as the paired comparator after S1; if S1 flattens the curve S2 becomes irrelevant, if not S2 wins by default |
| S3 train at 1.5x (train at test scale) | training + 2.25x compute per unit area, 512 crops see 44 % less context | crack study: matched scale is best | Not recommended: bakes the historical convention into the network, halves the context per crop, and G2 already puts 1.5x in-distribution |
| S4 per-frame pixel-size normalisation (Cellpose/TARDIS) | inference | TARDIS, Cellpose | Diagnostic and fallback for acquisitions outside [0.75, 1.5]x (frame sizes 730–2048 px suggest pixel sizes vary more than that) — implement as a *measured* rescale factor, logged per frame |

Success criterion for S1 (pre-declared): on roi303 VAL (141 frames) and HTW (66), paired
per-frame, (a) mean-F1 ≥ baseline + 0.02 in the 1.5x frame, (b) |F1(1.0x) − F1(1.5x)| < 0.01,
(c) HTW not lower than baseline by more than 0.01. Failure of (c) with success of (a) means the
photometric set (§3.2) is diluting like generator-DR did; re-run with G-only augmentation to
separate the two.

---

## 4. Proposal for close-parallel separation: target, head, decoder

Notation: polylines {P_k}, distance from pixel x to polyline k: d_k(x) (px, native). Output grid
at r = 2x native (so 1 output px = 0.5 native px). Eval frame produced by resampling the 2x output
by 0.75 (bilinear on probabilities), not by zooming the input.

### 4.1 Targets (rasterised directly from polylines at 2x, all free from the generator)

**Channel H — Gaussian centerline heatmap**

$$
H(\mathbf{x}) = \max_k \exp\!\left(-\frac{d_k(\mathbf{x})^2}{2\sigma_t^2}\right),
\qquad \sigma_t = 0.9\ \text{px native} = 1.8\ \text{output px}.
$$

Separability in a *max*-combined target is not the classical 2σ rule (that rule is for a *sum*
of Gaussians): the max of two equal bumps at gap g has a dip at the midpoint of depth
exp(−g²/8σ_t²) relative to the peaks for every g > 0, so the criterion is a minimum dip depth
matched to the decoder's twin-ridge rule (§4.3, δ):

$$
\frac{H_{\text{mid}}}{H_{\text{peak}}} = \exp\!\left(-\frac{g^2}{8\sigma_t^2}\right) \le 1-\delta
\;\Longleftrightarrow\;
g \ge \sigma_t\sqrt{-8\ln(1-\delta)} ,
$$

i.e. with σ_t = 0.9 px native a 20 % dip (δ = 0.2) needs g ≥ 1.34σ_t = 1.2 px native, a 50 % dip
g ≥ 2.35σ_t = 2.1 px native (3.2 eval px). Either is well below the ~3 px native physical limit
of §2B.1, so the target is not the bottleneck; the image is. (The alternative, a *sum* clipped at
1, H = min(1, Σ_k exp(−d_k²/2σ_t²)), obeys the classical rule — bimodal after clipping iff
g > 2.355σ_t = 2.1 px — and encodes multiplicity in the pre-clip amplitude; it was not chosen
because the clipped plateau at g ≤ 2.1 px hides exactly the pairs channel N is meant to carry.)
Target FWHM = 2.355σ_t = 2.1 px native, i.e. ~0.6x the real filament FWHM — sharper than the
image, which is allowed because the network localises the ridge, not the intensity profile (crack
study; SynthSeg). Using max keeps H ≤ 1 without a plateau and leaves the count to channel N.

**Channel N — multiplicity (how many filaments here)**

$$
N(\mathbf{x}) = \sum_k \mathbb{1}\!\left[d_k(\mathbf{x}) < r_N\right],\qquad r_N = 1.5\ \text{px native},
$$

predicted as a 3-class softmax over {0, 1, ≥ 2} with cross-entropy, computed only where
H > 0.05 (elsewhere the class is trivially 0). This is the channel that reads the double contrast
in the g < 3 px regime where H is unimodal. The generator must render bundles with additive
contrast for this to be learnable — check the generator's bundle rendering against a real bundle
profile before training (the audit measured no bundle contrast ratio; add that measurement).

**Channel G — between-parallels gap (optional, U-Net weight map turned into a target)**

$$
G(\mathbf{x}) = \mathbb{1}\!\left[2 < d_1(\mathbf{x}) + d_2(\mathbf{x}) < 8\ \wedge\ |d_1 - d_2| < 1\ \wedge\ \angle(\mathbf{t}_1,\mathbf{t}_2) < 15^\circ\right],
$$

with d_1 ≤ d_2 the two smallest polyline distances and t_i the local tangents; G marks the
midline between two near-parallel filaments 2–8 px apart. It gives the decoder an explicit
"there are two" cue in the regime where the H dip is shallow. Equivalent in spirit to the
boundary class of the DSB/Caicedo three-class formulation, and to Ronneberger's weight map
placed on the *target* side.

**Loss**

$$
\mathcal{L} = \frac{1}{|\Omega|}\sum_{\mathbf{x}} w(\mathbf{x})\,\ell_{\text{focal}}\!\big(\hat H(\mathbf{x}), H(\mathbf{x})\big)
\;+\; \lambda_N \,\mathrm{CE}\big(\hat N, N\big)\big|_{H>0.05}
\;+\; \lambda_G\,\mathrm{BCE}(\hat G, G)
\;+\; \lambda_S\,\mathcal{L}_{\text{skel-recall}},
$$

with CenterNet's penalty-reduced focal term (1904.07850):

$$
\ell_{\text{focal}} =
\begin{cases}
(1-\hat H)^{\alpha}\log \hat H, & H = 1 \\
(1-H)^{\beta}\,\hat H^{\alpha}\log(1-\hat H), & \text{otherwise}
\end{cases},
\quad \alpha = 2,\ \beta = 4,
$$

where the H = 1 set is the polyline rasterised 1 output-px wide (0.5 px native); the U-Net
separation weight (1505.04597)

$$
w(\mathbf{x}) = 1 + w_0 \exp\!\left(-\frac{(d_1(\mathbf{x}) + d_2(\mathbf{x}))^2}{2\sigma_w^2}\right),
\qquad w_0 = 10,\ \sigma_w = 3\ \text{px native},
$$

up-weights everything within ~two filament widths of a pair; and the skeleton-recall term is
Kirchhoff et al.'s soft recall on the tubed skeleton (radius 2 at native, 2404.03010) to keep
coverage. Suggested λ_N = 0.5, λ_G = 0.25, λ_S = 0.1 (their Table 2 uses w ∈ {0.1, 1.0}). Keep the
current BCE + Dice on a 3 px mask as a fourth auxiliary channel during the transition so the old
decoder still runs (A/B on the same weights).

### 4.2 Head

ResEnc-M decoder output F ∈ R^{C×h×w} at native resolution →
3x3 conv (C → 4·C') → **PixelShuffle(2)** (1609.05158) → C' × 2h × 2w → 3x3 conv → 3x3 conv →
outputs {Ĥ (sigmoid), N̂ (3 logits), Ĝ (sigmoid), mask (sigmoid)}. C' = 32. Memory cost is confined
to the head (4x the output maps), training time ~1.2–1.3x; no change to the encoder or to the
512 px native crops. Alternative with more capacity but 4x activation memory everywhere: an
up-sampling encoder (IMN 2110.13428) — not recommended first. Guided upsampling (1807.07466)
is a middle option if the PixelShuffle ridges look blocky.

### 4.3 Decoder (replaces threshold 0.98 → skeletonise)

1. Orientation: θ(x) from the structure tensor of Ĥ (σ = 1 output px), normal n̂ = (−sin θ, cos θ).
2. Ridge NMS (Steger 1998, 10.1109/34.659930): keep x if Ĥ(x) ≥ τ_lo and Ĥ(x) ≥ Ĥ(x ± n̂)
   (bilinear samples); sub-pixel position from the parabola through the three samples.
3. Link ridge pixels into polylines (8-neighbourhood along t̂, gap bridging ≤ 2 output px).
4. Accept a polyline if median(Ĥ along it) ≥ τ_peak and length ≥ L_min; twin ridges closer than
   3 output px are accepted only if the minimum of Ĥ between them is ≤ (1 − δ)·min(peak_1, peak_2).
5. Multiplicity: along an accepted polyline, where the majority vote of N̂ is "≥ 2" for a run ≥
   L_bundle, emit a second polyline offset by ±δ_b along n̂ with δ_b = max(0.5, (w_½ − w_iso)/2)
   px, where w_½ is the local width of {Ĥ > 0.5} and w_iso the same width on isolated filaments
   (audit: 4.14 eval px), and flag it `amodal_parallel` for the instancer.
6. Resample polylines to the 1.5x eval frame (multiply by 0.75); hand to the existing junction
   matcher.

Calibration of (τ_lo, τ_peak, δ, L_bundle) **on synthetic frames with exact GT only**
(memory: synthetic tuning beats real-VAL tuning). Starting values τ_lo = 0.3, τ_peak = 0.6,
δ = 0.15, L_bundle = 20 native px.

### 4.4 What this changes in the audit's numbers (expected, to be measured)

| Audit quantity | Now | Expected after §4 | Why |
|---|---|---|---|
| gap 4.5–6 eval px: two ridges at the shipped threshold | 43 % | ≥ 80 % | NMS at 0.3–0.6 instead of 0.98–0.99; audit already sees 86 % at 0.5 |
| gap 3–4.5 eval px (2–3 native): recoverable | n = 2, no conclusion | via N̂ only | image unimodal (§2B.1) |
| p > 0.99 ridge width on isolated filaments | 2.08 eval px | ≥ 4 output px at 2x | output grid halved |
| bundle recovery (MT-34 VAL) | 0.232 (oracle 0.717) | ≥ 0.45 | pre-declared target; measured on physically verified bundles only |
| crossings identity | 0.752 | unchanged | not addressed here |

---

## 5. What NOT to do (with the evidence)

1. **CLAHE / local histogram normalisation / per-tile z-score.** Sun et al. 2007.15883 §1: such
   preprocessing yields "limited improvement yet require[s] additional inference time"; here it
   would additionally destroy the two signals the parallel proposal relies on — the ~2x contrast
   of bundles and the sign of the interference contrast (polarity flips are physics, memory
   "generator-photometry-defect"). Global percentile stretch is what all synthetic frames were
   trained with; keep it and randomise its parameters instead (I1).
2. **nnU-Net's additive-noise variance verbatim.** var (0, 0.1) is on z-scored inputs; on a
   [0, 1] IRM frame with a few-percent filament contrast this is orders of magnitude too strong.
   Sun's RGN (σ = 20/255) reduced STARE F1 0.670 → 0.627. The generator already models
   Poisson + read noise; add at most σ ≤ 0.05 (I8).
3. **Elastic deformation.** Dropped from nnU-Net v2's defaults; the −4.3 Dice ablation in
   SynthSeg concerns anatomy label maps, not rendered thin structures; here it would change
   local curvature and invalidate the derived κ_max = 0.25 rad/px, which the paper claims is
   physics, not a tuned parameter.
4. **Widening the generator's physics priors again** (memory: HTW 0.620 → 0.504 → 0.422 with
   v4b → v6 → v7-DR). The literature's "randomise beyond realism" (SynthSeg §6; AnyStar) is
   correct for a broad unknown target, but the project's benchmark contains no far-domain lab,
   so it can only measure the cost. Photometric augmentation of the *rendered frame* (§3.2) is a
   different lever (export pipeline, not physics) — but it is subject to the same dilution risk,
   hence the HTW no-regression gate.
5. **Blur augmentation at σ up to 1 px on native frames without scale augmentation.** It is a
   one-sided width perturbation (only wider) and would push synthetic FWHM further from the real
   native 3.5 px; use G2/G3 for scale and keep I6 mild.
6. **Fixing the scale mismatch with test-time multi-scale ensembling.** S3-TTA (2310.16783)
   shows naive scale TTA produces "undesirable results" for many inputs unless selected; the
   root fix (S1) is cheaper than a robust selection scheme, and the memory rule is retrain over
   inference tricks.
7. **A thinner binary target (1 px) or a wider box target.** 1 px: project null; DDT's argument
   (skeleton pixel vs neighbour "similar features but different labels", 1912.03383) and IMN's
   negative skeleton-target result (2110.13428 §4.3). Wider: unimodal below the box width (§2B.1).
   Use a graded target with a σ chosen from the separation requirement (§4.1).
8. **Hysteresis or closing as the parallel fix.** Project null; both operate on connected
   components and cannot create two ridges where the threshold removed one. Ridge NMS (§4.3) is
   the operation that does.
9. **Fine-tuning on real annotations / real-annotation training.** Project: 0.357 vs 0.433
   cross-lab, and fine-tuning worse; the human GT is incomplete (memory "tune on synth") and
   69 % model-drawn on the 585 block (memory "agreement bias").
10. **Citing SynthMT for "what realism components were necessary for a trained network".** It
    trains no network and ablates no component (§2A.5); it supports the DINOv2-calibration
    precedent only.
11. **Evaluating the multiplicity channel on the raw roi303 GT.** 13 exact-duplicate polyline
    runs and 26 runs within 2 px (audit §B.5) will be "recovered" by any count channel; restrict
    the bundle metric to bundles verified against the image (double contrast present).

---

## 6. Cheap screens to run before committing (pre-declared criteria)

| # | Screen | Cost | Decision rule |
|---|---|---|---|
| E1 | Retrain SPARSE35 with §3.1 G1–G3 only (no photometrics); evaluate at 1.0x and 1.5x on roi303 VAL + HTW | 1 GPU-day | adopt if mean-F1 ≥ +0.02 and |F1(1.0x) − F1(1.5x)| < 0.01 and HTW ≥ −0.01 |
| E2 | E1 + §3.2 photometrics | 1 GPU-day | adopt if HTW ≥ E1 − 0.01 and pooled F1 ≥ E1; otherwise bisect (gamma/illumination vs noise/blur) |
| E3 | Ridge-NMS decoder on the *existing* SPARSE35 maps (no retraining) at τ_peak ∈ {0.5, 0.7, 0.9} calibrated on 20 synthetic frames | 1 day | ship if two-ridge rate in the 4.5–6 px bin ≥ 75 % **and** precision on isolated filaments unchanged (false twin-ridge rate < 2 %) |
| E4 | Generator bundle-contrast check: render 50 synthetic bundles (gap 0–3 px), measure ridge contrast ratio vs single; compare with ≥ 10 real verified bundles | 0.5 day | if the ratio does not match within 20 %, fix the generator before training channel N |
| E5 | 2x heatmap head (§4) on E1/E2's best recipe | 3–5 days + 2 GPU-days | adopt if bundle recovery ≥ 0.45 on verified bundles and MT-34 VAL pooled F1 ≥ E2 |
| E6 | Per-frame FWHM estimator as inference rescale (S4) on the 23 acquisitions; log the factor per frame | 1 day | if factors span > [0.75, 1.5] for any acquisition, extend G2's range or keep S4 for those frames |

Statistics for every screen: paired per-frame differences, report mean-F1 and micro-F1 pooled
and per source (memory "report combined benchmark"), Wilcoxon + bootstrap CI, Holm correction
over the family of screens; the native-inference +0.02/+0.06 at p = 0.23 shows 141 frames is
marginal power for effects of 0.02 — decide on effect size + CI, not p alone.

---

## 7. References

arXiv IDs are given where the version read is on arXiv; DOIs otherwise. The non-arXiv DOIs for
Cellpose, Steger 1998 and Eisenberger 1964 were resolved through Crossref during this session;
venue tags (CVPR/ECCV/WACV/PRCV/BMVC) come from the arXiv listings or memory and were not
separately verified.

- Billot B. et al. *SynthSeg: Segmentation of brain MRI scans of any contrast and resolution
  without retraining.* Medical Image Analysis 2023. arXiv:2107.09559.
- Billot B. et al. *A Learning Strategy for Contrast-agnostic MRI Segmentation.* MIDL 2020.
  arXiv:2003.01995.
- Dey N. et al. *AnyStar: Domain randomized universal star-convex 3D instance segmentation.*
  WACV 2024. arXiv:2307.07044.
- Isensee F. et al. *nnU-Net: Self-adapting Framework for U-Net-Based Medical Image
  Segmentation.* 2018. arXiv:1809.10486.
- Isensee F. et al. *nnU-Net: a self-configuring method for deep learning-based biomedical image
  segmentation.* Nature Methods 18, 203–211 (2021). DOI 10.1038/s41592-020-01008-z. Default
  augmentation parameters read from `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py`
  (github.com/MIC-DKFZ/nnUNet, `get_training_transforms`).
- Isensee F. et al. *nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image
  Segmentation.* MICCAI 2024. arXiv:2404.09556 (ResEnc presets; not read for augmentation).
- Sun X. et al. *Robust Retinal Vessel Segmentation from a Data Augmentation Perspective.*
  OMIA/MICCAI 2020. arXiv:2007.15883.
- Uysal E. S. et al. *Exploring the Limits of Data Augmentation for Retinal Vessel
  Segmentation.* 2021. arXiv:2105.09365.
- Wang Z. et al. *Data Augmentation is More Important Than Model Architectures for Retinal
  Vessel Segmentation.* ICIMH 2019. DOI 10.1145/3348416.3348425 (cited via 2105.09365).
- Liu B. et al. *DGSSA: Domain generalization with structural and stylistic augmentation for
  retinal vessel segmentation.* 2025. arXiv:2501.03466.
- Koddenbrock M. et al. *Synthetic data enables human-grade microtubule analysis with foundation
  models for segmentation (SynthMT).* PLOS Computational Biology 22(5), 2026.
  DOI 10.1371/journal.pcbi.1013901; bioRxiv DOI 10.64898/2026.01.09.698597;
  github.com/ml-lab-htw/SynthMT.
- Kiewisz R. et al. *Accurate and fast segmentation of filaments and membranes in micrographs
  and tomograms with TARDIS.* bioRxiv 2024. DOI 10.1101/2024.12.19.629196.
- Stringer C. et al. *Cellpose: a generalist algorithm for cellular segmentation.* Nature
  Methods 18, 100–106 (2021). DOI 10.1038/s41592-020-01018-x (canonical diameter 30/17 px; docs
  cellpose.readthedocs.io).
- Ait Laydi A. et al. *A novel attention mechanism for noise-adaptive and robust segmentation of
  microtubules in microscopy images (ASE_Res_UNet).* 2025. arXiv:2507.07800.
- Sid El Moctar S. M. et al. *MTFlow: Time-Conditioned Flow Matching for Microtubule
  Segmentation in Noisy Microscopy Images.* 2026. arXiv:2601.14841.
- Liu Y., Zhang Y. *A Conditional Generative Framework for Synthetic Data Augmentation in
  Segmenting Thin and Elongated Structures in Biological Images.* 2025. arXiv:2512.10334.
- Moshkov N. et al. *Test-time augmentation for deep learning-based cell segmentation on
  microscopy images.* Scientific Reports 10, 5068 (2020). DOI 10.1038/s41598-020-61808-3.
- Xie K. et al. *S³-TTA: Scale-Style Selection for Test-Time Augmentation in Biomedical Image
  Segmentation.* 2023. arXiv:2310.16783.
- Li M. et al. *Image Magnification Network for Vessel Segmentation in OCTA Images.* PRCV 2022.
  arXiv:2110.13428.
- Xu L. et al. *Pushing the Envelope of Thin Crack Detection.* 2021. arXiv:2101.03326.
- Kirchhoff Y. et al. *Skeleton Recall Loss for Connectivity Conserving and Resource Efficient
  Segmentation of Thin Tubular Structures.* ECCV 2024. arXiv:2404.03010.
- Shit S. et al. *clDice — A Novel Topology-Preserving Loss Function for Tubular Structure
  Segmentation.* CVPR 2021. arXiv:2003.07311.
- Wang Y. et al. *Deep Distance Transform for Tubular Structure Segmentation in CT Scans.* CVPR
  2020. arXiv:1912.03383.
- Audebert N. et al. *Distance transform regression for spatially-aware deep semantic
  segmentation.* 2019. arXiv:1909.01671.
- Ronneberger O., Fischer P., Brox T. *U-Net: Convolutional Networks for Biomedical Image
  Segmentation.* MICCAI 2015. arXiv:1505.04597.
- Zhou X., Wang D., Krähenbühl P. *Objects as Points (CenterNet).* 2019. arXiv:1904.07850.
- Shi W. et al. *Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel
  Convolutional Neural Network.* CVPR 2016. arXiv:1609.05158.
- Mazzini D. *Guided Upsampling Network for Real-Time Semantic Segmentation.* BMVC 2018.
  arXiv:1807.07466.
- Li Y. et al. *Scale-aware Super-resolution Network with Dual Affinity Learning for Lesion
  Segmentation from Medical Images.* 2023. arXiv:2305.19063.
- Hirsch P., Mais L., Kainmueller D. *PatchPerPix for Instance Segmentation.* ECCV 2020.
  arXiv:2001.07626.
- Mais L. et al. *FISBe: A real-world benchmark dataset for instance segmentation of long-range
  thin filamentous structures.* CVPR 2024. arXiv:2404.00130.
- Chen L., Wu Y., Merhof D. *Instance Segmentation of Dense and Overlapping Objects via
  Layering.* 2022. arXiv:2210.03551.
- Keshavarzi A., Angelini E. *Boundary-Emphasized Weight Maps for Distal Airway Segmentation.*
  2025. arXiv:2503.00242.
- Caicedo J. C. et al. *Evaluation of Deep Learning Strategies for Nucleus Segmentation in
  Fluorescence Images.* Cytometry A 95(9), 2019. DOI 10.1002/cyto.a.23863.
- Caicedo J. C. et al. *Nucleus segmentation across imaging experiments: the 2018 Data Science
  Bowl.* Nature Methods 16, 1247–1253 (2019). DOI 10.1038/s41592-019-0612-7.
- Li S., Ach T., Gerig G. *Improved Counting and Localization from Density Maps for Object
  Detection in 2D and 3D Microscopy Imaging.* 2022. arXiv:2203.15691.
- He S. et al. *Deeply-Supervised Density Regression for Automatic Cell Counting in Microscopy
  Images.* 2020. arXiv:2011.03683.
- Steger C. *An Unbiased Detector of Curvilinear Structures.* IEEE TPAMI 20(2):113–125, 1998.
  DOI 10.1109/34.659930.
- Eisenberger I. *Genesis of Bimodal Distributions.* Technometrics 6(4):357–363, 1964.
  DOI 10.1080/00401706.1964.10490199 (two-Gaussian bimodality condition g > 2σ for a *sum*;
  classical, also Helguero 1904).
- Mahamdeh M., Simmert S., Luchniak A., Schäffer E., Howard J. *Label-free high-speed wide-field
  imaging of single microtubules using interference reflection microscopy.* Journal of
  Microscopy 272(1):60–66, 2018. DOI 10.1111/jmi.12744.
- Tuna Y., Al-Hiyasat A., Howard J. *Imaging Dynamic Microtubules and Associated Proteins by
  Simultaneous Interference-Reflection and Total-Internal-Reflection-Fluorescence Microscopy.*
  2022. arXiv:2201.07911.
- Kuo Y.-W., Howard J. *In vitro reconstitution of microtubule dynamics and severing imaged by
  label-free interference reflection microscopy.* 2021. arXiv:2107.01501.

Project sources (not external): `docs/research/2026-09-02-scale-and-gap-audit.md`
(widths, ridge profiles, gap-bin statistics, coincident GT); memory notes
`biased-corpus-domain-randomization` (DR verdict), `segmenter-state-and-dead-ends`,
`instance-seg-research` (bundle recovery 0.232 / crossings 0.752), `tune-on-synth-not-real-val`,
`prefer-retrain-over-inference-tricks`, `report-combined-benchmark`; `scripts/train_v5.py`
(current preprocessing: percentile [1, 99] stretch, polarity inversion, rot90 + flips, 3 px
target, 1.5x zoom at eval).
