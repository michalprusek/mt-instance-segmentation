# Scale and gap audit — SPARSE35 on roi303 VAL (2026-09-02)

Measurement only; nothing was trained or tuned. Script: `scripts/diag_scale_and_gap.py` (untracked).
Outputs: `data/enc_sensitivity_testset/scale_gap/` (`synth_width.json`, `real_width.json` [tulen],
`real_width_local.json`, `pred_ridge.json`, `sanity.json`, `gap.json` with every stretch record,
`gt_coincident_polylines.json`, `width_samples_local.npz`, both run logs). Model: SPARSE35 cached maps
`/disk2/prusek/mt_work/preds/SPARSE35_roi` (binary head, 1.5x eval frame, FOV-masked). Data: roi303 VAL
(141 frames, 7 445 human-drawn curves); 30 evenly spaced frames for the width parts, all 141 for the gap part.
Synthetic: 20 IRM frames, 640x640, `calib_V6_c070_narrow.json` built exactly as `train_v5.py` builds it
(`gen_train.build_cfg(best_params, mask_hw=1.0)` + `gen_overrides`; SPARSE35's launcher `scripts/run_sparse_sweep.sh`),
composited on `irm_backgrounds_v2` at NATIVE resolution as `OnlineDS` does. SPARSE35 also saw 35 % TIRF
frames; those were not measured.

## Interpretation (the five lines)

1. **Train/test scale mismatch: yes, and it is symmetric around ~1.3x, not fixed by either 1.0x or 1.5x.**
   Synthetic training filaments at native resolution: per-sample FWHM p50 **4.03 px** (n=2460 samples,
   20 frames; noise-biased low, see §A.3 — SNR>=10 subset 4.43, n=906),
   unaligned mean profile **4.81 px**, config expectation 2.355·sqrt(ws²+psf²) = **4.53 px** (p25–p75 4.01–5.05).
   Real roi303 filaments at native: **3.51 px** (n=16715; per-frame medians p25/50/75
   3.17/3.48/3.81 over 30 frames) = 0.73–0.87x synth.
   After the eval pipeline's 1.5x zoom (paired on the same points, ratio 1.53): **5.51 px** = **1.15–1.37x synth**
   (1.22x the config expectation); 37 % of real-1.5x samples are wider than the synth p95 (6.11 px),
   65 % wider than the synth p75. The network trained to put a 3 px target on a ~4.5 px filament and is
   asked to put it on a 5.5 px one. The scale at which real width would match synth is 1.15–1.37x
   (1.29x vs the config expectation): width alone does not single out native inference — at 1.0x real is
   23 % narrower than trained, at 1.5x 22 % wider (vs the config expectation). The native-scale
   F1 gain reported by the training-free screen (0.4999 → 0.5202) is therefore at most partly a width effect;
   the noise/texture scale (bilinear zoom smooths pixel noise the network trained on) was not measured here.
2. **The predicted ridge is narrower than the filament.** On isolated real filaments in the eval
   frame the p>0.99 band is **2.08 px** wide (p25–p75 1.82–2.50, n=15504; 7.2 % of samples never reach 0.99),
   p>0.9: 3.06 px, p>0.5: **4.14 px** (n=16427), while the image FWHM there is 5.51 px and the training target 3 px.
   Probability falls to 0.51 at 2 px and 0.18 at 3 px from the centerline (medians over n=5667 isolated runs, 190208 samples).
3. **The probability dips between close parallels: at the shipped 0.99 threshold two ridges survive from ~6 eval px
   (4 native) upward; in the 4.5–6 px bin the pair's own ridges drop below 0.99 and separation exists only at ~0.5–0.9.** Gap 4.5–6 eval px (3–4 native; n=7 stretches,
   148 samples, 6 frames): median min-between **0.35**, but the median ridge maximum of the pair is only
   0.964 (exact centerline 0.88; isolated filaments 0.998). Two ridges both >= thr with a minimum < thr between them:
   at 0.99 in 43 % of stretches (ridge lost in 57 %), at 0.9 in 71 %, at 0.5 in **86 %** (along 60 % of stretch length).
   Gap 6–7.5 px (n=10): min-between 0.06, ridge max 0.999; two ridges at 0.99 / 0.9 / 0.5 in 90 / 90 / 70 %.
   Gap 7.5–9 px (n=35): 0.067; 74 / 89 / 89 %. Gap 9–13.5 px (n=29): 0.029; 72 / 93 / 97 %.
   So the map is not flat between close parallels: from 6 px up, 72–90 % of stretches show two ridges at 0.99 (the rest
   lose a ridge, against 7 % of isolated samples that never reach 0.99); in the 4.5–6 px bin only 43 % do, because the
   ridge confidence itself is depressed (57 % lose a ridge at 0.99, 0 % at 0.5) — a training-side finding, not a decoder one. The raw IMAGE resolves the two closest bins worse than the
   network does (median intensity dip 0.26 and 0.43 of ridge contrast; 0 % and 30 % of stretches with a >50 % dip).
   Whether the mid-gap probability at 4.5–6 px (0.62) exceeds two superposed isolated ridges is not settled: the isolated
   reference at gap/2 is 0.18 by the per-side, per-run aggregation (B.2) and 0.34 by the side-averaged per-sample curve (B.3),
   and two ridges at 0.34 already superpose to ~0.6. The 3–4.5 px bin has n=2: no conclusion.
4. **Close parallels a probability map could separate are rare in this GT; coincident ones are not.** Only
   19 stretches with gap < 7.5 eval px (5 native) exist in 141 frames (410 samples, 16 frames), against
   **26 near-parallel runs >= 40 px lying within 2 px of each other** (6792 px native, 15 frames; 17 at
   median offset < 0.5 px, 13 of them exact duplicates with identical vertex count — e.g. roi_img_207 curves 6/7, 977 vertices each).
   No foreground map can separate those. Which population the bundle-recovery 0.23 was measured on must be
   checked before that number is attributed to the network.
5. **Transpose gate passed**: median prob at 561953 exact GT points = 0.999 (mean 0.907), max within 2 px = 1.000,
   transposed convention = 3.9e-05, 15 px off the filament = 1.4e-04 median / 0.007 mean (n=796964).

## A. Apparent filament width (FWHM of the perpendicular profile)

| set | scale | accepted / sampled | p5 | p25 | p50 | p75 | p95 | mean-profile FWHM (peak-aligned) | (unaligned) | per-frame median p25/50/75 | SNR p50 | GT–peak offset p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| synthetic (as trained) | native 1x | 2460 / 3082 (80 %) | 1.59 | 3.04 | **4.03** | 4.95 | 6.11 | 3.71 | 4.81 | 2.97 / 3.53 / 4.42 | 7.6 | 0.50 |
| real roi303 VAL | native 1x | 16715 / 20653 (81 %) | 1.83 | 2.89 | **3.51** | 4.34 | 6.13 | 3.29 | 4.37 | 3.17 / 3.48 / 3.81 | 6.4 | 0.50 |
| real roi303 VAL | eval 1.5x | 16715 / 16715 (100 %) | 3.17 | 4.58 | **5.51** | 6.80 | 9.64 | 5.15 | 6.65 | 5.09 / 5.40 / 5.98 | 7.1 | 0.75 |

Training target: **3 px** line (`ori_channels_soft`, half_width 1, 3x3 dilation) at the training (native) scale.
Synthetic config expectation from `width_mean=1.785`, `width_std=0.35`, clip [0.7, 2.4], PSF sigma [0.49, 0.92]:
FWHM p25/50/75 = 4.01/4.53/5.05 px (200 000 draws).
Paired 1.5x/native ratio on identical sample points: p25/50/75 1.486/1.532/1.611 (n=16715); bilinear zoom adds ~2 % on top of 1.5.
Tulen and local runs of the real part agree (n 16715 vs 16674; native p50 3.505 vs 3.509; 1.5x p50 5.506 vs 5.515); the tail fractions below use the local per-sample arrays.

### A.1 Ratios and tails

| quantity | value |
|---|---|
| real-1.5x / synth, per-sample p50 | 1.367 |
| real-1.5x / synth, unaligned mean profile | 1.145 |
| real-1.5x / synth config expectation | 1.216 |
| real-native / synth (per-sample, unaligned, expectation) | 0.870, 0.729, 0.774 |
| real-1.5x samples wider than synth p50 / p75 / p90 / p95 | 87 % / 65 % / 47 % / 37 % |
| real-native samples wider than synth p50 / p95 | 34 % / 5 % |
| inference scale at which real width = synth (per-sample / expectation / unaligned) | 1.15x / 1.29x / 1.37x |

### A.2 Width by SNR band (per-sample p50)

| set | SNR>=3 | SNR>=6 | SNR>=10 | SNR>=20 |
|---|---|---|---|---|
| synthetic native | 4.03 (n=2460) | 4.33 (n=1521) | 4.43 (n=906) | 4.58 (n=282) |
| real native | 3.51 (n=16715) | 3.45 (n=9179) | 3.32 (n=4082) | 3.16 (n=1136) |
| real 1.5x | 5.51 (n=16077) | 5.41 (n=10141) | 5.17 (n=4915) | 4.89 (n=1383) |

Synthetic widths rise towards the config expectation with SNR (noise bias); real widths are flat or fall with SNR
(high-contrast real filaments are narrower — an acquisition/physics effect, not an estimator one).

### A.3 Estimator validation (controlled Gaussian ridges, straight + sinusoidal, multiplicative on a flat field)

| true FWHM | measured SNR ~6 (per-sample / aligned / unaligned) | SNR ~10 | SNR >= 20 |
|---|---|---|---|
| 2.83 | 2.86 / 2.56 / 3.03 | 3.01 / 2.98 / 3.09 | 3.04 / 3.07 / 3.06 |
| 4.47 | 4.11 / 3.98 / 4.84 | 4.49 / 4.46 / 4.63 | 4.57 / 4.62 / 4.61 |
| 5.89 | 4.85 / 5.04 / 5.79 | 5.64 / 5.64 / 5.96 | 5.77 / 5.81 / 5.82 |

At the synth SNR (7.6 median) the per-sample and peak-aligned estimates run 8–18 % low for 4.5–6 px ridges and the
unaligned mean profile is nearest the truth; at the real native width (3.5 px) all three are within ~5 %. Hence the
synth width is bracketed 4.03 (per-sample, low) – 4.53 (expectation) – 4.81 (unaligned mean, high: width mixture over frames), and every ratio above is given for all three.

### A.4 Predicted probability ridge across isolated real filaments (eval frame, SPARSE35)

| level | band width p25 / p50 / p75 (px, eval) | n measured | samples whose peak never reaches the level |
|---|---|---|---|
| p > 0.5 | 3.73 / **4.14** / 4.82 | 16427 | 1.7 % |
| p > 0.9 | 2.42 / **3.06** / 3.44 | 16173 | 3.2 % |
| p > 0.99 | 1.82 / **2.08** / 2.50 | 15504 | 7.2 % |

Peak probability at the ridge: p25/50/75 0.9992/1.0000/1.0000; >= 0.99 in 92.8 %, >= 0.5 in 98.3 % of n=16715 samples.
Ridge-vs-GT offset p50 0.50 px (eval). Median probability at distance d from the predicted ridge peak (both sides):
0 px: 1.00, 1 px: 0.98, 2 px: 0.53, 3 px: 0.07, 4 px: 0.00, 5 px: 0.00, 6 px: 0.00 (means 0.98, 0.91, 0.56, 0.17, 0.07, 0.05, 0.04).

## B. Parallel-gap probability profile (all 141 roi303 VAL frames, eval frame)

Pairs of GT polylines with nearest-point distance 2–9 px native, tangent difference < 15°, >= 40 px of consecutive
qualifying samples (4 px step), nearest polyline to the segment midpoint one of the pair (211 samples dropped
by that guard; 7338 polyline pairs screened). Per stretch: medians over its samples; per bin: medians over stretches.

### B.1 Pairs, by gap in the eval frame

| gap (eval px) | n stretches | n samples | n frames | p at centerlines (ridge max) | p at midpoint | min between (p25–p75) | stretches min<0.99 | min<0.9 | min<0.5 | length with min<0.99 | length with min<0.5 | two ridges >= thr & min < thr at 0.99 / 0.9 / 0.5 | ridge max < thr at 0.99 / 0.9 / 0.5 | image dip (min) | image dip>0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3.0-4.5 | 2 | 33 | 2 | 0.928 (0.952) | 0.834 | **0.703** (0.56–0.85) | 50 % | 50 % | 50 % | 70 % | 33 % | **0 / 50 / 50 %** | 50 / 0 / 0 % | -0.03 | 0 % |
| 4.5-6.0 | 7 | 148 | 6 | 0.880 (0.964) | 0.619 | **0.353** (0.14–0.38) | 100 % | 100 % | 86 % | 100 % | 60 % | **43 / 71 / 86 %** | 57 / 29 / 0 % | 0.26 | 0 % |
| 6.0-7.5 | 10 | 229 | 10 | 0.996 (0.999) | 0.166 | **0.060** (0.04–0.41) | 100 % | 100 % | 70 % | 100 % | 70 % | **90 / 90 / 70 %** | 10 / 10 / 0 % | 0.43 | 30 % |
| 7.5-9.0 | 35 | 543 | 22 | 0.990 (0.998) | 0.118 | **0.067** (0.01–0.16) | 100 % | 97 % | 91 % | 99 % | 76 % | **74 / 89 / 89 %** | 26 / 9 / 3 % | 0.65 | 66 % |
| 9.0-13.5 | 29 | 595 | 21 | 0.997 (0.998) | 0.046 | **0.029** (0.01–0.08) | 100 % | 100 % | 100 % | 100 % | 93 % | **72 / 93 / 97 %** | 28 / 7 / 3 % | 0.93 | 93 % |

"Ridge max" = max probability within ±2 px of each GT centerline along the joining line (isolated filaments: 0.998). "Two ridges" = both ridge maxima >= thr and the minimum between them < thr, i.e. what a threshold decode would see as two objects; "ridge max < thr" = at least the median sample of the stretch loses a ridge at that threshold. Two-ridge fractions with the exact centerline value instead of the ridge max are in `gap.json` (`pairs_two_ridge_by_thr`).
Image dip = 1 − min|I−bg| between the ridges / min ridge contrast (1 = drops to background, 0 = no dip). All stretches
in every bin are same-polarity pairs. In every bin with n>=7, no stretch has an image dip >0.5 with a flat (>=0.99) probability.

### B.2 Isolated filaments, virtual gap of the same width (control; n=5667 runs of >= 40 px, 190208 samples)

| virtual gap (eval px) | p at centerline | p at gap/2 | p at gap (= min between) | runs min<0.5 | image dip (min) |
|---|---|---|---|---|---|
| 3.0-4.5 | 0.998 | 0.680 | 0.009 | 94 % | 0.66 |
| 4.5-6.0 | 0.998 | 0.182 | 0.002 | 99 % | 0.89 |
| 6.0-7.5 | 0.998 | 0.020 | 0.001 | 100 % | 0.96 |
| 7.5-9.0 | 0.998 | 0.006 | 0.001 | 100 % | 0.98 |
| 9.0-13.5 | 0.998 | 0.002 | 0.000 | 100 % | 0.99 |

### B.3 Probability vs distance from an isolated GT centerline (eval px; median [p25, p75], mean; image |contrast| normalised)

| d | 0 | 1 | 1.5 | 2 | 2.5 | 3 | 3.5 | 4 | 5 | 6 | 8 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prob median | 0.998 | 0.913 | 0.716 | 0.512 | 0.389 | 0.185 | 0.046 | 0.011 | 0.003 | 0.002 | 0.001 | 0.000 |
| prob p25–p75 | 0.97–1.00 | 0.73–0.98 | 0.54–0.89 | 0.46–0.66 | 0.19–0.51 | 0.03–0.47 | 0.00–0.32 | 0.00–0.18 | 0.00–0.05 | 0.00–0.02 | 0.00–0.01 | 0.00–0.00 |
| prob mean | 0.910 | 0.823 | 0.704 | 0.550 | 0.393 | 0.267 | 0.183 | 0.131 | 0.074 | 0.046 | 0.021 | 0.010 |
| image | 0.91 | 0.83 | 0.76 | 0.68 | 0.59 | 0.50 | 0.43 | 0.36 | 0.27 | 0.22 | 0.19 | 0.17 |

### B.4 The 19 stretches with gap < 7.5 eval px

| frame | curves | n | len (native px) | gap eval (min–max) | p center | p mid | min between | length min<0.5 | image dip |
|---|---|---|---|---|---|---|---|---|---|
| roi_img_20 | 21/78 | 18 | 68 | 4.15 (3.1–4.9) | 0.859 | 0.668 | 0.413 | 67 % | -0.04 |
| roi_img_75 | 24/25 | 15 | 56 | 4.19 (3.4–10.3) | 0.997 | 1.000 | 0.994 | 0 % | -0.02 |
| roi_img_20 | 93/98 | 18 | 68 | 4.76 (3.5–12.7) | 0.408 | 0.734 | 0.103 | 94 % | 0.11 |
| roi_img_55 | 152/153 | 33 | 128 | 5.00 (3.1–13.3) | 0.880 | 0.822 | 0.353 | 61 % | -0.02 |
| roi_img_199 | 23/24 | 30 | 116 | 5.21 (4.7–10.4) | 0.993 | 0.638 | 0.401 | 53 % | 0.47 |
| roi_img_42 | 136/137 | 12 | 44 | 5.36 (3.1–7.7) | 0.781 | 0.492 | 0.187 | 67 % | 0.42 |
| roi_img_184 | 8/9 | 11 | 40 | 5.49 (5.2–8.9) | 0.985 | 0.078 | 0.065 | 73 % | 0.12 |
| roi_img_26 | 64/65 | 15 | 56 | 5.82 (4.2–8.2) | 0.785 | 0.619 | 0.588 | 20 % | 0.36 |
| roi_img_26 | 20/21 | 29 | 112 | 5.89 (3.3–12.9) | 0.993 | 0.412 | 0.354 | 52 % | 0.26 |
| roi_img_19 | 59/60 | 13 | 48 | 6.28 (3.5–9.6) | 0.913 | 0.883 | 0.578 | 46 % | -0.00 |
| roi_img_96 | 48/49 | 11 | 40 | 6.62 (3.9–11.1) | 0.994 | 0.392 | 0.115 | 91 % | 0.28 |
| roi_img_192 | 31/32 | 97 | 384 | 6.78 (3.5–8.8) | 0.998 | 0.150 | 0.074 | 76 % | 0.40 |
| roi_img_278 | 31/32 | 14 | 52 | 6.94 (4.2–13.1) | 1.000 | 0.034 | 0.005 | 100 % | 0.72 |
| roi_img_58 | 119/120 | 13 | 48 | 6.96 (3.2–12.2) | 0.990 | 0.702 | 0.503 | 46 % | 0.90 |
| roi_img_166 | 107/108 | 24 | 92 | 7.00 (3.0–12.7) | 0.976 | 0.847 | 0.585 | 50 % | 0.71 |
| roi_img_98 | 50/51 | 13 | 48 | 7.08 (3.4–11.7) | 0.999 | 0.045 | 0.044 | 62 % | 0.42 |
| roi_img_26 | 45/51 | 19 | 72 | 7.33 (4.6–7.7) | 0.273 | 0.183 | 0.046 | 100 % | 0.36 |
| roi_img_167 | 10/11 | 11 | 40 | 7.42 (3.1–13.0) | 1.000 | 0.039 | 0.035 | 55 % | 0.47 |
| roi_img_80 | 6/7 | 14 | 52 | 7.43 (3.1–13.1) | 1.000 | 0.009 | 0.008 | 79 % | 0.44 |

### B.5 GT caveat: coincident and duplicate polylines

Nearest-other-polyline distance of near-parallel (<15°) GT samples, native px: 0-1: 1994, 1-2: 337, 2-3: 391, 3-4: 423, 4-5: 421, 5-6: 549, 6-7: 635, 7-8: 719, 8-9: 686, 9-10: 806, 10-11: 795, 11-12: 720.
The 0–1 px bin is 5.1x the 2–3 px bin. Runs >= 40 px within 2 px of another polyline: **26** (15 frames,
6792 px native, median run length 146 px, longest 960 px); median offset < 0.5 px in 17, 0.5–1 px in 7, 1–2 px in 2;
13 runs are exact duplicates (distance 0.0 throughout, identical vertex counts). Frames: roi_img_18, roi_img_19, roi_img_199, roi_img_20, roi_img_207, roi_img_240, roi_img_26, roi_img_35, roi_img_39, roi_img_41, roi_img_42, roi_img_46, roi_img_53, roi_img_80, roi_img_9.
These were excluded from B.1 by the 2 px lower bound and are listed in `gt_coincident_polylines.json`. Not fixed here.

## C. Sanity gate (transpose trap)

| check | value | n |
|---|---|---|
| median prob at exact GT points, [row=y·1.5, col=x·1.5] | 0.999 (mean 0.907) | 561953 |
| median of max prob within 2 px perpendicular | 1.000 | 561953 |
| median prob under the TRANSPOSED convention | 3.9e-05 | — |
| median / mean prob 15 px (eval) off isolated filaments | 1.4e-04 / 0.007 | 796964 |

Gate: median at GT > 0.8, displaced < 0.2, transposed < half of exact — **passed**; the script exits without writing prediction-derived output otherwise.

## Method

- Profiles: bilinear (`map_coordinates`, order 1) along the normal of a chord tangent (±4 px of arc after 1 px resampling),
  every 8 px of arc (Part A), ±8 px window at native / ±12 px at 1.5x (same physical window), local background = median of the outer
  3 px (4.5 px) per side, polarity normalised by the sign of the strongest excursion within ±2.5 px (±3.75 px) of the GT point
  (IRM flips dark/bright), 3-tap (0.75 px) boxcar before the peak search (SNR judged on the raw profile), FWHM by linear
  interpolation of the half-maximum crossings. Accepted at SNR >= 3 (peak / 1.4826·MAD of the outer region) decided at NATIVE
  resolution; the 1.5x set re-measures exactly those points. Mean-profile FWHM from peak-normalised profiles, peak-aligned and
  unaligned (see A.3). Synthetic centerlines are exact; real GT is human-drawn (peak offset p50 0.50 px native).
- Isolation (Part A, B.2, C-displaced): no other polyline within 12 px native (20 px for the displaced control), >= 8 px from
  polyline ends, >= 16 px from the frame border, inside the field-of-view mask eroded by 16 px (predictions are zero outside the FOV).
- Part B sampling: every 4 px of arc along polyline A, nearest point of B, probability along the joining segment at 0.25 px
  (values at both ends, midpoint, minimum), image intensity along the segment extended by 8 px each side for the dip.
  Bins in the eval frame (native gap x1.5).
- Predicted band widths: contiguous interval with p > level around the ridge peak searched within ±3.75 px of the GT point.
- Runs: local Python 3.13 / scipy 1.16 (synth, real); tulen `nice -n 10`, CPU only, nnunet env Python 3.12 / scipy 1.18 (real, pred, gap).
