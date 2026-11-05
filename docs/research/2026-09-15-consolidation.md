# Consolidation, 2026-09-15: what is known, what the metric and test set are, and what to do next

Written on the user's instruction to step back ("udělej odstup, utřiď si myšlenky a zjištění").
Every number here is in `docs/TODO.md` with its provenance; this document orders them, decides
the metric and the test set, and sets the plan. Nothing in it is new measurement.

---

## 1. The benchmark, as it actually is

| block | frames (VAL / TEST) | GT provenance | crossings / frame | median GT length (native px) | in the generator's calibration corpus? |
|---|---|---|---|---|---|
| **roi303** | 139 / 137 | human, 14 123 curves | 17.3 | 107 | **YES — every frame** (content hash) |
| 586 (in MT-34) | 11 / 11 | 69 % model-drawn | 32.1 | 173 | 22 of 34 |
| alice (585) | 6 / 6 | human | 2.2 | 358 | no |
| **htw** | 33 / 33 | 2 annotators from scratch, SKIoU 0.80 | **0.7** | **29** | **no** |

Three facts that constrain everything below:

1. **Crossing density and contamination are perfectly confounded.** Every crossing-dense block
   was seen by the generator's calibration; every clean block is crossing-sparse. No block is
   both clean and dense. This is a data defect and no method repairs it.
2. **The contamination is calibration-only.** The generator's priors were fitted to these frames'
   image statistics (label-free MMD) and, in two channels, to MT-34 / HTW ground truth. The model
   never trained on a real label. Consequence: a paired A/B of two arms on the same frames carries
   identical contamination and is valid; a single absolute number on roi303 is in-distribution,
   not a generalisation estimate.
3. **Annotation convention is worth 0.10–0.20 on identical predictions** (model-drawn vs human GT
   +0.20–0.28; end tolerance +0.113; strict → 0.80/0.80 +0.145). Any number quoted without its
   convention is meaningless.

## 2. The metric — DECIDED

**Centerline-F1, one-to-one matching, tolerance 5 px in the 1.5× eval frame, coverage 0.80 AND
precision 0.80, end tolerance 0, POOLED MICRO-F1 as the primary statistic; per-frame macro over
frames with ≥10 GT reported alongside; the all-frames macro is NOT quoted.**

Why each clause:
- 0.80/0.80: the user's decision of 2026-09-08 ("běžná metrika v oboru"); under the strict
  0.95/0.95 rule 0.90 is unreachable even for an oracle mask (0.888).
- Micro as primary: the 3-seed spread is **0.036 on the all-frames macro and 0.003 on micro**.
  The macro gives 23 % of its weight to 32 near-empty frames holding 1 % of the filaments. A
  metric that cannot resolve a 0.03 effect cannot drive iteration.
- end_tol 0: the 25 px variant credits predictions past the annotator's end, which is
  annotation-owned, not model-owned. Reported as a secondary column, never as the headline.
- **Instancer at the DERIVED constants** (`params_a_derived.json`, `min_length` 15.0, rule
  declared before the numbers): the fitted 44.74 deleted 49 % of htw ground truth before matching.
  Every number quoted before 2026-09-15 carries the fitted constant and is a lower bound.

## 3. The test set — DECIDED, with the assumption stated so it can be overridden in one word

**Primary: roi303 TEST (137 frames; the 2026-09-10 repair removed the empty/saturated ones). Mandatory secondary: htw TEST (33 frames), reported beside
it, never pooled with it.** Development iterates on roi303 VAL (139), paired per frame.

Why roi303 and not the clean block: the method exists for the crossing regime; htw carries 0.7
crossings and 19 px of parallel arc per frame, so a method's mechanism has nothing to act on
there, and 33 frames cannot resolve an effect below ~0.10. roi303 has 4× the frames, human GT,
and 17.3 crossings per frame. Its contamination is calibration-only and is stated in every table.

**Assumption flagged:** if the user wants the clean block as THE test set, that is a legitimate
choice — it changes the paper from "a crossing-regime method" to "a transfer result on sparse
data", and it is the user's call, not mine.

## 4. The best method and its number

**SPARSE35**: nnU-Net ResEnc-M trunk, single binary head, trained 40 epochs on online synthetic
data (`calib_V6_c070_narrow.json`, seed 20260827, TIRF 0.35 sparse), assembled by the
curvature-bounded instancer A (`kappa_max` 0.25, derived). Three seeds exist.

| block | convention | number | instancer constant |
|---|---|---|---|
| roi303 TEST | 0.80/0.80, all-frames macro | 0.614 ± 0.072 (3 seeds) | fitted 44.74 |
| htw TEST | 0.80/0.80, macro / micro | 0.5385 / 0.5281 | fitted 44.74 |
| roi303 VAL | 0.80/0.80, macro | 0.672 | fitted 44.74 |
| oracle ceiling, roi303 VAL | 0.80/0.80 | 0.920 | fitted 44.74 |
| oracle ceiling, htw TEST | 0.80/0.80 | 0.634 → 0.947 as `min_length` 44.74 → 5 | — |

**Scored under §2 on 2026-09-16 (`2026-09-16-step1-results.md`):** roi303 TEST **0.641 ± 0.006**
micro (oracle 0.935), htw TEST **0.601 ± 0.012** (oracle 0.871), 3 seeds, ep040, derived
constant, thresholds chosen on VAL. The derived constant costs roi303 0.018 net (recall +0.04 = real short curves the oracle also
recovers, precision −0.06 = the model's fragments now reaching the matcher) and buys htw 0.113;
it stays (rule first). Gap ≈ 0.28 on both blocks, but roi303 loses on PRECISION (0.59 / 0.71) and
htw on RECALL (0.62 / 0.58).

**Provenance defect found while inventorying (2026-09-15 23:30):** every cached TEST dump
(`*_TEST_native`) and every cached htw dump (`*_htw_native`) was produced from `model.pth`,
which is the trainer's PICK — ep036 for SPARSE35 and S35_SEED2, ep040 for S35_SEED3 (tensor
equality against the epoch files). The pre-declared checkpoint rule is ep040 (the VAL arms were
all read at `*_roi_ep40_native`), and pick-vs-ep040 is worth 0.018–0.052. So the 0.614 ± 0.072
above is a pick number, and step 1 re-dumps TEST at ep040 for the two seeds where they differ.

## 5. Where the loss is — the measured anatomy, model only, no instancer

| block | pixel precision at the threshold carrying 80 % of curves over ≥95 % length |
|---|---|
| synth (in domain) | **0.985** |
| alice (clean, long) | 0.969 |
| roi303 (contaminated) | 0.803 |
| **htw (clean, cross-lab)** | **0.491** |

The failure is **the per-curve MINIMUM, not the mean**: median per-curve mean probability is
0.95–0.99 everywhere; median per-curve minimum is 0.94 in domain and 0.51–0.87 on real, falling
with length. The image contrast along a filament is NOT the cause — its heterogeneity in the
render sits inside the real range (min/median 0.40 synth, 0.35–0.50 real).

In domain, 91.2 % of instance breaks sit on a crossing (40.9 %) or a 2–6 px parallel (50.3 %)
and only 5.4 % are a hole in the mask — but the generator over-produces true parallels 9–11× per
unit GT length, so the parallel half of that is a generator property, not a real burden.

Instance recall on roi303 is flat at ~0.52 across thresholds 0.94–0.99; 85 % of curves are
covered at 0.50 but cannot be extracted there. The gap is **ordering** — on real data the
confidence the network adds lands on background, in domain it lands on filament.

## 6. What is closed, by what, and whether it could reopen

| route | closed by | could a clean block flip it? |
|---|---|---|
| post-hoc mask repair, 11 decoders, instancer re-tune | foreground-matched control | no (within-arm) |
| generator prior width — narrow / wide / more epochs | 8/8 replicated cost / null / null | narrow: no; **wide: only on a clean block** |
| noise texture, amplitude, hard-negative distractors | null / null / catastrophe (distractors ≡ faint MTs) | no |
| connectivity losses (cbDice, clDice, pos_weight) | structural / negative | no |
| multiplicity target, crossing-region weight | scrambled control reproduces 60 % / +0.014 | no |
| worst-point (softmin) loss | mechanism fired, precision at matched coverage fell, threshold moved +0.2–0.4 → **rescale** | no |
| **CURVEQ** (overlap-permitting per-query masks) | 5 runs, ~40 GPU-h: position-invariant mask cannot separate identical thin objects (AUC 0.483); with relative coords AUC 0.80 but instance F1 0.000 **and the shared dense head fell to 0.068** | no |
| feature-level domain adaptation | the unlabelled corpus IS the benchmark (leakage) | only with new data |

## 7. What is genuinely open

1. **The model has never been scored under the declared metric with the derived constant.**
   Cheap; first.
2. **Ordering on real data** is the loss and no lever has moved it. Precisely what HAS been
   tried, so research does not re-propose it: the loss (pos_weight, cbDice, clDice, junction
   weight, worst-point softmin), the generator's priors (narrow, wide, volume, noise texture,
   amplitude, distractors), the trunk (DINO+ASPP with an `hr` branch → nnU-Net ResEnc-M, and
   an 11-variant decoder screen), the input (a classical prior bank `--priors`, scale
   augmentation, photometric augmentation, low-res simulation), the emission (heat, overlap,
   multiplicity, CURVEQ), and post-hoc repair. **What is actually virgin:** (a) context ALONG a
   filament beyond one crop — every trunk so far integrates evidence over a 2-D receptive field,
   none over the 1-D structure the object is; (b) how the network is taught what background IS —
   real empty-field backgrounds are composited under synthetic filaments, so the model has never
   seen a real background structure labelled negative next to a real-looking filament, only a
   synthetic one; (c) the ordering statistic itself as a training signal (ranking on-filament
   above near-background per frame), as distinct from the per-pixel BCE that was found to move
   calibration rather than ordering.
3. **Data**: a clean block with crossings is the only thing that makes a crossing-regime claim
   evidencable. Not a method question.

## 8. The plan

1. Score SPARSE35 (3 seeds, **ep040**) under §2 on roi303 TEST and htw TEST with the derived
   constant AND the fitted one (comparison column), plus the oracle ceiling at the same constant
   and convention. Threshold rule: per seed, the pooled-micro optimum on the block's VAL frames,
   applied to TEST; htw additionally at the roi303-VAL threshold (pure transfer). Batch:
   `step1_score.sh` on tulen → `runs/step1/`; reader `scripts/step1_read.py`. One exposure.
2. From the same dumps, re-read the anatomy on TEST for the WRITE-UP only: precision at matched
   coverage, per-curve minimum, crossing-break share. **The weak part is already named on VAL
   (§5) and every lever is chosen against VAL; TEST anatomy confirms, it does not inform.**
3. Research, on the VAL-named weak part only, with §6 and §7.2 as the exclusion list.
4. Iterate on roi303 VAL, paired, micro-resolved, one lever at a time, falsifier declared first.
   Threshold is chosen on VAL and applied to TEST — never chosen on TEST.
