# SPARSE35: weaknesses, proposals and decisions (synthesis, 2026-10-04)

**Bottom line.** No model-side training arm survives at >= +0.01 on roi303. The only measured positive effect is an instancer change (RELINK, roughly +0.01 roi303 / +0.02 htw on VAL), which needs two rulings from you. The largest model-owned row, htw detection (+0.113 on VAL), has no surviving attack. The rest of the movement in the reported number sits in your benchmark decisions.

## 1. Which numbers, and the split defect

- **Declared numbers:** roi303 TEST (137 frames) 0.641 +- 0.006 and htw TEST (33 frames) 0.601 +- 0.012. Three seeds, ep040, centerline-F1 at tol 5, coverage 0.80/0.80, pooled micro, min_length 15, threshold from each block's own VAL. The blocks are never pooled.
- **Split defect:** `roi_img_208` (VAL) and `209` (TEST) are byte-identical, as are 172=173 (both TEST) and 206=207 (both VAL). About 94 sibling pairs cross the split (`docs/TODO.md:2702-2710`).
- **How proposals are read:** paired on roi303 VAL and htw VAL. TEST is read once, after the re-split by field.
- **Anatomy is VAL only.** TEST has counts but no FP/FN classes, and no TEST dump was re-read today.
- **Frame counts disagree and must be reconciled before any re-split:** tulen RAW has 281 frames (141/140), the sweep rows cover 276 (139 + 137), the local 8-bit export has 277 (139/138).
- **The +-0.006 is training-seed sd, not a confidence interval.** A refuter measured the roi303 TEST 95 % half-width at about +-0.040 (frame bootstrap) and +-0.050 (group bootstrap).

## 2. Weaknesses of SPARSE35 (priced anatomy, VAL)

Operating points (digest; `docs/research/2026-09-16-step1-report.json`):

| Block | Threshold | VAL TP/FP/FN | VAL P / R / F | TEST P / R / F | Oracle VAL / TEST |
|---|---|---|---|---|---|
| roi303 | 0.98 | 5212 / 3304 / 2167 | 0.612 / 0.706 / 0.6558 | 0.589 / 0.705 / 0.6414 | 0.927 / 0.935 |
| htw | 0.995 | 453 / 196 / 345 | 0.698 / 0.568 / 0.6261 | 0.614 / 0.589 / 0.6013 | 0.857 / 0.871 |

roi303 loses on precision, htw on recall. htw TEST carries 45 % more FP than VAL at the same threshold, so VAL prices do not transfer in magnitude.

Priced rows (ladder rows at band half-width 2.0 are additive to about 0.01; rows marked "cut" are not additive):

| Item | Block | Max delta | Owner | Evidence (confidence) |
|---|---|---|---|---|
| Off-annotation foreground, total | roi303 | +0.091 | mixed | delete-far-5: 0.6558 -> 0.7471, FP 3304 -> 1832 (high) |
| of which unannotated filaments + short rods | roi303 | ~0.05 | convention | 0.626 of off-annotation length; ~10 of 27 montage crops are undrawn filaments (medium) |
| of which field-stop edge | roi303 | ~0.02 | generator | 23.1 % of off-annotation length; ARM 50 negative (medium) |
| of which particles / debris | roi303 | ~0.01-0.02 | generator | eye estimate, no machine count (low) |
| of which long predictions 10-50 % on an annotation | roi303 | ~0.01 | ambiguous | 0.143 of off-annotation length (low) |
| Detection / continuity, total | roi303 | +0.107 | model | perfect band, model FP kept: 0.7624 (high) |
| of which crossing erosion | roi303 | +0.014-0.030 | model | share below p 0.98 near crossings 0.28 vs 0.15 plain; synth 0.03 vs 0.01 (high) |
| of which weak plain stretches | roi303 | ~0.07-0.08 | model (ordering), convention share unresolved | network AUC 0.95 vs displaced background, median p 0.89 (medium) |
| Merging at close parallels | roi303 | +0.012-0.017 | model | 0.8639 vs 0.8813 at hw 2.0 (high) |
| Instancer on a perfect map | roi303 | 0.073 VAL / 0.065 TEST (K=6); 0.10 one-channel | instancer | oracle FN is 98 % covered_split (high) |
| K=6 orientation channels | roi303 | +0.027 on the oracle; realised -0.088 | instancer | ARM 51 (high) |
| Short-fragment FP from min_length 15 | roi303 | 0.018 net on TEST (cut) | convention | constant is rule-derived and stays (high) |
| Detection of annotated filaments | htw | +0.113 | model | 0.626 -> 0.739; FN is 64 % "missed" (medium) |
| Off-annotation foreground | htw | +0.072 | unresolved, probably mostly convention | never classified on htw (low) |
| Merging / crossing erosion / field stop | htw | ~0 | n/a | 0.7 crossings per frame, no field stops (high) |
| Instancer + min_length on a perfect map | htw | 0.143 VAL / 0.129 TEST (K=6) | instancer | count-matched to short curves, not verified per curve (medium) |
| Threshold transfer from roi303 | htw | 0.025 | convention | already avoided by the own-VAL rule (high) |

Class anatomy by count:
- **roi303 FP:** off_other 39 %, fragment 35 %, junction 14 %. **FN:** covered_split 65 %.
- **htw FP:** off_other 52 %, fragment 24 %. **FN:** missed 64 %.

**Reconciliation of model-owned headroom**
- **roi303:** 0.06-0.07 is a lower bound (crossing <= 0.03, merging 0.012-0.017, edge ~0.02, particles ~0.01-0.02). The upper bound is 0.13-0.14 if the weak-stretch row is the model's. The "evidence-free, single-frame limit" label on that row is not established.
- **htw:** the old "detection 0.17 / off-annotation 0.06 / merging 0.08" was the 49-frame block at half-width 1.0. htw-only it is 0.113 / 0.072 / ~0.
- **htw is recall-limited under the declared metric.** The older "precision-limited, not fixable" headline is retired; its fact (81 % of over-long excess lies on unannotated pixels) stands.
- **The largest single model-owned row on either block is htw detection.** Missed htw curves are short (median 13 native px), visible (AUC 0.97 vs a displaced twin), and carried at p >= 0.5 in 78 % of cases. Instance recall is flat at 0.52-0.60 across thresholds.
- **Every roi303 model-owned row is individually <= 0.03.** That is the structural reason ARM 50-55 all read null.
- **In-domain claims about curves under 15 px are unmeasured.** The synth eval set was built with `--min-len 15`, and 15.5 % of htw GT is under 12 px.

New facts from today's lanes (exploratory, VAL only, one model seed unless stated):
- **Length-keyed miss on htw at 0.995:** 0-12 px 0.66, 12-20 px 0.47, 20-35 px 0.24, 60-120 px 0.12. The same gradient exists in domain (12-20 px 0.146 vs >= 120 px 0.019).
- **Missed htw short curves are thin and faint:** FWHM 2.5 px for misses vs 3.0 for hits; roi303 sits at 4.0-4.5 px.
- **The K=6 oracle gain is not a representation effect.** Orientation profiles computed from the binary band reproduce 0.9270 exactly, so ARM 51's premise was void.
- **`w_ori` is inert on the binary head.** The instancer runs geometry-only at junctions.
- **Close parallels remain unsolved:** bundle recovery is 67/266 (25 %) on roi303 and nothing moved it.

## 3. Open leads and status

| Lead | Status | What settles it |
|---|---|---|
| Tile fusion / context (uniform probability average, 24 % overlap, crop 512) | Open, never read | Native logit re-dump, under 0.3 GPU-h; htw is the negative control |
| Cached arms under the declared metric (NOISE, NOISE_S2, COMBO, SCALE, SCALE_S2, JWEIGHT, TIRF 0/20/35/50) | Old metric only | One CPU chain; SCALE on htw first |
| TIRF fraction 35 % (defines the recipe) | Old metric only (strict macro) | Same CPU chain; 35 % of batches go to a modality neither test block contains |
| PAINT_AA second seed | One seed: roi303 +0.0105, htw -0.014 | About 6 GPU-h; it also removes the unrecorded bright-tip defect (tips 1.56x, filaments 3.3 px short) |
| Comparator row (SAM3, TARDIS, ORION) on roi303 + htw | Never run under the declared metric | Inference only; the paper needs it |
| htw filament width / pixel scale | Open, critic-identified | See section 4 |
| roi303 crossing erosion cause | Open, nobody asked why | `diag_crossing_angles.py`, `diag_crossing_compounding.py` exist, unread |
| Generator over-produces close parallels 9-11x (`docs/TODO.md:3462-3467`) | Open, untouched | Price is small (+0.017 / ~0); skipping is a decision, not an omission |
| Screen B (`docs/TODO.md:3397`, `:3579`) | Declared, never read | CPU read |
| `gen_train.build_cfg` not in git (`scripts/train_v5.py:1193-1194`) | Reproducibility gap | Commit the file; prior widths cannot be audited without it |
| `along_intensity_std` dead in `render_irm` (`synth/mt_generator.py:739` vs `:770`) | Recorded-wrong-reason | Second pointer today: synth prefers short links because the in-domain mask has almost no long breaks |
| Temporal model on synthetic video | Old metric only | Not settleable on the declared set: roi303 h5 has no stack index or dt |
| Two-frame input from siblings | Closed | Sibling GT overlap is median 0.02; filaments are not co-registered |
| CURVEQ / per-curve emission | Closed | Emission F1 0.0000 in domain |
| ARM 29 synth-tuned params | Closed today, NEGATIVE | roi303 -0.037 (p 6e-11), htw -0.021 at min_length 15 |
| Learned linker, gated retrain, SETPRED, XCOMP | Closed | Nothing owed |
| Feature-level UDA | Closed on leakage | 252 Wiggly + 428 background frames are benchmark-disjoint, but that is ~80 scenes of 8-bit mp4 frames; needs new 16-bit frames and your ruling |

## 4. Proposals, ranked by expected gain per cost

Deltas are the refuters' corrected values on VAL, not the proposers'.

| # | Proposal | Anatomy item / price | Mechanism | Delta roi303 / htw | Pre-screen and kill line | GPU-h | Rule conflicts | Refuter 1 / Refuter 2 | Owner |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **RELINK 1-D** (merged salvage of RELINK-D) | Over-linking in the free-end linker; pool is 77 of 345 base FN on htw | One multiplier on `w_theta`, derived on synth under 0.80/0.80 micro; a filament fused to a neighbouring fragment is cut free | +0.008 to +0.015 (R1), modal +0.012 (R2) / +0.010 to +0.020 (R1), modal +0.024 (R2) | Smallest multiplier within 0.005 of the synth maximum; in-domain identity and bundle guard; `diag_fp_anatomy.py` on the x2 dump; earning rule on re-split TEST | 0 (GPU-minutes for a second synth dump) | No retrain; CLAUDE.md says instance tuning is deferred; real VAL shaped the search space | Refuted, SERIOUS / Not refuted, SERIOUS | Your ruling, then us |
| 2 | **Decision B: dedupe + group handling** | Duplicate/sibling defect; < 0.002 | Drop 209, 173, 207; group-wise split; cluster bootstrap | Dedupe +0.0015 measured; B1 at 0.99 -0.006 measured / unmeasured | Group definition fixed before any score is read | 0 | TEST already read under v1; must be disclosed | Not refuted, MINOR / Refuted as a lever, MINOR | Your decision |
| 3 | **Decision A: blind annotation completion** | roi303 unannotated filaments ~0.05; htw off-annotation +0.072 | Human pass with no model overlay re-assigns ownership of the row | Primary 0. Secondary +0.01 to +0.03, central ~+0.02 (scenario arithmetic) / 0.00 to +0.04, unpriced | Kill line on length within 5 px of an ADDED curve only; >= 24 VAL frames | 0 (4-5 annotator-days) | Annotators must see no model output; VAL only until the TEST read | Not refuted, SERIOUS / Refuted as a lever, SERIOUS | Your decision + BIOCEV |
| 4 | **htw width / scale support** (unjudged, critic-identified) | htw detection +0.113 | Missed htw curves (FWHM 2.5 px) may lie below every trained width; training width ~4.5 px, SCALE minimum ~3.6 px (critic's estimate across two estimators) | No predicted delta | SCALE and SCALE_S2 htw dumps re-read under the declared metric; FWHM-vs-miss within each htw saturation cluster; pixel pitch | 0 for the pre-screen; ~13 for a 2-seed arm (assumed from the FLATSAT estimate of ~6 GPU-h per seed) | Found by reading htw VAL; must be disclosed. A widening, not a fit | No refuter read | Us (pre-screen) |
| 5 | **Tile fusion** (unjudged) | roi303 and htw detection rows | Logit-space Gaussian fusion at 50 % overlap or a whole-frame pass instead of a uniform probability average | Unmeasured | Native logit re-dump; weak-point share vs distance to tile edge | < 0.3 | Changes the inference path; your ruling whether it is a defect fix | Deferred by both | Your ruling, then us |
| 6 | **PAINT_AA second seed** (unjudged) | Renderer defect (two mechanisms) | Anti-aliased `_paint`; removes axis-aligned brightness and bright tips | Seed 1: +0.0105 / -0.014 | Second seed must replicate sign on roi303 | ~6 | None | No refuter read | Us |

Notes on RELINK:
- **The effect is real on a bit-identical map.** `w_theta` x2 gives roi303 VAL +0.0188 (p 0.0016) and htw +0.0243 (p 0.0027), replicated on three seeds. Synth exact GT agrees in sign (+0.045).
- **Refuter 1's objection is procedural.** Real VAL was read before synth and shaped the frozen set, and the earning rule was written after the numbers were seen.
- **Refuter 2 corrected the mechanism.** On htw all 27 recovered curves were covered by one over-long prediction that was itself an FP. The event is "1 FP + 1 FN -> 1 TP + 1 FP", not a chimera split.
- **The gain is recall only.** Precision is flat, FP rises (+61 on roi303), junction identity moves 829 -> 819 of 2043, bundle recovery 67 -> 65 of 266. It is not a crossing or parallel fix.
- **Over-restriction turns negative on roi303** (-0.008 on a 28-frame subset) while synth still reads +0.027. That is why the selection rule is the smallest multiplier on the synth plateau.
- **The htw plateau is flat and robust** (+0.024 to +0.030 across all restrictive variants). The roi303 side is the fragile one.

## 5. Refuted today (so they are not rediscovered)

The 101-row closure ledger in the digest stands. Today's kills:

Judged proposals:
- **DECODE-ORDER, variant (b)** (threshold at native, then zoom the mask): measured null. htw -0.005 at own threshold on all 33 frames; roi303 +0.002 on a 53-frame subset. **CONTESTED for variant (c) and tile fusion:** the FATAL verdict was called without running either, and 8.5 % of on-filament native support on htw is erased by the resample.
- **FLATSAT** (render the background-subtracted, saturated export chain): the support gap is real (32 of 33 htw frames outside the generator's range), but the dose-response inside htw has the wrong sign. More saturated curves are missed less (0.309 / 0.181 / 0.096 by tercile). Caveat from the critic: n = 33 frames in two clusters, pseudo-replicated p-values.
- **ENCPROBE / VFMSKIP** (frozen foundation encoder): premise contradicted (DINOv2 features separate synth from real at ~0.98, no better than the trunk's 0.95); the probe's kill line is void because its populations are selected on SPARSE35's own output.
- **RELINK-D as a 7-dim Optuna:** superseded by the 1-D version; a joint synth optimum can overshoot.
- **Decision B option B1 as written:** the outcome depends on the correlation threshold (-0.006 at 0.99, -0.03 to -0.06 at 0.95).

Ordering lane:
- **Spatially heteroscedastic noise:** synth is already more heterogeneous than real (MAD-CV 0.120 vs 0.052).
- **Clutter co-location:** AUC 0.51-0.57 against a 0.60 kill line.
- **Wider within-frame contrast spread:** synth CV 0.32 vs roi303 0.19; narrowing would be fitting.
- **Length-normalised loss:** equated with WORSTPOINT, which floods htw (miss 0.310 -> 0.502 / 0.692 at matched off-annotation foreground). **Soft kill:** blob loss is a different redistribution, and the dumps were trainer picks, not ep040.
- **Ranking / AP loss, PASTA, RandConv, GIN, distractors, MixStyle, flat-minima optimisers:** in-domain AP 0.996 leaves no mis-ordered pairs, or the class is AUG_PHOTO / ARM 41.

Continuity lane:
- **CAPE, widest-path, SCNP:** all reduce to raising the weakest link, which is WP03/WP10.
- **Skeleton Recall, CoLeTra dropout:** reduce to pos_weight or a refuted premise (render dips already in the real range).
- **Tip restoration before min_length:** retraction is only 0.5-1.8 px; returns about 25 htw curves on a perfect map.
- **Along-filament integration:** closed on roi303 (recall 0.04-0.05 vs kill line 0.20). **Alive on htw:** 0.36-0.40, on 287 weak samples of long filaments; ceiling about +0.02-0.03.

Instance lane:
- **Orientation side-channel, frozen-trunk K=6 head, double-angle regression:** equal `w_theta` x2 within 0.002; the trained AMODAL channels give only +0.009 / +0.004.
- **Shorter link range:** x0.5 is -0.029 on roi303; link off is -0.128.
- **CondInst, curve queries, geodesic tracking, gap-weighted loss for parallels:** prize is <= 0.03 roi303 and 0 on htw.

Generator and encoder lanes:
- **Background subtraction alone, isolated quantisation, raising noise at fixed contrast:** too weak, prices at +0.005, or is ARM 44.
- **SSL pretraining, ResEnc-L/XL, DINOv3 trunk, SAM fine-tune, normalisation randomisation:** no clean data, not capacity-bound, or expected <= 0.005.

Benchmark lane:
- **Ignore rule on weak annotated stretches, model-derived ignore band, shorter-curve exclusion, length-weighted primary, looser tolerance:** each fits the convention to a model failure or changes the instrument after TEST was read.

## 6. Decisions that are yours

| Decision | Honest effect on the reported number |
|---|---|
| Is a synth-derived instancer constant with no retrain admissible, and is the "instance tuning deferred" note lifted? | If yes: about +0.01 roi303, +0.01 to +0.024 htw on VAL; TEST unverified. Reported as an instancer change, not a model gain |
| Drop the byte-identical frames 209, 173, 207 | +0.0015 on roi303 TEST (measured, 3 seeds). Mandatory before print |
| Re-split by acquisition group | 0 as a model effect. B1 at correlation 0.99 moves 49 frames and lowers the 3-seed TEST mean to about 0.635. Needs a group definition from BIOCEV stack index or dt, or a pre-declared threshold |
| Report a confidence interval instead of seed sd | The bar grows from +-0.006 to about +-0.04 to +-0.05. SPARSE35 vs V5H survives the group bootstrap (lower bound +0.015 to +0.020) |
| Blind annotation completion on >= 24 VAL frames | Primary number moves 0. Secondary score +0.01 to +0.03 on roi303 (not measured). A result below +0.02 means the model-owned share rises toward 0.13-0.14 |
| Is changing tile fusion or decode order a defect fix or an inference trick? | Unmeasured; applies equally to the baseline and every arm |
| Ask the SynthMT authors for both annotators' label sets and the crop-to-video map; ask for nm/px on htw and roi303 | 0 on the score. Gives the human reference row and settles the width question |
| Do the htw VAL statistic reads (FLATSAT, FWHM) weaken the "untouched cross-lab block" claim? | 0 on the score; a disclosure in the paper |
| New benchmark-disjoint 16-bit real frames for label-free alignment | Unpriced; conflicts with the biased-corpus rule unless you rule otherwise |

## 7. Recommended order of work

**Step 1, CPU only, cached dumps (about a day):**
1. Reconcile the roi303 frame counts (281 / 277 / 276).
2. Run `diag_fp_anatomy.py` on the `w_theta` x2 dump. This settles RELINK's label and its mechanism gate.
3. Re-read the cached arms under the declared metric, SCALE and SCALE_S2 on htw first, then NOISE, COMBO, TIRF 0/20/35/50.
4. Read FWHM against miss within each htw saturation cluster.
5. Check the oracle's 121 perfect-band misses on htw against per-curve length.
6. Build the htw off-annotation montage (the +0.072 row was never classified).
7. Paired frame and group bootstrap of ARM 50-55 and PAINT_AA on cached per-frame rows.
8. Rebuild the synth eval set with min-len 5 and re-read the length-binned miss.

**Step 2, under 0.3 GPU-h, needs your go:** one native logit re-dump of SPARSE35 ep040 on roi303 VAL and htw VAL. It reads tile fusion, decode variant (c), and weak-point share against tile-edge distance in one pass.

**Step 3, gated arms:**
- **Width/scale-support arm (2 seeds):** only if the SCALE htw re-read and the within-cluster FWHM read both point out of support.
- **PAINT_AA seed 2 (~6 GPU-h):** justified as a defect fix; it earns a recipe place only if roi303 replicates and htw does not fall.
- **RELINK 1-D (0 GPU-h):** only on your ruling, and only if step 1.2 shows over-long single-prediction FN falling with total FP up by no more than about 3 %.
- **Tile-fusion change:** only if step 2 moves micro beyond +0.006 roi303 or +0.012 htw at each variant's own threshold, and you rule it a defect fix.

**Step 4, before any TEST read:** dedupe, fix the group definition, re-declare, then read TEST once per seed.

Everything from today's lanes is exploratory: nothing was written to `runs/`, the repo, docs or memory, and no GPU job ran. The scratch outputs live under `/disk2/prusek/tmp/` on tulen and in the session scratchpad, and are ephemeral until moved.

---

# Appendix: completeness critic (verbatim)

**COMPLETENESS CRITIQUE** (record claims grep-verified where a file:line is given; "digest" means taken from the audit digest and not re-opened; nothing was run on tulen)

**0. Overall picture**
- **htw detection (+0.113, the largest model-owned row) has no surviving proposal.** Every lane that attacked it died: ordering (0 proposals), FLATSAT (wrong-sign dose-response, both refuters), ENCPROBE (premise contradicted, kill line void), DECODE-ORDER (measured null).
- **RELINK-D is the only live item, and its verdict is split.** Refuter 1 says refuted/SERIOUS, refuter 2 says not refuted/SERIOUS. Their salvages are compatible and should be merged into one proposal:
  - a 1-D `w_theta` multiplier derived on synth under 0.80/0.80 micro;
  - the smallest multiplier within 0.005 of the synth maximum;
  - an in-domain identity/bundle guard;
  - the earning rule attached to re-split TEST.
- **RELINK-D's anatomy label is wrong.** Refuter 2 showed the event is "1 FP + 1 FN -> 1 TP + 1 FP" (27 of 27 on htw), not a chimera split. `diag_fp_anatomy.py` exists in `scripts/` and was not run on the variant.

**1. Model-owned anatomy nobody attacked**
- **roi303 crossing erosion (<= 0.03, the largest cleanly model-owned roi303 row).** The instance lane declined it on price, and no lane asked why it happens.
  - The share of samples below p 0.98 within 5-20 px of a real crossing is 0.27-0.30 against 0.15 on plain stretches; on synth it is 0.02-0.03 against 0.01, with identical image contrast (`docs/research/2026-09-16-step1-results.md:122-136`).
  - The record says "neither explained by a contrast-magnitude difference" and stops there.
  - Candidate causes never tested: the crossing-angle distribution, and local filament density around real crossings.
  - `scripts/diag_crossing_angles.py` and `diag_crossing_compounding.py` exist, and no lane cited a read from them.
- **Close-parallel over-production by the generator (9-11x per unit GT, `docs/TODO.md:3462-3467`).** It is listed as open in `docs/research/2026-09-16-generator-research.md:118-120`. The generator lane did not touch it, and the instance lane reports bundle recovery of 25 % "unsolved and untouched". It is a declared project bottleneck with zero attack. The price is small (+0.017 roi303, ~0 htw), so skipping is defensible, but it should be stated as a decision rather than an omission.
- **Filament width / pixel scale on htw.** This is the gap I would rank highest.
  - FLATSAT refuter 2 measured missed htw curves at FWHM 2.5 px, hits at 3.0, and roi303 at 4.0-4.5.
  - The record's own training width is about 4.5 px (`docs/research/2026-09-04-parallels-preprocessing-domain-gap.md:83`).
  - The only scale arm used `--scale-aug 0.8,1.5` (`docs/TODO.md:1101`), which gives a minimum width of about 3.6 px. So the missed htw population (ratio about 0.55-0.67) lies below the support of every arm ever trained. This is my arithmetic across two different FWHM estimators, so treat it as an estimate.
  - SCALE was only read under strict coverage with min_length 44.74, which deletes exactly these curves.
  - No lane proposed a width-support arm such as SCALE with a lower bound near 0.5, or LOWRES isolated (open, `generator-research.md:120`).
  - It would be a widening, not a fit, but it was found by reading htw VAL, which needs disclosing.
- **roi303 "weak plain stretches" (0.07-0.08, relabelled model-owned by the digest).** Three lanes named it as a target and none produced a pre-screen that could re-price it. Every argument reduces to "in-domain AP 0.996, so nothing to learn". The row stands unpriced between model and convention.

**2. Open leads nobody picked up**
- **CONTEXT / tile fusion** (uniform probability averaging at 24 % overlap against Gaussian logit fusion or a whole-frame pass; digest code finding 3). DECODE-ORDER refuter 2 deferred tile fusion "to the CONTEXT lead", but no such lane existed. `--tile 0` whole-frame decode already exists (`docs/TODO.md:3133`, `:3343`). The cost is under 0.3 GPU-h, with htw as the built-in negative control.
- **CPU re-read of cached arms under the declared metric** (NOISE, NOISE_S2, COMBO, SCALE, SCALE_S2, JWEIGHT, and TIRF 0/20/35/50). Three lanes called it "step 0" or "hygiene" and none owned it. Two reasons it matters beyond hygiene:
  - 35 % of every batch is TIRF, a modality neither test block contains, and that choice rests on a strict-macro read (digest, TODO:5943-5985).
  - The SCALE htw re-read is the first step of the width question above.
- **PAINT_AA second seed.** It is the only arm under the declared metric at +1.7 seed-sd, and the record says "a second seed would be needed to call it" (`docs/TODO.md:2525`). The generator lane also found that it removes a second, unrecorded defect: tips 1.56x brighter, filaments 3.3 px short. That makes about 6 GPU-h for an arm with two mechanisms and one seed. Caveat: htw read -0.014.
- **Comparator row** (SAM3, TARDIS, ORION under the declared metric on roi303 and htw). Two refuters recommended it and nobody owns it. It also answers the encoder lane's question, whether a real-trained extractor orders these pixels better, without the void ENCPROBE kill line.
- **Screen B.** Declared and never read (`docs/TODO.md:3397`, `:3579`).
- **`gen_train.build_cfg` is imported from a hard-coded tulen path** (`scripts/train_v5.py:1193-1194`) and is not in git. Every "prior width is closed" verdict refers to priors that cannot be audited from the repo.
- **Dead `along_intensity_std` in `render_irm`** (`synth/mt_generator.py:739` against `:770`). Two lanes noted it and dismissed it. The instance lane found synth prefers short links and real needs long ones "because the in-domain mask has almost no long breaks". That is a second, independent pointer to the same omission, and it also explains why the RELINK synth optimum may not transfer.

**3. Refutations that look weak**
- **Ordering kill #4.** A length-normalised loss was equated with WORSTPOINT. WORSTPOINT is a softmin level-raise and blob loss is a per-instance Dice. "The redistribution alone fails" is inferred from a different redistribution, and on trainer-pick dumps rather than ep040. It is a soft kill, not a measured one.
- **The htw along-filament price bar was applied unevenly.** The evidence lane rejected it on htw as "+0.02-0.03, not resolvable at sd 0.012". RELINK-D is carried forward at the same htw size. ENCPROBE refuter 2 independently measured coh_40/inc_40 beating the network's strong-vs-blob ordering on htw (0.85-0.86 against 0.75).
- **DECODE-ORDER refuter 2 called it FATAL on incomplete evidence.** The roi303 read covers 53 of 139 frames at 2 thresholds, with no paired test. Variant (c) and tile fusion were not run. The same refuter found that 8.5 % of on-filament native support is erased on htw. "Inert" is established for variant (b) only.
- **Decision A refuter 2's +0.016 "central" is scenario arithmetic on assumed conversion rates.** It is useful as a sanity bound and is not a measurement.
- **FLATSAT refuters' within-htw dose-response is at n = 33 frames in two source clusters.** The curve-level p-values are pseudo-replicated, as refuter 2 admits. The kill holds on direction, but "saturation helps" may simply be "brighter filaments saturate".

**4. Killed on one block, alive on the other**
- **Along-filament integration.** Closed on roi303 by STEP 2c (recall 0.04-0.05 against the 0.20 kill line, digest TODO:2886-2903); passes on htw (0.36-0.40). The honest caveat is 287 weak samples on filaments of 50 px or more. That is not the short-curve population making up the 21 % unmatched GT.
- **Width / scale support** (item 1). Irrelevant on roi303 (FWHM 4.0-4.5, in support), potentially the owner of the htw short-curve misses.
- **RELINK-D.** The htw plateau is flat and robust (+0.024 to +0.030 across all restrictive variants, 3 seeds). The roi303 side is fragile: it turns negative on over-restriction. If the user wants a low-risk version, htw is where the effect lives.
- **PAINT_AA runs the other way.** It is +0.0105 on roi303 and -0.014 on htw.

**5. Cheap pre-screens worth running regardless**
1. **Paired bootstrap of the closed arms.** Decision B refuter 2 found the roi303 TEST 95 % half-width is ±0.040 by frame and ±0.050 by group, 7-8x the quoted 0.006. He did not re-read paired deltas. A CPU frame- and group-level paired bootstrap of ARM 50-55 and PAINTAA on cached per-frame rows shows whether "<= seed noise" was the right bar for same-seed paired reads.
2. **`diag_fp_anatomy.py` on the `w_theta` x2 dump.** CPU only; it settles RELINK's label and its mechanism gate.
3. **htw pixel pitch.** Ask the SynthMT authors or BIOCEV for nm/px on htw and roi303 (no value is recorded in `docs/`). Then repeat the per-curve FWHM-against-miss read within each of the two htw saturation clusters. This re-prices the +0.113 row between "width out of support" and "annotation convention for short seeds".
4. **Short-curve identity on htw.** Check the oracle's 121 perfect-band misses against per-curve length, CPU only. The 0.13-0.14 instancer/min_length row is count-matched only (digest), and no lane verified it by identity.
5. **htw off-annotation montage or classifier.** This is the +0.072 row with ownership "low confidence, never classified" (digest). No lane ran the 27-crop montage that exists for roi303, and it needs no annotation.
6. **Native logit re-dump** (under 0.3 GPU-h). It serves three things at once: tile fusion, variant (c), and the weak-point share against distance to a tile edge.
7. **Synth eval set floor.** The in-domain set has `--min-len 15` (ordering-lane side finding), so every in-domain statement about the shortest htw curves (15.5 % of htw GT is under 12 px) is unmeasured. Rebuild with min-len 5 and re-read the length-binned miss.

**Inconsistency to resolve before any re-split:** the roi303 frame counts disagree. RAW on tulen has 281 frames (141/140), the sweep rows cover 276 (139 + 137), and the local 8-bit export has 277 (139/138).