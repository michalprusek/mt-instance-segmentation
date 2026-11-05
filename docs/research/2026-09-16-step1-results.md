# Step 1 result (2026-09-16 04:43): SPARSE35 under the declared metric, on the declared test sets

Batch: `step1_score.sh` → `step1b/c/d.sh` on tulen, `runs/step1/` (108 sharded roi303 nodes +
12 htw-block sweeps + 6 oracle runs + 6 anatomy runs, 0 failures). Reader:
`scripts/step1_read.py` (summary JSON: `2026-09-16-step1-report.json`). Every rule below was
declared in `2026-09-15-consolidation.md` §2/§8 and `docs/TODO.md` (STEP 1 entry) BEFORE any
number was read.

**Convention:** centerline-F1, tol 5 px (1.5× frame), coverage 0.80 / precision 0.80, end_tol 0.
**Checkpoint:** ep040 for every seed (the cached TEST dumps were the trainer's pick = ep036 for two
of three seeds; re-dumped). **Threshold:** per seed, the pooled-micro optimum on the block's own
VAL frames, applied to TEST; every optimum interior (roi303 0.97–0.98 derived / 0.92–0.96 fitted;
htw 0.99–0.995 / 0.97–0.99). **Instancer:** `params_a_derived.json` (`min_length` 15.0, the
rule-derived constant) as the headline; `params_a_model_v5H.json` (44.74, fitted) as the
comparison column. Dumps raw (`--no-fov --include-empty`), native inference scale.

## The table (TEST, 3 seeds, mean ± sd over seeds)

| block | constant | **pooled micro** | P / R | macro ≥10 GT (n) | oracle ceiling (micro) | gap |
|---|---|---|---|---|---|---|
| **roi303 TEST (137)** | derived 15.0 | **0.641 ± 0.006** | 0.59 / 0.71 | 0.659 ± 0.003 (105) | 0.935 | 0.29 |
| roi303 TEST | fitted 44.74 | 0.659 ± 0.003 | 0.65 / 0.67 | 0.671 ± 0.007 (105) | 0.904 | 0.25 |
| **htw TEST (33)** | derived 15.0 | **0.601 ± 0.012** | 0.62 / 0.58 | 0.616 ± 0.037 (25) | 0.871 | 0.27 |
| htw TEST | fitted 44.74 | 0.488 ± 0.003 | 0.67 / 0.38 | 0.497 ± 0.012 (25) | 0.614 | 0.13 |
| htw TEST, threshold transferred from roi303 VAL | derived | 0.576 ± 0.010 | 0.56 / 0.60 | 0.594 ± 0.016 | 0.871 | — |
| alice TEST (6) | derived | 0.825 ± 0.035 | 0.79 / 0.86 | 0.841 ± 0.037 (6) | 0.984 | — |
| 586 TEST (11) | derived | 0.478 ± 0.016 | 0.42 / 0.56 | 0.495 ± 0.026 (6) | 0.867 | — |

Per-frame all-frames macro (for continuity with the older numbers only): roi303 derived
0.598 ± 0.008, fitted 0.625 ± 0.017; htw derived 0.575 ± 0.036, fitted 0.508 ± 0.030.

## What the table says

1. **The headline under the declared metric: roi303 TEST 0.641, htw TEST 0.601.** Seed spread on
   micro is 0.006 / 0.012 — the metric resolves a 0.02 effect on roi303, ~0.03 on htw.
2. **The instancer constant trades the two blocks against each other.** The rule-derived
   `min_length` 15 lifts htw by **+0.113** (its GT median is 29 px; the fitted 44.74 had deleted
   half the block before matching) and costs roi303 **−0.018 net**. The net hides two opposite
   moves, and the oracle row separates them: the oracle on roi303 ALSO rises under the derived
   constant (0.904 → 0.935), and a GT-rasterised mask has no fragments, so ~0.03 of real short
   roi303 curves had been deleted by 44.74 as well. Decomposed: recall **+0.04** (those real short
   curves come back; matches the oracle gain) and precision **−0.06** (the model's own fragments,
   which the fitted constant had been deleting before matching, now reach it). **The fragment
   cost of the model is therefore ~0.06 on roi303, not 0.018.** The constant stays derived (rule
   declared first; the fitted value is a fit to one block's length distribution, and the oracle
   says it is wrong on both blocks); the −0.06 is a measured cost of the model's fragmentation.
3. **The gap to the ceiling is 0.27–0.29 on both blocks, but NOT the same shape:** roi303 is
   P 0.59 / R 0.71 (loses on precision), htw is P 0.62 / R 0.58 (loses on recall, consistent with
   its anatomy: 80 % curve coverage is barely reachable). Two blocks, two weak parts. On htw the
   transferred threshold (from roi303 VAL) costs 0.025 — the calibration does not transfer
   perfectly, but that is a tenth of the gap.
4. **Anatomy on TEST (write-up only; `step1_anat_read.py`) reproduces VAL:** pixel precision at
   the threshold carrying 80 % of curves = roi303 **0.785 ± 0.017** (VAL 0.803), htw **0.40**
   (VAL 0.49; one seed never reaches 80 % coverage at any threshold), alice 0.965. Per-curve
   minimum probability median 0.70 (roi303) / 0.75 (htw) with 34 % of curves below 0.5 on both.
   The 586 frames never reach 80 % coverage at ≥95 % length at any threshold.

## Step 2: where the 0.28 goes — FP / FN decomposition on VAL, same instrument (06:20, `runs/step2/`)

`diag_fp_anatomy.py --cov 0.80` and `diag_failure_split.py --cov 0.80 --no-fov` (both given a
`--cov` flag; default 0.95 stays bit-identical to every earlier read), derived constant, each
seed's VAL-chosen threshold. SPARSE35 shown; SEED2/SEED3 agree within a few points.

**roi303 VAL (139 frames, tp 5212 / fp 3304 / fn 2167):**

| unmatched PREDICTIONS (FP) | by count | by length | median len (eval px) |
|---|---|---|---|
| fragment — lies on ONE annotated curve, covers < 80 % of it | 35 % | 32 % | 108 |
| junction — touches ≥ 2 annotated curves | 14 % | 30 % | 302 |
| gap_bridge — spans two annotated pieces end-to-end | 4 % | 10 % | 298 |
| overshoot — covers a curve ≥ 80 % but runs past it | 5 % | 6 % | 149 |
| **off_other — < 50 % of its length near ANY annotation** | **39 %** | 18 % | **37** (40 % shorter than 30 px) |

| unmatched GROUND TRUTH (FN) | share |
|---|---|
| **covered_split — ≥ 80 % of the curve IS covered by predicted centerlines, but by more than one piece** | **65 %** |
| partial — 50–80 % covered | 18 % |
| missed — < 50 % covered | 17 % |

Fitted → derived on the same maps: off_other 576 → 1281 (the +705 are almost all < 60 px: the
short off-annotation blobs 44.74 had deleted), fragment 835 → 1145. So the 0.06 precision cost of
the derived constant is half short off-annotation blobs, half real-filament fragments.

**htw VAL (33 frames, tp 453 / fp 196 / fn 345):** FP = off_other 52 % (median 29 px), fragment
24 %, gap_bridge 9 %; **FN = missed 64 %**, covered_split 25 %, partial 11 %. Failure-split:
21 % of GT has no matching prediction at all, and 81 % of the excess of too-long predictions
lies on unannotated pixels (the 2026-09-0x "82 % on unannotated" finding reproduces at the new
constant).

**In domain, same instrument (`data/synth_eval_s35`, 48 exact-label frames, SPARSE35 native
dump, thr 0.98, derived):** FP = junction 149 / fragment 144 / gap_bridge 32 / **off_other 30
(8 %)**; FN = **covered_split 404 (80 %)** / partial 63 / missed 35. So the short off-annotation
blobs are a REAL-data defect (8 % of FP in domain vs 39 % on roi303), splits outnumber chimeras
on real (fragment : junction = 2.6 : 1) but not in domain (1 : 1), and the covered-but-split
class is the dominant FN on both — a crossing problem the model has in domain as well.

**Named, per block:**
- **roi303 loses to FRAGMENTATION.** Two thirds of its FN are curves the map covers but in
  pieces; 62 % of its FP length is pieces of annotated filaments (fragment + junction). The
  break is along the filament, at the places the campaign already located (crossings, 2–6 px
  parallels; on real data 97 % of crossing failures are a broken MASK, not the assembler).
  Fixing only covered_split would move roi303 recall 0.71 → ~0.90 and remove most fragment FP.
- **htw loses to DETECTION of short faint filaments.** Two thirds of its FN are not covered
  even to 50 % at the operating threshold (0.995; lower thresholds buy coverage but drown in
  short off-annotation blobs). Its FP are mostly those blobs.
- **Common to both:** the map's response is not CONTIGUOUS along a real filament and not SILENT
  off it — short pieces on, short pieces off. Both are a failure of along-filament consistency,
  not of per-pixel evidence (the per-pixel map covers 85 % of curves at 0.50).

## Step 2b: the lift-over pre-screen (06:40, `scripts/diag_crossing_contrast.py`, `runs/step2/cc_*.json`)

Hypothesis tested for CPU-minutes, model-free first: in IRM the filament that climbs OVER the other
at a crossing is lifted over a ramp, so its contrast should fade / flip over 5–20 px on either side,
and the generator (which composites filaments at their own heights) would not render that. Instrument:
perpendicular-max |I − bg| along every GT filament as a ratio to the filament's own median ≥ 40 px from
any crossing, binned by distance to the crossing, with pseudo-crossings on the same filaments as the
null; the WEAKER of the two filaments per crossing; the SIGN flip rate; and the model's probability in
the same bins.

| block | crossings | weaker-of-pair ratio p50, 5–20 px (null) | sign flips 5–20 px (null) | model p < 0.98, 5–20 px (null on plain stretches) |
|---|---|---|---|---|
| **synth** (exact labels) | 483 | 0.85–0.87 (1.01) | 0.07–0.09 (0.03) | **0.02–0.03 (0.01)** |
| **roi303 VAL** | 1904 | 0.85–0.86 (1.00) | 0.08–0.10 (0.04–0.05) | **0.27–0.30 (0.15)** |
| 586 VAL | 323 | 0.68–0.77 (1.03) | **0.20–0.25 (0.02–0.03)** | **0.17–0.23 (0.03)** |
| htw VAL | 26 | 0.79–0.96 | 0.02–0.07 | 0.09–0.11 |

Reading: **on roi303 the image near a crossing looks the same as the render on this instrument** — the
weaker-of-pair dip (0.85) is the min-of-two selection effect and synth reproduces it exactly; the flip
excess (+5 pp) is the same size on both. So the lift-over fade is NOT a generator omission on roi303;
the hypothesis dies there. On 586 there IS an image-level excess of sign flips near crossings
(20–25 % vs 8 % in the render) — a bright-regime block; kept as a note, not a lever (586 is 11 VAL
frames of 69 % model-drawn GT). **The model, meanwhile, erodes near real crossings beyond its own
plain-stretch deficit** (roi303 0.28 vs 0.15; 586 0.20 vs 0.03; synth 0.03 vs 0.01), while the plain-
stretch deficit itself is the larger real-data fact: **15 % of samples on ordinary real filament
stretches sit below 0.98 against 1 % in domain.** Two model-side facts, neither explained by a
contrast-magnitude difference the generator could be asked to render.

## Step 2c: what curing each defect is WORTH, and whether the along-filament lane has a mechanism (07:15)

**Ceilings on the declared instrument** (`diag_ceiling_union.py`, roi303 VAL, SPARSE35 ep040 native map,
thr 0.98, derived constant, the model's own foreground KEPT, GT band painted in and p raised to 1 there):

| painted region | micro | P / R | tp / fp / fn | Δ vs baseline |
|---|---|---|---|---|
| none (baseline, same code path) | 0.6558 | 0.612 / 0.706 | 5212 / 3304 / 2167 | — |
| GT band within 20 px of a crossing | 0.6713 | 0.624 / 0.727 | 5361 / 3232 / 2018 | **+0.016** |
| GT band within 40 px of a crossing | 0.6852 | 0.640 / 0.737 | 5441 / 3061 / 1938 | **+0.029** |
| **GT band everywhere** (a PERFECT map on every annotated filament, model FP kept) | **0.7575** | 0.710 / 0.812 | 5992 / 2449 / 1387 | **+0.102** |
| oracle band alone (no model foreground) | 0.927 | — | — | +0.271 |

Reading: curing the crossing erosion is worth ≤ 0.03; curing the ENTIRE along-filament deficit — a
map that is perfect on every annotated filament — is worth 0.10. **The other 0.17 of the 0.28 gap is
the model's foreground OFF the annotation**: with a perfect band it still costs 2449 FP and, through
merges and overshoots, 1387 FN (recall 0.81 with a perfect band). Precision, not continuity, is the
larger half on roi303.

**Along-filament pre-screen — DEAD** (`diag_line_integral.py`, roi303 VAL all 139 frames, 695 456
plain-stretch samples, 35 483 weak (5.1 %), 74 581 samples on 1 322 off-annotation predictions,
median blob length 25 px):

| statistic (raw image) | AUC weak vs blob | AUC strong vs blob | recall of weak at 80 % blob rejection | AUC weak vs displaced 5–8 px |
|---|---|---|---|---|
| network p | 0.012 (by construction) | 0.798 | 0.000 | 0.950 |
| pixel contrast | 0.335 | 0.458 | 0.038 | 0.515 |
| coherent line integral, L = 20 / 40 | 0.324 / 0.345 | 0.477 / 0.496 | 0.041 / 0.051 | 0.573 / 0.593 |
| incoherent, L = 20 / 40 | 0.339 / 0.349 | 0.460 / 0.466 | 0.039 / 0.043 | 0.535 / 0.542 |

Two facts. (1) **The weak stretches carry no raw-image evidence at 10–40 px scale**: against
laterally displaced background they are at AUC 0.52–0.59 (pixel 0.515), and integrating 40 px along
the tangent adds 0.08 — the annotator drew through stretches that are invisible in this frame. Kill
line (≥ 20 % recovered at 80 % blob rejection) missed by 4×. In domain the same holds (AUC 0.55–0.62,
0.9 % weak). (2) **The model's off-annotation blobs have the contrast of STRONG annotated filament
stretches** (AUC strong-vs-blob 0.46–0.50 on every statistic): they are filament-like ridges in the
image — unannotated filaments or filament-shaped debris — not noise hallucinations, and no along-axis
image statistic separates them from annotated filaments.

## Step 2d: what the off-annotation foreground is (08:40, montage + `diag_offgt_classes.py`)

Montage: `data/enc_sensitivity_testset/step2_offgt/offgt_fp_{short,long}.png` (raw crops, GT green,
prediction red). Four things recur: the **field-stop edge** (bright saturated surround of the
aperture; 55 % of roi303 VAL frames have one, 82 % of 586, none of htw/alice), **short bright rods**
(unannotated short filaments / seeds), **defocused particles**, **dark-polarity filaments skipped or
partly drawn**. Machine count over all 1322 off-annotation predictions at the operating point:

| class | by count | by length | snr p50 (annotated curves: 3.3) |
|---|---|---|---|
| field-stop edge (≥ 50 % of points within 10 px of the aperture boundary) | 6.5 % | **13 %** | — |
| not edge, < 30 px | 37 % | 9 % | 3.9 |
| not edge, 30–60 px | 25 % | 12 % | 3.2 |
| not edge, 60–120 px | 17 % | 17 % | 2.9 |
| not edge, ≥ 120 px | 15 % | **49 %** | 3.0 |

63 % by count / 57 % by length of the non-edge predictions are at least as bright as the
25th-percentile annotated curve; frames with a stop hold 79 % of all off-annotation predictions.
**With the edge rule at 25 px** (the montage showed edge predictions running 10–25 px inside the
aperture and along saturated frame borders): field-stop edge **12.9 % by count / 23.1 % by
length**; ≥ 120 px non-edge 12 % / 41 %. **Larger montage of the ≥ 120 px class**
(`offgt_fp_long36.png`, 27 crops, 160 px): ~13 are aperture edges or saturated frame borders,
~10 are clearly visible filaments the annotators did not draw or drew only in part (bright wavy
filaments, dark-polarity filaments, continuations past an annotation's end), ~3 are the
interference fringes of large debris, 1 is nothing. So of the off-annotation LENGTH: ≈ 25–30 %
aperture/border (a model target — ARM 50), ≈ 30–35 % unannotated real filaments (annotation
convention — not a model target), ≈ 10 % short bright rods / seeds (convention), ≈ 10–15 %
particles and debris fringes (model target, small), the rest faint and ambiguous.
**The evaluation-side field-of-view mask is not a fix:** re-scoring the same map at 0.98 WITH
`fov_mask` (4 px erosion, the MT-34 convention) moves micro **0.6558 → 0.6575** (FP 3304 → 3264):
the edge predictions run inside the aperture along the transition, where the mask does not reach,
and eroding further deletes ground truth (fov.py docstring). **SPARSE35 and every arm were trained
with `--fov-prob 0`** — the field-stop augmentation in
`train_v5.py` was never switched on. ARM 50 (FOV50) is declared on that fact (TODO).

## Step 2d′: the non-edge classes split by how much of each prediction touches an annotation (08:05)

The "unannotated real filament" reading of the long montage mixed two things: predictions that lie
entirely off the annotation (the annotator drew nothing there) and predictions that lie 10–50 % on
an annotation and continue (the annotator stopped early, OR the model ran onto a neighbour — the
montage cannot tell). From the stored classifier rows (25-px edge rule, 1 322 predictions,
111 872 eval px), share of off-annotation LENGTH:

| class (not field-stop edge) | on_gt < 0.10 | on_gt 0.10–0.50 |
|---|---|---|
| < 60 px | 0.202 | 0.004 |
| 60–120 px | 0.135 | 0.022 |
| ≥ 120 px | **0.289** | 0.117 |
| field-stop edge (25 px rule) | 0.231 | |

Only the on_gt < 0.10 bucket is cleanly "convention" (nothing annotated anywhere near); 0.63 of the
off-annotation length. The ≥ 120 px on_gt 0.10–0.50 bucket (0.117) is the ambiguous one — its median
contrast (snr 3.1) is the same as the clean bucket's (2.9), so it is not debris, but whether it is an
annotation stopping early or a model merge onto a neighbour is a per-case call.

## Step 2f: is the 0.17 the benchmark's or the model's? Two more ceiling variants (advisor check, 08:05)

The 0.17 in Step 2c was labelled "model foreground off the annotation" by subtraction. But with a
perfect band painted in, 2 449 FP remain on roi303 while only 1 322 predictions are off-annotation —
so roughly 1 100 of the remaining FP sit ON annotated filaments: junction, gap-bridge and overshoot
chimeras the model's own foreground creates by bridging between annotated filaments (CLAUDE.md
bottlenecks 1 and 2 by definition). Two variants split the 0.17 into a model-owned and a
benchmark-owned part, same code path (`diag_ceiling_union.py --delete-far D`: model foreground
farther than D eval px from the GT band set to 0):

| variant | roi303 VAL micro (P / R) | htw VAL micro (P / R) |
|---|---|---|
| baseline (none) | 0.6558 (0.612 / 0.706) | 0.5697 (0.574 / 0.565) |
| off-annotation foreground deleted (none + delete-far 5) | **0.7489** (0.752 / 0.745) | 0.6169 (0.655 / 0.583) |
| GT band everywhere, model FP kept (all) | 0.7575 (0.710 / 0.812) | 0.7371 (0.738 / 0.736) |
| GT band everywhere AND off-annotation foreground deleted (all + delete-far 5) | **0.8602** (0.869 / 0.851), fp 947 / fn 1096 | **0.7962** (0.839 / 0.757) |
| oracle band alone, same code path (all + delete-far 0.01) | 0.8995 (0.917 / 0.883), fp 592 / fn 864 | 0.8435 (0.909 / 0.787) |
| oracle band, step-1 path (`run_oracle_eval.py`, for reference) | 0.927 | 0.873 (0.928 / 0.823) |

htw-only (the 33 htw VAL frames inside the block, the source the htw threshold is chosen on):
none 0.626 → none + delete 0.698 → all 0.711 → all + delete 0.792 → oracle same path 0.816
(P 0.927 / R 0.729: the oracle band itself loses 27 % of htw recall through the derived-constant
instancer). So on htw: detection 0.085, off-annotation 0.07–0.08, merging 0.024.

roi303 (139 frames): deleting the off-annotation foreground at baseline is worth **+0.093** on its
own, painting the band +0.102, and both together reach **0.860** (P 0.869 / R 0.851) — the
advisor's two lines were "≤ 0.85: model-owned prize ≥ 0.08, retract" and "≥ 0.90: framing stands";
it landed between. 947 FP and 1 096 FN remain with a perfect band and nothing off the annotation;
against the step-1 oracle (494 / 577) that is ≈ 450 FP + 520 FN of merging and thickening the
model's own near-filament foreground makes between annotated filaments — bottlenecks 1 and 2 —
worth **0.039** (same-path oracle 0.8995 − 0.860; both hand the instancer ONE channel). The step-1
oracle (0.927) is not a second estimate of the same thing: `run_oracle_eval.py --masks oracle`
passes the K = 6 ORIENTATION channels (`oracle_ori_channels`, a crossing pixel set in two channels)
to the same assembler, so **0.927 − 0.900 = +0.027 is what orientation channels buy the fixed
instancer on a perfect map**. The head that produces them (`--head amodal`, the trainer's default)
was dropped for `--head binary` on the 2026-08 factorial (binary +0.021, n.s., old metric, old VAL)
and has never been scored under the declared metric at the derived constant — an existing-code
arm with a priced ceiling of +0.027 on a perfect map and an unknown value on the real one.

**The decomposition, roi303 VAL, both owners named** (gap to the same-path oracle 0.244; to the
step-1 oracle 0.271):

| part | size | how measured | owner |
|---|---|---|---|
| model foreground near annotated filaments: merging / thickening (class split in Step 2f′) | **0.039** | same-path oracle − (all + delete-far 5) | **model** |
| orientation channels on a perfect map (K = 6 amodal head vs one channel, fixed instancer) | +0.027 | step-1 oracle − same-path oracle | model head — never scored under the declared metric |
| crossing erosion of the map | ≤ 0.03 | cross40 painting | model |
| evidence-free stretches (no ridge within the metric's tolerance, Step 2g) | ≈ 0.07 | all − cross40 | single-frame limit / annotation |
| off-annotation: field-stop edge + saturated borders (0.23 of off-annotation length) | ≈ 0.02 | classifier share × 0.093 | model — ARM 50 |
| off-annotation: unannotated filaments and seeds, on_gt < 0.10 (0.63 of length, minus particles) | ≈ 0.05 | classifier share × 0.093 | annotation convention |
| off-annotation: particles / debris fringes (eye-estimated 0.10–0.15 of length) | ≈ 0.01 | montage estimate | model, small |
| off-annotation: long, 0.10–0.50 on an annotation (0.12 of length) | ≈ 0.01 | classifier share | ambiguous |

Model-owned total **≈ 0.11–0.13** (with the orientation-channel item counted); convention / single-frame total **≈ 0.12–0.13**. STEP 2e's
"model-side prize ≈ 0.05" is RETRACTED: it is about half the gap, and the largest single
model-owned item is the one STEP 2e had no row for — the model's own foreground bridging
annotated filaments at crossings and close parallels. A single-frame model that fixed every
model-owned item would sit near **0.77–0.80** on roi303 VAL as annotated; 0.90 still needs the
convention items. The off-annotation shares above are length shares of the classifier applied
to an F1 delta, i.e. a proxy, not a per-class ceiling; the residual-anatomy run
(`fp_*_ceilresid.json`) names the merging term's classes.

The block first (49 frames, minutes): painting the band is worth +0.167 (detection is htw's deficit, as
Step 2 said), deleting every off-annotation prediction on top of it +0.059 more, and the result is
still 0.80 with 183 FP and 306 FN left — those are merges and thickening the model's own
near-filament foreground makes between annotated filaments. Whatever the distance from 0.796 to the
same-path oracle band is, it is model-owned.

## Step 2f′: what the merging term IS, by class (residual anatomy under the ceiling minus the same-path oracle's, 08:40)

`diag_fp_anatomy.py` on the dumped maps of "band + nothing off the annotation" and of the
same-path oracle band, same threshold and constant (`scripts/anat_table.py`):

| htw-only (33 VAL) | tp / fp / fn | FP: fragment / junction / gap_bridge / overshoot | FN: covered_split / partial / missed |
|---|---|---|---|
| baseline | 453 / 196 / 345 | 47 / 12 / 17 / 15 (+102 off-annotation) | 86 / 37 / 222 |
| band + delete-far 5 | 573 / 76 / 225 | 32 / 16 / 24 / 4 | 97 / 17 / 111 |
| oracle band, same path | 582 / 46 / 216 | 5 / 11 / 29 / 1 | 93 / 2 / 121 |
| **residual − oracle** | −9 / +30 / +9 | **+27** / +5 / −5 / +3 | +4 / **+15** / −10 |

On htw the model-owned residual is not junction chimeras: it is **fragments** — with the annotation
band painted in and the model's own near-filament foreground kept, the instancer breaks filaments
that it traverses whole on the clean band (+27 fragment FP, +15 partial FN), and junction FP rise by
only 5. So the model's foreground within 5 px of a filament is lumpier / wider than the 2-px band
it was trained to produce, and the lumps spawn branches that the fixed instancer treats as ends.
That is a property of the map (crispness near the filament), not of the assembler — the same
assembler is clean on the band. The roi303 table follows when its run lands.

### Step 2f″: width of the map near filaments, and the same pair in domain (09:15)

Is the residual a WIDER / LUMPIER foreground than the 2-px band (`diag_mask_width.py`: run of
mask pixels through each GT centerline sample along the normal, at the operating threshold)?

| set (thr) | on-mask share of GT samples | model width p10 / p50 / p90 (eval px) | ≥ 6 px | lumpiness | band p50 / p90 |
|---|---|---|---|---|---|
| roi303 VAL (0.98) | 0.807 | 3.0 / 3.5 / 4.5 | 0.026 | 0.014 | 3.0 / 3.5 |
| htw VAL (0.995) | 0.584 | 2.5 / 3.5 / 4.0 | 0.005 | 0.007 | 3.5 / 3.5 |
| synth VAL (0.98) | 0.953 | 3.0 / 4.0 / 6.5 | 0.127 | 0.053 | 3.0 / 3.5 |

On real data at the operating threshold the mask is NOT wider or lumpier than the band (roi303
+0.5 px at p50, 2.6 % of samples ≥ 6 px); in domain it is (p90 6.5, 12.7 % ≥ 6 px). So "map
width" does not explain the real-data fragments; what remains is foreground within 5 px of the
band that is not part of the centerline run — side strands, crossing wedges, end extensions —
located by `diag_break_locus.py` (next).

The same ceiling pair in domain (synth VAL, 48 frames, SPARSE35 at 0.98):

| variant | synth VAL micro |
|---|---|
| baseline | 0.617 |
| band painted | 0.665 |
| band + delete-far 5 | 0.678 |
| oracle band, same path | **0.747** |

Two things. The in-domain oracle is only 0.747: the synthetic VAL holds configurations the
derived-constant instancer cannot assemble even from a perfect band (the generator is harder than
roi303's 0.900), which is the known in-domain fragmentation. And the residual − oracle delta in
domain is **0.069**, larger than roi303's 0.039 — the model's near-filament foreground breaks the
assembler on any ribbon, real noise texture is not what makes it, so the lever is on the map's
representation / training target, not generator realism.

### Step 2f‴: where the residual fragments break, and what the map looks like there (09:30)

`diag_break_locus.py` (ends of fragment predictions under the "band + nothing off" ceiling,
htw-only, 35 fragments / 70 ends): at a GT endpoint 0.586 · near a crossing 0.029 · near ANOTHER
filament (≤ 8 px) 0.057 · **mid-filament 0.329**. The oracle band has 4 fragments. So the breaks
are not at junctions and not at close parallels; they sit on plain stretches.

The montage of those mid-filament break ends (`data/enc_sensitivity_testset/step2_offgt/
brk_htw_montage.png`, `viz_break_points.py`: raw + GT + predictions | the map the assembler saw |
the model's original p) shows the mechanism, and it is the CEILING'S, not the model's: the
painted 2-px annotation band runs 1–3 px BESIDE the model's own ridge — the annotator's polyline
and the image ridge disagree laterally by an amount the metric tolerates (5 px) — so band ∪ model
foreground is a double track with ledges, and the assembler breaks on the ledge. The model's
original p in the same crops is a clean single ribbon. Consequence: the 0.039 "merging" term is an
UPPER bound that contains a construction artifact; the true model-owned merging between
annotated filaments is smaller and, on htw, close to zero. Re-measured with the painted band
widened to half-width 2.0 (swallows the offset): `ceil_*_hw2.0_*` — launched 09:50, with hw 3.3
(the metric's 5 eval px) as a second point and the K = 6 same-path oracle (TODO declaration 09:40).

roi303 at hw 1.0 (contaminated by the same artifact, recorded for the delta's shape only):
residual − oracle = FP +355 (fragment +227, **junction +78**, off_other +31, frag_overshoot +15)
and FN +232 (covered_split +204, partial +24). Fragment ends (565 residual fragments vs 323 on
the oracle band; the EXTRA 484 ends): at a GT endpoint 215 · mid-filament 142 · near another
filament (≤ 8 px) 89 · near a crossing 38. roi303 is denser than htw, so a fifth of the extra
breaks sit beside another filament and junction chimeras rise by 78 — whether those survive a
band wide enough to cover the annotation offset is exactly what the hw 2.0 / 3.3 rows decide.

### Step 2f⁗: the ceilings at matched band width — htw first (10:00, `ceil_hb_hw*`, `scripts/ceil_table.py`)

Declared read (TODO 09:40): (band + delete-far 5) − (oracle) at the SAME band half-width.

| htw-only (33 VAL, thr 0.995) | hw 1.0 (2 px) | hw 2.0 (4 px) | hw 3.3 (= metric tolerance) |
|---|---|---|---|
| baseline | 0.626 | 0.626 | 0.626 |
| none + delete-far 5 (off-annotation gone) | 0.698 | 0.698 | 0.703 |
| cross20 / cross40 painted | — | 0.626 / 0.630 | 0.622 / 0.629 |
| band painted, model FP kept (all) | 0.711 | 0.739 | 0.735 |
| band + delete-far 5 | 0.792 | **0.821** | **0.822** |
| oracle band, same path | 0.816 | 0.812 | 0.805 |
| **(band + delete) − oracle** | +0.024 | **−0.009** | **−0.017** |

Once the painted band is wide enough to cover the annotation-vs-ridge offset, the model's
near-filament foreground costs NOTHING against the oracle band — it is slightly better than the
band alone (the model's ridge is where the image ridge is; the annotation is not). **The
"bridging / merging" item is dead on htw** (declared line ≤ 0.01: met with margin). The block
(49 frames) reads the same: hw 2.0 0.8205 vs 0.8198, hw 3.3 0.7918 vs 0.7798.

Two side readings. (1) The band-width cost on the instancer: the wide-band oracle drops (block
0.844 → 0.820 → 0.780; htw-only 0.816 → 0.812 → 0.805) — a 6–7 px ribbon merges crossings in the
assembler, which is why the hw-1.0 rows were used in the first place and why the artifact went
unnoticed. (2) Crossing erosion on htw ≈ 0.00 (few crossings); the detection ceiling on htw is
+0.11 (0.626 → 0.739 at hw 2.0), not +0.085.

### Step 2f⁗ (cont.): roi303 at matched band width, and the decomposition that stands (13:35)

| roi303 VAL (139, thr 0.98) | hw 1.0 (2 px) | hw 2.0 (4 px) | hw 3.3 (= tolerance) |
|---|---|---|---|
| baseline | 0.656 | 0.656 | 0.656 |
| none + delete-far 5 (off-annotation gone) | 0.749 | 0.747 | 0.744 |
| cross20 / cross40 painted | 0.671 / 0.685 | 0.670 / 0.685 | 0.660 / 0.677 |
| band painted, model FP kept (all) | 0.758 | 0.762 | 0.748 |
| band + delete-far 5 | 0.860 | 0.864 | 0.846 |
| oracle band, same path, 1 channel | 0.900 | 0.881 | 0.858 |
| **(band + delete) − oracle** | +0.039 | **+0.017** | **+0.012** |
| oracle band with K = 6 channels (hw 1.0) | **0.927** (6802 / 494 / 577 = the step-1 oracle, bit-identical) | | |

Declared lines (TODO 09:40): ≤ 0.01 dead, ≥ 0.03 real. roi303 lands at 0.012–0.017: **small, and
not dead by the letter of the rule** — read as "the model's near-filament foreground costs at most
0.01–0.02 against a perfect band on roi303, ≈ 0 on htw". The wide band cannot remove the offset
artifact completely without merging crossings (the oracle itself falls 0.900 → 0.881 → 0.858 with
band width), so 0.012 at hw 3.3 is the tightest upper bound this instrument gives. The 08:30
"bridging 0.04–0.07 is the largest model-owned item" is withdrawn for good; the 08:00 estimate was
the closer one.

The other rows are stable under band width, so they stand: off-annotation foreground **+0.093**;
crossing erosion **+0.014 / +0.030** (20 / 40 px); detection ceiling **+0.107** (of which
evidence-free stretches ≈ 0.07, Step 2g); orientation channels on a perfect map **+0.027**
(isolated: the K = 6 same-path oracle reproduces the step-1 oracle to the count).

**Decomposition of the 0.27 gap on roi303 VAL that this round leaves standing:**

| part | size | owner |
|---|---|---|
| evidence-free stretches (no ridge within the metric's tolerance) | ≈ 0.07 | single frame / annotation |
| unannotated filaments and seeds off the annotation | ≈ 0.05 | annotation convention |
| crossing erosion of the map | ≤ 0.03 | model |
| orientation channels (K = 6) on a perfect map | (≤ 0.027 on the oracle) | **struck: ARM 51 negative** (−0.088 roi303, −0.16 htw) |
| field-stop edge + saturated borders | ≈ 0.02 | model — **ARM 50 negative** (edge share unchanged) |
| near-filament foreground vs a perfect band ("merging") | 0.01–0.02 | model, small |
| particles / debris; ambiguous long predictions | ≈ 0.02 | model, small / ambiguous |

Model-owned ≈ 0.06–0.07 (the channel item struck after ARM 51); convention / single-frame
≈ 0.12–0.13. A single-frame model that bought every model-owned item sits near 0.74–0.76 on
roi303 VAL as annotated.

## ARM 50 — FOV50 read (13:35, `scripts/read_arm.py`, VAL only, ep040 native, own thresholds)

| | SPARSE35 | FOV50 | Δ | paired (frames ≥ 10 GT) |
|---|---|---|---|---|
| roi303 VAL micro (thr 0.98 / 0.98) | 0.6558 (P 0.612 / R 0.706) | 0.6518 (P 0.604 / R 0.708) | **−0.004** | mean −0.011, better 36 / worse 67, **p 0.047** |
| roi303 precision at matched recall | 0.612 | 0.590 (node 0.97) | **−0.022** | |
| htw-only VAL micro (thr 0.995 / 0.99) | 0.6261 | 0.6076 | −0.019 | p 0.19; 0.608 ≥ SEED2 0.588 → inside seed spread |
| block / alice / 586 | 0.570 / 0.727 / 0.430 | 0.555 / 0.705 / 0.429 | −0.015 / −0.021 / −0.001 | |

Headline: null on the pooled line (−0.004, inside −0.010) but the paired read is a significant
small NEGATIVE and precision at matched recall is lower by 0.022 — the "precision not lower"
gate FAILS. htw passes the amended gate.

**Mechanism gate (13:40, `diag_offgt_classes.py` on the FOV50 roi303 VAL dump at its own threshold):**

| | SPARSE35 | FOV50 | gate |
|---|---|---|---|
| off-annotation predictions / length | 1 322 / 111 872 px | 1 436 / 112 422 px | |
| field-stop edge share by length, 10 px rule | 0.130 | **0.113** | ≤ 0.05 — **FAIL** |
| field-stop edge share by length, 25 px rule | 0.231 | **0.229** | ≤ 0.09 — **FAIL** |

The synthetic field stop in training (`--fov-prob 0.5`, surround 1.3–5×) did not move the
model's response to the real aperture edge: the edge share is unchanged within noise and the
off-annotation count rose. **ARM 50 verdict: NEGATIVE** — headline null-to-small-negative
(paired p 0.047, precision at matched recall −0.022), mechanism absent. The synthetic edge
(a hard circular boundary with a uniform bright surround) is evidently not what the model sees
at the real aperture; what the real edge looks like at the pixel level (gradient width,
fringes, the saturated border) was never measured, and that would be the prerequisite for a
second attempt — not another launch.

## ARM 51 — AMODAL read (15:25, `scripts/read_arm.py`, VAL only, ep040 native, own thresholds)

| | SPARSE35 | AMODAL | Δ | paired (frames ≥ 10 GT) |
|---|---|---|---|---|
| roi303 VAL micro (thr 0.98 / 0.97) | 0.6558 (P 0.612 / R 0.706) | 0.5675 (P 0.495 / R 0.664) | **−0.088** | better 18 / worse 88, p 1.4e-13 |
| roi303 precision at matched recall | 0.612 | 0.482 (node 0.96) | −0.130 | |
| htw-only VAL micro (thr 0.995 / 0.94) | 0.6261 | 0.4657 | **−0.160** | better 2 / worse 24, p 1.3e-5 |
| alice / 586 | 0.727 / 0.430 | 0.727 / 0.433 | +0.000 / +0.003 | |

Decisive negative on both test-relevant blocks, null on alice and 586. The trainer's own 16-frame
selection VAL (alice/586-heavy) had AMODAL ahead (0.569 vs 0.485 at ep040), which is exactly why
the 2026-08 factorial (old VAL, old metric) called the head "exonerated": the orientation-keyed head
fails on roi303 and htw specifically. The amodal grid is lower everywhere on roi303 (optimum
0.567 at 0.97, 0.40 at 0.995 where binary still holds 0.60), i.e. the max over six sigmoid
orientation channels is a less confident, noisier foreground on dense 16-bit frames. Traps
checked: same derived constant, ep040 by md5, native scale, VAL-chosen threshold, and eval decodes
max over the six orientation channels exactly as the trainer's validate does.

Two follow-up reads, diagnostic only: the FP/FN anatomy of the amodal map at 0.97 (what the −0.088
is made of), and the same model decoded from its UNION channel (`--fg-head`) — if the union matches
SPARSE35, the multi-task training left the map intact and the channel decoding is what fails; if
the union is also worse, the amodal objective damaged the map itself.

**Follow-up reads (16:10).** Union channel, htw-only: 0.482 (vs ori-max 0.466, SPARSE35 0.626) — the
union is also worse, so the amodal objective damaged the map itself. Anatomy on roi303 VAL (AMODAL
0.97 vs SPARSE35 0.98): off-annotation FP 2 629 vs 1 281, off_border 111 vs 37, overshoot 311 vs
173, fragment 1 211 vs 1 145, junction 435 vs 447 (unchanged); FN covered_split 1 609 vs 1 411,
missed 465 vs 369. The orientation-keyed head over-fires off the annotation — the failure mode of
the 2026-07 WEAVE/amodal models — and buys nothing at crossings even at the class level. **ARM 51
closed, negative.** In the standing decomposition the "orientation channels ≤ 0.027" row is struck
as a model-side item: the K = 6 ceiling is a fact about the instancer on perfect channels, not
about a head this recipe can train. Model-owned total ≈ 0.06–0.07.
roi303 union channel (17:20): 0.617 vs ori-max 0.568 vs SPARSE35 0.656 — the map itself is −0.039
and the channel decoding costs a further −0.050 on roi303; on htw the map alone is −0.14.

## ARM 52 — PAINT_AA read (2026-09-17 01:20) and what the rotation probe taught

Background: the research round's one survivor was a renderer defect (`_paint` rounds 0.5-px offsets,
so axis-aligned filaments render 20–40 % brighter; invariant under rot90/flip). The static angle
pre-screen missed it on roi303 (57 % of filaments are already axis-aligned) but the ROTATION PROBE
(`diag_rotation_probe.py`: the same frames at 90° / ±45°, maps rotated back, read per original
tangent bin) showed the model IS grid-anisotropic on the primary block: oblique filaments rotated
onto the axes get 2.5× more weak points (0.045–0.050 → 0.122–0.126), horizontal ones rotated to
45° fewer (0.053 → 0.040), 90° lossless-identical.

| ARM 52 = SPARSE35 + `--paint-aa` (isotropic stamp, gain 1.0574) | SPARSE35 | PAINTAA | Δ | paired |
|---|---|---|---|---|
| roi303 VAL micro (thr 0.98 / 0.99) | 0.6558 (P 0.612 / R 0.706) | 0.6663 (P 0.630 / R 0.708) | **+0.0105** | better 53 / worse 49, **p 0.45** |
| roi303 precision at matched recall | 0.612 | 0.630 | +0.018 | |
| htw-only VAL micro (0.995 / 0.99) | 0.6261 | 0.6117 | −0.014 | p 1.0; ≥ SEED2 0.588 |
| alice / 586 / block | 0.727 / 0.430 / 0.570 | 0.785 / 0.451 / 0.573 | +0.059 / +0.022 / +0.003 | |
| **mechanism gate** (probe on the new checkpoint) | oblique→axes +0.077 | **+0.061** (2.8×) | not < half | **FAIL** |
| renderer marginals (mean contrast on mask) | 0.0647 | 0.0664 | **+2.6 %** (> 2 %) | breached |

Verdict: **null** on the declared read (the pooled +0.0105 crosses the line by 0.0005 but the
paired test does not; 1.7 seed-sd), the mechanism gate fails, and the renderer carries 2.6 % more
mean contrast. The defect is real and stays fixed behind a flag (default OFF); it is not the cause
of the model's grid anisotropy. The next rot90-symmetric candidate is the TRAINING TARGET: the 3×3
square dilation of the rasterised centerline supervises 3.03 target px per unit length at 0/90° but
3.35–4.02 at 15–75° (`tests/test_target_band.py`); ARM 53 (`--target-band euclid`, exact 1.5 px
Euclidean band, 3.00–3.03 at every angle) is training with the same falsifiers.

## ARM 53 — TARGET_EUCLID read (2026-09-17 07:05)

| ARM 53 = SPARSE35 + `--target-band euclid` | SPARSE35 | EUCLID | Δ | paired |
|---|---|---|---|---|
| roi303 VAL micro (thr 0.98 / 0.98) | 0.6558 (P 0.612 / R 0.706) | 0.6563 (P 0.611 / R 0.710) | **+0.0005** | better 50 / worse 53, p 0.62 |
| htw-only VAL micro (0.995 / 0.99) | 0.6261 | 0.5816 | **−0.045** | p 0.065; < SEED2 0.588 → gate fails |
| alice / 586 / block | 0.727 / 0.430 / 0.570 | 0.783 / 0.469 / 0.561 | +0.057 / +0.039 / −0.009 | |
| mechanism (probe, own thr): oblique → axes | +0.077 | +0.10 (all bins +0.06–0.10) | relative swap 0.09 → 0.04 | gate FAILS on the letter |
| mask width p50 / p90 (roi303, own thr) | 3.5 / 4.5 | 3.0 / 3.5 | thinner, as intended | |

Null on the primary, negative-leaning on htw, and the mechanism gate fails: the isotropic target
halves the relative grid swap but makes the ridge detector brittle under bilinear resampling
(overall weak share at ±45° 0.124 vs SPARSE35's 0.064). Together with ARM 52: neither the renderer
nor the target band explains the model's grid anisotropy alone. What remains rot90-symmetric is
the network itself under an augmentation group that contains only rot90 and flips — the model has
never seen off-axis resampled content in training, and every probe at ±45° says it cannot handle
it. `--target-band` stays default square.

## ARM 54 — ROTAUG read (2026-09-17 21:55) and the close of the grid-anisotropy lane

| ARM 54 = SPARSE35 + `--rot-aug-prob 0.5` | SPARSE35 | ROTAUG | Δ | paired |
|---|---|---|---|---|
| roi303 VAL micro (thr 0.98 / 0.98) | 0.6558 (P 0.612 / R 0.706) | 0.6572 (P 0.613 / R 0.709) | **+0.0014** | better 42 / worse 60, p 0.35 |
| htw-only VAL micro (0.995 / 0.98) | 0.6261 | 0.6020 | −0.024 | p 0.53; ≥ SEED2 0.588 |
| alice / 586 / block | 0.727 / 0.430 / 0.570 | 0.748 / 0.466 / 0.569 | +0.021 / +0.036 / −0.000 | |
| mechanism 1: oblique → axes loss (own thr) | +0.077 | **+0.041** | relative swap halved | fails the "< half" line by 0.003 |
| mechanism 2: ±45° overall penalty | +0.013 | **+0.013** | unchanged | fails |
| mask width p50 / p90 (roi303) | 3.5 / 4.5 | 4.0 / 5.0 | thicker | |

Null. The augmentation halves the grid swap but leaves the resampling penalty untouched and moves
F1 by nothing. **Lane closed after three arms:** the model's grid anisotropy is real (rotation
probe, 2.5–2.8× on the primary block) and partially removable by each of the three isotropy fixes,
but it is not a lever for the declared metric — on real filaments the weak points are dominated by
evidence-free stretches and annotation convention, which have no orientation. The three flags stay
OFF; the code, tests and the probe remain.

**Where this leaves the model-side space (2026-09-17):** five arms read in 36 hours under the
declared instrument — ARM 50 field stop (negative), ARM 51 amodal head (negative), ARM 52 isotropic
renderer (null), ARM 53 isotropic target (null), ARM 54 rotation augmentation (null) — plus the
research round's 12 refuted proposals. Every model-side item of the standing decomposition is ≤ 0.03
and none of the mechanism-backed levers tried moved it. The reachable target for "improve the
model" on roi303 as annotated stays the single-frame bound 0.74–0.76; the distance to 0.90 is the
benchmark's (evidence-free stretches ≈ 0.07, unannotated filaments ≈ 0.05, a split with
byte-identical and sibling frames across VAL/TEST) and is decided by ruling, not training.

## ARM 55 — pos_weight 2 / 4 read (2026-09-18 20:55): the last knob with a measured hint

| branch (own thr) | roi303 VAL micro | paired | P at matched R | htw-only | fragment FP / covered_split FN / off-annotation FP (SPARSE35 1 145 / 1 411 / 1 318) |
|---|---|---|---|---|---|
| PW2S, pw 2 (0.99) | 0.6509 vs 0.6558 (−0.005) | worse 62 / better 41, p 0.10 | −0.018 | 0.597 vs 0.626 (−0.029), p 0.22 | **1 317** / 1 394 / 1 152 |
| PW4S, pw 4 (0.99) | 0.6593 vs 0.6558 (+0.0035) | worse 57 / better 46, p 0.31 | +0.004 | 0.594 vs 0.626 (−0.032), p 0.17 | 1 122 / 1 428 / 1 303 |

Both null, both fail the mechanism gate (fragment and covered_split had to fall ≥ 10 %): at a
per-model threshold the fragmentation is unchanged (pw 4) or worse (pw 2, +15 %, with a thinner
mask and 13 % fewer off-annotation FP). The August hint (cc/gt falling with the weight at ONE
shared threshold) was the threshold-calibration trap: a lower positive weight lowers every
probability, so at a shared threshold the pw-2 map carries less foreground and fewer
low-confidence blobs. cc/gt counted vanishing blobs, not fewer breaks on filaments; a thinner
mask breaks more, which is what the +15 % fragments at its own threshold show. `pos_weight`
stays 8. The knobs never tried (learning rate, Dice weight, deep-supervision depth) carry no
mechanism reading and fall under the stopping rule.

**Correction (21:45).** pw 2 was not untried. The same recipe with the same seed was trained
as PW2S on 2026-09-05 and read on 2026-09-06 under the previous primary (strict 0.95/0.95
macro): −0.058, resolvable; at its own threshold 0.70 the map was more connected (fragments
1 632 → 1 056, MAP_BREAK 589 → 274) and the gain was absorbed by chimeras (328 → 675),
instancer cuts (71 → 351) and off-annotation FP (+524). The 2026-09-17 launcher wrote into the
same run directory and overwrote those checkpoints (all `ep*.pth` dated 2026-09-17/18; the
roi303 dump files 2026-09-18 00:13, so the read above is of the re-trained weights). ARM 55's
new content is the declared-metric read of pw 2 and the first read of pw 4. Both reads of pw 2
agree: the map is under-confident (foreground 1.70 % at 0.70 vs SPARSE35's 0.98 % at 0.98),
its extra low-threshold foreground buys no recall under the declared metric (micro R 0.696 at
0.70 vs 0.697 at 0.99, micro F1 monotone 0.582 → 0.651 to 0.99), and at the optimum the
declaration picks the mask is thin and fragments more.

**Round closed (2026-09-18).** Six arms in 60 hours under the declared instrument — field stop
(negative), amodal head (negative), isotropic renderer, isotropic target, rotation augmentation,
pos_weight 2 / 4 (all null; pw 2 a same-seed re-train of the arm read 2026-09-06 under the
old metric) — and 12 refuted research proposals. No mechanism-backed lever moved
roi303 VAL micro beyond seed noise, every mechanism gate failed, and the standing decomposition
holds: model-owned ≈ 0.06–0.07, convention / single-frame ≈ 0.12–0.13, single-frame bound
0.74–0.76. SPARSE35 ep040 remains the model. What is left belongs to the benchmark.

## Step 2g: was "evidence-free" just polyline drift? The line-integral read at the metric's own tolerance (08:12)

Objection (advisor): the network's p and the contrast in Step 2c were read within ±2 px of a hand-
drawn polyline, but the metric tolerates 5 eval px because those polylines drift; a stretch drawn
3 px off the ridge reads "weak" on both instruments while the matcher counts it covered. Re-run
with the perpendicular windows at the tolerance (`--perp 3.3` native = 5 eval px for the contrast,
`--net-window 5` eval px for the network), same samples, same negatives:

| window | weak share (p < 0.98 on plain stretches) | AUC weak vs displaced, pixel / coh 20 / coh 40 | recall of weak at 80 % blob rejection, best statistic |
|---|---|---|---|
| ±2 px (Step 2c) | 5.1 % | 0.515 / 0.573 / 0.593 | 0.051 |
| ±5 eval px / 3.3 native (the metric's tolerance) | **4.4 %** | 0.507 / 0.561 / 0.582 | 0.052 |

The weak share does not collapse (4.4 %, not < 2 %), and the raw-image evidence does not appear
when the window is widened (AUC unchanged, all < 0.60; kill line 0.20 missed by 4× again). So the
weak stretches are not drift of the polyline: the annotator drew through image that carries no
ridge at this frame's noise, within the tolerance the metric grants. Declared reading (TODO STEP 2f):
"evidence-free stretch" stands, the along-filament lane stays closed, and the only model-side route
to those stretches remains information from outside the frame (time), which is a hypothesis, not a
measurement.

## Where this leaves the 0.90 goal on the declared benchmark (08:00)

| part of the 0.27 gap (roi303 VAL, SPARSE35) | size | owner |
|---|---|---|
| map not contiguous on annotated filaments, of which crossing erosion ≤ 0.03 and evidence-free stretches (no image signal at ≤ 40 px) the rest | ≤ 0.10 | model (crossings) / single-frame limit (evidence-free) |
| aperture edge + saturated frame borders | ≈ 0.04 | model — ARM 50 running |
| unannotated real filaments, short seeds | ≈ 0.07–0.08 | annotation convention |
| particles, debris fringes | ≈ 0.02 | model, small |
| merges/overshoots induced by the above with a perfect band | (inside the 0.17) | — |

~~A single-frame synthetic-trained model on this benchmark, as annotated, is bounded near
**0.70–0.74** on roi303 VAL micro; the remainder is the annotation's own convention and the
frame's own missing evidence.~~ **Superseded by Step 2f (08:30):** the table above missed the
model's own merging between annotated filaments (0.04–0.07); the single-frame bound is nearer
0.77–0.80 and the model-owned prize ≈ 0.11–0.14, about half the gap. The two decisions that move that bound are the benchmark's, not the
method's: (a) an annotation-completion pass on a few frames to measure the class-B tax properly
(requested on 2026-09-09, never done), or an IGNORE rule for predictions on visibly real
unannotated filaments; (b) whether evidence-free stretches drawn by the annotator count against a
single-frame model.

## Sensitivity: pooled micro along the whole grid (VAL | TEST, derived constant)

roi303 SPARSE35: 0.50 0.552|0.528 · 0.70 0.586|0.574 · 0.80 0.608|0.592 · 0.90 0.633|0.618 · 0.94 0.641|0.630 · 0.96 0.653|0.638 · 0.97 0.654|0.644 · **0.98 0.656|0.641** · 0.99 0.649|0.640 · 0.995 0.635|0.618 · 0.999 0.537|0.516
htw SPARSE35: 0.90 0.539|0.468 · 0.96 0.576|0.534 · 0.98 0.608|0.565 · 0.99 0.622|0.585 · **0.995 0.626|0.601** · 0.999 0.539|0.513

**These TEST rows are a sensitivity record only: no future threshold choice may cite them.**
VAL and TEST curves are parallel within ~0.02 on roi303 and ~0.05 on htw (TEST htw sits below
VAL htw at every threshold: the 33 TEST frames are simply harder). No selection on TEST would
have changed the operating point by more than one grid node.

## Exposure ledger after this read

roi303 TEST: 2 (09-08 pick read, this). MT-34+HTW TEST: 3. Both are re-reads of the same three
checkpoints; no new arm was scored on TEST.
