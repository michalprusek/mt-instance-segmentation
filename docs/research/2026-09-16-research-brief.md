# Research brief, 2026-09-16: one weak part, one exclusion list, three virgin lanes

Step 3 of the user's directive of 2026-09-15 ("identifikuj slabé části, udělej research a iteruj").
Steps 1–2 are in `2026-09-16-step1-results.md`; everything a proposal must respect is here.

## 1. The target, stated on the declared instrument

Metric: centerline-F1, tol 5 px (1.5× frame), 0.80/0.80, end_tol 0, pooled micro; instancer at
the derived constants; ep040; threshold chosen on VAL. Best method SPARSE35, 3 seeds:

| block | TEST micro | P / R | oracle | what is lost |
|---|---|---|---|---|
| roi303 (137 TEST, 139 VAL dev) | **0.641 ± 0.006** | 0.59 / 0.71 | 0.935 | **fragmentation**: 65 % of FN are curves the map covers ≥ 80 % but in pieces; 62 % of FP length is pieces of annotated filaments (fragment 32 %, junction chimera 30 %) |
| htw (33 TEST, 33 VAL) | **0.601 ± 0.012** | 0.62 / 0.58 | 0.871 | **detection**: 64 % of FN are < 50 % covered at the operating point; 52 % of FP are short off-annotation blobs (median 29 px) |

Goal: 0.90. Gap ≈ 0.28 on both blocks. Development iterates on roi303 VAL, paired per frame,
micro-resolved (seed sd 0.006); htw VAL is the mandatory second reading (sd 0.012).

## 2. The mechanism, as measured (model-side, real data)

1. **A broad confidence deficit along ordinary real filament stretches:** 15 % of on-filament
   samples sit below the operating threshold 0.98 on roi303, 1 % in domain, on image stretches
   whose contrast is unremarkable (`diag_crossing_contrast.py` null positions). Per curve: median
   minimum probability 0.70 on real vs 0.94 in domain; 34 % of real curves dip below 0.5.
2. **An extra, crossing-specific erosion on top of it:** within 5–20 px of a real crossing the
   below-threshold share doubles (roi303 0.28 vs 0.15; 586 0.20 vs 0.03) while it barely moves in
   domain (0.03 vs 0.01). The image contrast near crossings, measured with one instrument on both,
   is the SAME in the render and on roi303 (weaker-of-pair 0.85 both; flip excess +5 pp both) — so
   this is not a contrast fade the generator omits (lift-over hypothesis tested and dead on roi303,
   2026-09-16 06:40; a sign-flip excess survives on the bright 586 block only).
3. **Short off-annotation responses on real background:** 39 % of roi303 FP by count (median
   37 px, 40 % under 30 px), 8 % in domain. They set the operating threshold high, which is what
   turns the dips of (1)–(2) into breaks.
4. Consequence at the instancer: a broken arm turns a 4-arm junction into a 3-arm one, and 3-arm
   junctions resolve at 0.33 vs 0.94 (junction-arity finding); the assembler is innocent wherever the
   mask carries through (0.90–0.95 on real).

**Fragmentation is NOT absent in domain.** On the 48 exact-label synthetic frames the same model
at the same convention scores F1 ≈ 0.62 (recall ≈ 0.58, 502 FN of ~1200 curves, FP 361) — the same
as real — with covered_split at 80 % of FN, through a DIFFERENT mechanism: the mask is intact
(1 % below threshold) and the assembler makes chimeras at crossings and at the generator's
9–11×-over-produced close parallels (Screen A: 40.9 % of in-domain breaks on a crossing, 50.3 % on a
parallel). What is absent in domain is the confidence deficit (1 % vs 15 %) and the background blobs
(8 % vs 39 %). The in-domain figure is therefore neither the ceiling nor a clean comparator.

In one sentence: **on real data the network's response is neither contiguous along a filament nor
silent off it; those two defects are real-specific; the crossing-identity failure exists in domain
as well; and every image-statistic lever aimed at the real-specific gap has been null.**

**Ceilings on the declared instrument (`diag_ceiling_union.py`, roi303 VAL cached maps, 2026-09-16):**
paint the GT band into the model's map (a) within 20 px of crossings, (b) within 40 px, (c)
everywhere, keep the model's own FP, score through the unchanged instancer — see the table in
`2026-09-16-step1-results.md` §Step 2c once it lands. A proposal is priced against these numbers,
not against intuition.

## 3. Constraints every proposal must satisfy (standing rules, not preferences)

- The model itself must change (architecture, target, loss, or the generator's PHYSICS): every
  post-hoc map repair and every decoder variant is closed by measurement (TODO 476–485, 154–163,
  4199–4222), and the user asked to test the model, not a repair (2026-09-11).
- No training, pseudo-labelling or feature alignment on real frames; the only real corpus IS the
  benchmark (content-hash leakage, TODO 3433–3459).
- Generator priors are not fitted to real statistics (replicated 8/8 cost); physics mechanisms may
  be added and must be randomised wide.
- One lever at a time; a training-free pre-screen with a kill line before any GPU hour; a paired
  per-frame A/B on roi303 VAL at each arm's own swept threshold; the foreground-matched control
  (an arm must beat "the baseline with the threshold lowered to the same foreground"); micro is the
  resolving statistic; falsifiers declared before the run.
- Budget for one arm: 6 GPU-h (40 epochs nnU-Net ResEnc-M) + 1 CPU-h of sweeps.

## 4. Exclusion list — closed, with the ledger line that closed it

Levers below are CLOSED; a proposal that reduces to one of them is dead on arrival.

| family | what was run | verdict | TODO line |
|---|---|---|---|
| decoders / map repair / instancer re-tune | 11-variant decoder screen, hysteresis, closing, prune, Optuna re-tune, crossing-gated growth (XCOMP), curve-level accept, free-end growth bound | all negative / closed by pre-screen | 476–485, 154–163, 4199–4222, 2371–2393 |
| training target | THIN 1 px, prior bank (Frangi/LoG/structure tensor, PRIOR_THIN/THICK), HEAT existence+heatmap, multiplicity/overlap head (+ scrambled control), crossing-region weight (JWEIGHT) | null-to-negative; multiplicity gain = multi-task regularisation artefact | 171–180, 1258–1280, 3945–3973, 1847–1886 |
| loss | pos_weight 2 (negative), cbDice (negative), clDice (structural), worst-point softmin WP03/WP10 (mechanism fired, pure rescale) | closed | 1081, 5232, 5208, 3084–3148 |
| generator statistics | NOISE texture ×3 seeds/doses, amplitude (ARM 42/44/45), streak distractors (ARM 41, catastrophic), DENSE, wide/narrow physics priors, along-filament contrast (pre-screen), lift-over ramp (pre-screen 2026-09-16) | null / negative | 915–926, 3867–3889, 3917–3944, 3460–3484, 3149–3172 |
| augmentation / scale | AUG_PHOTO (negative at native), SCALE (null on F1, robustness only), 1.3× inference, support-matched normalisation | null | 4747–4754, 1452–1475, 300–304, 3773–3797 |
| emission | SETPRED (DETR curves, 0.013), CURVEQ (per-query masks, 0.000 and damages the trunk) | negative | 4931–4945, 2475–2499 |
| training length / trunk | 100 epochs (worse), frozen DINOv2 (null), learned arm-pairing (identical TP) | closed | 3511–3529, 6062–6074 |
| domain adaptation | image-level (= the generator arms above), feature-level MMD/adversarial (leakage) | closed | 3433–3459 |
| review-killed designs (never built, do not rebuild as written) | SENSOR-TWIN, MINRANK, SNR-RANK, MATCHFIELD, CHAINWALK, TopoBDA-as-CURVEQ | fatal ×2 each | `2026-09-09-beat-sparse35.md`, 2872–2914 |

## 5. What is genuinely untried (from the ledger map, 2026-09-16)

- **R3 path-connectivity losses** (CAPE / Oner maximin / constrained MALIS on the probability map)
  — zero ledger footprint. Note the difference from WP03/WP10: those raised the minimum along GT
  only and rescaled; a maximin loss has a NEGATIVE half (the strongest path between disconnected
  points must be weak), i.e. it constrains ordering, not level. Crossings need masking.
- **R5 DeepFlux context-flux target**, **R7 homotopy-warping loss**, **R8 Sat2Graph graph-tensor
  encoding** — never entered the ledger (R8 dismissed in a survey as "not our failure").
- **Continuous orientation-field regression head**, **instance-gap valley head** — never built.
- **Along-filament inductive bias in the trunk** (orientation-adaptive long kernels, dynamic snake
  convolution [DSCNet, arXiv 2307.08388], line-integral features at 20–40 px) — no occurrence
  anywhere in the record; the prior bank was isotropic Frangi/LoG at 1.5–6 px. This is the only
  lane that targets §2(1) directly. **Its pre-screen is `diag_line_integral.py`** (2026-09-16): on
  the plain-stretch samples the model leaves below 0.98, is a coherent / incoherent line integral
  of the raw image along the GT tangent (L = 10/20/40 px) able to separate them from laterally
  displaced near-filament negatives at 1 % FPR? Kill line: < 20 % of the weak positives recovered
  at L = 20 or 40 → the information is not there along the filament either and the lane is dead.
  **Discriminating reading for any along-filament lever:** it must raise htw recall (64 % of htw FN
  are missed faint filaments); a lever that moves roi303 but not htw is crossing-specific, not
  along-filament.
- **Path-connectivity losses to read first:** CAPE (arXiv 2504.00753, 2025), Oner et al. maximin
  paths (2207.06832, MICCAI 2022), widest-path reachability fields (2607.07123, 2026).
- **Ranking objective** (per-frame on-filament vs near-background ordering) — designed as
  MINRANK and killed in review as "a per-image pos_weight in disguise"; only a formulation that
  answers that objection may be proposed.
- **Screen B** (frozen-feature AUC along the sub-threshold arm vs per-pixel map) — declared, never read.

## 5b. Closed and opened on 2026-09-16 (after the ceilings and the montage)

- **Along-filament inductive bias — CLOSED before a build** (`diag_line_integral.py`): the weak
  stretches have no raw-image evidence at 10–40 px, and the model's off-annotation blobs outrank
  them on every along-axis statistic (recall 0.04 at 80 % blob rejection vs a 0.20 kill line).
- **Crossing erosion is worth ≤ 0.03; along-filament continuity ≤ 0.10; the off-annotation
  foreground ≈ 0.17** (`diag_ceiling_union.py`). The FP side is the larger half.
- **What the off-annotation foreground IS (montage `data/enc_sensitivity_testset/step2_offgt/`):**
  the FIELD-STOP EDGE (55 % of roi303 frames carry an aperture; SPARSE35 and every arm were
  trained with `--fov-prob 0`, the flag never appeared in a launcher), short BRIGHT RODS
  (unannotated short filaments / seeds — the +705 blobs the derived constant admitted; annotation
  convention, not a model target), defocused PARTICLES, and a few dark-polarity filaments the
  annotators skipped. Quantified in `2026-09-16-step1-results.md` §Step 2d.
- **Opened: ARM 50 — the field stop in training** (`--fov-prob 0.5`, surround 1.3–5× measured;
  the augmentation existed, tested, and was never switched on). Physics of the instrument, not a
  prior fit. Pre-declared in `docs/TODO.md`.
- **For the user to decide (benchmark convention, not method):** (a) apply the field-of-view mask
  on roi303 at evaluation as MT-34 does (annotators drew nothing outside; the re-score at the
  operating point is in §Step 2d), and (b) short unannotated filaments: annotate them, or make
  predictions shorter than the annotators' implicit minimum an IGNORE class rather than FP.

## 5c. Corrected on 2026-09-16 08:30 — the target is the model's foreground BETWEEN filaments

The 0.17 in §5b was labelled "off-annotation foreground" by subtraction. Split by two further
ceilings (`diag_ceiling_union.py --delete-far 5`: delete every model pixel farther than 5 eval px
from the GT band; same code path as the metric), roi303 VAL, SPARSE35:

| variant | micro |
|---|---|
| baseline | 0.656 |
| off-annotation foreground deleted | 0.749 |
| GT band painted, model FP kept | 0.758 |
| band painted AND off-annotation deleted | **0.860** |
| oracle band, same path (step-1 path) | 0.900 (0.927) |

**Model-owned ≈ 0.11–0.14** — merging / thickening of the model's own foreground BETWEEN
annotated filaments at crossings and close parallels **0.04–0.07** (the largest single item;
bottlenecks 1 + 2 of CLAUDE.md), crossing erosion ≤ 0.03, field-stop edge ≈ 0.02 (ARM 50),
particles ≈ 0.01. **Convention / single-frame ≈ 0.12–0.13** — evidence-free stretches ≈ 0.07
(NOT polyline drift: re-read within the metric's own 5 eval px the weak share is 4.4 % and AUC
vs displaced background 0.51–0.58), unannotated filaments and seeds ≈ 0.05. htw-only: detection
0.085, off-annotation 0.07–0.08, merging 0.02.

**So the research target is:** a foreground representation / training signal under which two
filaments 2–6 px apart and the four arms of a crossing stay separable in the MAP — not a
post-processing step (the user tests the model), not the instancer (the assembler is innocent
wherever the mask carries through, §2), and priced first against the 0.04–0.07 + ≤ 0.03 it can
buy. The residual anatomy under the "band + nothing off" ceiling minus the same-path oracle's
(`runs/step2/fp_*_ceilresid.json` − `fp_*_oracle.json`) names the classes (junction / gap-bridge /
fragment) the merging term consists of; read it before proposing.

## 6. What a proposal must contain

1. The lever, in one paragraph, and WHICH of §2(1)/(2)/(3) it acts on.
2. Why it is not a member of a §4 family (name the nearest closed relative and the difference).
3. A training-free or < 1 GPU-h pre-screen with a numeric kill line.
4. The full-arm falsifiers: primary (paired micro on roi303 VAL vs SPARSE35 at own threshold, ≥ +0.02),
   the htw VAL recall reading (the discriminator between along-filament and crossing-specific),
   the foreground-matched control, the rescale falsifier (precision at matched coverage must rise;
   threshold shift alone is a fail), and the htw VAL reading.
5. Cost in GPU-h and the modal outcome, priced honestly against the seed sd.
