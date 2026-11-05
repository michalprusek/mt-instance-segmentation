# Paper plan — annotation-free instance segmentation of crossing filaments

> **VENUE DECIDED 2026-08-25: NATURE COMMUNICATIONS.** Nature Methods was assessed and is not reachable
> on the current evidence (W2 and W3 both negative; see docs/TODO.md §4 and the 2026-07-11 honest verdict).
> **The operative plan is now `docs/TODO.md` §0 — this file is the claim/positioning document and is
> SUPERSEDED by §0 wherever the two disagree.**
>
> Original north-star (2026-07-05): a methodologically novel, Nature-family contribution. Must DIFFERENTIATE
> from **SynthMT** (Wieslander/…, PLOS Comput Biol 2026), which already published "synthetic IRM MT +
> foundation model, annotation-free param tuning." The bar is a genuinely new capability + rigor + generality.
>
> **Spine after the venue decision:** the headline is **N2 + annotation-free end to end** (including the
> instancer's hyperparameters, §17q). **N1 is DEMOTED to an ablation** — it is not causally supported, and
> reporting its negatives honestly is a rigor asset rather than a loss. The capability claim is a **REGIME**
> (dense crossings, where TARDIS scores 0.000 and PySOAX preserves 0 of 288 junction identities), not a
> higher pooled F1 — see TODO §0.1.

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
- **C1.5 (identifiability / VARIANCE-recovery) — RUN, and NEGATIVE. Claim DROPPED.** The design was: on a
  SYNTHETIC target with a KNOWN property std, sweep that std and show region_mmd has a well at the true std
  while frame-pooled residual_mmd is flat. **It did not reproduce on test** (2026-07-11, [[nature-comms-target]]),
  and C2ST additionally saturated — synth and every real lab all separate at ~0.98 in DINOv2, so "synth is a
  distinct source, like each lab is" is a modest statement, not a mechanism. This entry stood for weeks as
  "unrun, cheap, decisive" and was stale; W3 is therefore closed NEGATIVE, not open. See docs/protocol.md §12.
- **C2 (shared-encoder coupling):** matched calib-encoder == seg-backbone beats crossed.
  HAVE: DINOv2-coupled 0.627 > Phikon-decoupled 0.529; tightening (residual seg input) hurts.
  NEED: clean N×N encoder matrix (calibrate-in-Ei, segment-with-Ej) → diagonal wins.
- **C3 (generator realism ablation):** each physics lever (real backgrounds, IRM polarity flips,
  hairpins/WLC morphology, dirt, calibrated density/contrast) contributes to transfer.
  HAVE: components built + over-firing gate; **four measured-defect fixes that were NULL on F1**
  (SNR-stratified labels, length prior, pos_weight, photometry — protocol §21); and, 2026-09-05, the
  first lever with a MECHANISM: the **noise texture** (real 16-bit frames' residual is spatially
  correlated and heavy-tailed, synthetic noise was white Gaussian; injecting the real texture at
  inference cuts the trained model's label recall at matched SNR 1–2 from 0.93 to 0.56–0.75, i.e.
  onto the real curve, while white noise of equal amplitude does nothing — protocol §23). The C3
  story is therefore "realism levers are null unless they change what the network must integrate
  over", with the injection check as the paper's discriminating experiment (no training needed to
  show the mechanism). NOISE's F1 read (2026-09-05): native NULL, 1.5× negative, although the
  mechanism moved on real maps (coverage +5 % of filaments, fragments −123) — the strict metric
  charges the gain as overshoot / gap bridges / unannotated filaments (protocol §23, "what the
  strict metric can and cannot see"). So C3's honest form is: **realism levers change what the
  network integrates over, and the human-drawn benchmark cannot credit the part of that change
  that lies beyond the annotation.** NEED: the leave-one-out ablation restricted to levers that
  pass an injection-style check, reported under the primary AND the end-tolerance variant.
- **C3b (scale robustness, 2026-09-07; replicated n = 2):** train-time scale augmentation (render
  and polylines rescaled 0.8–1.5 with p 0.7, low-resolution simulation p 0.2, target rasterised
  after the rescale) closes the native-vs-1.5× inference gap on every reading — baseline three
  seeds lose 0.052 micro / 0.036 dense-frame macro on upscaling, SCALE loses 0.003 / 0.010 and
  its replicate 0.006 / 0.0005 (pick) and 0.001 / 0.005 (ep40). It is NOT an F1 gain: null at
  native for three of the four checkpoint-rows (both picks = epoch 36, seed 1 ep40); seed 2's
  ep40 checkpoint is WORSE at native beyond the seed spans on every reading (own threshold 0.80
  on a 0.70–0.9995 grid: dense −0.016, micro −0.023, 1–9-GT macro ≤ 0.25 at every threshold vs
  the baseline's 0.42–0.50), so the paper states "3 of 4 rows null at native, 1 worse". **Scope of the claim (amended 2026-09-07 after diagnosis 30(c)):** the gap closes on roi303 **at each
  scale's own operating point**. At a SHARED threshold on MT-34 the two scales still diverge, and
  SCALE's checkpoint curves at the two scales are further apart than the baseline's (0.034 vs
  0.025) — so this is operating-point robustness, not scale invariance. Paper form: a robustness
  claim with the pre-registered gap as the experiment (protocol §23, "Scale augmentation: robustness, not
  gain"); the first read's "+0.03 / +0.05 beyond the seed span at 1.5×" is one exceedance among
  ~32 resolving-reading reads and is NOT claimed. Report with the seed mean and the caveat.
- **C3c (the operating point is the crossing lever, 2026-09-07):** the strict-F1-optimal decode
  threshold (0.99) is not the identity-optimal one. On one cached map, 0.99 → 0.97 costs 0.003
  of all-frame macro and buys +0.093 junction identity, +0.036 bundle recovery, +0.013 micro and
  +0.011 dense-frame macro, with fragmentation flat. A training-side crossing-region loss weight
  (arm 32, the research note's #1 recommendation, radius derived from our own 8–20 px failure
  measurement) buys +0.014 identity at matched thresholds and is null on F1. Paper form: report
  the crossing figure at an identity-chosen operating point as a separate column, and state the
  7:1 ratio between the two levers as the evidence that the crossing loss is confidence, not
  geometry (protocol §23 and the new section).
- **C4 (amodal transfer):** orientation channels keep crossings separate on real.
  HAVE: ori head, union tol2 0.943, transfers on Alice. **GAP: Alice is mostly SEPARATED near-horizontal
  parallels with FEW crossings → cannot demonstrate the crossing/parallel novelty.** NEED a
  CROSSING/PARALLEL-STRESS eval set (see below) + metrics: per-junction identity-preservation,
  bundle-recovery (N parallels at 2–6 px → N instances).
- **Metric reporting (SUPERSEDED 2026-09-08 by the user: "use the metric that is common in the
  field"). The HEADLINE is now the 0.80 / 0.80 / 5 px convention; the strict 0.95 / 0.95 rule is
  demoted to a secondary column as the instance-exactness measure.** On roi303 TEST (3 seeds,
  identical predictions): headline **0.614 ± 0.072**, strict 0.473 ± 0.052; ORACLE ceiling 0.920
  headline vs 0.888 strict, so the 0.90 target is reachable in principle under the headline and
  not under the strict rule. Note the headline's seed span is LARGER (0.072 vs 0.052), so it
  resolves differences less well; campaign verdicts were read on the strict macro and have not
  been recomputed. Historical note, the superseded rule: the strict 0.95 / 0.95 / 5 px
  centerline-F1 was the
  primary number; every table also carries the 0.80 / 0.80 / 5 px convention (same predictions:
  SPARSE35 0.527 → 0.672 on roi303 VAL; the 95 % rule costs ~0.15, the tolerance ~0.03) so the
  reader sees the instance-exact and the community-standard numbers side by side. Added
  2026-09-05: a third column, the primary rule with a symmetric **25-px end tolerance**
  (`end_tol=25`, fixed from the baseline's overshoot distribution — 13 % of its false positives
  are predictions that follow the filament past the annotation's end), and every table reports
  the macro over frames with ≥ 10 GT next to the all-frame macro, because 23 % of the roi303
  macro weight sits on 32 near-empty frames holding 1 % of the filaments. Measured on the
  baseline (SPARSE35, roi303 VAL, 139 frames, same predictions): primary 0.527 · end_tol 25
  **0.640** (+943 TP, a quarter of the strict misses lie within 25 px of the annotation's ends) ·
  0.80 convention 0.672 · 0.80 + end_tol 25 0.704. No variant changed an arm's verdict. **Seed reporting rule (2026-09-06):** three seeds of the
  baseline recipe span 0.041 on the primary (all on 32 near-empty frames) and ≤ 0.01 on the
  ≥ 10-GT macro and micro; every arm table therefore reports the seed mean ± span of the
  recipe it compares against (never a single lucky seed — the campaign's baseline sat +0.018
  above its own seed mean), and a difference is claimed only if it exceeds the seed span of the
  reading it is claimed on (protocol §23, "The resolution of the primary, measured"). The 0.90 target is
  stated against the primary on the named benchmark; under that rule against this annotation it
  is not model-reachable (the overshoot and gap-bridge classes alone are ~17 % of the FPs and
  are annotation-owned), and the paper says so with the numbers.
- **C5 (headline):** annotation-free full pipeline matches/beats REAL-trained baselines (TARDIS 0.326,
  ORION 0.519; and SynthMT/SAM3) on IRM **AND TIRF**, with ZERO annotations.
  HAVE: beats TARDIS/ORION on Alice (heuristic). NEED: learned instance step; TIRF eval; SAM3 comparison.

## What the 2026-09-08/09 diagnostic campaign established (paper-grade, all pre-registered)

These are results, not progress notes: each was read against a falsifier written before the number
existed, and several killed an apparent gain. They change the paper's spine from "we built X" to
"we located the bottleneck and proved where it is not".

**R1 · The assembler is not the bottleneck on real data, and that is measurable per crossing.**
On 2 243 GT crossings of roi303 VAL: wherever the mask carries the filament through, the
curvature-bounded assembler resolves the crossing at **0.898** (junction present in the graph) and
**0.951** (none); **97.1 %** of all crossing failures are a mask that breaks the filament.
`mask_connected` is 0.252 and an oracle mask lifts junction identity 0.402 → 0.835 through the
SAME assembler. A companion in-domain reading (exact GT) shows the opposite balance, so the two
regimes must never be pooled — the campaign had been conflating them.

**R2 · What decides identity at a crossing is the ARITY of the arc-graph junction, not its
presence.** In domain a 4-arm junction resolves **0.944** (n = 107) and a 3-arm one **0.333**
(n = 33), and the gap holds inside every angle band, so arity is not a proxy for crossing angle.
It replicates on real (0.993 vs 0.561). A 3-arm junction is a crossing split into two clusters
joined by a shared arc of 25–120 px the graph can traverse once — **bottleneck #1 (close
parallels) appearing inside the crossing statistics**. `_absorb_crossing_bridges` names this
trade-off as irreducible in its own docstring; a predicted multiplicity map is the evidence that
decides it. Implemented (`instance_a(..., overlap=...)`) with a unit test that reproduces the
failure and its removal with no network; worth **+0.013 strict F1 with a PERFECT map**, not
resolvable. **Publishable as a mechanism, explicitly not as a gain.**

**R3 · A synth-only network localises microtubule overlaps on REAL images, from free supervision.**
The generator's multiplicity channel, never previously consumed, trains a second head to
overlap AP **0.778 in domain / 0.332 on real** (chance 0.00048 in a full frame), both risen
between epoch 12 and 40. Its F1 effect is null (−0.0013, p 0.92). At matched foreground it moves
the map (`mask_connected` +0.051, junction identity +0.026); replication and a scrambled-channel
control are running, and no claim is made before both.

**R4 · The recall ceiling is an ORDERING failure, and it is quantified.** Instance recall is
**flat at ~0.52 across every usable threshold** on two independent models — 23 % more foreground
buys +0.014 recall. Yet the evidence is present: the losing stretches have median minimum
probability **0.830**, and **at threshold 0.50 the map already covers 85 % of GT curves to ≥ 95 %
of their length**. It cannot be extracted because decoding there floods the frame. **The available
headroom is recall 0.52 → 0.85, and it is entirely a ranking problem.** Fixing it would also fix
precision, since 50 % of false positives are fragments and 15 % junction chimeras — both
consequences of breaks — which is why an oracle mask reaches 0.888 through the same assembler.

**R5 · The ordering failure is a DOMAIN GAP with two measured causes.** Pixel AP against GT
proximity is **0.996 in domain and 0.945 on real**; the extractable 0.5–0.97 band is 97.4 % on-GT
in domain and 66.5 % on real (only 13 % of the excess is the known edge-band artefact). Two
generator defects account for the direction: the render carries **40 % less elongated ridge
structure** off-filament than a real frame (0.0044 vs 0.0062, q75 0.0069 vs 0.0193) while being
**1.6x NOISIER** in amplitude (high-pass std 0.0448 vs 0.0285). **The network is trained to reject
loud white noise and never has to reject quiet filament-shaped structure.** Arms 41 and 42 test
each half.

**R6 · A methods contribution the field will recognise: how to compare two maps honestly.**
Four apparent gains died to pre-registered checks in one day — a `--params` override that halved
junction identity while the mask stayed bit-identical, a criterion that could not fire on any real
arc, a checkpoint-protocol mismatch that manufactured the campaign's only "positive" arm, and a
shared-threshold comparison of a statistic that moves 1.23 per percentage point of foreground.
**Mask statistics must be compared at MATCHED FOREGROUND, found by bisection on the cached dump,
never at a shared threshold** — a shared threshold measures calibration, not map quality.

## What the 2026-09-15/18 round established (declared instrument, six pre-declared arms)

**R7 · One instrument, declared before any number.** Centerline-F1 (tol 5 eval px, coverage and
precision 0.80) pooled micro on roi303 TEST (137 frames, human-drawn) with htw TEST (33) as the
cross-lab secondary; per-model threshold on the block's own VAL; ep040 by rule; derived instancer
constants. SPARSE35 ep040: **0.641 ± 0.006 / 0.601 ± 0.012** against oracle-mask ceilings
0.935 / 0.871. Every number in the paper is re-read under this instrument; earlier MT-34 numbers
are traceability only (agreement bias, R-series above).

**R8 · The gap is decomposed and most of it is the annotation's.** Paint-and-keep ceilings at
matched band width bound the roi303 VAL terms: evidence-free stretches ≈ 0.07 (not polyline
drift — the weak share survives widening the window to the metric's tolerance), unannotated
filaments ≈ 0.05, crossing erosion ≤ 0.03, field-stop edge ≈ 0.02, near-filament merging
0.01–0.02, particles ≈ 0.02. **Model-owned ≈ 0.06–0.07; the single-frame bound as annotated is
0.74–0.76.** This is the sentence that replaces "we could not reach 0.90".

**R9 · Six pre-declared arms, six nulls or negatives, every mechanism gate failed.** Field stop
(−0.004, paired p 0.047, and the edge-share gate failed), amodal head (−0.088; −0.16 cross-lab);
then four nulls — anti-aliased stamp, Euclidean target, rotation augmentation, pos_weight 2 / 4 —
all inside ±0.010 micro with paired p ≥ 0.10. The
grid-anisotropy finding is real (oblique filaments rotated onto the axes lose 2.5–2.8× more
stretches; the renderer's stamp aliases 1.4× in peak) and is reported as a characterised property
that three isotropy fixes did not move. Twelve of thirteen research proposals were refuted by
adversarial verification before any GPU time. Written up as a methods result: how a declared
falsifier plus a mechanism gate kills headline gains that seed noise would have passed.

**Must-fix before any TEST number is written:** roi303 VAL and TEST share fields. **Re-split
done 2026-10-05 (v2, by field, chosen blind): VAL 145 / 7 386 curves, TEST 128 / 6 506,
`docs/roi303_split_v2.json`.** The leak was larger than recorded (~100 of 137 v1 TEST frames
shared a field with a VAL frame). v2 TEST is unread; SPARSE35 is re-read on it once, on the
user's word. Every roi303 TEST number above is v1 and provisional.

## What 2026-10-05 established (declared reads; `docs/protocol.md §25`)

**R10 · Honest intervals.** Field-cluster bootstrap on the stored v1 rows: roi303 TEST
0.641 ± 0.040, htw TEST 0.601 ± 0.054. Seed-to-seed spread of one recipe equals the "leads" of
arms 50–55 — the paper reports cluster CIs, not seed sd.

**R11 · The two blocks fail on different axes.** roi303 misses are SNR-driven (30 % of curves
at SNR < 2, miss 0.42), htw misses are length-driven (< 12 px miss 0.88). One per-curve figure.

**R12 · In domain the instancer is the ceiling (exploratory).** Perfect band on synthetic 0.686,
model 0.636–0.646 (93 %); on real 70–72 % of the ceiling. The sim-to-real loss is in the mask.

**R13 · Correlated noise: a trade when it replaces, a gain when it is mixed.** Replacing the
shipped noise: htw +0.023 / +0.028, roi303 −0.018 on both seeds. A per-frame 50/50 MIXTURE
(ARM 56, `calib_V12_noisemix.json`, read 2026-10-06): htw VAL +0.024 / +0.072, roi303 VAL
+0.010 / +0.010 against same-seed baselines; mechanism gate and synth guard pass; all four
mixture runs above all three baseline seeds on htw. roi303 claim is non-inferiority (cluster CI
includes 0). Flat-fielded backgrounds on top (ARM 57) are null. Scale augmentation, photometric
augmentation and junction weighting are null to negative. **Paper use:** this is the cleanest
domain-randomization result of the project (widen, do not swap) — two seeds, VAL only; needs a
third seed and the v2 TEST read before it is a headline.

**R14 · Tile-seam defect in the inference path, fixed.** Uniform-mean fusion at stride 0.757
tile cut filaments at seams (30.6 % weak points within 8 px of a seam vs 16–19 % elsewhere).
Gaussian-weighted logit mean at stride tile/2: +0.012 / +0.013 / +0.015 on roi303 VAL on three
seeds, cluster CI excludes 0 each; htw is single-tile and unchanged. User ruled it a defect fix;
it is the declared instrument from 2026-10-06. Methods-section item, not a contribution.

**Closed negative:** RELINK 1-D (one `w_theta` multiplier). First rule picked 1.75 and failed its
guards; the repaired rule (guards inside the selection) picked 1.25, and the one confirmation
read gave roi303 +0.007 (met) and htw +0.008 (line +0.012, not met). Not adopted.

**R15 · A "novel decoder" screened and not supported (LPC, 2026-10-06).** Selection ceiling on the
unchanged mask: +0.17 in domain, +0.20 on htw VAL of which ~half is the annotation convention. A
strip CNN verifier beats the mean map logit but not five hand features in AUC, and the raw image
adds nothing; junction pairing is already 0.92 / 0.997 at clean through-crossings, so the crossing
failures are in arc-graph construction. Paper use: a negative that supports the standing claim
"the synthetic→real gap is the wall, not the decoder" with a measured decoder ceiling, and a
pointer to where a decoder could still gain (12–36 px links between arbitrary arc ends, in domain).
**Amended 2026-10-06 13:20 — the pointer paid off, so R15 is no longer a pure negative.** A link-score
decoder ("instancer L": the same arc graph, a logistic on five hand features fitted on synthetic links only,
max-weight matching on arc ends, κ ≤ 0.25 fixed) gains on VAL on five models: roi303 v2 +0.030 … +0.042
(all field-cluster CIs exclude 0), htw +0.013 … +0.052 (four of five exclude 0), junction identity +19 %
relative, bundle score +29 %. Honest wording for the paper: the *learned* parts of the proposed decoder
(strip CNN, raw-image evidence, junction head) are measured negatives; what gains is replacing greedy
bridging by a global assignment with a score fitted in domain. "Not the decoder" must be softened to
"the mask is the larger share; the assembler was leaving ≈ 0.03–0.04 on the table". TEST not read for L.
Block order: htw is the decisive block (clean annotation) and there the baseline seed's gain is +0.013
with an interval that includes 0 — the htw case rests on the other four models; roi303's gain may
partly be fragments of unannotated filaments merging, which the annotation-completion data will settle.
**17:10 — the network variant of the scorer is closed.** The same strip CNN used as the link score
(L-strip) is level with the five hand features on htw VAL (−0.007 … +0.007, all intervals include 0)
and worse on roi303 v2 VAL (−0.027 … −0.033, all exclude 0; junction identity −9 % relative), and is
not ahead even in domain (0.668 vs 0.676). Declared instancer = L-hand. Count of the learned parts:
raw image negative, junction head void, strip CNN not better than hand features in either role.
**ARM 56 (noise mixture) does not become the recipe:** third seed htw +0.063 but roi303 v2 VAL −0.007
against a −0.006 line; three seeds htw +0.024 / +0.072 / +0.063, roi303 +0.001 / +0.001 / −0.007.
Paper use: a replicated cross-lab gain that costs nothing measurable in the home block but did not
pass a strict pre-declared non-inferiority rule — report as such, do not relabel.

**R20 · Debris content + crossing-gap bridge (2026-11-02; third adaptive TEST use).** Recipe V16 = V14 + compact debris (0-40 objects per 640 px training crop, axis ratio <= 2.5, never labelled; zero inference cost); decoder L+H = L-hand + bridge hysteresis (p > 0.98 pieces near skeleton junctions only where they join >= 2 own-cut components; selected on half of the roi303 VAL fields, confirmed on the other half).

| TEST, 4 seeds, kajman | roi303 v2 | htw |
|---|---|---|
| V14 + L | 0.693 | 0.681 |
| V16 + L+H | 0.715 | 0.699 |
| paired, pooled | +0.022 [-0.003, +0.041] | +0.018 [+0.004, +0.036] |

Run level vs six V14 runs: +0.021 (p 0.001) / +0.019 (p 0.14). Debris removes the debris (labelled off-annotation tiles still predicted 0.58 -> 0.32) and keeps unannotated filaments (0.82 -> 0.75); the cost is the shortest curves (htw < 12 px 0.339 -> 0.257). VAL gain of H concentrates in the one crossing-dense field; identity on TEST does not follow VAL. Disclose: H selected on real VAL (user ruling), TEST read after both VAL decisions, dose per 640 px crop. Rejected in the same round: interference rings (no net gain though firing on rings fell ~20x).

Addendum, match reach 60 -> 80 px (2026-11-02; TEST NOT read, the table above is 60 px). Selected on synthetic links; paired on the same maps at VAL: roi303 +0.007 [+0.003, +0.009], htw +0.001, MT-34 +0.035, synth +0.005, identity 0.642 -> 0.665, no added complexity. Disclose: it missed the declared +0.010 line (block mean +0.0037) and was accepted by the user after that read. Current VAL: 0.740 / 0.730 / 0.660 (roi303 / htw / MT-34). Any table with 80 px needs TEST read #4.

Replication of debris (2026-11-02; DEB_S1, DEB_S2, never used in a decision), L+H at 80 px, VAL: htw +0.011, roi303 +0.005; six V16 runs vs six V14 runs: +0.013 / +0.006. Quote debris as worth about +0.01, not the +0.015 of the first four pairs under L. Rejected: a thin-filament regime at dose 0.35 (htw +0.032, roi303 -0.017, MT-34 -0.028), a trade between blocks rather than a gain, and not by the recall it was built for: htw short recall did not move, its precision did. Filament width by one estimator: htw 2.61 px, hub 3.30, MT-34 3.49, roi303 4.43, V16 synth 4.62, so the blocks bracket the deployment data and neither alone stands in for it.

Half doses (2026-11-02, VAL, L+H at 80 px, TEST not read). Debris x 0.5 fails its repair rule: roi303 -0.006, recall of GT < 20 px not back. Thin x 0.5 misses the gain rule: htw +0.014, roi303 -0.006, MT-34 +0.009 (2/4), block +0.004. Post hoc, not for a claim without saying so: four thin x 0.5 runs vs six debris runs, recall < 20 px +0.076 (htw, p 0.001), +0.119 (roi303, p 0.001), longer curves unharmed, roi303 FP 2878 -> 3067 and unlabelled. Paper use: a method note that a recipe's F1 hides a recall / FP trade on short curves; recall-by-length is reported next to F1.

Deployed (2026-11-03): V16 DEB_S4 + L+H, reach 80, cut 0.995. Disclose: MT-34 VAL numbers of the 10-28 to 11-02 round were read under uniform fusion (DEB_S4 at 0.995: 0.666 uniform, 0.681 Gaussian); any MT-34 cell in the paper must be re-read under Gaussian fusion. TEST cells remain per-block cut, reach 60.

Neighbour votes in the link score (2026-11-04, no training): one-to-one partition at lag 3 +0.026 CI [+0.018, +0.034], below the declared +0.03; synthetic F1 +0.003, tracks per object 1.48 to 1.38. Temporal input retrain (ARM 65, six seeds against the six V16 seeds as exact control): synthetic clips F1 +0.029, tracks per object 1.46 to 1.37; real video partition +0.004 [-0.006, +0.013], mover length -19 %, htw single-frame -0.013. Not adopted. The decoder vote replicated on six seeds, +0.026. Integrated in the hub 2026-11-04 on the user's word after the read (below its declared line, disclosed as such); hub code bit-identical to the research path on 211 of 211 real interior frames; not deployed yet. Deployed in SpheroSeg 2026-11-05 (hub #613): same-card re-decoding identical on production, 7 of 7 interior frames of a synthetic test video changed, 26 s for 7 frames of 1 Mpx; cross-card (L40S vs A5000) not bit-identical, 39 of 40 curves within 1 px.

Temporal window without training (2026-11-03, same videos, read at lag 3 because consecutive windows share frames): mean image -0.004 one-to-one partition, map fusion -0.09 to -0.23 and it deletes movers; no support for a retrain, not a refutation. Of the places where one curve becomes two, 37 % are map breaks, 35 % show no break within 2 native px, 28 % are side-by-side filaments.

Frame-to-frame stability on real video (2026-11-03, unlabelled, 12 videos): the model is per-frame. V16 vs SPARSE35: persistence 0.907 vs 0.895, long-track length +19 %, tracks per filament 2.24 vs 2.38; one-to-one partition unchanged at 68 % of length (81 % on an unchanged scene). A tracking claim in the paper needs this limit stated, and has no ground truth behind it yet.
The Essays batch runs the same model since the same day (10:33 CET), so any assay readout in the paper must state which side of that date its runs fall on; `results.csv` names the checkpoint per row from then on.

**R19 · Two nulls and the run-level noise floor (2026-10-31; TEST not read).** Encoder-level DANN on unlabeled SpheroSeg
frames (ARM 59b) and loss-ignore of MT-like lines in the empty backgrounds (ARM 60) both fail the block-mean line
(+0.0065, +0.003 vs +0.010). Both beat their four same-seed controls on htw by +0.011–0.012, but the controls are the four
lowest of six AA runs; vs all six the gap is +0.007–0.008 (Welch p 0.16–0.28). ARM 60 is close to a no-op (0.022 % of crop
area) and is the design's empirical null. Paper use: (i) methods state that frame / field bootstrap CIs exclude
training-run variance and every recipe claim carries a run-level test; (ii) run sd on htw 0.0075–0.012 F1; (iii) label-free
alignment to unlabeled real frames is null on both branches (output space ARM 59, encoder ARM 59b), a reportable negative;
(iv) label-free domain-gap readouts: seed disagreement real / synth 3.3x, bottleneck probe AUC. Ensembling as an
inference-time option: two seeds +0.017 block mean at 2x cost, six +0.021 (roi303 +0.033).

**R18 · Anti-aliased rendering joins the recipe (2026-10-29; second adaptive TEST use).** A renderer defect fix
(`paint_aa`: the old `_paint` drew axis-aligned filaments 20–40 % brighter; the earlier "~3.3 px short of its label" story was REFUTED by the 2026-10-31 render audit, so do not write it)
adds no parameters and no inference cost. Six same-host seed pairs (kajman) against the noise mixture, L-hand, own VAL cut:

| | roi303 v2 | htw |
|---|---|---|
| VAL, fresh pairs 3–6 pooled | +0.007 [−0.007, +0.014] | +0.014 [+0.001, +0.028] |
| TEST, fresh pairs 3–6 pooled | −0.001 [−0.006, +0.005] | +0.014 [−0.001, +0.028] |
| TEST seed mean, AA + L (fresh 3–6) | 0.693 | 0.681 |
| TEST seed mean, NMIX + L same host | 0.695 | 0.667 |

The gain is recall of short curves (< 20 native px) on htw, positive in all six pairs; roi303 is null. Disclose the path:
the htw hypothesis came from looking at ARM 58; a one-seed confirmation (ARM 58b, line +0.024) failed; the line was then
lowered to +0.010 by the user's criterion (+1 pp is worth it at zero added complexity) and a four-fresh-seed rule was
written before seeds 4–6 existed; TEST was read once after that VAL decision. Final recipe: noise mixture + `paint_aa`,
instancer L-hand. Host offset: kajman controls read +0.007 above tulen on roi303 TEST, as on VAL — compare within a host.
**Run-level caveat (2026-10-31):** the CIs above resample frames only. At the level of training runs the htw gain is
VAL +0.014 (4 vs 4, Welch p 0.13) and TEST +0.012 (6 vs 6, p 0.14, 5/6 seeds positive): consistent sign on two disjoint
frame sets, mechanism 6/6, not resolved at six runs. Write it exactly so; never cite the frame CI alone.

**R17 · The held-out 2×2 (2026-10-06, read once).** Pooled micro, 3-seed mean, each cell at its VAL cut:

| | roi303 v2 TEST (128 fr / 39 fields) | htw TEST (33 fr) |
|---|---|---|
| SPARSE35 + instancer A (deployed) | 0.637 | 0.601 |
| SPARSE35 + instancer L | 0.674 | 0.629 |
| noise mixture + A | 0.651 | 0.638 |
| noise mixture + L | 0.687 | 0.668 |

L − A: +0.038 [+0.014, +0.044] roi303, +0.028 [−0.002, +0.054] htw. Mixture − baseline under A: +0.014
[−0.003, +0.023] roi303, +0.037 [+0.011, +0.067] htw. Both together: +0.050 [+0.013, +0.060] and +0.067
[+0.037, +0.095]. Instancer and recipe were fixed on VAL before the read; the mixture had failed its
VAL rule by 0.001, so its TEST row is a report, not a selection (state this in the paper).
**17:50 — the user adopted the mixture as the recipe after this read.** The paper must say so: the
recipe choice saw TEST; only the instancer comparison (L − A) is a clean held-out estimate. Project
criterion from the same ruling: a compromise of compute cost, accuracy and novelty — every method row
in the paper carries its inference cost.

**R16 · Baseline on the field-disjoint test split (2026-10-06).** SPARSE35 recipe, instancer A, roi303 v2
TEST (128 frames, 39 fields), one evaluation per seed at the VAL cut: 0.640 / 0.637 / 0.633, mean 0.637,
seed sd 0.003, field-cluster 95 % CI ≈ ±0.08. The between-field interval, not the seed sd, is the
uncertainty the paper must quote.

## Critical-path gaps (blocking a strong paper)
0. ~~Re-split roi303 by field~~ (done 2026-10-05, v2) — **re-read the declared TEST number once
   on v2 (user's word)**; then an annotation-completion pass on ~10 VAL frames and a declared
   convention for evidence-free stretches — the three items the decomposition (R8) names as the gap.
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
2. ~~C1.5 variance-recovery figure~~ — **RUN, NEGATIVE, closed.** Do not re-attempt without new evidence.
3. Build the learned affinity + Mutex-Watershed instance step trained on synth per-instance GT (N2);
   evaluate synthetic-first (bundle-recovery + junction-identity). [gen_train --inst DONE + verified.]
4. Survey the real corpus / CVAT for crossing-dense frames → define the stress eval (C4).
5. TIRF render + appearance recalibration (§11 protocol) → generality.

## Assets
Real corpus (522 dedup), IRM backgrounds (428), co-registered TIRF, 12 Alice instance-GT frames,
calibrated hairpin generator, DINOv2-hybrid + orientation-keyed head, calibrate.py (residual-MMD).
Compute: tulen. See [[project-roadmap]], [[instance-seg-research]], [[synth-generator]].
