# Protocol — the path we took (lab notebook)

> Living research log for the annotation-free synthetic-MT → segmentation project. Records the PATH:
> what we tried, what worked, what did NOT, and why. Feeds the paper's Methods + Limitations.
> Companion to the structured docs: PAPER_PLAN.md, INSTANCE_SEGMENTATION_RESEARCH.md,
> ENCODER_SENSITIVITY_EXPERIMENT.md, LABELFREE_FOREGROUND_ENCODING.md, MT_SEGMENTATION_KNOWLEDGE.md.

## 0. Goal & setting
2D instance segmentation of microtubules (MTs) in **label-free IRM** (and TIRF) microscopy, trained on
**synthetic data only — ZERO manual annotations**. Eval = strict centerline-F1 on 12 real "Alice"
frames. The wall is the **synthetic→real domain gap**, not the decoder (prior synth-trained decoders
all failed on real; real-trained TARDIS works). So the effort goes into synthetic realism + a
principled, annotation-free calibration + an instance representation that survives crossings/parallels.

## 1. Data assembled (what we already HAVE)
- Real MT frames: 522 deduped (dedup by pixel-correlation — dHash over-merged low-content IRM frames).
- **Real empty-field IRM backgrounds: 428** (`data/real/irm_backgrounds_v2`). KEY DECISION: we do NOT
  synthesize backgrounds — we **composite synthetic MTs onto REAL backgrounds**. This side-steps the
  hardest appearance component (background mottling, dirt, vignetting, correlated texture) by USING the
  real thing. (An inpainted background set was tried and DELETED — unrealistic.) So calibration effort
  is on **morphology + MT appearance**, not background. SynthMT, by contrast, uses synthetic uniform
  backgrounds — a real differentiator.
- 12 Alice frames with human instance polyline GT (the benchmark) + co-registered TIRF channels
  (free instance-GT transfer to a 2nd modality).

## 2. Choosing the encoder (empirical experiment, docs/ENCODER_SENSITIVITY_EXPERIMENT.md)
We needed a feature space whose distance REACTS to MT changes but is stable to background — to drive a
label-free calibration. Metric R = dMT/dBG (embedding move when MTs appear vs when the background is
swapped), measured on a fixed real background.
- **What did NOT work: global pooling (FID/CLS/mean embedding) is FOREGROUND-BLIND** — GLOBAL R < 1
  for EVERY encoder (swapping the background moves the global embedding MORE than 15 MTs appearing).
  ⇒ do NOT calibrate on global FID/KID/cosine. (This is exactly SynthMT's layer-5 global cosine.)
- **What worked: mid-layer PATCH tokens + a foreground-focused readout.** Top-5% patch-token distance
  ~doubles sensitivity vs mean-pool. Best encoder = **DINOv2 ViT-L/14** (peak R at blocks ~11–17);
  **DINOv3 is the WORST** (smooth high-level features suppress thin low-contrast lines); CLIP close;
  C-RADIOv4 / SAM2 mediocre. Later: Phikon-v2 (pathology DINOv2) has 3× higher global R but is WORSE
  as a segmentation backbone — **R is a calibration proxy, not a transfer predictor**.
- Decision: **DINOv2 ViT-L/14, mid-block patch tokens.**

## 3. The residual (background subtraction) — how it came about & why
Because global features are foreground-blind (§2), we needed to make the metric attend to the sparse
(~2%) MT foreground. Insight: we KNOW the background (we composite onto it), so **subtract it**.
- **residual `r = I / gaussian_lowpass(I, sigma=40) − 1`** — a local background-normalized image where
  the smooth background → ~0 and MTs (dark or bright) stand out in the tails.
- Effect: lifts R from 0.85 (raw, foreground-blind) to ~1.65 (R>1). Residual beats raw at EVERY DINOv2
  block. This is the "foreground-aware" input. (Honest caveat found later: even residual + mean-pool
  still under-weights the sparse foreground — see §12.)

## 4. Distributional MMD (not FID, not single-frame) + the identifiability finding
- Single-frame feature distance is too noisy → we match DISTRIBUTIONS over many frames via **MMD**
  (RBF-kernel, median-heuristic bandwidth), NOT FID (sample-size/resize-fragile) and NOT single-frame.
- Parameter recovery PROVEN: on synthetic targets with KNOWN params, MMD's argmin sits at the truth for
  density and contrast (residual gives a ~2× sharper well).
- **What did NOT work as hoped: MMD recovers a property's MEAN but NOT its VARIANCE.** Width-std /
  contrast-std sweeps are flat/noisy with the wrong argmin (~6× smaller dynamic range). ⇒ we calibrate
  MEANS only and keep each property's std at a fixed domain prior. (Root cause diagnosed later, §12:
  frame-pooling averages away within-frame per-MT spread.)

## 5. The generator (morphology + IRM appearance), `synth/mt_generator.py`
- **Morphology (modality-agnostic — the microtubules themselves):** stiff worm-like centerlines
  (straight baseline + heavily-smoothed bounded lateral deviation — first attempt used cumulative gamma
  bends which made CIRCLES; fixed to lateral-deviation). Frame-level waviness. IRREGULAR parallel bundles
  (variable gap, angular divergence, staggered ends — fixes too-regular parallels). **Sharp bends /
  hairpins** added on user request: occasional SMOOTH large turns up to ~180° reversal via re-integrating
  the tangent angle with a smootherstep (C2) ramp → finite curvature, no corner; multi-bend cases
  alternate sign → S-curves not closed loops. Spatial clustering (Thomas point process).
- **IRM appearance:** MTs are DARK (attached to coverslip) — corrected from an early wrong dark/bright
  mix. Occasional LOCAL bright DETACHMENT segments (smooth transitions). Whole FRAMES sometimes inverted
  (incl. background). Parametric cross-section = Gaussian core + opposite-sign interference halo.
  Multiplicative composite img = bg·(1+contrast); PSF; Poisson+read+correlated-texture noise; dirt spots.
- **Frame-level conditions vs per-MT:** properties that are acquisition/buffer CONDITIONS (waviness,
  contrast level, halo, detachment activity, inversion, hairpin activity) are sampled ONCE per frame and
  applied to ALL its MTs (per-MT only phase/jitter); per-MT: geometry/length/position/width.
- Every modelled property is a Normal(mean, std); calibration tunes MEANS (variances = priors, §4).

## 6. Label-free calibration loop, `synth/calibrate.py`
Optuna/TPE tunes ~14 GenConfig knobs (density, contrast, width_mean, curve, halo, length, waviness
freq/amp, detach freq/prob, invert prob, PSF, texture, cluster_frac) to MINIMIZE residual-patch RBF-MMD
vs a 320-frame real corpus (NOT the eval set — avoids leakage). Deterministic (fixed seeds → reproducible;
we back up best_dinov2.json after a Phikon run overwrote it once). pysqlite3 shim (base python lacks _sqlite3).

## 7. Downstream validation gates (what makes a calibration "good")
- **Over-firing test (the key acceptance gate):** train a foreground on the calibrated synth, predict on
  the 12 Alice frames; ratio = alice_fg% / synth_fg%. <3× = CLEAN. Historically synth-trained foregrounds
  over-fired ~6×; ours are 0.6–0.9× (clean). A foreground that over-fires breaks every downstream step.
- **Strict centerline-F1** on Alice (heuristic instance step, for reference only): reached 0.586 (U-Net)
  → **0.627 (DINOv2-hybrid)**, beating TARDIS 0.326 and ORION 0.519 with ZERO annotations.

## 8. What did NOT help (honest negatives — these are paper assets)
- **SimGAN GT-preserving appearance refinement = a WASH** (0.615→0.612). Perfect GT preservation
  (in-mask change 0.003) but no F1 gain: the sim→real gap is STRUCTURAL (scene statistics), not local
  filament appearance (already well-calibrated). A local GT-preserving refiner can't (and mustn't) close it.
- **Phikon-v2 backbone** (3× higher calibration-R) is WORSE end-to-end (0.495–0.529 < 0.627): higher R ≠
  better transfer; patch16 localizes 2px filaments worse; its own calibration over-densified.
- **Tightening the shared-encoder synergy** (residual input to the SEGMENTER = same as calibration) HURT
  0.627→0.559: optimal input for CALIBRATION (residual, foreground-sensitive) ≠ for SEGMENTATION (raw,
  full context). Loose coupling is the sweet spot.
- **Thinner training masks** improve semantic localization (tol2 0.926→0.941) but drop the HEURISTIC
  instance-F1 (thin FG fragments the tracer) — irrelevant once the tracer is replaced by a learned step.

## 9. Semantic-eval pivot + orientation-keyed amodal head
The heuristic instance tracer conflates generator quality with tracer quality (its FP=122 are tracer
errors; over-firing is clean → foreground is good). So we TUNE the generator on SEMANTIC foreground
quality (tolerant centerline-coverage F1, no instance grouping), on a **6-val/6-test split of Alice** (no
test leakage). Chose mask_half_width=1.0. Built the research-#1 **orientation-keyed "overpass" head**
(K=6 30° bins; crossings split across channels = amodal). Union coverage 0.943 ≈ single-channel 0.941 —
no semantic loss, adds amodal channels; orientations transfer to real. tol5 saturates (0.99), use tol1/2.

## 10. Instance step — research + decision (docs/INSTANCE_SEGMENTATION_RESEARCH.md)
Deep research verdict: a pure point-cloud/DIST head is NOT better for 2D — its max-degree-2 graph cut
collapses 2D crossings (the WEAVE wall). The fix: orientation-keyed multi-point + **learned connectivity**
+ **Mutex-Watershed hard-repulsion** for parallels (never-merge constraint, unlike fixed-radius embedding
which intrinsically merges — our Object-Condensation F1~0.01). DECISION: the instance step is a **LEARNED
model trained on the generator's FREE per-instance annotations** (short attractive + long repulsive/mutex
affinities from the exact per-instance masks) → Mutex-Watershed. Crossings handled by the orientation
channels. Evaluate SYNTHETIC-FIRST (controlled bundle gaps + crossing angles) before real crossing-stress
data. `affogato` (mutex_watershed) is installed — don't reimplement.

## 10b. Learned instance step — first iteration (2026-07-05)
- **MWS implemented from scratch** (affogato on tulen is a DIFFERENT package, no mutex_watershed).
  ~40-line union-find with mutex constraints, on the foreground graph. Offsets: 4 short attractive +
  8 long repulsive (P(same) per offset, from gen_train --inst labels).
- **ORACLE test (perfect affinities): bundle-recovery 0.859, 0 merges** → the MWS+mutex MECHANISM is
  sound; mutex correctly separates parallels (bottleneck #1). Fragmentation at crossings (single label
  map breaks the under-MT) = the amodal issue.
- **LEARNED model (first attempt, 30 ep): recovery 0.333 → 0.526** (with a mutex-threshold fix),
  still heavy fragmentation (pred_inst ~14× GT). DIAGNOSIS (measured): attractive affinity on
  same-instance pairs = only 0.75 (weak), repulsion on different-instance = 0.96 (over-confident) →
  confident mutex edges fragment the weakly-attracted MTs. Mutex-threshold (ignore mutex < 0.85-0.95)
  mitigates (0.33→0.53) but the ROOT is attractive-affinity quality. Visually decent at thresh 0.85
  (most MTs whole + parallels separated). Viz `data/enc_sensitivity_testset/simgan/aff_model_thr.png`.
- **NEXT iterations:** (a) improve attractive learning — upweight the attractive loss, add mid-range
  attractive offsets (0,2)(2,0)… to bridge small gaps, longer training, maybe Dice/structured loss;
  (b) ORIENTATION-KEYED amodal affinities (compute affinities per orientation layer) → crossings stop
  fragmenting (the label-map crossing issue). Approach is SOUND (oracle 0.86); learned-affinity quality
  is the work ahead.

## 11. IRM vs TIRF — one morphology, swap the render (REFACTORED 2026-07-05)
The generator now literally SEPARATES modality-agnostic **morphology** from the per-modality **render**
in code: `sample_scene()` produces geometry (centerlines + instance masks) — the microtubules themselves,
identical across modalities, and the co-registered IRM/TIRF channels are the SAME physical MTs, so sharing
morphology is physically exact, not an approximation. Two renderers: `render_irm()` (dark, multiplicative
bg*(1+field), interference halo, bright detachment, inversion) and `render_tirf()` (bright fluorescent
filaments on a DARK background, ADDITIVE bg+signal, NO halo, detachment DIMS — less evanescent excitation,
fluorescence shot noise). `generate_frame(bg, rng, cfg, modality="irm"|"tirf")`, default irm = backward
compatible (verified: calibrate/gen_train/eval all still run). PLAN: calibrate morphology on IRM (data-rich
+ 12 Alice), FREEZE those morph params, inject into the TIRF generator, recalibrate ONLY the TIRF appearance
on real TIRF — which also SOLVES the TIRF-data scarcity (only a few appearance knobs to fit). Params split
into MORPHOLOGY / IRM-appearance / TIRF-appearance groups in GenConfig. TODO: real DARK TIRF backgrounds
(from co-registered channels); move whole-frame INVERSION to augmentation + |residual| calibration.
Parameter classification (shared / IRM-only / concept-shared-values-differ) documented in TODO #5.

## 11b. NEW morphology diversity parameters (2026-07-05, user request)
Added to `sample_scene`/`sample_centerline`/`sample_instance` for more diverse MTs (grounded further by the
running morphology deep-research): **per-MT STIFFNESS bimodality** (`straight_prob`/`straight_curve_frac` —
some MTs rigidly straight, some flexible, vs one global curvature); **KINKS / lattice defects** (`kink_*` —
localized sharp small-angle bends via the smootherstep `_apply_bends`, distinct from the large smooth
hairpins); **frame-level ORIENTATION ANISOTROPY / nematic alignment** (`orient_kappa_range` — von Mises
concentration, 0=isotropic .. high=aligned); **LENGTH bimodality** (`short_prob`/`short_length_log_mean` —
a population of short dynamic seeds); plus appearance diversity **tip TAPER** (`tip_taper_*`) and
**along-length intensity heterogeneity** (`along_intensity_std`). Visually verified (IRM+TIRF montage,
diversity visible): `data/enc_sensitivity_testset/hairpins/refactor_viz.png`.

## 11c. Morphology research (deep-research wa7a0iz91, 106 agents) + what it changed
Key cited findings: (P1, highest gain) MT persistence length is MILLIMETER-scale (Gittes/Howard 5200µm;
1–5mm) — far bigger than any field of view — so ALL observed 2D curvature is NON-thermal (buckling,
pinning, compression), NOT thermal WLC. Crucially Lp RISES with contour length (Pampaloni PNAS 2006:
~110µm at 2.6µm → ~5035µm at 47.5µm) ⇒ **curvature (lateral-dev/length) should FALL with length, with
broad per-MT log-normal scatter** — short MTs wavier, long straighter, from ONE rule. (P2) baseline
near-straight, curvature = imposed modes. (P3) local curvature = half-Gaussian bulk + EXPONENTIAL fat
tail of sharp bends; frame-mean condition-dependent (~0.12µm⁻¹ in-vitro vs ~0.39µm⁻¹ cells). (P4)
frame-level NEMATIC order S knob (isotropic<0.10, lanes 0.10–0.20, coexistence 0.20–0.25, nematic>0.25).
(P5) curvature-dependent breaks (breaking curvature 0.43µm⁻¹). (P6) length family selectable (lognormal/
exp/Gaussian) + short-seed population. FIXED PRIORS (don't tune): protofilament defects (sub-diffraction),
crossing-angle / aster-radial (not well-constrained). IMPLEMENTED (2026-07-05): **length-coupled stiffness**
(`curve_base`/`curve_len_ref`/`curve_len_exp`/`curve_log_std` — replaced the crude straight_prob bimodality);
kept kinks (fat tail), orient_kappa (nematic S), short_prob. Calibrated knobs now include curve_base,
orient_kappa, kink_prob, short_prob (see §12/autonomous run below).

## 12b. AUTONOMOUS improved-generator run (pipeline10, 2026-07-05→06)
All agreed improvements applied together: **region_mmd** calibration (per-region, variance-recovering,
foreground-focused) with **|residual| POLARITY-INVARIANT** input (dark synth matches inverted real);
**whole-frame inversion moved to training AUGMENTATION** (generator now always-dark, `frame_invert_prob=0`;
dino_seg/dino_aff invert 50% of crops); **length-coupled stiffness + new morph knobs** in `build_cfg`;
**400 optimizer trials**; affinity model **upweights attractive channels 3×** + eval **mutex-threshold**.
Pipeline: calibrate → gen IRM(train_final) → IRM+TIRF samples → train orientation head → train affinity →
eval both. Diagnosis + ranked improvements to follow.

## 12. Calibration bottleneck + the fix (per-region matching)
Critical analysis: the approach is correct but its core weakness is **FRAME-LEVEL POOLING** (mean over the
whole frame → matches the distribution of frames). Two coupled consequences: (A) a frame's mean ≈ the mean
over its MTs → BLIND to per-MT VARIANCE (the §4 finding); (B) even with the residual, mean-pool averages
over ~98% background → foreground DILUTION. **Fix (building now): per-REGION distributional matching** —
pool patch tokens per connected foreground component (synth: free instance masks; real: components of the
residual foreground) and MMD-match the per-region distribution. The SPREAD across regions carries per-MT
variance (fixes A); only foreground contributes (fixes B). `CALIB_OBJECTIVE=region_mmd` in calibrate.py.
Secondary (honest) limits: (C) parametric expressiveness ceiling; (D) proxy gap (MMD ≠ transfer).

## 13. Positioning vs SynthMT (PLOS Comput Biol 2026) — the competitor to beat
SynthMT: synthetic IRM MT + foundation model, annotation-free — BUT foreground-blind global layer-5 cosine,
SYNTHETIC backgrounds, NO shared encoder, does NOT train on synth (only HPO-tunes SAM3 text-prompt), IRM
only. Our differentiators: foreground-aware residual-patch-(now region)-MMD on REAL backgrounds, in the
segmenter's own space ("calibrate-what-you-segment"); a synth-TRAINED amodal + affinity/mutex instance
model for crossings/parallels; IRM+TIRF generality; and rigorous negatives (§8).

## 13b. C1 result (2026-07-05) — nuanced, partly against the N1 hypothesis (HONEST)
Head-to-head on the SAME generator, only the calibration objective differs:
- **Param recovery (clear foreground-blindness):** global_cosine (SynthMT) picks density=**7**, residual_mmd
  picks density=**14** (real ≈ 14). global_cosine is blind to MT COUNT — confirms §2/§12.
- **Downstream semantic tol2:** residual 0.935 vs global_cosine 0.941 — **COMPARABLE** (global slightly
  higher!). So "residual beats global on semantic F1" is NOT supported by the coverage metric.
- **Over-firing (calibration faithfulness):** residual **0.7× (clean, faithful)** vs global_cosine
  **1.5×** — global_cosine calibrated to a sparse/low-contrast synth (fg% 0.89) → on real it fires 1.5× more.
LESSON: the tol-coverage metric is OVER-FIRING-TOLERANT (recall-dominated within tol=2), so it does NOT
discriminate calibration quality. The N1 evidence must be framed on **calibration FAITHFULNESS**
(param-recovery + over-firing + variance-recovery via region_mmd), NOT tol2 coverage — and ultimately on
the INSTANCE-F1 (where over-firing produces spurious instances). ⇒ region_mmd + the C1.5 variance-recovery
figure become the DECISIVE N1 evidence, not C1's downstream tol2.

## 14. Current state / next
Working label-free pipeline (calibrated hairpin generator → DINOv2-hybrid + orientation-keyed head →
over-firing clean, semantic tol2 0.94, heuristic instance 0.627). Building: (1) per-region calibration
(§12), (2) learned affinity+mutex instance step (§10), (3) crossing-stress eval data, (4) TIRF (§11).
Compute on tulen. Running experiment: C1 head-to-head global-cosine (SynthMT) vs residual-MMD.

## 16. Instance linker upgrade — gap-bridging features (2026-07-09, Phase 1 of Alice+HTW-max)
Metric = strict centerline INSTANCE-F1; HTW fully SEALED (tune only on Alice). The learned amodal linker
(`learn_amodal.py`/`amodal_eval2.py`) decided "same MT?" from endpoint-local geometry only (7 feats) → couldn't
resolve close PARALLELS (look linkable, nothing connects them) or CROSSINGS (need linking THROUGH the junction).
**Added 5 gap-bridging features (NF 7→12):** foreground-continuity sampled along the bridge between the two closest
endpoints (`_bridge`: mean/min of max-over-channel prob — does signal actually connect?), collinearity offset
(perp deviation → parallels high), gap-vector alignment (both arms ~along the gap). Retrained the MLP on 250
synth_inst_v6 frames (`amodal_mlp_v7link.pt`). **RESULT: Alice 0.653 → 0.697 (+0.044), robust across p_thr 0.5–0.7
(0.686–0.697) — synth-only MATCHES real-trained v19 (0.696). HTW 0.234 → 0.236 (flat).** HTW flatness is
DIAGNOSTIC: HTW instance is SEMANTIC-limited (foreground 0.620 → ~38% of centerline uncovered → no arcs to group),
NOT grouping-limited. ⇒ HTW headroom is in the SEMANTIC model = Phase 2 (thin-filament decoder). v4b foreground
unchanged (primary). Design: `docs/superpowers/specs/2026-07-09-instance-linker-gap-bridging-design.md`.

**Phase 2 — semantic decoder upgrade (v8) = REGRESSION (2026-07-09/10).** Added `SEG_ARCH=aspp` to dino_seg.py:
ASPP multi-scale fusion (dilated convs — DINO blocks are all /14 so a block-FPN gives no spatial pyramid) + deep
supervision (aux head on DINO features) + train-time SCALE augmentation (variable crop→resize, hypothesis: HTW's
tol2→tol5 gap = scale mislocalization). Retrained online, tight calib. **RESULT: semantic REGRESSED on BOTH labs —
Alice tol2 0.940→0.914, HTW 0.620→0.554; fg% ROSE (Alice 1.43→1.86) = predictions thicker/more diffuse.** NOT a
collapse (the high train loss ~1.02 was deep-sup aux inflation; fg% clean). Lesson (fits the SimGAN/Phikon/DR-wide
pattern): **for sub-pixel filaments, added decoder complexity (ASPP dilation, coarse deep-sup) COARSENS predictions
and destroys strict-tol localization — the minimal high-res directional v4b decoder was already near-optimal.** Since
ALICE regressed too (tunable signal), Phase-2 lost without needing HTW. **ISOLATION ablation (scale-aug ALONE on the
base v4b decoder, `dino_seg_ori_scaleaug.pth`): Alice 0.940→0.920, HTW 0.620→0.539 — scale-aug ALONE hurts, and hurts
HTW WORST (−0.081).** So the scale-mislocalization hypothesis for HTW is REFUTED (scale-robustness training degraded
thin-filament localization). **Phase 2 is a DEAD direction in all forms; every lever regresses.** DECISION: BANK PHASE 1.
**Final deliverable = v4b foreground + Phase-1 enriched linker → Alice 0.697 / HTW 0.236 (synth-only SOTA; Alice matches
real-trained v19 0.696).** KEY REFRAME: HTW semantic 0.620 resisted FOUR interventions (DR-wide, ASPP, deep-sup,
scale-aug) — all made it WORSE. A held-out score that robust to model/data changes points to the BENCHMARK not the
model: HTW GT is documented incomplete/noisy (correct-but-unannotated MTs scored as FP) → a MEASUREMENT ceiling no
training change lifts. ⇒ treat HTW as a benchmark to CHARACTERIZE (GT completeness audit), not a model to fix.
Artifacts `dino_seg_ori_v8.pth`, `dino_seg_ori_scaleaug.pth`, `eval_v8_sem.log`, `eval_scaleaug.log`.

## 15. Physics generator + DOMAIN-RANDOMIZATION pivot (2026-07-08/09)
Two big moves after the calibration line above.
**(a) Physics rendering + multi-regime morphology.** Rewrote render from a parametric profile to a
**two-beam IRM interference** forward model (I = B + A·cos(2k·h)·E_INA(h), k=2π·n_water/λ; height h sets
DARK-at-contact ⇄ BRIGHT-when-elevated), ~40→~15 physical params, dirt heavy-tail FIT to real skew/kurt.
Morphology split into **static / gliding / dynamic** regimes (WLC per-regime persistence length; rings,
hairpins, kinks). Two peer-review rounds fixed 4 real bugs (WLC √(2ds/Lp) factor, ring over-coiling,
contrast_floor=0.22 so labeled MTs are never invisible at the zero-crossing, false cross-fringe docstring).
Leakage-free: HTW carved OUT of calibration → held-out test. Online parallel generation (no disk).
Result vs corpus-calibrated v4b: Alice sem 0.940→0.921, instance 0.653→0.509 (multi-regime v6).

**(b) The biased-corpus reframe → DOMAIN RANDOMIZATION (task #17).** The real corpus is a BIASED, incomplete
sample of an UNKNOWN experiment space. So this is domain GENERALIZATION, not adaptation — tight
MMD-calibration to the corpus *overfits* the observed experiments. Research (SynthSeg, DRIFTS, ADR;
wf w2ez5ernh): (i) randomize appearance **beyond realism**; (ii) DRIFTS shows fine-tuning to the observed
target DROPS unseen-domain Dice 80.1→66.7 — the exact MMD-calibrate-to-biased risk; (iii) but generalization
is strongest along the RANDOMIZED axis and SynthSeg gets morphology from real label maps we lack → morphology
needs engineered per-regime FLOORS, not just wide appearance. **Implemented `scripts/dr_cfg.build_cfg_dr`**
(DR=1 online path): anchor only universal physics (λ, n_water, interference law), WIDEN every nuisance beyond
the calib band — ina 0.6–1.15, height_base 15–110 nm (dark⇄bright polarity coverage), contrast 0.4–2.0×,
PSF 0.7–2.0, cross-sensor noise, per-frame dirt 10–85; per-regime FLOORS (static 0.43 / gliding 0.33 /
dynamic 0.24); density 3–26 and orient_kappa 0–3 wide (not corpus-centered). **Expect Alice/HTW at/below v4b
— that is the CORRECT DR cost, not a regression** (the biased benchmark can't reward unseen-experiment
coverage). Validation = leave-one-experiment-out worst-group F1 (report per-lab, never pooled) + a DINOv2
feature-COVERAGE check.

**(c) EMPIRICAL VERDICT (2026-07-09) — DR HURT the near-domain held-out set; keep v4b, DR = evidence.**
Ran the v7-DR chain (online DR train → 2-lab semantic + amodal instance eval) + a DINOv2 feature-coverage
analysis. Results (strict): **widening MONOTONICALLY hurt HTW** (the far-ish held-out lab we can score):

| metric              | v4b (tight calib) | v6 (first wide) | v7-DR (full DR) |
|---------------------|-------------------|-----------------|-----------------|
| Alice semantic tol2 | 0.940             | 0.921           | 0.920           |
| Alice instance      | 0.653             | 0.509           | 0.636           |
| HTW  semantic tol2  | **0.620**         | 0.504           | **0.422**       |
| HTW  instance       | 0.234             | 0.199           | 0.169           |
| train loss          | 0.45              | ~0.60           | 0.73            |

**Why (coverage analysis settles it):** in the residual (foreground) space DR raises synth internal diversity
0.087→0.103 ≈ real 0.108 (DR works as intended) AND pulls every lab's real→synth distance down — BUT a
centroid offset (~0.51) persists and, crucially, **HTW is the CLOSEST lab to synth (0.35 vs 0.55)** → HTW is
NEAR-domain, so tight calibration is optimal and randomizing away only dilutes it (loss 0.45→0.73 = capacity
spread thinner). NOTE the residual op is illumination/polarity-invariant → structurally blind to DR's appearance
widening, so this space understates DR's appearance-axis coverage. **The benchmark has NO genuinely-far
experiment**, so it can only measure DR's COST (near-domain dilution), never its BENEFIT (far-scope coverage) —
exactly the biased-benchmark limitation. **Decision (user): keep v4b as the primary model; retain the DR
machinery (`dr_cfg.build_cfg_dr`, DR=1) + this v4b→v6→v7-DR ablation + the coverage figure as HONEST EVIDENCE of
the calibration↔randomization tradeoff; DEFER the DR verdict until a genuinely out-of-domain test set exists.**
Paper framing: NOT "DR generalizes to unseen experiments" (unsupported by a biased benchmark) but "we quantify
the calibration-vs-randomization tradeoff and honestly scope the generalization claim to held-out conditions
within corpus coverage." Artifacts: `dino_seg_ori_v7dr.pth`, `eval_v7dr_sem.log`, `amodal_v7dr_eval.log`,
`dr_coverage2.log`, `dr_cov_resid_emb.npz`.

## 17. MT-34 benchmark + curvature-bounded instancing (2026-08-10)

User brief: PySOAX works but **breaks microtubules at crossings**; the derivative along a
predicted polyline must be bounded (a kinked microtubule never occurs). Build an extended real
test set, tune instancing on oracle masks first, then on model masks; consider nnU-Net.

### 17a. The benchmark — MT-34 (`data/real/mt34_eval/`, 34 frames, 957 GT polylines)
CVAT task **585** turned out to BE the Alice eval set (job 557, same 12 scenes); re-exporting it
picked up the 2026-06-04 human corrections (229 polylines vs the stale local 231). CVAT task
**586** frames 0-21 are exactly the human-reviewed block — `source="manual"` edits stop at frame
id 21 — 728 polylines, 33 MT/frame. Split 6/6 (Alice) + 11/11 (new), tuning on VAL only.

**Why this benchmark finally lets N2 be measured** (TODO W9): crossings per frame
**Alice 2.2 vs new-22 32.1** (706 crossings total), close-parallel pairs per frame 0.9 vs 5.4,
MT/frame 19 vs 33, median length 358 vs 173 px. Alice genuinely cannot test crossings.

**kappa_max is derived, not tuned.** Over all 957 GT microtubules, |dtheta/ds| at an 8 px
baseline has p99 0.029, p99.5 0.040, p99.9 0.096 and a **maximum of 0.239 rad/px**; the extreme
tail is real hairpins (the top offenders are long `manual` polylines), not annotation noise.
kappa_max = **0.25 rad/px** admits every microtubule ever annotated. Measurement scale matters:
the same data gives max 1.015 at a 2 px baseline, because coarse human polylines (median 5
vertices) make vertex-level turns meaningless. Curvature is therefore always measured at 8 px.

Gotchas fixed on the way: CVAT exports at NATIVE 1x (like Alice; HTW's 1.5x is the outlier, and
`CLAUDE.md`'s "both GTs at 1.5x" is wrong for Alice); frames come in 5 different sizes; the
benchmark contains BOTH IRM polarities (Alice dark only), so it also tests the inversion
augmentation.

### 17b. Why PySOAX breaks at crossings — four structural causes (source read, not guessed)
1. **Single-layer skeleton with first-come-first-served junction ownership.** `_trace_path`
   marks `used[]`; the second microtubule through an X finds the junction consumed and breaks.
2. **45-degree-quantised direction.** The "straightest continuation" compares two single
   8-connected steps, so a shallow crossing is decided at random.
3. **Greedy and order-dependent.** Nothing forces the 4 arms of an X into 2 consistent
   through-paths, and a wrong choice is never revisited.
4. **Skeletonisation deforms the junction** into two degree-3 nodes joined by a bridge (Y-Y),
   biasing the pairing and injecting a kink.
The re-link step (`direction_threshold` ~29 degrees) is a soft, endpoint-local filter that
bounds neither curvature along the polyline nor junction-level consistency.

### 17c. Two instancers (`src/instance/`)
**A — curvature-bounded junction matching.** Skeleton -> junction-cluster contraction -> arcs ->
tangents from a PCA over ~12 px -> per-junction **minimum-cost perfect matching** with a HARD
forbid above kappa_max and an explicit priced "leave this arm open" option -> chain -> smooth
until the bound holds. Fixes causes 2, 3, 4 and half of 1.
A closed form fell out of the shallow-angle case: two bands of half-width r crossing at angle
alpha stay skeletally FUSED over ``L ~= 4r/sin(alpha)`` (15.5 px predicted at 15 degrees, 14.4
measured), so A's shallow-crossing reach is set by MASK WIDTH, not by the algorithm. Hence
`bridge_max_len ~= 4*half_width/sin(alpha_min)` -- a derived parameter with a real cost (it also
absorbs genuine short segments between nearby crossings).

**B — curvature-constrained beam tracing in the orientation-lifted (x, y, theta) graph.** Uses
the K=6 amodal channels as a learned, discretised orientation score; consuming a traced path
retires only the orientation slices it used, so a crossing filament keeps its own slice and
survives at full length -- the one cause A structurally cannot fix. Beam search rather than
Dijkstra because the lifted representation already removes the ambiguity Dijkstra would pay for.
**This is NOT the per-layer approach that scored 0.11**: that segmented each bin independently
and shattered wavy microtubules; here a bin transition is a priced edge in ONE joint graph
(regression-guarded by a test).
Three real bugs found and fixed during development, each worth remembering: consuming only the
exact traced bin leaves duplicates (refine_theta lights 2-3 adjacent bins); at a 1 px step
kappa_max=0.25 allows 14.3 degrees which is LESS than one 15-degree bin, so without quantisation
slack a curving filament cannot turn at all; and the beam prefers 2 px steps, skipping pixels
that then re-trace the same microtubule, so consumption must be spatial as well as angular.

### 17d. ORACLE-mask result, MT-34 VAL (defaults, kappa_max=0.25)
| method | Alice F1 | new-22 F1 | pooled | junction-id (new-22) | frag | bundle | max kappa |
|---|---|---|---|---|---|---|---|
| PySOAX (its shipped tuned params) | 0.863 | 0.587 | 0.684 | **0.045** | 1.33 | 0.194 | **0.512** |
| A (curvature-bounded) | 0.954 | 0.813 | **0.862** | **0.780** | 1.11 | 0.773 | 0.250 |
| B (orientation lift) | 0.823 | 0.801 | 0.809 | 0.785 | 1.15 | 0.479 | 0.141 |

The headline is not F1 but **junction identity 0.045 -> 0.78**: PySOAX keeps a microtubule's
identity through a crossing in 4.5% of cases. And **PySOAX's max curvature 0.512 rad/px is more
than double the sharpest bend in any of the 957 annotated microtubules (0.239)** -- direct
evidence it emits physically impossible centerlines, not merely worse ones.
(Budget-parity caveat: these are PySOAX's OWN tuned parameters, fitted months ago at 1x on
different data. A same-budget Optuna re-tune of all three on this oracle VAL is running.)

### 17e. Model masks — the drop is NOT mainly the instancer
v4b on MT-34: mean predicted fg% 1.93 (vs ~1.6 in-domain synth) = **no over-firing**, and
semantic RECALL is 0.92-0.999 on every VAL frame -- the foreground finds the microtubules.
Semantic precision, however, is 0.86-0.94 on Alice but 0.33-0.84 on task-586. Diagnosis:
- **Not a polarity failure.** Dark frames semP 0.73, bright frames ~0.75 -> the inversion
  augmentation transfers. (First real test of it; Alice is dark-only.)
- **Field-of-view stop.** The task-586 frames were acquired through an OCTAGONAL field stop that
  Alice frames lack. Detections away from GT carry **2.06x the residual contrast of an annotated
  microtubule** (3.65x background) -- too strong to be faint unannotated filaments -- and
  rendering them puts them on the octagon's corner wedges. The generator composites onto real
  EMPTY IRM backgrounds that carry no field stop, so the model never learned a field boundary is
  not a filament. Added `mt_bench.fov` (validated by "every GT vertex must be inside the mask").
  It removes ~40% of the spurious mass at a 10 px erosion, so it is part -- not all -- of the gap.

### 17f. Model masks — the numbers, and what actually limits them
All on MT-34 **VAL**, kappa_max 0.25, instancer parameters tuned on ORACLE masks (a model-mask
re-tune is running; these are therefore a lower bound, not the method's ceiling).

| input | method | Alice F1 | new-22 F1 | pooled | junction-id (new-22) | inst/frame |
|---|---|---|---|---|---|---|
| oracle | A | 0.954 | 0.813 | 0.862 | 0.780 | 37 |
| v4b, whole-frame norm | A | 0.643 | 0.156 | 0.338 | 0.368 | 76 |
| v4b, FOV-based norm | A | 0.616 | 0.134 | 0.315 | **0.687** | 117 |
| v4b, whole-frame norm | B | 0.488 | 0.135 | 0.268 | 0.315 | 99 |
| v4b, FOV-based norm | B | 0.468 | 0.115 | 0.247 | 0.362 | 124 |

**The oracle -> model drop is a SEGMENTATION-side problem, not an instancer one.** Instancer A
recovers 0.86 pooled from a perfect foreground and 0.34 from the predicted one, on identical code.

**Input-normalisation finding (systemic, affects the whole established pipeline).**
`learn_amodal.norm01` takes its 1st/99th percentiles over the WHOLE frame. Every MT-34 frame has
either a field stop or a dark border, which pushes p1 out onto it, so the imaged interior is
compressed into a fraction of [0, 1]: measured interior spread after normalisation is
**5.6-6.4x smaller** than with an FOV-based stretch (Alice included -- 5.58x). The synthetic
training frames have no stop and use the full range, so this is a train/inference mismatch in the
INPUT. Fixing it is preprocessing fidelity, not an inference trick.
Effect, at an UNCHANGED binarisation threshold of 0.35: **recall and junction identity improve
markedly** (new-22 TP 108 -> 138, junction identity 0.368 -> **0.687**, bright-frame micro-recall
0.307 -> 0.399) while precision collapses (FP 729 -> 1147, 76 -> 117 instances/frame), so pooled F1
is flat-to-slightly-worse. The model now sees the contrast it was trained on and fires more; the
threshold has to move with it. That is why the model-mask tuning searches `prob_thr` -- on thick,
noisy predicted foreground the binarisation threshold dominates every matching weight (the
project's own SAM 3 record has a confidence sweep moving F1 0.39 -> 0.71).

**Field-of-view stop.** `mt_bench.fov` masks the imaged field. A 10 px erosion removes 40% of the
spurious prediction mass but deletes up to 6% of the GT vertices on the worst frame; at a GT-safe
4 px it removes ~16%. The stop is therefore a real but PARTIAL cause, and the knob cannot be
pushed further without destroying ground truth. The real fix belongs in the generator: composite
onto backgrounds that have a field stop, or synthesise one.

**Do NOT read the polarity split as a polarity result.** The bright VAL frames are exactly the
dense ones (31-101 MTs), so the instance-level dark-vs-bright gap is polarity x density and is
undecidable at n=16. The SEMANTIC-level claim stands on its own: semantic recall is 0.92-0.999 on
both polarities, so the inversion augmentation does transfer.

### 17g. Tuning discipline
Every instancer gets the SAME Optuna budget (120 trials) on the SAME objective on oracle VAL --
PySOAX's shipped parameters were fitted months ago at 1x on different data, and this project has
already been burned once by an under-tuned baseline (W10, the SAM 3 confidence sweep).
`kappa_max` is NOT a search dimension: it is derived from the ground truth, and letting Optuna
raise it to ~0.34 would buy a little F1 while permitting bends no real microtubule exhibits --
which is the argument the method exists to make. It is reproduced as an explicit ablation
(`--tune-kappa`). **Instancer A tuned on oracle VAL with kappa frozen: 0.9182** (defaults 0.862),
i.e. the physics-derived bound costs essentially nothing.
Frames with zero GT (`training_img_102`) are excluded from the macro mean -- per-frame F1 is
undefined and `centerline_f1` hands out a free 1.0 when both sides are empty -- but keep
contributing false positives to the micro totals, where an empty field is a real test.

### 17h. Baseline parity, and the two negative results
**Same-budget parity (the W10 lesson, applied to ourselves).** Every instancer got 120 Optuna
trials on the SAME oracle-VAL objective, kappa frozen at its GT-derived 0.25:

| method | oracle-VAL mean F1, tuned |
|---|---|
| PySOAX | **0.6662** |
| A (curvature-bounded matching) | **0.9118** |

So the gap is not a tuning-budget artifact -- PySOAX's shipped parameters were already close to
the best its formulation can do on this data. That was the one alternative explanation worth
ruling out before claiming anything.

**nnU-Net (synth-only, ResEnc-M 2D, 250 epochs, same generator + calibration as v4b) MATCHES
v4b but does not beat it.** MT-34 VAL semantic centerline-coverage F1:

| model | Alice tol1 / tol2 / tol5 | new-22 tol1 / tol2 / tol5 | fg ratio (Alice / new-22) |
|---|---|---|---|
| v4b | 0.732 / **0.950** / 0.989 | 0.586 / **0.777** / 0.857 | 1.09 / 1.25 |
| nnU-Net @1.0x | 0.730 / 0.947 / 0.992 | 0.556 / 0.741 / 0.818 | 1.56 / 1.95 |
| nnU-Net @1.5x | 0.729 / 0.941 / 0.987 | 0.550 / 0.737 / 0.812 | 1.15 / 1.49 |

(v4b Alice tol2 0.950 reproduces the documented 0.940, which is the check that the harness is
wired correctly.) nnU-Net was given both scale conventions -- v4b's 1.5x inference scale and its
own 1.0x training scale -- rather than v4b's hand-me-down. Its plan was inspected before
training: 512x512 patches, stage 0 stride [1,1], so the first skip is at full resolution and the
v8/ASPP coarsening failure mode does not apply. Two consequences:
- **v4b stays the primary semantic model.** nnU-Net is the strong standard baseline reviewers
  will ask for, and it is now answered with a number rather than an opinion.
- **nnU-Net cannot feed instancer B.** Its softmax head assigns each pixel exactly one class, so
  it structurally cannot express the amodal orientation overlap that makes B work. That is an
  architectural difference, not a configuration gap.

**The FOV-based input normalisation is diagnosed but NOT a fix.** The contrast measurement in
17f is real (interior spread 5.6-6.4x larger with an FOV-based stretch), and it does raise
junction identity (0.368 -> 0.687) and recall. But semantically it is WORSE on the frames it was
meant to help (new-22 tol2 0.777 -> 0.740) because the extra contrast also raises the foreground
ratio (1.25 -> 1.34): it buys recall with precision. **The whole-frame normalisation stays the
default.** The finding belongs in the generator instead -- synthetic frames have no field stop
and no dark border, so the training distribution, not the inference preprocessing, is what
should change. Recorded as a negative result rather than quietly dropped.

Practical notes for reproducing on tulen: all CPU work runs at
`/home/prusek/mt_enc_exp/mt34_work` (32 cores; `--n-jobs 12` leaves room for nnU-Net
dataloaders). `optuna` needs the same `pysqlite3` shim `synth/calibrate.py` uses, and `rsync`
over ssh breaks on the client PQ-handshake banner -- transfer with `tar` + `scp`.

### 17i. Process hygiene (a near-miss worth recording)
Three separately-launched tuning drivers ended up with **two stale Optuna studies of a
different configuration writing into the same log file and the same `params_b.json`** as the
live one. This does not crash: it produces a plausible number drawn from an unknown mixture.
Two causes, both easy to repeat:
- killing the study children BEFORE their driver -- the driver then advances its loop and
  spawns an orphan that outlives the kill;
- a lock race -- two drivers both passed the "is a run live?" check before either wrote its PID.
Fixed by consolidating everything into `scripts/tulen_finish_all.sh` with a PID lock, killing
driver-then-children, and verifying `pgrep` shows exactly one of each before trusting anything.
Any tuned parameter file produced before this cleanup was discarded and regenerated.

### 17j. FINAL TEST RESULTS (scored once, 2026-08-11)

**Wiring check passed first.** Re-running the tuned parameters through the evaluation
reproduced each study's own best value exactly: A 0.912 vs 0.9118, B 0.855 vs 0.8552,
PySOAX 0.666 vs 0.6662. Only then was TEST scored.

#### MT-34 TEST, ORACLE foreground (instancer quality, human GT)
| method | Alice F1 | new-22 F1 | pooled | junction-id (new-22) | frag | bundle | max kappa |
|---|---|---|---|---|---|---|---|
| PySOAX (tuned, same 120-trial budget) | 0.787 | 0.483 | 0.590 | 0.154 | 1.25 | 0.044 | 0.351 |
| **A — curvature-bounded matching** | **0.939** | **0.868** | **0.893** | **0.916** | 1.08 | 0.544 | 0.249 |
| B — orientation-lifted tracing | 0.833 | 0.802 | 0.813 | 0.793 | 1.14 | 0.370 | 0.119 |

**Junction identity BY CROSSING ANGLE is the result worth reading twice** (new-22, n=66/139/149):

| method | shallow <30 deg | oblique 30-60 | steep >60 |
|---|---|---|---|
| PySOAX | 0.258 | **0.000** | **0.000** |
| A | 0.773 | 0.863 | 0.826 |
| B | 0.606 | 0.899 | **0.946** |

PySOAX preserves a microtubule's identity through a non-shallow crossing **zero times out of
288**. Its shallow-angle 0.258 is not skill: at a shallow crossing the two filaments are nearly
collinear, so even a wrong pairing still covers both probe points.

And the two instancers' angle profiles match the mechanisms they were built from, which is
stronger evidence than the pooled numbers: **B is best exactly where amodality helps (steep,
0.946 > A's 0.826) and worst exactly where its angular consumption width merges the filaments
(shallow, 0.606 < A's 0.773)**. A, working on a single-layer skeleton, is flatter across angle.
The theory predicted both shapes before the numbers existed.

#### MT-34 TEST, PREDICTED foreground
| foreground | instancer | Alice | new-22 | pooled | junction-id (new-22) |
|---|---|---|---|---|---|
| v4b | A | 0.638 | 0.238 | 0.379 | 0.511 |
| v4b | B | 0.564 | 0.168 | 0.308 | 0.277 |
| **nnU-Net** | A | 0.681 | 0.274 | **0.418** | **0.720** |

nnU-Net's foreground instances BETTER than v4b's (0.418 vs 0.379, junction identity 0.720 vs
0.511) despite near-identical semantic scores -- a cleaner, less fragmented mask matters more
downstream than tolerant coverage F1 shows. v4b still wins overall as the pipeline model because
it supplies the amodal channels B needs, but this is a real point for the paper: **semantic
coverage F1 does not rank foregrounds by their downstream instancing value.**
Input normalisation was chosen on VAL (whole-frame 0.438 > FOV-based 0.412), consistent with 17h.

#### SYNTHETIC TEST, exact ground truth (20 frames, in-domain)
| foreground | method | F1 | junction-id | shallow / oblique / steep | max kappa |
|---|---|---|---|---|---|
| oracle | PySOAX | 0.397 | 0.322 | 0.483 / 0.000 / 0.043 | **0.832** |
| oracle | **A** | **0.695** | 0.823 | 0.800 / 0.673 / 0.723 | 0.250 |
| oracle | B | 0.677 | **0.850** | 0.667 / 0.808 / **0.894** | 0.117 |
| v4b | A | 0.180 | 0.454 | — | 0.249 |
| v4b | B | 0.127 | 0.301 | — | 0.147 |

The same ordering, the same angle profile, and PySOAX again at 0.000-0.043 on non-shallow
crossings -- on ground truth that is exact by construction, with no annotator in the loop. Its
max curvature reaches **0.832 rad/px**, 3.5x the sharpest bend any real microtubule exhibits.

**What the synthetic set settles, and one thing it refutes.** Absolute F1 is LOWER than on MT-34
(0.695 vs 0.893 with a perfect mask) because synthetic ground truth is *complete*: it contains
every microtubule the generator drew, including ones a human annotator would not have marked.
The obvious explanation -- "the extra ones are invisible" -- was tested and **does not hold**:
microtubules the foreground misses carry 0.77x the contrast of the ones it finds and are the
same length (median 260 vs 242 px), so they are only marginally fainter. Measured with the same
metric, v4b's semantic tol2 is **0.749 on synthetic vs 0.779 on real new-22**: it does NOT do
better on its own training distribution.

That is the load-bearing conclusion of the whole exercise:

> With a perfect foreground the instancer reaches 0.89 (real GT) / 0.70 (exact GT). With v4b's
> foreground it reaches 0.38 / 0.18 -- and the collapse is just as large IN-DOMAIN, where there
> is no domain gap and no annotation error to blame. **The remaining bottleneck is the semantic
> foreground, not the instancer and not the synthetic-to-real gap.**

#### kappa ablation
Letting Optuna fit the curvature bound instead of deriving it: oracle VAL 0.9214 vs **0.9118**
frozen at 0.25. Permitting bends no real microtubule exhibits is worth **+0.010 F1** -- so the
physics-derived bound costs essentially nothing, and the claim that it encodes physics rather
than a fit survives its own ablation.

## 17k. Three levers, measured (v2, 2026-08-11)

Driven by 17j's conclusion that the foreground -- not the instancer -- was the bottleneck.

### The foreground metric question, settled
Which property of a foreground predicts how well it INSTANCES? Measured over 4 foregrounds
(v4b, v4b+FOV-norm, nnU-Net@1.5x, nnU-Net@1.0x) x 33 frames = 174 paired comparisons, scoring
each property by how often the foreground it prefers is also the one that instances better:

| property | rank accuracy | Spearman rho vs instance F1 |
|---|---|---|
| `cc_per_gt` (components per GT microtubule) | **0.82** | -0.73 |
| `endp_per_kpx` (skeleton endpoints) | **0.79** | -0.75 |
| `gaps_per_mt` (foreground dropouts per microtubule) | **0.79** | -0.66 |
| `rec2` -- half of what we tune on | 0.73 | +0.63 |
| `prec2` -- half of what we tune on | **0.58** | **+0.87** |

**A property can correlate at rho = +0.87 across frames and still rank models at chance.** Both
prec2 and instance F1 track frame difficulty, which manufactures the correlation; between two
foregrounds on the SAME frame, prec2 is a coin flip. Fragmentation measures are not. Concretely
v4b vs nnU-Net: nearly equal coverage F1, but v4b has 2.6x the skeleton endpoints, 2.7x the
components per microtubule and 2x the dropouts -- and instances worse.

**Retraining v4b was deliberately DEFERRED.** nnU-Net trained on identical data fragments 2.6x
less, so the lever is architecture/loss, not data -- and there is no measured hypothesis for
which v4b change lowers `gaps_per_mt`. The project's record on unfounded levers (SimGAN,
Phikon, ASPP, scale-aug, DR) is five negatives. The symptom was compensated in the instancer
instead, and a retrain now has a measured target to be gated on.

### What changed in the instancer
The parallel-merge and the gap-bridging problems turned out to be **one mechanism**: the pairing
cost was blind to DISPLACEMENT. It charged the direct turn ``|theta_a + pi - theta_b|``, so two
parallel microtubules 4 px apart scored EXACTLY zero -- both arms collinear -- while the real
path has to jog sideways and back. The turn is now charged in two parts, via the gap direction.
Added with it: **gap linking** across foreground dropouts (same formula, gated by curvature and
by weak image evidence along the bridge) and an optional **orientation-agreement** term
importing instancer B's amodal evidence, which vanishes without channels so A still runs on
nnU-Net's channel-less mask.

### Lever ablation, VAL (each lever off by a PARAMETER, so the shipped code is what is tested)
| variant | oracle F1 | junction | bundle | model F1 |
|---|---|---|---|---|
| FULL | **0.938** | 0.947 | 0.934 | **0.441** |
| - displacement (direct turn) | 0.560 | 0.075 | 0.182 | 0.252 |
| - gap linking | 0.866 | 0.768 | 0.797 | 0.384 |
| - orientation | 0.928 | 0.945 | 0.906 | 0.441 |

**Read the first ablation row carefully: 0.560 is NOT v1's score.** It is v2's tuned parameters
with the displacement term switched off, which is a broken configuration. v1, with its own
tuned parameters, scored 0.912 on this same oracle VAL. The honest improvement is
**0.912 -> 0.938**; the ablation only orders the levers' importance *within* the tuned config.

**The orientation term is a NEGATIVE result.** +0.010 on oracle VAL and exactly nothing on model
masks (0.441 vs 0.441). The amodal evidence B exploits does not transfer into A's cost as hoped
-- once the displacement term is present, geometry already resolves what the channels would
have. Kept because it costs nothing and is off by default, but it is not the hybrid win that
was hypothesised.

### TEST v2 (scored once, alongside v1)
| setting | v1 | **v2** | delta |
|---|---|---|---|
| MT-34 oracle, pooled | 0.893 | **0.920** | +0.027 |
| MT-34 oracle, new-22 | 0.868 | **0.916** | +0.048 |
| MT-34 v4b foreground, pooled | 0.379 | **0.416** | +0.037 |
| synthetic, exact GT, oracle | 0.695 | **0.710** | +0.015 |
| synthetic, exact GT, v4b foreground | 0.180 | 0.183 | flat |
| junction identity (new-22, oracle) | 0.916 | **0.965** | +0.049 |
| bundle recovery (new-22, oracle) | 0.544 | **0.634** | +0.090 |

**A prediction the data corrected.** 17c argued shallow crossings were mask-width-bound: two
bands of half-width r at angle alpha stay fused over ~4r/sin(alpha), and v1's junction identity
duly sagged at shallow angles (0.773 vs 0.826 steep). In v2 the profile is FLAT --
0.924 / 0.914 / 0.933 across shallow / oblique / steep. The reason is the same fusion length:
a long fused stretch puts the two arm tips far apart, which makes the gap DIRECTION reliable and
hands the displacement term its strongest signal. The fusion is a problem for skeleton topology
and a gift to the cost function. The earlier "mask-width-bound" conclusion was too pessimistic
and is superseded.

### What did not move
Model-mask F1 is still 0.416 against an oracle 0.920, and on synthetic data with exact GT and no
domain gap it is 0.183 against 0.710 -- **v2 does not change 17j's conclusion, it sharpens it.**
Bundle recovery on predicted foreground remains poor (0.174 on new-22). The foreground is still
the bottleneck, and now there is a measured target to aim a retrain at: `cc_per_gt`,
`endp_per_kpx`, `gaps_per_mt`.

### 17l. Two testing lessons from v2
- **Unit tests were silently coupled to a tuned artifact.** ``instance_a`` falls through to
  ``default_params()``, which reads ``params_a.json`` -- so copying a tuning result into the
  repo redefined what the unit tests were testing, and one started failing for reasons that had
  nothing to do with the code. The instancer tests now build their parameters from pristine
  ``DEFAULTS`` explicitly. Any test that can be re-defined by a tuning run is not a test.
- **One test asserted something the physics does not forbid.** It expected gap linking to refuse
  a join between two microtubules offset 15 px over a 12 px gap; that path turns 51 deg and back
  over 19 px, i.e. kappa = 0.093 rad/px, comfortably BELOW the 0.25 bound -- because real
  microtubules do bend that hard. The hard constraint cannot and should not reject it; only the
  cost and ``c_open_link`` can. The test now asserts that instead, which is the mechanism that
  actually exists.

## 17m. Making the two open claims checkable (2026-08-11)

Two things §17k left as prose rather than machinery. Both are now shipped, tested, and runnable;
neither has been executed on tulen yet, because the host was unreachable
(`No route to host`) for the whole session.

### The foreground gate — `src/mt_bench/fg_quality.py`

§17k found that the metric the segmenter is tuned on cannot rank foregrounds by their downstream
instancing value. Reproduced here from the saved 132-row measurement, as **pairwise ranking
accuracy over 198 same-frame model pairs** (within-frame pairs, so frame difficulty cancels):

| property | ranking accuracy | direction |
|---|---|---|
| `cc_per_gt` | **0.82** | lower better |
| `gaps_per_mt` | **0.80** | lower better |
| `endp_per_kpx` | **0.79** | lower better |
| `skel_px_per_gt_px` | 0.58 | — |
| `prec2` (the control) | **0.58** | chance, despite rho = +0.87 |
| `fg` (raw foreground fraction) | 0.75 | *higher* better |

That last row is the finding that shaped the implementation. Across four calibrated foregrounds,
**more foreground predicts better instances** — and every one of the three winners improves
monotonically under dilation, since a flooded mask has one component and no dropouts. An
unconstrained search on these metrics would dilate its way to a perfect score and land exactly on
the failure the standing constraint calls fatal. `select_checkpoint` is therefore **constrained
minimisation**: minimise the continuity score subject to `fg <= 3x` in-domain synth (the same
over-firing ceiling used at prediction time; v4b sits at 1.2x) and `rec2 >= 0.90`, which blocks
the opposite cheat — a nearly empty mask also has few components and, because a microtubule it
misses entirely contributes no dropouts, few gaps. When nothing passes, it returns `None` rather
than the least-bad over-firing checkpoint.

The module is instancer-free (milliseconds, not seconds per frame) so it can run every epoch, and
it returns the full battery — `fg`, `rec2`, `prec2` — rather than only the three winners, because
the constraint has to be checkable from the same call. Selection runs on the **real VAL split**,
which is where the ranking accuracy was measured and is inside the synth-only rule: training data
stays synthetic, TEST stays sealed. The synthetic VAL split is the no-domain-gap control, not a
substitute.

### Uncertainty — `paired_bootstrap` / `bootstrap_ci`

Every head-to-head number in §17 rests on 17 TEST frames. A vs B at 0.89 vs 0.81, or v1 vs v2 at
0.893 vs 0.920, is not a claim until it has an interval. Three design points, each of which
changes the answer:

- **Paired.** One frame multiset per replicate, scored by *both* methods, CI on the difference.
  Two marginal intervals overlap at n=17 for almost any pair of methods — demonstrated in
  `tests/test_bootstrap.py`, where a method that wins on *every single frame* still has
  overlapping marginals. The frame-difficulty variance the two methods share must cancel, not be
  counted twice.
- **Stratified by source task.** Pooled MT-34 is a fixed 6 Alice + 11 new-22 design, and the two
  halves differ enormously (2.2 vs 32.1 crossings/frame). Resampling them jointly injects
  composition variance the measurement does not have.
- **The empty-frame policy re-applied inside every replicate.** `aggregate_benchmark` runs on the
  resampled rows, not on a pre-computed number, so `training_img_102`'s free 1.0 never leaks into
  a macro mean and its false positives keep counting in micro.

`run_oracle_eval.py` now dumps per-frame rows under `_frames`, and `scripts/bootstrap_report.py`
re-derives any interval from a saved report. That separation is the point: **attaching a CI to a
published TEST number must never require re-running the benchmark**, because a second run under
changed code is a second TEST shot however it is labelled. The existing v1/v2 reports predate the
dump, so they need one re-instrumentation pass on tulen under byte-identical frozen params before
the numbers can be restated with intervals — re-instrumentation, not re-scoring, and only if
nothing else changes.

### 17n. What the re-instrumentation found (2026-08-11)

**The intervals.** Oracle MT-34 TEST, 17 frames, 20 000 replicates, stratified by source task:

| method | mean F1 | marginal 95% CI |
|---|---|---|
| A (curvature-bounded matching) | 0.920 | [0.870, 0.966] |
| B (orientation-lifted beam) | 0.813 | [0.717, 0.903] |
| PySOAX (same tuning budget) | 0.590 | [0.470, 0.724] |

| paired difference | value | 95% CI | p |
|---|---|---|---|
| A − B | **+0.107** | [+0.047, +0.182] | <0.001 |
| A − PySOAX | **+0.330** | [+0.219, +0.432] | <0.001 |
| B − PySOAX | **+0.222** | [+0.118, +0.330] | <0.001 |

A and B's *marginal* intervals overlap (0.870 < 0.903) — placed side by side they would read as
indistinguishable. Paired, the difference is resolved with room to spare. That is the whole
argument for §17m's design, arriving on the project's own data rather than a constructed
example.

The same 17 frames on the **v4b predicted foreground** (whole-frame norm):

| method | mean F1 | marginal 95% CI | paired difference |
|---|---|---|---|
| A | 0.416 | [0.337, 0.500] | — |
| B | 0.308 | — | A − B = **+0.108** [+0.063, +0.160], p < 0.001 |

A beats B by essentially the same margin on predicted foreground (+0.108) as on oracle (+0.107),
even though both absolute scores more than halve. **The ranking of the two instancers does not
depend on foreground quality** — which is what makes the foreground, not the instancer choice,
the thing worth spending on.

**A reproducibility failure, found by the gate.** `tulen_ci_reinstrument.sh` was written to
abort if a re-instrumented point estimate disagreed with the published one, and it disagreed
immediately: running "v1" params gave 0.920, not the published 0.893. Cause — `tune_instancer.py`
writes `src/instance/params_a.json` in place, and `tulen_v2_chain.sh` copies that file to
`params_a_v2.json` *after* tuning. So the v2 run **overwrote v1's parameters**, and every later
reference to `params_a.json` has silently meant v2. `params_a.json` and `params_a_v2.json` are
byte-identical, on tulen and locally.

The same holds for the model-mask branch: `params_a_model.json` and `params_a_model_v2.json` are
identical too, so the published v1 model number (0.379) has the same status as 0.893.

The bootstrap then confirmed it independently: the paired v2 − v1 difference is **+0.000
[+0.000, +0.000]** on both the oracle and the model branch — not a small difference, a
bit-identical one, across 20 000 replicates.

v1's tuned values were recovered from `tune_a.log` (trial 107, oracle VAL 0.9118) into
`params_a_v1_recovered.json`, but **that file cannot reproduce the published 0.893**: v1's search
space covered 9 keys and the rest fell back to the DEFAULTS *of the time*, and the cost function
itself changed in v2 (the displacement term and `gap_floor`). The repository is not under version
control, so the v1 code is gone too.

**Fixed 2026-08-11:** `tune_instancer.py` now takes `--tag` and treats params files as
**write-once** — an existing file makes it exit rather than overwrite, and `--force` has to be
asked for explicitly. The very next tuning run (§17q) would otherwise have overwritten
`params_a_model.json` a second time.

Consequences, stated plainly:
- The **v1 → v2 delta (0.893 → 0.920) is not reproducible** and must not be presented as a
  measured improvement with an interval. The v2 numbers stand on their own; the delta is
  development history.
- Everything about the *current* system is fully instrumented and reproducible: the table above,
  and the model-mask numbers below.
- Two process fixes this earns: tuning must write a **new, named** params file rather than
  mutating a canonical one, and this repository needs version control before any paper number is
  frozen. Neither is optional for a reviewer-proof claim.

### 17o. The gate, validated against a known foreground (2026-08-11)

`train_gated.py` re-implements `learn_amodal.prob_channels` so the gate can score a *live*
model instead of a checkpoint on disk. A re-implementation that quietly disagreed would corrupt
every selection decision it makes, so it was checked against v4b, whose standalone numbers are
already on record (`scripts/check_gate_on_v4b.py`, MT-34 VAL vs the published all-frame values):

| property | gate path (VAL, 16 frames) | standalone (all 33) |
|---|---|---|
| `prec2` | 0.735 | 0.728 |
| `rec2` | 0.975 | 0.971 |
| `gaps_per_mt` | 1.396 | 1.560 |
| `endp_per_kpx` | 21.02 | 23.57 |
| `cc_per_gt` | 7.86 | 10.69 |
| `fg` | 2.04 % | 1.93 % |

The two coverage numbers agree to three decimal places; the continuity numbers sit below the
all-frame values because VAL holds fewer of the crossing-dense new-22 frames. The path is sound.

Two things fell out of the check:

- **`fg_empty` = 0.000 %.** On MT-34's zero-GT frame, v4b predicts *not a single* foreground
  pixel. That is the strongest possible over-firing evidence and the benchmark had never
  produced it, because the empty frame is excluded from every scored metric. The gate now uses
  it as a probe, and applies the over-firing ceiling to the worst of the two foreground
  fractions — an annotated frame can hide over-firing behind real microtubules; an empty field
  cannot.
- **v4b's continuity score on VAL is 0.841**, not 1.0, because `REFERENCE` anchors the scale to
  its all-frame values. That number is the operational bar: a checkpoint must score *below*
  0.841 on VAL to be more continuous than the model currently in use.

The smoke run also confirmed the failure path end to end: after one epoch the decoder predicts
nothing, `foreground_quality` returns `None` for every frame, and `select_checkpoint` refuses to
select rather than shipping the least-bad collapse.

## 17p. The gated foreground retrain — a negative result (2026-08-11)

30 epochs, online synthetic generation, the v4b recipe unchanged (`SEG_MODE=ori`, `MASK_HW=1.0`,
`POS_W=8`, `CLDICE_W=0.1`). Validation every 2 epochs on the real MT-34 VAL split, selection by
`fg_quality.select_checkpoint`. This is the first time the project has selected a checkpoint at
all: `dino_seg.py` runs N epochs and overwrites one path with the final weights.

### What the training curve showed

**Coverage F1 barely distinguishes checkpoints.** Across the 15 validated epochs it spans
0.825–0.873 — a 5.8 % relative range — while the continuity score spans 0.634–1.289, a 103 %
range. The metric the segmenter has always been selected on is nearly flat over exactly the
choice it would be making. That is §17k's finding reproduced *within* a single training run,
not across models.

**But on this run the two rules agreed.** Both `argmin` continuity and `argmax` coverage F1
picked epoch 28, so the gate's marginal value *over coverage F1* was untestable here. What it
did change is the actual prior practice — shipping the last epoch — which would have taken
epoch 30 at continuity 0.781 instead of epoch 28 at 0.634.

The selected model is genuinely more continuous than v4b by the gate's own measure:
`cc_per_gt` 7.86 → **3.71**, `endp_per_kpx` 21.0 → **14.9**, `prec2` 0.735 → 0.793, at
`fg` 1.95 % (v4b 2.04 %) and `fg_empty` 0.00 %. Continuity score **0.634 vs 0.841**, a 25 %
improvement.

### What happened downstream: nothing good

| split | v4b | gated | paired difference | p |
|---|---|---|---|---|
| VAL (17 frames) | 0.441 | 0.409 | −0.031 [−0.070, +0.008] | 0.123 |
| **TEST (17 frames, scored once)** | **0.416** | **0.393** | −0.023 [−0.061, +0.013] | 0.230 |
| TEST · Alice (6) | 0.692 | 0.614 | −0.078 [−0.166, +0.008] | 0.080 |
| TEST · new-22 (11) | 0.265 | 0.272 | +0.007 [−0.022, +0.039] | 0.670 |

**A 25 % improvement in the gating metric produced no improvement in the target metric.**

### Two effects that do survive their intervals, pulling opposite ways

On the crossing-dense new-22 half:

| diagnostic | v4b | gated | paired difference | p |
|---|---|---|---|---|
| false positives | 404 | **327** | **−77** [−116, −43] | **<0.001** |
| fragmentation (instances per GT MT) | 1.182 | **1.293** | **+0.111** [+0.018, +0.239] | **0.006** |
| junction identity | 0.500 | 0.613 | +0.113 [−0.066, +0.396] | 0.461 |
| bundle recovery | 0.174 | 0.282 | +0.109 [−0.231, +0.519] | 0.562 |

The false-positive drop is real and visible: `training_img_101` shows v4b firing along the
octagonal field stop (17 instances, most of them rim artefacts) where the gated model fires
almost nowhere (12 instances, clean rim). The junction-identity and bundle-recovery gains look
large and are **not** distinguishable from noise at n = 11 — without the interval they would
have been reported as a headline.

### The mechanism, and why it matters for the gate's premise

`cc_per_gt` — connected components of the **mask** per ground-truth microtubule — halved.
`fragmentation` — predicted **instances** covering one ground-truth microtubule — got
significantly *worse*. The mask became more connected while the instancer still cut it into
more pieces. **The proxy moved without the target moving**, and the strict metric's 0.95
coverage requirement means extra pieces cost recall roughly as fast as fewer false positives
buy precision. The two cancel; F1 does not move.

`fg_quality`'s 0.79–0.82 ranking accuracy was measured **across four different foregrounds**.
This run tests something it was never validated for — ordering checkpoints *within one
training trajectory* — and it does not transfer. The honest scope of the claim is therefore
narrower than §17m implied.

**Decision: v4b stays the primary foreground.** The retrain is kept as evidence, not shipped.
The instancer numbers in §17n are unaffected.

## 17q. Removing the last real annotations — and getting a better system (2026-08-11)

The pipeline was described as annotation-free. That was true of the semantic model, which
never sees a human label, and **false of the instancer**: its 17 hyperparameters were fitted
by Optuna against human polylines on the real MT-34 VAL split. The weakest sentence in the
paper, and the easy fix was never tried.

Same 17 knobs, same 100-trial budget at `n_jobs 12`, fitted instead on **synthetic VAL** —
20 frames whose ground truth is exact and free, because the centerlines *are* the objects the
generator drew. Then MT-34 TEST, scored once, paired against the real-VAL-tuned parameters on
the same 17 frames.

| split | real-VAL-tuned | synth-tuned | paired difference | p |
|---|---|---|---|---|
| MT-34 VAL | 0.441 | 0.466 | — (development check) | — |
| **MT-34 TEST pooled** | **0.416** | **0.457** | **+0.041** [+0.018, +0.065] | **<0.001** |
| TEST · Alice (6) | 0.692 | 0.695 | +0.003 [−0.022, +0.029] | 0.855 |
| TEST · new-22 (11) | 0.265 | **0.327** | **+0.062** [+0.029, +0.095] | **<0.001** |

**Parameters fitted without a single human annotation beat parameters fitted directly on the
real validation split** — including on that split itself (0.466 vs 0.441). The gain is entirely
on the crossing-dense half; Alice is flat.

### Why, and why it is not a fluke

Unlike §17p's retrain, both error modes improve at once. On new-22:

| diagnostic | real-VAL-tuned | synth-tuned | paired difference | p |
|---|---|---|---|---|
| false positives | 404 | **326** | **−78** [−120, −41] | **<0.001** |
| fragmentation | 1.182 | **1.145** | **−0.037** [−0.087, −0.002] | **0.028** |
| false negatives | 194 | 182 | −12 [−30, +4] | 0.168 |
| junction identity | 0.500 | 0.537 | +0.038 [−0.014, +0.091] | 0.149 |

§17p cut false positives by an almost identical 77 and paid for it with *more* fragmentation,
so F1 did not move. Here the same-sized FP cut comes with *less* fragmentation, and F1 moves.

The parameters say what changed:

| knob | real-VAL | synth | direction |
|---|---|---|---|
| `w_kappa` | 8.99 | **16.11** | trust the curvature physics nearly twice as much |
| `min_length` | 33.8 | **44.7** | discard short fragments far more aggressively |
| `window` | 20.2 | **28.4** | fit tangents over a longer, more stable baseline |
| `smooth_size` | 7 | 11 | more smoothing |
| `bridge_max_len` | 30.4 | 11.5 | absorb only genuinely short crossing bridges |

Every one of those is a move toward being **more conservative about what counts as a
microtubule** and **more committed to the physical constraint**. That is what exact ground
truth rewards. Human ground truth cannot reward it: MT-34's annotations are human-corrected
model output and are demonstrably incomplete on sparse frames, so a tuner scored against them
is pushed to be permissive in order to recover filaments the annotator drew — and permissive
settings manufacture false positives everywhere else. **Tuning on real VAL was not just
unnecessary, it was actively fitting the annotation noise.**

### Consequences

- **The pipeline is now annotation-free end to end.** No human label enters training or tuning
  at any stage. This is the claim the project wanted and could not previously make.
- The shipped configuration changes to `params_a_model_synthtuned.json`; MT-34 TEST on the
  predicted foreground is **0.457**, up from 0.416.
- The instancer's *oracle-mask* parameters (`params_a_v2.json`) are still real-VAL-tuned. That
  path is a diagnostic ceiling rather than the shipped system, but the same treatment is owed
  before the oracle numbers go in a paper.
- Multiplicity, stated honestly: this session scored MT-34 TEST several times (v1/v2
  re-instrumentation, the gated retrain, this). Each comparison was declared before it ran and
  each is reported with its interval, but the accumulated exposure is real and belongs in any
  write-up.

## 18. Cross-frame tracking (2026-08-12)

The generator produced only stills, so nothing could train or measure a tracker. `mt_sequence`
adds time at the seam that already separated morphology from appearance, and
`instance.tracker` associates across frames with the junction matcher pointed sideways.

### The representation

A microtubule is a **window of arclength on a longer fixed path**, which makes all three
regimes one mechanism: static holds the window, gliding advances both ends together (so the
filament slides along its own contour and conserves length by construction), dynamic
instability moves the ends independently under catastrophe/rescue. Measured: gliding tip
advance tracks the sampled speed at ratio 1.00, r = 0.989; the arclength window is conserved
to 2e-13. `n_frames=1` is byte-identical to `generate_frame`, pinned by a test.

### Two bugs that only measurement could find

- **Velocity came out halved.** Projecting the head of frame t+1 onto frame t **saturates**
  once the filament has advanced past t's end — the nearest point is the last vertex however
  far it went — so averaging that with the correct tail estimate reported 3.55 px/frame too
  little. Fixed by using interior matches only.
- **Drift estimation measured motility.** A gliding filament's centroid travels along its own
  contour at the full gliding speed, and in a gliding field *every* filament does, so the
  median centroid shift returned 2.91 px of drift on sequences that had none. Only the
  component perpendicular to a filament's own tangent is drift-free, so drift now comes from
  normal-flow constraints pooled across filaments at different orientations — the aperture
  problem, solved by orientation diversity. On a drift-free sequence the estimate fell to
  0.32 px.

### Results — synthetic sequences, 24 sequences x 5 frames

| metric | oracle val | oracle test | model val | model test |
|---|---|---|---|---|
| detections / frame | 11.4 | 8.4 | 11.9 | 8.8 |
| identity switch rate | 0.006 | 0.011 | 0.023 | 0.012 |
| track completeness | 0.650 | 0.655 | 0.472 | 0.491 |
| **tracks per object** | **1.77** | **1.59** | **3.50** | **3.21** |
| velocity error (px/frame) | −1.50 | −1.87 | −1.01 | −1.96 |

Val and test agree closely, which is the first thing to check on a benchmark this new.

**Association is not the problem.** The identity switch rate is 0.6–2.3 % and barely moves
between a perfect foreground and the real one: 2–5 switches in 170–354 links. The geometry-first
bet held — at these displacements a learned association has nothing to add, and it was not
built.

**Fragmentation is the problem, again.** Tracks per object doubles from ~1.7 to ~3.4 when the
foreground becomes real, and completeness falls from 0.65 to 0.48. Each microtubule is followed
by three disconnected tracks instead of one. This is the same bottleneck §17p and §17q found,
now visible in the time dimension, and it is exactly the quantity a temporal semantic model
would attack.

### Open, not fixed

- **Velocity is systematically 1–2 px/frame low in every condition, including oracle.** Against
  speeds of ~5 px/frame that is a 20–40 % underestimate, and it is not the halving bug (that
  one is fixed and pinned). The remaining bias is present with a perfect foreground, so it is
  in the estimator or in the instancer's endpoint trimming, not in the segmentation. No
  velocity number should be quoted until this is understood.
- Oracle tracks-per-object of 1.6–1.8 is higher than MT-34's oracle fragmentation (~1.05)
  would suggest. Whether that is the metric's 50 %-coverage assignment rule, or filaments
  leaving the field in the gliding regime, is not yet established.
- **A dropped foreground was the first result and it was my error, not the model's.** The
  initial sequences were built on uncropped 2048² backgrounds while the reference set uses a
  768² centre crop; the same filament count spread over seven times the area gave 0.01 %
  foreground against the expected ~1.6 %. It looked exactly like a broken model. The generator
  now crops to match, and predicted foreground came back at 1.3–1.8 %.

### 18a. The velocity underestimate, diagnosed (2026-08-12)

§18 left a 1-2 px/frame velocity bias open and said no velocity number should be quoted until
it was understood. It now is, and the answer is not an estimator bug.

**What it is not.** Four candidates were tested and eliminated:

- *Curvature.* Nearest-neighbour arclength projection is exact on arcs down to R = 50 px
  (κ = 0.02 rad/px): measured error 0.00 at every radius.
- *Drift subtraction.* It works, and demonstrably: with drift switched off the error is
  +0.27 px/frame; at 1.0 and 2.5 px/frame of drift, the error without subtraction rises to
  +0.51 and +1.71, and **with** subtraction stays at +0.26 regardless. Removing drift is
  removing drift, not signal.
- *End trimming.* Sweeping `edge_frac` from 0.05 to 0.45 changes the answer by nothing at all.
- *The estimator.* On ground-truth polylines from the actual sequence files, the median
  per-pair ratio of measured to true shift is **0.984 and 1.030** on two gliding sequences,
  and `track_velocity` reproduces it exactly.

One genuine defect *was* found and fixed on the way: `eval_tracking.py` compared a shift
measured in the 1.5x eval frame against a speed stored at native scale. That 1.5x mismatch was
partially cancelling the real bias, which is why fixing it made the reported number look worse
(−1.87 → −4.48 px/frame). An error that hides another error is the worst kind.

**What it is: fragmented detections cannot carry a velocity.** With the same estimator, same
drift handling and the same sequences, the ratio of measured to true speed is

| detections | ratio to truth |
|---|---|
| ground-truth polylines | **0.98–1.03** |
| instancer output, per frame pair | **0.543** |
| instancer output, tracks surviving all 5 frames | **0.315** |

Two effects compound. A detection's extent is set by where the mask is strong, not by which
material the filament is made of, so as the filament glides the detection re-forms in roughly
the same place and reports less motion than the material has. Then a **selection effect** on
top: the tracks that survive every frame are the ones whose detection is stable, and a
detection is stable precisely when it is anchored to something that is not moving — so the
population you can measure velocity from is skewed towards the slowest apparent motion. The
longer the track, the worse the bias, which is the opposite of the usual intuition that longer
tracks are better evidence.

**Consequence.** Gliding velocity is not measurable from this foreground at all, and no
estimator change will fix it: the information is destroyed upstream. The fix is the same one
§18 already identified for identity — less fragmentation — which is what the temporal training
is testing. Until that lands, velocity stays unreported rather than reported with a caveat.

## 19. Temporal context: it works, and the control is what proves it (2026-08-12)

Two 30-epoch runs on kajman's free L40S, identical except the one variable: `p_single 0.35`
(65 % of batches see `(t-1, t, t+1)`) against `p_single 1.00` (always `(t, t, t)` — same
generator, same schedule, **zero temporal information**).

No architecture change was needed. `DinoSeg` already takes three input channels because the
frozen DINOv2 backbone expects RGB and the high-res branch starts with `Conv2d(3, 48, ...)`;
today the same frame is replicated into all three. Putting neighbouring frames there makes the
temporal model the same network, and the single frame its exact degenerate case.

### The gate: single-frame quality holds

MT-34 TEST, single-frame mode, paired against the deployed model on the same 17 frames:

| model | mean F1 | vs deployed | p |
|---|---|---|---|
| deployed (v4b + synth-tuned) | 0.457 | — | — |
| control | 0.441 | −0.016 [−0.050, +0.023] | 0.373 |
| **temporal** | **0.471** | +0.014 [−0.031, +0.056] | 0.523 |

Nothing is separable here, which is the point: the gate asks only that single-frame quality not
regress, and it does not. Worth noting the split underneath: the temporal model is *worse* on
sparse Alice (0.632 vs 0.695) and clearly better on the crossing-dense half (**0.382 vs 0.327**).

### The result: fragmentation falls, and only for the temporal model

Synthetic sequences, TEST split, bootstrapped over **sequences** (frames within one are
near-copies; resampling frames would fake precision):

| metric | v4b | control | temporal | temporal − control | p |
|---|---|---|---|---|---|
| **tracks per object** | 3.181 | 3.547 | **2.738** | **−0.810** [−1.354, −0.328] | **<0.001** |
| **track completeness** | 0.559 | 0.523 | **0.614** | **+0.091** [+0.011, +0.170] | **0.023** |
| identity switch rate | 0.007 | 0.011 | 0.007 | −0.004 | 0.763 |

Against its matched control the temporal model is significantly better on both fragmentation
measures. Each microtubule is followed by 2.74 tracks instead of 3.55, against an oracle
reference of 1.59 — roughly **40 % of the gap between the real foreground and a perfect one,
closed by information that was already in the video.**

**Against the deployed v4b the same difference is not separable** (−0.443 [−0.954, +0.087],
p = 0.100). v4b sits between control and temporal, and 12 sequences cannot resolve that
narrower gap. So the honest claim is: *temporal context demonstrably reduces fragmentation
relative to an identically trained single-frame model*, not yet *the temporal model beats the
deployed one*.

### Why the control earned its GPU

Without it this would have read as "the new sequence generator improves tracking". It does not:
the control is **worse than v4b on every metric** (3.55 vs 3.18 tracks per object, 0.523 vs
0.559 completeness, and −0.016 on MT-34). The generator and schedule contributed nothing; the
temporal channels contributed all of it. Half the compute bought the only interpretation that
survives scrutiny.

### Still not fixed

Velocity remains unmeasurable: the ratio to truth is 0.397 for the temporal model against 0.418
for v4b. §18a explains why — a detection's extent is set by the mask rather than the material,
and the tracks that survive longest are the ones anchored to something stationary. Less
fragmentation is necessary but not sufficient, and no velocity number is quoted.

**Next, and cheap:** more synthetic sequences. Twelve is too few to resolve temporal against
v4b, and sequences cost nothing but GPU time.

## 20. Bottleneck, located by elimination (2026-08-13)

The predicted foreground was assumed to be the bottleneck, and it is — but not for any of the
reasons previously written down. Four candidate causes were measured and eliminated:

| candidate | measurement | verdict |
|---|---|---|
| missing coverage | 99.2 % of GT length has foreground within 5 px (the metric's own tolerance) | **not it** |
| poor localisation | 89.3 % of skeleton pixels lie within 5 px of a GT centerline | **not it** |
| mask too thick | predicted width median **4.00 px**, oracle **4.00 px** | **not it** |
| false branch points | predicted 2621 branch points vs oracle 2689 | **not it** |

What is left, and it is stark:

| | branch points | endpoints | **connected components** | skeleton px |
|---|---|---|---|---|
| oracle | 2689 | 968 | **294** | 219 486 |
| predicted | 2621 | **4302** | **2043** | 243 060 |

**The mask is shattered: 7.0x more components and 4.4x more loose ends for the same 494
microtubules**, with the same width, the same branch topology and complete coverage. The breaks
are sub-pixel — a one-pixel hole splits a component while costing one pixel of coverage, which
is why every coverage-based metric said the foreground was fine.

That single fact explains the numbers that did not add up. The instancer repairs most of it
through arc-level gap linking, so instance fragmentation is only 1.13 — but the repair fails
exactly where the project's two stated bottlenecks live: **junction identity 0.965 → 0.500** and
**bundle recovery 0.634 → 0.238** when the oracle mask is swapped for the predicted one. A
filament broken near a crossing arrives as a stub rather than an arm, and a broken parallel
cannot be told from its neighbour.

### What was tried and did not work

- **Morphological closing**, isotropic radius 1–3: fragmentation 1.13 → 1.10, F1 unchanged
  (0.457 → 0.455 at r=2, and 0.425 at r=3). All differences except r=3 sit inside the ±0.08
  interval on 17 frames, so the honest reading is *no measurable effect*.
- **Anisotropic closing along the orientation channels** (line element per channel, so gaps are
  bridged along a filament and never across it): fragmentation 1.11, F1 0.451 — again flat. One
  side effect is worth keeping: **bundle recovery 0.238 → 0.301**, the first measurable use ever
  found for the orientation channels, whose `w_ori` term measures exactly 0.000.
- Both add pixels, and the strict metric wants 95 % *precision* coverage, so what is gained in
  connectivity is paid back in precision.

### Two corrections to earlier claims in this document

- The "model is confidently blind on 15.8 % of each microtubule" line of reasoning was **an
  artefact of my own diagnostic**, which sampled the probability at the exact GT pixel with zero
  tolerance. At 1 px it is 3.7 %, at the metric's 5 px it is **0.8 %**. Ninety-three percent of
  the apparent blindness was one pixel of annotation offset, visible directly in
  `gap_viz/`: five of the six longest "gaps" have a clearly visible filament with the GT line
  drawn beside it.
- The follow-on hypothesis that the generator lacks long faint stretches is **refuted**: under
  real training conditions it produces 10.6 % of faint length in runs ≥10 px against reality's
  2.5 %, i.e. four times too many. The earlier contrary measurement used `data/synth_eval`,
  which predates the physics renderer.

### 20a. Connectivity improves 42 % and F1 does not move (2026-08-13)

`scripts/mask_topology.py` measures the quantity §20 identified as the failure. On MT-34 TEST:

| model | components / MT | endpoints / MT | branch points | fg % |
|---|---|---|---|---|
| **oracle** (target) | **0.60** | **1.96** | 2689 | 1.58 |
| v4b (deployed) | 3.82 | 7.95 | 2402 | 1.85 |
| **temporal** | **2.20** | **5.30** | 2771 | 1.87 |
| control (same recipe, no temporal input) | 3.74 | 8.32 | 2858 | 1.98 |

The temporal model cuts components per microtubule by **42 %** and endpoints by **33 %**, and
its matched control does not move at all. That is the strongest evidence yet that temporal
context repairs the shattering, and it lands exactly on the metric §20 nominated.

**And its MT-34 F1 was +0.014, not separable from zero.** Connectivity improved by 42 % and the
target metric did not move.

The likely reason is that the instancer was already repairing it: instance-level fragmentation
is 1.13 even at 3.8 mask components per microtubule, because arc-level gap linking bridges the
breaks before they reach the metric. If so, improving mask connectivity is work the pipeline
was already doing, and §20's diagnosis — while correct that the mask is shattered — was wrong
that this is what limits F1.

This is recorded **before** the cbDice runs finish, because it recalibrates what they can be
expected to show: cbDice targets connectivity, and connectivity has just been improved 42 %
with no F1 gain. The runs continue on otherwise idle cards and may still differ — the temporal
model raised branch points (2402 → 2771) while lowering breaks, so "better connectivity" is not
one axis — but a third independent null would say the limit is in the instancer or in the
ground truth, not in the mask.

### 20b. cbDice: a clean negative, and a lesson about reading one's own instruments

Two weights, 45-epoch recipe, prodA/prodB as the `cbdice_w = 0` control.

| run | weight | foreground | gate | continuity over training |
|---|---|---|---|---|
| cbA | 0.5 | 2.1–2.8 % | PASS throughout | 0.563 → 0.603 → 0.671 → 0.762 → 0.762 |
| cbB | 2.0 (the reference value) | **12–92 %** | **FAIL at every validation** | 0.485 → 0.281 → … |
| control | 0 | 1.9–2.0 % | PASS | best 0.610 / 0.611 |

**At the reference weight the model collapses to all-foreground** — the same failure this
project already recorded for clDice at 0.5. Predicting MT-34 with cbB's best-scoring checkpoint
gives 25.3 % foreground against the ~1.8 % of every healthy model, and its apparently excellent
connectivity (0.93 components per microtubule, close to the oracle's 0.60) is that collapse: a
mask covering a quarter of the frame has few components and few endpoints by construction.

**At the safe weight it makes things worse**, monotonically, from 0.563 to 0.762 while the
control sits at 0.610.

Both runs were killed at epochs 11 and 13 rather than run to 45. cbA was degrading with every
validation and cbB could never produce a shippable checkpoint, so the remaining ~10 GPU-hours
would have bought nothing.

**The lesson is about the instrument, not the loss.** `fg_quality.passes_overfiring_gate` exists
precisely because the continuity metrics are gameable by dilation — that is written in its
module docstring, with the measurement (raw foreground fraction ranks foregrounds at 0.75)
behind it. It flagged `gate=FAIL` on every single cbB validation. I read the score column out of
the training log without the gate column beside it and called 0.281 promising. The guard worked;
the reading did not. Any future report of a continuity score must carry its gate verdict in the
same sentence.

### What this makes three

Three independent interventions have now improved mask connectivity or fragmentation without
moving F1: the gated retrain (§17p), morphological closing in both forms (§20), and the temporal
model, which cut components per microtubule by 42 % for +0.014 F1 (§20a). cbDice adds a fourth
data point of a different kind — it could not improve connectivity at all without collapsing.

The weight of evidence now says the mask is **not** what limits F1, and the search should move
to the instancer or to the benchmark's own ground truth. TODO lever #3 — how many of the 326
false positives are unannotated real microtubules — is the cheapest next test and is unblocked.

## 21. A day of interventions, and a consistent null (2026-08-13)

Five independent attempts to raise F1, all measured, none of which moved it:

| intervention | what it did move | F1 |
|---|---|---|
| gated retrain (§17p) | continuity metric −25 % | −0.023, p=0.23 |
| morphological closing (§20) | fragmentation 1.13 → 1.10 | flat, −0.032 at r=3 |
| anisotropic closing (§20) | bundle recovery 0.238 → **0.301** | flat |
| temporal model (§20a) | mask components/MT **3.82 → 2.20** | +0.014, p=0.52 |
| cbDice (§20b) | collapsed at w=2, degraded at w=0.5 | not shippable |
| field-of-view erosion (below) | false positives 364 → 327 | **+0.001 on VAL** |

### The false positives are not what limits F1 either

`audit_false_positives.py` split the 367 false positives on TEST into what they actually are:

| category | count | share |
|---|---|---|
| fragment of an **annotated** microtubule | 265 | 72.2 % |
| away from every annotation | 102 | 27.8 % |

**Seventy-two percent are not false detections at all** — they are pieces of annotated
microtubules that failed the 95 % coverage rule, counted once as a false positive and again as
a false negative for the ground truth they failed to cover. "Precision is the larger loss on
the crossing-dense half" was a misreading of the same fragmentation, wearing a different label.

The remaining 102 carried 7.66x the contrast of the same curve dropped at random, which looked
like unannotated real filaments and would have meant the measurable ceiling is below 1.0.
**Looking at them refuted that**: they are the field-stop edge. 70.6 % lie within 20 px of the
aperture boundary, where **0.0 %** of true positives do (median distance 11 px against 489 px).

Eroding the field-of-view mask further does remove them — 364 → 327 false positives on VAL —
and F1 does not move: 0.466 at the current 4 px against 0.467 at 12 px, with the whole sweep
flat between 0.449 and 0.467.

### Two near-misses the discipline caught

- **cbDice's collapse.** Its continuity score of 0.281 was the best number seen all day and I
  reported it as promising. Every one of those validations carried `gate=FAIL` with 12–92 %
  foreground in the same line of the log. The over-firing gate was right; reading the score
  column without the gate column was not.
- **A TEST-tuned +0.017.** The erosion sweep was run on TEST first and showed 0.457 → 0.474 at
  12 px. Repeating the selection on VAL showed +0.001 and a flat curve. Taking the first number
  would have been fitting the test split, and it would have looked like the day's best result.

### Where this leaves the search

Coverage, localisation, width, branch topology, mask connectivity, and false positives have now
each been measured and each eliminated as the limiting factor. What has not been examined is the
**instancer's 95 % coverage rule itself** — the point at which a well-formed mask becomes an
instance that either matches a ground-truth microtubule or does not. 72 % of the false positives
are created exactly there, and the instancer's own parameters were fitted on synthetic data with
exact ground truth (§17q), against which this rule behaves differently than it does against
human polylines that are 1 px offset (§20).

## 22. Tracking measured on v5H, and a metric that was flattering it (2026-08-26)

§18 and §19 measured tracking on v4b and on a temporal variant of it. The deployed model is
v5H — a different architecture, a different generator, its own instancer parameters — and the
tracker had never been run on its foreground. The deployed package carried `instance/tracker.py`
but nothing that called it.

### Reproducing the benchmark before comparing anything to it

The sequence set was lost with the rest of tulen's tree; only `manifest.json` survived. It was
regenerated from the same seed and the same **original** calibration — deliberately not v5H's
own SNR-corrected one, which would have handed the model its own generator's video and made the
comparison a home fixture. The regenerated manifest is **byte-identical** (md5 `f4a319abe3`),
and the oracle arm reproduces §18 to the exact switch counts:

| | det/frame | switches | switch rate | completeness | tracks/obj |
|---|---|---|---|---|---|
| oracle VAL, §18 / re-run | 11.4 / 11.4 | — / 2/354 | 0.006 / 0.006 | 0.650 / 0.650 | 1.77 / 1.77 |
| oracle TEST, §18 / re-run | 8.4 / 8.4 | — / 3/276 | 0.011 / 0.011 | 0.655 / 0.655 | 1.59 / 1.59 |

Identical rasterised ground truth means the **morphology** is reproduced exactly; the frames'
appearance cannot be verified against files that no longer exist, so the caveat on any
cross-run comparison is scoped to appearance alone.

### The metric was rewarding a model for missing microtubules

`tracks_per_object` averages over every GT object, and an object nobody detected contributes
**zero tracks**. A foreground that simply misses more therefore scores as *less* fragmented —
an error in the measured model's favour, in exactly the direction that would have been quoted.
`eval_tracking.py` now also reports `detected_frac`, `tracks_per_detected_object` and
`completeness_detected`. The patch is additive and was **proven a no-op**: all ten pre-existing
keys are identical across all four arms of the run that preceded it.

Measured, the suspicion was wrong in an interesting way: v5H's detected fraction (0.88 TEST) is
*higher* than the oracle's (0.80), because the oracle loses filaments shorter than
`min_length`. The comparison was not inflated.

### Results, TEST split, both benchmarks

| benchmark | arm | switch rate | trk/obj | found | compl\|det | trk/det obj |
|---|---|---|---|---|---|---|
| original (§18/§19) | oracle | 0.011 | 1.59 | 0.804 | 0.815 | 1.98 |
| original (§18/§19) | **v5H** | 0.014 | **2.52** | 0.882 | 0.633 | 2.86 |
| v5H's own generator | oracle | 0.006 | 1.46 | 0.840 | 0.854 | 1.74 |
| v5H's own generator | **v5H** | 0.014 | 3.10 | 0.975 | 0.614 | **3.18** |

**§18's central finding survives the change of model.** Association is not the problem: 3
switches in 207 links, against the oracle's 3 in 276. Fragmentation is: each detected
microtubule is followed by 2.9–3.2 tracks against an oracle 1.7–2.0.

Against the recorded v4b, on the same benchmark and instancer: **2.52 against 3.181** tracks per
object, with the temporal model of §19 at 2.738. v5H is less fragmented in time than either,
without using any temporal information — but this is the mixed metric (the only form v4b's
number exists in) and a cross-run comparison, since v4b's predictions were lost.

### The open negative: these sequences are not v5H's training distribution

Single-frame strict F1 on the sequences is **0.067–0.152**, against 0.495 for the same weights
on real MT-34 VAL. Threshold was the obvious suspect and is **ruled out** — a sweep over cached
probability maps is flat while the foreground fraction moves through its target:

| thr | 0.50 | 0.70 | 0.85 | 0.92 | 0.97 | 0.99 |
|---|---|---|---|---|---|---|
| F1 | 0.073 | 0.066 | 0.077 | 0.071 | **0.078** | 0.067 |
| foreground | 1.58 % | 1.41 % | 1.28 % | 1.20 % | 1.09 % | 0.97 % |

At thr 0.50 the foreground fraction is exactly on target (1.58 % against ~1.6 %) and F1 does not
move. The mask is shattered, not under-thresholded — §20's finding again. What this points at is
`build_synth_seq.py` building its config from a calibration JSON via `gen_train.build_cfg`,
while v5H was trained through `train_v5.py`'s own dataset path; the two do not produce the same
appearance. **Consequence: the tracking numbers above characterise the tracker on a degraded
foreground, and are a lower bound on what this model does on real video.** Not fixed here.

### Shipping it: `track.py`

The entry point the package lacked. It reuses `predict.py`'s `Segmenter` and `instance_a`
unchanged, associates in the **1.5x working frame** where the tracker's gates were fitted, and
emits native-scale coordinates — the one thing that would have failed silently, since
associating at native scale runs fine and merely loosens every gate by 1.5x. Three checks:

- **against the evaluation path**, same sequence: 67 tracks / 97 linked detections against
  66 / 95. Not bit-identical by construction (the package predicts in fp32 at stride 388,
  the trainer in bf16 at 387) and not expected to be;
- **scale mutation**: the same detections tracked at native scale give 71 tracks and 41 linked
  detections against 66 and 47. The frame choice is load-bearing, and the test would have
  caught getting it wrong;
- **in the production container on cvat2**, the bundled sample duplicated into three frames
  named `00`, `02`, `10` (natural sort, not lexical): 65 instances per frame → **65 tracks,
  195 linked detections, zero births after the first frame**, which is the only correct answer
  for identical frames.

Velocity is deliberately absent from the output and from the printed table: §18a established it
is not measurable from this foreground, and a field in a JSON file will be used.

## 23. The mask campaign: where the loss is, and the noise texture (2026-09-02 – 09-05)

Nine days after §22 the question had narrowed to one number: strict F1 0.50–0.52 on the
human-drawn roi303 VAL against 0.95 with an oracle mask. Everything below was measured on that
block, pre-declared in `docs/TODO.md §0.0-CAMPAIGN` before any number was read, and judged by
paired per-frame bootstrap with Holm over the whole family.

### What did not move it

- **The target's width and a prior bank** (2×2 factorial THIN / PRIOR_THIN / PRIOR_THICK):
  null-to-negative in every cell. The only consistent effect in that factorial was **native-scale
  inference**, +0.02 to +0.05 everywhere, adopted as the screen default (p = 0.23 on its own).
- **Inference at the width-matching 1.3×**: +0.006, p = 0.67.
- **Every decoder on the SPARSE35 map** (11 variants on 139 frames: lower thresholds, valley cut,
  hysteresis, NMS ridges, watershed): all worse than the single 0.98 threshold, best −0.022,
  every p < 0.001 in the negative direction. Junction identity rises as the threshold falls
  (0.42 → 0.60) but fusion and false positives rise faster. The map has to change; the decoder
  cannot rescue it.

### Where the loss actually is

- **Crossings fail on a weak arm, not on geometry.** 1 083 GT crossings: identity 22 % with the
  model mask vs 79 % with the oracle mask; 97.6 % of failures have a disconnected mask at the
  crossing and 62 % an arm the model never found. Shallow crossings, the hardest for the
  instancer, are resolved most often.
- **The operating point is per frame and unpredictable.** The optimal threshold spans
  0.30–0.9995 across frames; an oracle per-frame threshold is worth +0.048; no image covariate
  predicts it (|ρ| ≤ 0.24), although density, SNR, bright share, FWHM and grey-level count all
  move the model's confidence (|ρ| ≈ 0.5–0.6).
- **The false positives are the same event as the false negatives.** At 0.9 and 0.98 alike,
  80–85 % of unmatched predictions are fragments or chimeras of ANNOTATED filaments; only 7–11 %
  are off-image texture. Micro F1 barely moves with the threshold (0.487 vs 0.503).
- **The break anatomy** (572 790 GT samples): 22.8 % of real GT samples fall under 0.98. The break
  rate is 66 % below local SNR 0.7 and still 10.6 % above SNR 5; polarity-flipped samples break at
  73 %, non-isolated ones at 37 %. Real and synthetic faint stretches match in width, roughness
  and along-length statistics — the defocus and background-junk hypotheses are dead — yet the
  model bridges the synthetic ones (97 % of label at ≥ 0.98) and not the real ones.

### The benchmark itself

Five frames whose image is empty or saturated but carry annotation, and 35 duplicate polylines in
23 frames, were found by measurement and repaired under a declared rule (`repair_roi_dataset.py`);
every arm from AUG_PHOTO on is scored on the repaired 139-frame VAL against a re-swept baseline.

**What the strict metric can and cannot see (2026-09-05).** The NOISE arm was the first whose
mechanism visibly moved on real maps — per-filament full coverage at p ≥ 0.98 rose from 27 % to
33 % of filaments, fragments fell by 123, missed GT by 110 — and the strict F1 still came out
null at native (−0.015, p 0.11) and negative at 1.5× (−0.030, p 0.013). Decomposing the number
showed two properties of the benchmark, not of the model. First, the macro mean over frames gives
23 % of its weight to 32 near-empty frames that hold 1 % of the filaments, so one or two extra
predictions there outweigh sixty extra true positives on the dense frames (NOISE: −0.063 on the
1–9-GT frames, +0.013 on the ≥ 80-GT frames; AUG_PHOTO has the same signature). Second, the
annotation stops where the human stopped seeing the filament: an anatomy of every unmatched
prediction (`diag_fp_anatomy.py`) put 13 % of the false positives on OVERSHOOT (the prediction
covers its filament and continues past the annotation's end, median 17 px), 4 % on GAP BRIDGES
(one filament stored as two pieces, the prediction runs through both, scored as one chimera plus
two misses), and, on the sparse frames, two thirds of the "off-annotation" predictions on
unannotated visible filaments and a dark band along the edge of cropped frames. The bridged
stretches have a median contrast of 0.42 σ — the models connect pieces through evidence a human
would not call filament. Fragments (50 %) and junction chimeras (15 %) remain the model-owned
majority.

Three consequences were declared and are applied to every arm alike: (i) secondary readings next
to the primary verdict — macro F1 over frames with ≥ 10 GT, micro F1, and the same metric with a
symmetric 25-px END TOLERANCE (`centerline_f1(end_tol=25)`, fixed from the baseline's overshoot
distribution before any arm was scored under it) — none of which replaces the primary; (ii) a
seed-to-seed resolution row (the baseline recipe re-trained with a different seed) that bounds
what any paired difference may claim; (iii) no repair of the GT and no border rule (a darkness
mask dropped a true positive at its smoke test; an edge-band rule would make 59 real filaments
unreachable for +0.002) — the gap and overshoot counts go to the agreement-bias section as
evidence of what human annotation of faint stretches does to a metric with a 95 % rule.

The union-covered-but-unmatched GT (35 % of all misses) was split by `diag_covered_split.py`:
43 % run through two or more mask components separated by 0–5 px cuts (the crossing-region cuts
the curvature-bounded linker does not close), 24 % are chimeras, 5 % instancer cuts, 28 % the
overshoot family. Instance-side ownership is therefore 24 % of the misses; the map side (partial
and missed) owns 65 % — which is why the next arm after NOISE is a LOSS arm on the map
(`--pos-weight 2`, the one segmenter knob never turned), not a sixth generator fix.

**The resolution of the primary, measured (2026-09-05/06).** Three seeds of the baseline recipe
score 0.527 / 0.486 / 0.513 on the primary at native — a span of 0.041 that lies entirely on
the 32 near-empty frames — while the macro over frames with ≥ 10 GT (0.536 / 0.532 / 0.540) and
micro F1 (0.494 / 0.498 / 0.497) agree to within 0.01. The baseline the campaign compared
against is the top draw of the three (+0.018 above the seed mean). Consequences applied to the
ledger: every single-seed arm verdict below 0.04 on the primary is "not resolvable"; the ≥ 10-GT
macro and micro are the resolving readings; at 1.5× the seed pair spans 0.052; on the 50-frame
combined benchmark the seed pair spans 0.07 (0.13 on the HTW block), so that set cannot rank
arms at one seed at all.

**The loss arm, and what it showed (PW2S, `--pos-weight 2`).** At its own threshold (0.70; the
map is flat, 0.46–0.47 from 0.50 to 0.97) it met every pre-registered prediction — fragments
1 632 → 1 056, mask breaks 589 → 274, missed + partial GT 2 514 → 1 678 — and the primary fell
by 0.058, the dense-frame macro by 0.023, micro by 0.016, all beyond seed noise. The gain went
to chimeras (328 → 675; a more connected map hands the instancer more junctions and it resolves
them worse), instancer cuts (71 → 351), and the annotation-owned classes (overshoot +481,
off-annotation +524, gap bridges +82). Map connectivity is not the binding constraint of the
strict F1 on this benchmark.

**The linker closes the crossing cuts (read-only diagnosis, 2026-09-06).** Of 1 286 mask cuts
along the 587 "map-break" filaments (cut length 1 px at the median), the arc graph's short
bridge joins 67 % and the free-end linker 24 %; linker-owned open cuts are 4 % (mostly a wrong
join 13 px away). 485 of 587 filaments have every cut joined and still fail, as chimeras (38 %
of the pool) or on precision and the one-to-one assignment. The instance-side loss is junction
identity under weak arms — the map's confidence at crossings — not linking; the linker's
parameters are not levers.

**The texture lever, replicated and bounded.** A second seed of NOISE reproduced the mechanism
(coverage 0.324 vs 0.325, fragments 1 494 vs 1 509, missed 1 001 vs 1 050) and the cost
(16-bit frames +37 TP against +92 FP); seed means on the resolving readings put the texture
recipe +0.008 above the baseline recipe on the dense-frame macro and on micro, inside the seed
spread. Tripling the dose (NOISE2) reproduced the effect without deepening it (break share below
0.98: 0.228 → 0.211 → 0.196; fragments unchanged) — the lever saturates at NOISE's dose.

### The noise texture, and the arm it earned

Two hypotheses were killed by their own pre-declared checks before costing GPU hours: a
label-visibility gate (the model FINDS the "invisible" synthetic label at p 0.999 — a straight
faint line is trivially integrated) and the generator's invisible-frame mode as a cause (it is
the noise normalisation on dirty background crops, but nothing the model fails on).

The one covariate on which real and synthetic faint stretches differ is the **noise itself**:
the real 16-bit frames' off-filament residual is spatially correlated (lag-1 ACF median 0.19,
p90 0.62) and heavy-tailed (kurtosis median 4), every synthetic frame is white and Gaussian. The
decisive check injected that texture into synthetic renders at inference: label recall at
MATCHED local SNR 1–1.5 fell from 0.90 to 0.56 (real 0.73) and at 1.5–2 from 0.94 to 0.75 (real
0.81), while white noise of the same amplitude changed nothing. **A line integrator trained on
white noise is blind to faint filaments in correlated texture.** The NOISE arm rewrites the
generator's noise model (photon-count reference, a per-frame correlated share at constant
variance, sparse impulses), calibrated with the same instruments against the real band and a
criterion that could fail; its predictions (recall at SNR 1–2 rises, the threshold spread narrows,
the injection check on its checkpoint shows corr ≈ white) are on record before it trains.

Read (2026-09-05): the recall and break-rate predictions were met (break share below 0.98
0.228 → 0.211 in every SNR bin; injection recall at SNR 1–2 0.556 → 0.616), the spread did not
narrow, and the strict F1 did not follow — see "The benchmark itself" above for why. The cost
was concentrated on the 16-bit (textured) frames: +141 false positives against +24 true positives
there, on stretches the annotation does not contain; on the coarse-quantised frames the arm was a
clean +48 / −10. NOISE is closed; a pre-registered dose escalation (NOISE2, correlated share
0.8–0.98, crop ACF 0.43 vs 0.13) runs with a falsifier: it fails if its textured-frame FP rise
again exceeds its TP rise.

### Two bugs caught before they cost a night

The evaluation chain's bash function expanded `"$@"` to its own arguments, which would have
killed every overnight sweep on stray positionals; and the first median-reference noise draft
subtracted `median − span/2` and clipped at zero before the Poisson draw, putting a floor inside
the dark-filament dip range for 40 % of frames — visible only as a lower filament-SNR median in
the calibration screen. Both were found in review, not by a failed run.

### Scale augmentation: robustness, not gain (2026-09-06 – 09-07)

The SCALE arm (render-and-polyline rescale 0.8–1.5 with probability 0.7, nnU-Net low-resolution
simulation on the image with probability 0.2, target rasterised after the rescale) was
pre-registered with one mechanism target: the gap between native and 1.5× inference must close.
It did, on two seeds and two checkpoints each, on every reading: the baseline's three seeds lose
0.052 micro and 0.036 dense-frame macro when the image is upscaled, SCALE loses 0.003 / 0.010 and
its replicate 0.006 / 0.0005 (pick) and 0.001 / 0.005 (ep40). At native the arm is null on the
dense-frame macro and micro for both seeds (within the seed spans of 0.008 and 0.004), while the
second seed's ep40 checkpoint is worse at native beyond the seed spans on every reading (own
threshold 0.80 once the grid was extended to 0.70: dense-frame macro −0.016, micro −0.023, and
a 1–9-GT macro that never exceeds 0.25 at any threshold against the baseline's 0.42–0.50). The
first read had that row at the fast grid's floor; the advisor's rule — an own threshold at a
grid edge is not a read — sent it back for a lower grid, which confirmed the loss as a property
of the checkpoint. So the claim is stated with its count: the gap closes on all four
checkpoint-rows, and native cost is null on three of them. The first read framed SCALE's +0.03 / +0.05 at 1.5× as "the campaign's one positive
reading beyond the seed span"; the advisor pointed out that one exceedance among ~32 such reads
against an n = 3 span is what chance delivers, and that the pre-registered target tells the
cleaner story: **scale augmentation makes the model scale-robust, it does not make it better.**
That is the claim the paper carries, with the sparse-frame caveat.

### The heat-ridge decoder: a lead closed by its own rule (2026-09-06)

The HEAT arm's Gaussian centerline head lost F1 like every other arm, but decoding its ridge
instead of thresholding the existence map raised junction identity from 0.41 to 0.53–0.55 on the
same instancer. Two pre-declared screens on that dump (spur pruning, an existence gate, and the
heat threshold) showed the trade is monotone in the heat threshold: t = 0.8 / 0.7 / 0.6 / 0.5
gives identity 0.41 / 0.52 / 0.54 / 0.55 and micro 0.474 / 0.468 / 0.445 / 0.430 against the
threshold decoder's 0.41 / 0.488, with 200–1 200 extra false positives. The pruning length and
the existence gate move under 1 % of ridge pixels, and even that gate costs 0.011 identity,
because the pixels it drops are the weak arm through the crossing. The identity gain and the
spurs are the same pixels. No variant reached the pre-declared bar (identity ≥ 0.50 with micro
not below the baseline by more than 0.01), so the lead is closed; it survives as the decoder-side
face of the crossing finding of §23 (the loss at a crossing is the confidence of its weak arm).

### The decode threshold is the crossing lever, not the loss (2026-09-07)

The campaign chose its operating point by maximising the strict macro F1, which put it at 0.99.
Sweeping only that threshold on one cached map shows what the choice costs: at 0.97 the macro
falls by 0.003 while junction identity rises from 0.332 to 0.425, bundle recovery from 0.341 to
0.377, micro F1 from 0.494 to 0.508 and the dense-frame macro from 0.536 to 0.547. Fragmentation
is flat across the whole range (1.057-1.084), so the identity is not bought by cutting instances.
The deployed threshold is optimal on the all-frame macro alone -- the reading whose 23 % weight
sits on 32 near-empty frames holding 1 % of the filaments -- and is beaten at 0.97 on every other
reading at once.

Against that, the training-side intervention aimed at the same target is small. Arm 32 put a
weight of 6 on the BCE within 14 px of every crossing (the research note's first recommendation,
with the radius derived from this project's own measurement that the failing arm sits 8-20 eval
px away rather than the note's 6 px). At matched thresholds it buys +0.014 of junction identity,
against +0.093 for the threshold step, and its F1 is null within the seed span. The gain does
survive the matched-threshold check, so it is real and attributable -- it is simply seven times
smaller than moving the operating point. Both results say the same thing in the language of
§23's crossing finding: the loss at a crossing is the confidence of its weak arm, and a region
weight did not make that arm confident enough to clear a high threshold, whereas a lower
threshold accepts it. Crossing metrics should therefore be reported at an identity-chosen
operating point, stated as its own column beside the F1-chosen one.


### Two crossing problems, and the one the arc graph owns (2026-09-08)

The campaign closed at 33 arms with nothing positive on the strict primary, so the question moved
from "which recipe" to "which mechanism". Two measurements settled it.

**In domain and on real, crossings fail for different reasons.** On real data the arm is usually
not in the map at all (51-62 % of failures) — an evidence problem. In domain that nearly vanishes
(5.4 %) and the mask instead runs straight through the crossing while identity is still lost. The
campaign had been treating these as one thing.

**Inside the in-domain half, the discriminator is the ARITY of the arc-graph junction, not its
presence.** Classifying every GT crossing against the graph the instancer itself walks (n = 454,
identity 0.6586, params `src/instance/params_a.json`):

| class | n | identity | with an oracle mask |
|---|---|---|---|
| broken (mask breaks the filament) | 79 | 0.228 | 0.544 |
| node (a junction cluster within 8 px) | 214 | 0.738 | 0.710 |
| fused (no junction) | 161 | 0.764 | 0.764 |

Whether the graph has a junction there carries **no information**. Within `node`, arity carries
almost all of it: 4 arms → 0.944 (n = 107), 3 arms → 0.333 (n = 33), and the gap holds inside
every angle band, so it is not a proxy for crossing angle. **Given a well-formed X,
`match_junction` resolves it 94 times in 100; the pairing cost is not the bottleneck.**

A 3-arm junction is the crossing represented as **two** junction clusters joined by a shared arc
of 25-120 px. Each 3-way junction can match only one pair, so one filament is always left open.
`_absorb_crossing_bridges` refuses to merge them for a documented reason — past `bridge_max_len`
such an arc could be a real segment — and its docstring calls that an irreducible trade-off. It
is irreducible from geometry alone. A predicted multiplicity map is evidence that decides it, and
`instance_a(..., overlap=...)` now consumes it: the outer arms of the two junctions are matched
across the shared arc in one `match_junction` at the arc's true length, and each pair gets its own
traversal. `overlap=None` is bit-identical, asserted by two tests.

This is bottleneck #1 (close parallels) appearing inside the crossing statistics: two filaments
within 3 px for tens of pixels, merged into one ridge the graph can traverse once.

**The other 39 % of failures are a broken mask, and the break is shallow** — 84 % keep their
weaker arm above 0.9 — so a plain threshold drop fills the same holes. Any overlap-conditioned
fill must be read against a threshold drop **matched on added foreground**.

### Two parameter traps that voided a read (2026-09-08)

Both cost a full measurement and both are silent, so they are recorded rather than fixed once.

1. **`--params` overrides, it does not select.** `instance_a` merges
   `{**default_params(), **params}` and `default_params()` already loads
   `src/instance/params_a.json`, the vector behind every published instancer number. Passing a
   tuned probe vector on the command line halved in-domain junction identity (0.630 → 0.311)
   while the mask stayed bit-identical. Only a pre-declared self-check caught it.
   `scripts/diag_crossing_map.py` now prints a WARNING naming every overridden key.
2. **The outer tree has no `params_a.json`.** `/home/prusek/mt_enc_exp/mt34_work` lacks the file,
   so `default_params()` there silently returns the hard-coded DEFAULTS. Any instancer script run
   from that tree without `--params` uses a different assembler than every published number.

## 24. One instrument, six nulls, and the gap that is not the model's (2026-09-15 – 09-18)

The user's directive on 2026-09-15 was to stop and fix the instrument before touching the model
again: one metric, one test set, then read the best model and improve it. What follows is what
that produced, in the order it happened.

### The declared instrument

Centerline-F1 at tolerance 5 px on the 1.5× frame, coverage and precision 0.80, **pooled micro**
as the primary with the ≥ 10-GT macro alongside; instancer constants derived, not tuned
(`params_a_derived.json`, min_length 15, κ_max 0.25); checkpoint ep040 by rule; the threshold
chosen per model on the block's own VAL; native inference. Test set: roi303 TEST (137 frames)
primary, htw TEST (33) secondary, each scored once. SPARSE35 ep040 is the model: 0.641 ± 0.006
and 0.601 ± 0.012, against oracle-mask ceilings of 0.935 and 0.871.

### Where the 0.27 on roi303 VAL goes

Paint-and-keep ceilings (`diag_ceiling_union.py`: the GT band painted into the model's own map,
the rest kept, optionally everything far from the band deleted) split the gap into named terms.
The first reading was wrong by construction — a 2-px band painted 1–3 px beside the model's
ridge makes a double track the assembler breaks on — and was retracted once the band was widened
to the annotation offset. Read at matched band width and taken as an upper bound: evidence-free
stretches drawn through image that carries no ridge ≈ 0.07 (not polyline drift: the weak share
does not fall when the window is widened to the metric's tolerance), unannotated filaments
≈ 0.05, crossing erosion ≤ 0.03, field-stop edge ≈ 0.02, near-filament merging 0.01–0.02,
particles ≈ 0.02. **Model-owned ≈ 0.06–0.07; the single-frame bound as annotated is 0.74–0.76.**

### The six arms

Every arm was declared with a falsifier and a mechanism gate before launch and read with
`read_arm.py` (paired roi303 VAL micro at per-model thresholds, Wilcoxon, precision at matched
recall, htw-only, anatomy at the own threshold).

| arm | change | roi303 VAL micro Δ | mechanism gate | verdict |
|---|---|---|---|---|
| 50 | field stop rendered (`--fov-prob 0.5`) | −0.004, p 0.047 | edge share of off-annotation length unchanged (0.130 → 0.113) | negative |
| 51 | amodal K-channel head | −0.088 (htw −0.16) | over-fires off annotation | negative |
| 52 | anti-aliased stamp (+2.6 % contrast) | +0.0105, p 0.45 | anisotropy unchanged | null |
| 53 | Euclidean target band | +0.0005 (htw −0.045) | brittle to resampling | null |
| 54 | rotation augmentation | +0.0014 | ±45° penalty unchanged | null |
| 55 | pos_weight 2 / 4 | −0.005 / +0.0035 | fragments +15 % / unchanged | null / null |

Arms 52–54 were aimed at a real finding: the model is grid-anisotropic on real frames (a
rotation probe puts oblique filaments rotated onto the axes at 2.5–2.8× the weak share) and the
renderer's stamp aliases by 1.4× in peak between angles. Three isotropy fixes moved neither the
anisotropy nor the score, so the finding stands and the lever does not. Arm 55 is discussed below.
A 14-agent research round in parallel (IRM physics, sim-to-real, annotator convention,
diagnostics, data side; adversarial verifiers) refuted 12 of 13 proposals; the survivor was the
aliasing, killed by its own gate in arm 52. Temporal input is dead on this benchmark: sibling
frames are not co-registered, and the kill line was missed by 2.5×.

### What the round taught about the instrument

- **A paint-and-keep ceiling needs matched band width.** A band narrower than the annotation
  offset creates a second ridge and scores the assembler, not the defect.
- **The mechanism gate is read before the headline.** Every headline gain of the round was
  inside seed noise (±0.010 micro, two seeds 0.04 apart on the strict macro); the only reading
  that ever discriminated was the anatomy at the own threshold.
- **Matched recall means the best-precision node with recall ≥ baseline − 0.005**, not the node
  nearest in recall; the first rule picked a low-precision node by 0.001 and flipped a sign.
- **The trainer's 16-frame VAL trace misjudges heads.** It had the amodal head ahead (0.569 vs
  0.485) on alice/586-heavy frames; on roi303 and htw the head loses 0.09–0.16. Gains on the
  6- and 11-frame sources recur in every arm and are inside seed spread (SEED2 alone moved 586
  by +0.035); they are not evidence.
- **Check the ledger, not the summary, before calling a knob untried.** Arm 55's pw 2 had been
  trained and read twelve days earlier under the old metric (−0.058; at its own threshold 0.70 a
  more connected map whose gain is absorbed by chimeras and instancer cuts); the declaration was
  written from a stale memory line, and the launcher re-trained into the existing run directory
  and overwrote the 2026-09-05 checkpoints. Same seed, so a replicate rather than a loss, but the
  rule is now: grep the TODO ledger by run name, and `ls runs/<NAME>` before any launch.
- **A shared-threshold statistic can invert.** The August cc/gt hint fell with the weight
  because the under-confident map carries fewer low-confidence blobs at a shared threshold, not
  because filaments break less; at its own threshold a thinner mask breaks more.

### The benchmark owns the rest

roi303's VAL and TEST share fields: roi_img_208 (VAL) is byte-identical to roi_img_209 (TEST),
172 = 173 inside TEST, 206 = 207 inside VAL; 163 of 277 frames have a sibling at correlation
≥ 0.99 and 94 sibling pairs straddle the split. No published TEST number until the split is
redone by field. With the model-owned share at 0.06–0.07 and every mechanism-backed lever null,
the next work is the benchmark — re-split by field, an annotation-completion pass on ~10 VAL
frames, and a declared convention for evidence-free stretches — and those are decisions for the
user, not arms.

## 25. Four rulings, a leaking split, and a ceiling the instancer owns (2026-10-05)

The user ruled "do 1, 2, 3, 4" on the 2026-10-04 research round and then authorised generator
changes and retraining (goal: the most realistic and most variable synthetic frames; widen, add
content, fix defects, fit nothing). Everything below was declared in `docs/TODO.md` before it
was read, except where marked exploratory.

### The roi303 split leaked far more than recorded; v2 is split by field (B2)

Field groups = connected components of high-pass image correlation ≥ 0.20 (the low-pass
correlation used on 2026-10-04 measured illumination, not the field). The v1 split had ~6 large
groups (18–40 frames, time-lapse stacks) and ~65 singletons; about 100 of 137 TEST frames shared
a field with a VAL frame. roi303 v2 (`docs/roi303_split_v2.json`, chosen blind, no score per
option): 273 frames, VAL 145 / 7 386 curves, TEST 128 / 6 506. v2 TEST is unread. Intervals from
the stored v1 rows under a field-cluster bootstrap: roi303 TEST 0.641 ± 0.040, htw TEST
0.601 ± 0.054 — the seed-to-seed difference of one recipe is as large as the "leads" of arms 50–55.

### RELINK 1-D failed its own guards

One multiplier m on `w_theta`, derived on fresh synth (96 frames, 3 seeds). Pooled synth F1 rises
0.637 (m = 1) → 0.665 (1.75) → 0.669 (2.25), by recall only, while junction identity falls
monotonically with m. The rule picked m* = 1.75; identity −4.8 % and bundle −3.8 % against a
−2 % limit → FAILED, real VAL not read, `params_a_derived.json` unchanged. Declaration flaw: the
rule selected by F1 and checked guards afterwards with no fallback; m = 1.25 passes (+0.017) but
taking it now would be a rule written after the read.

### The blocks fail on different axes, and in domain the instancer is the ceiling

Per-curve logistic regression of a miss: roi303 is SNR-driven (β −0.56; 30 % of curves have
SNR < 2 and miss at 0.42), htw is length-driven (β −1.22; < 12 px miss 0.88; width carries no
signal once length is in). Exploratory (no criterion): a perfect band from exact GT scores only
0.686 on the model's own distribution and the model reaches 0.636–0.646 there (93 %); on real it
reaches 70 % (roi303) and 72 % (htw) of the ceiling. The sim-to-real loss is in the mask; the
synthetic scenes are harder for the instancer than the real ones.

### Old arms re-read at ep040: correlated noise is a trade between the blocks

Paired vs SPARSE35 on v1 VAL. NOISE / NOISE_S2: htw +0.028 / +0.023 (recall 0.568 → 0.658 /
0.638, the 12–60 px curves), roi303 −0.018 on both seeds (precision; cluster CI excludes 0).
COMBO: +0.025 / −0.012. SCALE fails the width gate (miss of < 20 px htw curves 0.698 → 0.662 /
0.708) — "htw short filaments are out of scale support" is dead. AUG_PHOTO null on roi303,
−0.030 on htw. JWEIGHT negative. SPARSE20 / SPARSE50 / TIRF35 confirm the 35 % sparse choice.

### Tile fusion loses ~0.012 on roi303

Gaussian-weighted tiling (stride 256, declared decode) is +0.0117 [+0.006, +0.020] on roi303 VAL
(p 0.012), precision at equal recall (FP 3 304 → 3 058); a tile-free pass gives the same number,
so the loss sits at the seams of the uniform-mean / stride-387 fusion. htw frames fit one tile
(bit-identical). One seed, v1 VAL only. Mechanism read (139 multi-tile frames): within 8 eval px
of an interior tile edge 30.6 % of annotated-filament points fall below the cut under the declared
fusion, against 16–19 % elsewhere; Gaussian weighting brings the seam bin to 16.4 %, flat across
bins. Off-annotation foreground near seams is LOWER under the declared fusion (an addendum
criterion that expected the opposite was not met): the seam suppresses everything, so the
precision gain is not fewer stray pixels — fragments of curves cut at seams are the consistent
but unmeasured explanation. The comparison is at a shared cut, which is valid here only because
the two maps coincide away from seams (64+ px bin: foreground 0.00809 / 0.00812, weak share
0.190 / 0.188). Whether this is
a defect fix (nnU-Net's own default) or an inference trick is the user's ruling; neither the
declared path nor the hub was changed.

### ARM 56 NMIX / ARM 57 NMIXFLAT (launched 19:38; read 2026-10-06, see below)

Generator additions, both off by default and bit-identical when off
(`tests/test_noise_mix_flat_bg.py`): a per-frame choice between two noise regimes
(`noise_alt_prob`, `noise_alt`) and a flat-fielded background (`bg_flatten_prob`, sigma
log-uniform 3–60 px — content the generator never rendered: htw low-pass background std
0.007–0.05 vs synthetic ≥ 0.07). Hypothesis: a mixture keeps the htw gain of NOISE without the
roi303 cost. Two seeds each, earning rule, mechanism gate and synth guard in the TODO block of
19:50. Said before the read: htw noise amplitude (nMAD 0.11–0.28) is still only partly covered
(synthetic p99 0.156).

### The mixture earns, the flat background does not, RELINK fails its confirmation (read 2026-10-06 04:40)

Four reads, each against a line written before the dump existed (TODO 19:50 and 22:15).

- **ARM 56 NMIX earns.** A per-frame 50/50 choice between the shipped noise and the correlated
  regime: htw VAL +0.024 / +0.072 against the same-seed baseline (0.626 -> 0.651, 0.588 -> 0.660),
  roi303 VAL +0.010 / +0.010 (0.656 -> 0.666, 0.658 -> 0.667). Replacing the noise had cost roi303
  −0.018 on both seeds; mixing it costs nothing. The roi303 line was non-inferiority and the
  field-cluster intervals include 0, so the claim is "the cost is gone", not "a gain". Mechanism
  gate (htw miss of 12–35 px curves −11.5 % / −31.6 %, line −10 %) passes narrowly on the first
  seed; synth guard passes. Seed-free reading: all four mixture runs sit above all three
  SPARSE35-recipe seeds on htw, and both NMIX runs above all three on roi303.
- **ARM 57 NMIXFLAT is null against ARM 56** (htw +0.005 mean over NMIX, line +0.012). Rendering
  flat-fielded backgrounds — content the generator never produced — adds nothing on top.
- **RELINK with the repaired rule does not earn.** The rule now selects among multipliers that pass
  the identity and bundle guards. The fresh set picks 1.75, the old set 1.25; the rule takes the
  smaller. At 1.25: roi303 +0.0072 (line +0.006, met), htw +0.0081 (line +0.012, NOT met); same
  sign on all seeds, junction identity −0.2 %. The one confirmation read is spent; the lane is
  closed. What it taught: the two synthetic sets disagree on where identity breaks (1.5 vs 2.0),
  so "the largest safe multiplier" is not a constant one can derive from 96 frames.
- **The fusion fix replicates on three seeds** (+0.012 / +0.013 / +0.015 on roi303 VAL, every
  cluster interval excludes 0; `--fusion uniform` reproduces the cached dump exactly). `eval_v5.py`
  now defaults to Gaussian fusion; the declared instrument is "ep040, native, gauss fusion".
  `train_v5.predict` keeps `uniform` as its function default so old callers reproduce.

Lesson worth keeping: the noise regime was never the wrong content — it was the wrong
*replacement*. A generator arm that swaps one prior for another tests "which is closer to this
block"; a mixture tests "does the model need both", which is the question domain randomization
actually asks. Three earlier noise arms answered the first question and were read as a trade.

### LPC screened in a morning: a ceiling that exists, and three premises that do not hold (2026-10-06)

The user said go on screens S0–S2 of the Lifted Path Cover proposal. Each had a line written before
its code (`docs/TODO.md` 07:55, 08:25, 09:05).

- **S0, candidate ceiling — passed.** With the mask unchanged, an oracle choosing among the instancer's
  own arcs and curvature-admissible links scores 0.838 on a perfect band and 0.831 on htw VAL (instancer
  0.646 / 0.608). Two things it also showed: a *looser* cut makes the candidate set worse at every bridge
  length (neighbours merge into arcs no selector can split), so the "permissive skeleton" was wrong; and
  the bridge length chosen was the grid edge.
- **S1, strip vs pixel — passed its line, emptied its premise.** A CNN on rectified strips beats the
  mean map logit along the bridge (+0.053 AUC in domain, +0.035 on htw VAL on 91 true links). A logistic
  regression on five hand features equals it in AUC (it trails in average precision in domain, 0.79 vs
  0.89), and a map-only strip equals the full strip everywhere: the raw image contributes nothing to
  the presence decision.
- **S2, junction identity — void.** At junctions where two GT curves each own two facing arms, the
  instancer's matching is already right 92 % of the time in domain and 99.7 % on roi303 v2, so a +0.10
  line could not fire; the CNN reads 0.917. The useful part is what the unit excluded: only 218 of 749
  GT crossings form such a junction. The "connected mask, identity 0.50" failures are crossings where
  the arc graph itself is wrong (arity after contraction, shared arcs, unowned arms) — upstream of
  pairing.
- **Exploratory decomposition (not declared):** without the privilege of deleting unowned arcs the
  oracle keeps 0.780 in domain but drops to 0.710–0.727 on htw. In domain the +0.17 is link choice in
  the 12–36 px range between arbitrary arc ends; on htw about half of the ceiling is the annotation
  convention.

What it cost: one morning, ~20 GPU-minutes, no trunk retraining. What it bought: the proposal's three
mechanisms (raw-image cue, learned verifier over hand features, link-identity head) are each measured
and none carries; the surviving quantity is an instancer-side selection gap in domain. A design that
had gone straight to S3 would have spent days to find the same thing less clearly.

### The decoder gains after all — a link score fitted on synthetic links (2026-10-06, afternoon)

The user asked for the two instancer-side forks left by the screens. The crossing diagnostic on 701
synthetic crossings said most failures are arcs no curve owns (56 % on the model map) and predicted
that a link-level decoder had "almost nothing to win at crossings". That prediction was wrong, and the
oracle it rested on was not an upper bound: it chose among links the old bridging rule admitted.

Instancer L keeps the arc graph and replaces the greedy bridge by one assignment: every pair of ends of
different arcs within 60 px whose implied curvature is ≤ 0.25, scored by a logistic on five hand
features (mean and min map logit along the bridge, gap, turn, the old pair cost) fitted on synthetic
links only, solved as a max-weight matching. Selected on synthetic scenes (0.637 → 0.708), then read:

| block (VAL) | A → L, own cuts | interval |
|---|---|---|
| roi303 v2, three baseline seeds | 0.676 → 0.718 / 0.677 → 0.714 / 0.676 → 0.706 | all exclude 0 |
| htw, three baseline seeds | 0.626 → 0.639 / 0.588 → 0.628 / 0.598 → 0.650 | first includes 0 |
| htw, noise-mixture models | 0.651 → 0.698 / 0.660 → 0.693 | both exclude 0 |

htw is the decisive block (its annotation is clean), and there the deployed seed gains least, +0.013
with an interval that includes zero; roi303's larger gain is secondary until the annotation is completed.
Junction identity on roi303 rose 19 % relative. On htw the mixture and the decoder stack (+0.07 / +0.10
over today's model); on roi303 the mixture adds nothing under either decoder. What did not work stays
on record: the strip CNN, the raw image and the junction head. What worked is the least novel piece —
a global assignment instead of a greedy one, with weights learned where labels are exact.

The same afternoon the baseline was read once on the field-disjoint roi303 v2 TEST: 0.640 / 0.637 /
0.633 (VAL 0.676). Seeds agree to 0.003; fields do not (95 % interval about ±0.08).

### The network version of the link score loses to five hand features (2026-10-06, evening)

The user chose to build the network variant before the test read. The strip CNN from the morning's
screen was retrained as the link scorer (same strips, map only, two seeds), a rule was written first
(htw VAL decides; the simpler model keeps ties), and both scorers were run on identical arcs and
candidate links.

| | L-hand | L-strip |
|---|---|---|
| link AP in domain | 0.79 | 0.89 |
| F1 in domain (`synth_relink96b`) | 0.676 | 0.668 |
| htw VAL, strip − hand | — | −0.007 … +0.007, all include 0 |
| roi303 v2 VAL, strip − hand | — | −0.027 … −0.033, all exclude 0 |

A better ranking of single links did not become a better assembly, even in domain: the matching needs
the score to be right about *which of two competing links* wins at one arc end, and average precision
over all links does not measure that. L-hand is the instancer.

The noise mixture's third seed gained +0.063 on htw and read −0.007 on roi303 v2 VAL against a
−0.006 line, so by the rule written in the morning the recipe stays SPARSE35. Two operational notes:
58 of 72 jobs of the first launch died of GPU memory and were logged as `rc=0` (the log line took the
exit code of `date`); completion is checked by counting outputs.

### The held-out 2×2, read once (2026-10-06, 17:45)

With the instancer (L-hand) and the recipe (SPARSE35) fixed on VAL, the four cells were evaluated once
on the field-disjoint roi303 v2 TEST and on htw TEST, each at its own VAL cut, three seeds each.

| 3-seed mean | roi303 v2 TEST | htw TEST |
|---|---|---|
| SPARSE35 + A (deployed) | 0.637 | 0.601 |
| SPARSE35 + L | 0.674 | 0.629 |
| noise mixture + A | 0.651 | 0.638 |
| noise mixture + L | 0.687 | 0.668 |

The decoder's gain held out of sample at the size seen on VAL (+0.038 on roi303, interval excludes
zero; +0.028 on htw, interval touches zero). The mixture, which had missed its VAL rule by 0.001,
read positive on both TEST blocks (+0.014, +0.037). Its row is a report, not a selection: the rule was
written to be decided on VAL, and choosing the recipe on TEST would turn the test set into a selection
set — that decision was handed to the user as such. Junction identity on TEST exists for one seed
only (0.493 → 0.514), far below the +19 % seen on VAL.


## 26. The "ultimate model" round: one free lever survives, once the line is scaled to its cost (2026-10-28/29)

The validated parts (noise mixture, instancer L-hand, Gaussian tile fusion) were already one
configuration, so the round could only add new levers on top of it. Two were declared in `docs/TODO.md`
before launch, each against same-recipe controls, and trained on kajman because tulen was occupied.

**The training host matters as much as a seed.** The same recipe and seed trained on kajman and on tulen
differ by +0.007 on roi303 v2 VAL and −0.016 on htw VAL, outside the ±0.006 host rule. Decoding is not
the cause (kajman decodes tulen checkpoints within 0.0012). From here on every control is trained on the
same machine as the arm it controls (NMIX_K, NMIX_K2, NMIX_K3).

**ARM 59, output-space alignment to unlabeled SpheroSeg frames (AdaptSegNet, λ 0.001, 188 IRM frames
selected by rule and leak-checked against every VAL and TEST block): a clean null.** Pooled +0.001 on both
blocks, no over-firing, and the bottleneck probe did not move beyond its own run-to-run spread
(0.81–0.93 across controls of one recipe). Synthetic and real output maps already had the same density,
so an output discriminator had nothing to align; the gap lives in the encoder.

**ARM 58, anti-aliased rendering (`paint_aa`) on the noise-mixture recipe: does not earn.** roi303 v2 VAL
null (pooled −0.0004), and its declared mechanism, shorter end overshoot, failed the other way. htw VAL
rose in all four comparisons (pooled +0.030), and a post-hoc decomposition put the gain in recall of GT
curves shorter than 20 native px. That pointed at a plausible cause (the old renderer drew each filament
about 3.3 px short of its label), but the hypothesis was formed after looking, so it got its own
confirmation arm.

**ARM 58b, a third seed with the line written before any output existed: not confirmed.** On seed 3 alone,
htw +0.0072 [−0.019, +0.033] against a line of +0.024; roi303 +0.0009, synth +0.0044; the short-curve gate
read +0.042 against +0.05. The sign stayed positive in all five comparisons, but the seed that took no
part in choosing the hypothesis read 0.3 of the line, with an interval that still covers it. A one-seed
test at this noise level has about even odds of passing even when the effect sits exactly at the line, so
it cannot tell selection from a small real htw gain of about +0.02. By the declared rule `paint_aa` is not
adopted and stays off; settling the htw question would take at least three more same-host seed pairs.

Under the lines declared that night no arm earned, so TEST was not read then, and the final model stayed the
noise mixture with instancer L-hand as read on 2026-10-06 (superseded the next day by ARM 58c, below) (roi303 v2 TEST 0.687, htw TEST 0.668, with the disclosure that the mixture missed its
VAL rule by 0.001 and was adopted after that read). Cost: eight trainings (about 2.7 h each at 4 min/epoch)
plus dumps on 2× L40S, roughly 25 GPU-hours over 12 h of wall time (an estimate from the logs).

**ARM 58c, the same lever under a complexity-scaled line (2026-10-29): adopted.** The user's criterion is that
one point of F1 is a good gain when the model does not grow, and the renderer fix adds no parameters and no
inference cost. A new rule, written before seeds 4–6 existed, asked for htw ≥ +0.010 pooled over the four
pairs trained after the hypothesis (seeds 3–6), with no regression on roi303 or synth. It read htw +0.014
[+0.001, +0.028], roi303 +0.007, synth −0.0004; the htw sign was positive in all six pairs, and so was the
short-curve recall. The single TEST read for these twelve models (the second adaptive use of TEST) repeated
the htw gain at the same size, +0.014 [−0.001, +0.028], and found roi303 null (−0.001). The recipe is now the
noise mixture with `paint_aa`, instancer L-hand: TEST 0.693 on roi303 v2 and 0.681 on htw (mean of four fresh
seeds, kajman), against 0.695 and 0.667 for the same-host noise mixture. The lesson about lines: a line set
at the noise of one seed demands a large effect; a small free gain is found by averaging seeds, not by
waiting for one seed to be lucky.

**Run-level caveat on ARM 58c (added 2026-10-31).** The intervals above resample test frames only; they do not see
the variance between training runs, and the trainer is not deterministic, so the same seed fixes the synthetic data
stream but not the trajectory. Tested at the level of runs, the htw gain is +0.014 on VAL (four fresh runs per arm,
Welch p 0.13) and +0.012 on TEST (six per arm, p 0.14, five of six seeds positive). Its sign holds on two disjoint frame
sets and the short-curve recall moved in all six pairs, but six runs do not resolve it. The adoption stands on zero added
complexity, no roi303 regression and that consistency, and is reported that way.

**Morning of 2026-10-31: five checks without training.** Averaging the probability maps of six seeds gains roi303 v2
VAL +0.033 and htw +0.010 (two seeds already +0.017 as the mean of the two blocks, at twice the inference cost); weight
averaging over epochs 28–40 (SWA) gains nothing. Distilling the ensemble was ruled out before any training: on synthetic
frames the ensemble equals a single model (+0.002), and the seeds disagree 3.3 times more on real frames than on
synthetic ones, so a student taught on synthetic inputs, the only ones allowed, would learn the label again. A denser
generator (DENSE) re-read under the current metric costs htw −0.058. An audit of the renderer against the label
refuted the story that the old renderer drew filaments short; why `paint_aa` helps on htw remains unexplained.

**ARM 59b, adversarial alignment of the encoder (2026-10-31): null.** A gradient-reversal discriminator on the
nnU-Net bottleneck (DANN, λ 1.0, 188 unlabeled SpheroSeg IRM frames, four seeds against same-host same-seed controls)
held its discriminator at chance for all forty epochs, yet a linear probe still told synthetic from real features
(AUC 0.74–0.82 against 0.81–0.91 for the controls). It read htw +0.012, roi303 +0.001, a block mean of +0.0065 against a
+0.010 line. With the output-space arm of 10-28, label-free alignment to unlabeled real frames is closed on both branches.

**ARM 60, MT-like lines in the empty backgrounds ignored in the loss (2026-10-31): null.** About a quarter of the
backgrounds carry faint scratches and fibres that the model itself fires on (7 % of their pixels against 0.03 %
elsewhere); the generator taught them as background. A non-learned line detector (oriented filters at the generator's
own width range, an IAAFT noise null, a cluster-mass test) flagged 207 of 428 backgrounds at a precision of about one
half, and the trainer ignored those pixels when they were unlabelled. The dose was 0.35 % of the synthetic label length.
It read htw +0.011, roi303 −0.005, a block mean of +0.003, and the response inside the flagged regions did not rise.
Its htw "gain" is the useful result: an arm that changes the loss on 0.022 % of the crop area beat the four controls on
htw in all four pairs with a frame interval excluding zero. Those four controls are the four lowest of the six AA runs
on htw, and both 59b and 60 land where the other two sit. Against all six runs the difference is +0.007–0.008 (p 0.16–
0.28). Run-to-run sd on htw is 0.0075–0.012, so resolving +0.010 needs about nine runs per arm at the lower sd and about
twenty-three at the higher; what catches a lucky control set is the reference (every run of the recipe), not the test; a line on the mean
of htw and roi303 is what kept both arms from a false adoption.

**Debris, and a decoder that only fills gaps (2026-11-01/02): adopted.** Looking at the off-annotation predictions by eye showed a quarter of them on htw were particles and debris, and a fifth were arcs of interference rings; the generator had only round specks. Interference rings as negatives (ARM 61) made the model stop firing on synthetic rings about twentyfold and still gained nothing, paying in short filaments. Compact debris (ARM 62: aggregates of particles with the interference profile, irregular blobs, never elongated, so that no filament-shaped negative is taught as in ARM 41) gained on both blocks and on all four seeds, and stood against all eighteen other runs of the recipe. The arm removed its target: of the predictions labelled earlier, debris fell from 0.58 to 0.32 still predicted, unannotated filaments only from 0.82 to 0.75. On crossings, 94 % of identity failures on roi303 turned out to be a broken arm whose map sits just under the operating threshold, with p above 0.95 in three quarters of the gaps. Synthetic frames rarely show this, so the fix could not be selected there; with the user's ruling it was selected on half of the VAL fields and confirmed on the other half. Admitting weak pixels near junctions raised identity but added false positives; admitting them only where they join two different components gained F1. The combined candidate read +0.022 on roi303 TEST and +0.018 on htw TEST against the previous recipe (third adaptive TEST use). The dose I first described as per 2048 px background was in fact per 640 px training crop, about ten times denser.

**The reach of a link (2026-11-02): accepted after it missed the line.** The bridge of 60 px had been the top of the October grid and was never checked above it, and 22 % of the broken chains on roi303 were gaps over 60 px at a positive link score. On synthetic links 80 and 100 px tied, which selected 80. Paired on the same maps at VAL it gained roi303 +0.007 [+0.003, +0.009] and MT-34 +0.035, with htw +0.001, synth +0.005 and no added complexity; the block mean of +0.0037 missed the +0.010 line, which had been set for training arms, where run-to-run variance is the problem. The user accepted the constant after reading that, and the decision is disclosed as post-read and applies to this constant only. TEST at 80 px has not been read. Later arms (ARM 63, the ARM 62 replication) are judged at 80 px, amended before any of their outputs existed.

**A thin-filament regime, and what the replication said (2026-11-02): rejected, and smaller.** htw's largest row of misses was short thin filaments, and its filaments (FWHM 2.6 px) lie almost wholly below the training support (4.6 px), so a frame-level thin regime reached that band. It gained htw +0.032 and lost roi303 -0.017 and MT-34 -0.028, and not through the recall it was built for: short recall on htw did not move, its false positives fell by a quarter. The own cut sat on the top of the 9-point grid in 5 of 8 cells, which the 09-02 protocol treats as unreliable, so one more cut was computed for arm and controls before the verdict stood; every curve falls there. The same round replicated debris on two fresh runs: same sign, smaller, about +0.01 over twelve runs against the +0.015 first reported. Half doses of both levers are the follow-up.

**Halving the doses (2026-11-02): a repair that failed, and a mechanism that appeared only at half strength.** Debris had cost short curves, so half the debris was tried with a repair rule rather than a gain line: same F1, short curves back, false positives still down. It failed. Short recall barely moved on htw (0.529 with no debris, 0.478 at half, 0.485 at full), which means half a dose already carries the whole cost. The double dose was then dropped, because short recall falls with dose and more debris cannot bring it back. Half the thin regime also missed its declared rule (block mean +0.004 against +0.010, roi303 -0.0056 against a floor of -0.005), and it stays out. What it did to the anatomy is more interesting than the F1: at full dose the thin regime gained on htw through precision and left short recall flat, while at half dose the gain comes through recall, on both blocks, in the declared mechanism. Four half-dose runs against all six debris runs give recall of curves under 20 px +0.076 on htw and +0.119 on roi303, with longer curves unharmed. On roi303 the price is that the whole false-positive gain of debris is handed back, in short predictions off the annotation, for which we have no labels. This reading was post hoc and moves no verdict, but it changes what the next person should try first: debris and thin filaments appear to act as opposed dials on the model's willingness to report short structure (our inference, not a measurement), so the question for a short-curve arm is how much false-positive cost a given recall is worth, not which recipe wins on F1.

**Deployment (2026-11-03): what checking the port against the research path found.** The decoder ported to the hub reproduced the research polylines exactly on all 195 VAL frames, in both environments. The check was worth running for what else it caught. It showed that the hb block had been dumped under uniform tile fusion, on the reasoning that htw is a single tile, while the 17 MT-34 frames in the same block are not. So every MT-34 number of the round was read on the wrong instrument: for the deployed model, 0.666 under uniform and 0.681 under the declared Gaussian fusion, and the two fusions even disagree on which cut MT-34 prefers. No rule read MT-34, so no verdict moved. An independent review found what the identity check could not, because each was a difference in behaviour outside the benchmark path: a decoder with no time bound under the service's single inference lock, FRAP spots that could land on a link across a gap, and the export recording the cut that was sent rather than the one applied. The first would have blocked every model for minutes on a frame the network cannot read, which is a real input in this service.

**Essays follows (2026-11-03).** The batch assay moved to the same model on the user's word, with a standing rule that it always does, held by a parity test in the hub. Through the real queue on four VAL wells its curves match the research reference on every well (counts equal, p99 at most 0.43 px at 1.5x). A batch adds one thing the interactive path does not need: a table outlives the run, so each row now names its checkpoint and decoder. Essays numbers before and after the date are not comparable, and the arc-limit fallback would otherwise have been visible only in a log that dies with the container.

**Stability between frames (2026-11-03): a suspect that was cleared, and the partition left standing.** The user asked whether the model is temporal, because segmentations that differ between consecutive frames break tracking. It is not. Measured on 12 unlabelled IRM videos, 20 consecutive frames each, with the production tracker, V16 is more stable than the model it replaced: 90.7 % of centerline length persists to the next frame against 89.5 %, and the length in tracks covering most of the window rose by 19 %. The first reading was far larger, +6.7 points rather than +1.2, and it was wrong: it subtracted a drift from the tracker's own `estimate_drift`, which errs by more than 3.33 px on 40 % of pairs and by 106 px at worst, on frames whose true shift never exceeds 4 px. A per-pair look caught it, an exhaustive search for the best translation confirmed it, and the natural next conclusion, that this estimator is why tracks fragment, then failed its own test: the tracker with zero drift or with the true shift gives the same tracks. What remains is the partition. The pixels mostly persist, yet only 68 % of length maps to exactly one curve in the next frame, on both models, against 81 % when the scene does not change at all. Each filament is still followed by 2.2 tracks. There is no ground truth on these videos, so none of this says which curves are right.

**A temporal window without training (2026-11-03): no support, and one route closed.** The user asked for a model that decides frame t from t-1 and t+1. V16 already takes three channels and receives the same frame in all of them, so `(t-1, t, t+1)` needs no new architecture, only new data. Before paying for that, the same 12 videos went through V16 on the mean of three aligned frames, and through the decoder on the mean, the median and a foreground-matched mean of three aligned maps. Consecutive windows share two of three frames, so stability rises by construction; the read was therefore declared at lag 3, whose windows share nothing, and the trap proved real: the mean image gains +0.029 persistence at lag 1, with a confidence interval above zero, and +0.006 at lag 3, with one spanning it. The one-to-one partition did not improve in any arm: -0.004 for the mean image, and -0.09 to -0.23 for the map arms, which also delete moving filaments, because at the cut the map is a thin core that rigid alignment does not superimpose. It falls even on a static video (0.833 to 0.566-0.794), so motion is not the only cause; the screen cannot separate the thin core from L+H seeing a mask shape it was not tuned for. Map fusion, which had been suggested as the cheap first step, is closed for this model with the deployed decoder parameters. A mean is not a network and cannot learn to ignore a mover, so this withholds support from a temporal retrain rather than refuting it. The attribution says where to look instead: of 9107 places where one curve continues as two, 37 % have a break in the map, 35 % show no break within 2 native px (47 % within 3.33), and 28 % are filaments side by side. The flicker also barely grows with time, 0.677 at lag 1 and 0.663 at lag 3, so it behaves like a re-roll at each frame and not like accumulated motion.

**Votes from the neighbouring frames in the link score (2026-11-04).** L+H decides every link between arc ends with a frozen score, which makes a temporal term easy to add without training: a neighbour frame votes +1 when one of its curves covers both ends and runs along the bridge, -1 when two different curves cover them, and abstains otherwise. With no neighbour it is the deployed decoder exactly, checked on every frame. The weight was chosen on held-out synthetic clips with exact truth, where F1 rose by 0.003 and tracks per object fell from 1.48 to 1.38 against an oracle 1.04, and nothing was tuned on real video. On the 12 videos the one-to-one partition at lag 3 rose by 0.026, CI [0.018, 0.034], on all 12, with length unchanged and no harm to the moving video. That is below the declared +0.03, and every weight on the ladder landed between 0.020 and 0.026, so the weight is not what limits it. A vote can only re-decide links between arcs that exist: it cannot add a filament, mend a mask away from a link or part a bundle. The tracker on every third frame, the read free of shared input, did not separate; over all frames, which share input and which is how the tool runs, tracks per filament fell from 2.09 to 2.00. Accuracy with neighbours active can only be read on synthetic clips.

**Temporal input, trained (2026-11-04): it works on the generator's video and not on real video.** V16's network already takes three channels, so six runs of the V16 recipe were trained on `(t-1, t, t+1)` from a new clip generator whose frame t and target are byte-identical to the still recipe; the six existing V16 seeds were therefore its exact control. On held-out synthetic clips the temporal model is clearly better: F1 +0.029 on frames with both neighbours, 6 of 6 seeds, +0.054 when filaments are static, and tracks per object 1.46 to 1.37. On the 12 real videos it moved the one-to-one partition by +0.004, CI [-0.006, +0.013], drew 19 % less on the moving video, and on single images it lost 0.013 on htw. A neighbour on real video is not less informative, since background residuals decorrelate from frame to frame even more than on the clips. The likelier reason sits in the partition itself: even on the clips the temporal input moved it only by 0.015, and the clips are far steadier than real video for the single-frame model (0.84 against 0.68), steadier even than a real frame re-tiled (0.81). Real frames are more ambiguous in themselves, and the decoder breaks the same ambiguity differently in each frame. A cleaner map does not change that; a vote about the links does, which is why the decoder vote, +0.026 and replicated on all six seeds, is the only temporal route left standing.

**The vote goes into the tool (2026-11-04, the user's decision after the read).** It missed its declared line by 0.004 and was accepted because it needs no training and harms no guard. In the hub it runs before the tracker on a video segmented through the queue, one frame at a time: the map of frame t is computed again and the neighbours' stored curves vote. Unlike the tracker, which only adds an identity, the vote rewrites geometry, so it acts only where it was measured: frames the queue has just segmented, never a frame edited or re-segmented by the user, never twice, never the first or last frame, and the route itself refuses a frame whose recomputed per-frame curves are not the ones stored. The integration was held to identity rather than to a score: on all 211 interior frames of the 12 videos the hub code returns the research curves bit for bit, and a synthetic triplet pins this in the hub's CI. The cost is one more inference per interior frame, so a video's tracks arrive later than its curves. **Deployed 2026-11-05.** On production the one open question was whether the card re-decodes a frame to exactly the curves it stored, since the bit-for-bit replay ran on the research card. It does: on a fresh synthetic video on the test account no frame was refused and the vote changed all seven interior frames. The production card does not reproduce the research card's curves bit for bit (one curve of forty differs on the test frame), and it does not need to: storing and re-decoding happen on the same card, where two runs are identical. There the vote took 26 s for seven 1-Mpx frames, more than the segmentation of the video itself.
