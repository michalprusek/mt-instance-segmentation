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
