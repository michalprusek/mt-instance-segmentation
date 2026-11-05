# Replacing the geometric instancer with dense local shape patches

**Status:** design, approved in principle by the user 2026-08-27 ("udělej A").
**Goal:** strict centerline-F1 >= 0.90 on the combined MT-34 + HTW benchmark.
**Baseline to beat:** 0.4091 (v5H, combined TEST, 2026-08-27).
**User constraint, 2026-08-27:** *"chci abys ve finální metodě maximálně omezil postprocessing.
nefiltruj třeba podle délky. chci aby veškerou práci udělal segmentační model."*

All numbers below are **VAL only** and say which harness produced them. TEST carries a declared
exposure budget and is not touched during development.

---

## 1. The measured diagnosis

Everything in this section was measured on 2026-08-27, locally, without a GPU, on the full
combined VAL (50 frames). The harness was validated against the project's own recorded oracle
numbers *before* it was believed: 585 oracle 0.983, 586 oracle 0.881.

### 1.1 Where the pooled F1 actually goes

All figures are `aggregate_benchmark`'s **mean-F1** — the macro mean over non-empty frames, which
is what the recorded 0.3847 / 0.4091 are. Micro-F1 is quoted alongside where it differs, because
mixing the two is the harness confusion this project has already been bitten by.

| source | model (v5H) | oracle mask,<br>shipped params | oracle mask,<br>**no length filter** | oracle mask,<br>**no filter + thin target** |
|---|---|---|---|---|
| 585 Alice | 0.776 | 0.968 | 0.987 | 0.982 |
| 586 crossing-dense | 0.405 | 0.816 | 0.845 | 0.895 |
| **htw** | **0.375** | **0.536** | **0.916** | 0.913 |
| **POOLED mean-F1** | **0.3847** | **0.646** | **0.910** | **0.918** |
| *(pooled micro-F1)* | *—* | *0.668* | *0.860* | *0.884* |

Read down the columns, not across the rows. Three separate losses:

| term | cost in pooled mean-F1 |
|---|---|
| the assembler itself, even with a perfect mask and no filter | 1.000 → 0.918 |
| **instancer configuration** (`min_length`, target width) | 0.918 → 0.646 |
| **foreground quality** | 0.646 → 0.385 |

**The single largest term is a hyperparameter.** Removing the length filter is worth 0.264 pooled
mean-F1 on a perfect mask — more than twice the assembler's own residual defect (0.082).

**Correction to an earlier reading of these same runs.** In micro-F1 the assembler's ceiling is
0.884 and 0.90 sits *above* it; in the reported mean-F1 the ceiling is **0.918** and 0.90 sits
just *below* it. So the assembler is not the binding constraint — but the margin is essentially
zero, and reaching 0.90 with instancer A would require a near-perfect foreground and leave nothing
in reserve. That is the honest case for replacing it, and it is weaker than "the target is above
the ceiling".

### 1.2 The recorded diagnosis was right about MT-34 and wrong about HTW

The project's standing claim is *"the foreground is the bottleneck, about 2:1"*. That was
measured on MT-34 alone. On HTW — **61 % of the combined ground truth by count** — it inverts:

| source | instancer configuration costs | foreground costs |
|---|---|---|
| 585 / 586 | small | dominant |
| **htw** | **−0.380** (0.916 → 0.536) | **−0.161** (0.536 → 0.375) |

The mechanism is a single number. `min_length = 44.74` eval px is **longer than 49 % of HTW's
ground truth**, so the instancer deletes half of HTW by construction. A `min_length` sweep on
oracle masks improves **all three sources monotonically** — pooled mean-F1 0.646 (shipped) →
0.910 (at 3), and 0.918 with a thin target. There is no trade-off to balance.

### 1.3 The generator never made a microtubule short enough

v5H's real calibration (`calib_reg418_morph_snr3.json`, not the dataclass defaults) is
`length_log_mean = 6.60` → median **694 px**, under the clamp `length_range = (80, 950)`, with
roughly 35 % of samples degenerately pinned to exactly 950 px.

| source | median GT length | **% below 80 px = below anything the network ever saw** |
|---|---|---|
| synth (training) | 694 px | 0 % |
| 585 | 358 px | 19.2 % |
| 586 | 173 px | 21.6 % |
| **htw** | **29 px** | **88.1 %** |

Three alternative explanations were tested and refuted. HTW is **not at a different pixel
scale** (sub-pixel filament width sigma: htw 1.16 px vs 585 1.33 px — a 1.15x difference, not
the ~6x that would explain the length gap). HTW ground truth is **not chopped into fragments**
(only 1.2 % of polyline endpoints abut another polyline's endpoint, against a chained
annotation's near-100 %). And the curvature bound **holds**: kappa_max = 0.25 rad/px survives on
HTW at max 0.242. HTW's microtubules are genuinely short — about 3 um against MT-34's 15–30 um,
a different biological regime.

The contrast between kappa_max and min_length is the lesson worth carrying: the constant that
was **derived** from physics transferred to a new lab; the two that were **fitted** to MT-34 did
not, and failed silently.

### 1.4 PHASE 0 RESULT: the benchmark ceiling is 0.88-0.93, and the 0.90 target sits inside it

Run on tulen over the cached v5H combined-VAL predictions
(`scripts/phase0_nofilter_audit.py`, `phase0_val_v2.json` / `phase0_val_ml20.json`).
**The harness reproduces the recorded baseline exactly** — at `prob_thr 0.99, min_length 44.74`
it returns TP 477 / FP 517 / FN 784, micro-F1 0.4231, mean-F1 0.3925, matching the recorded
combined-VAL numbers cell for cell.

**0b — the length filter is worth +0.057 mean-F1 today, for free:**

| `prob_thr` | `min_length` | pooled mean-F1 | micro | TP | FP | FN | 585 | 586 | htw |
|---|---|---|---|---|---|---|---|---|---|
| 0.99 | 44.74 (shipped) | 0.3925 | 0.4231 | 477 | 517 | 784 | 0.798 | 0.299 | 0.347 |
| 0.99 | **20** | **0.4498** | 0.4666 | 608 | 737 | 653 | 0.757 | 0.311 | **0.436** |
| 0.99 | 3 (no filter) | 0.4059 | 0.4167 | 640 | 1171 | 621 | 0.693 | 0.285 | 0.390 |

Removing the filter *entirely* costs 0.044 against `min_length = 20` — false positives rise
737 → 1171. **That 434-false-positive gap is the fragmentation debt the new architecture has to
pay off**, and it is the concrete, measured meaning of "let the model do all the work".

**0a — how much of the ground truth is missing.** Every false positive was split by median
distance to the nearest annotated centerline (<= 5 px = a fragment of an annotated microtubule;
farther = a detection somewhere else) and then tested for real filament contrast against a
displaced-curve null:

| source | n GT | FP | near (fragments) | far | far **and real** | implied ceiling F1 |
|---|---|---|---|---|---|---|
| 585 | 104 | 55 | 35 | 20 | 11 | 0.950 |
| 586 | 359 | 404 | 277 | 122 | 24 | 0.968 |
| **htw** | **796** | 690 | 285 | **405** | **319** | **0.833** |
| **pooled** | **1259** | | | | **354** | **0.877** |

Those far detections carry **49x the contrast of the null** on HTW, against the annotated
ground truth's own 93.4 — they are real filaments the annotator did not draw.

**The count is inflated by fragmentation, so the ceiling is a range, not a number.** Repeating
the audit at `min_length = 20`, where the instancer breaks detections up far less, drops
far-and-real from 354 to 182 and lifts the pooled ceiling to **0.933** (htw 0.908). Several
fragments of one unannotated filament are counted as several false positives here, while a
non-fragmenting method would produce one. So:

> 🔴 **The combined benchmark's attainable F1 ceiling is roughly 0.88 to 0.93. The 0.90 target
> sits inside that interval** — it is at best marginally reachable, and on the pessimistic end
> it is not reachable at all. **HTW is the whole story: its own ceiling is 0.83-0.91**, against
> 0.95-0.97 for both MT-34 halves.

This is a decision for the user, not a detail: either the target moves, or HTW's ground truth is
completed, or progress is reported against a stated ceiling rather than against 1.0.

**What would sharpen it:** count *distinct* unannotated filaments rather than fragments (merge
collinear far detections), and have a human confirm a sample of them on the images. Until then
the interval, not either endpoint, is the honest statement.

---

## 2. What the design has to do

1. **No length filter, and no hand-tuned geometric constants** — the user's constraint, and
   independently what §1.2 says is destroying HTW.
2. **Stay amodal at crossings** — each microtubule keeps its identity through a shared pixel.
3. **Separate close parallels** — two filaments 2–6 px apart must not fuse. §1.1's own
   `parallel_screen` finding says this information is destroyed at the *mask* level, before any
   geometry runs, so it must be fixed in the representation.
4. **Handle both length regimes** — 29 px and 700 px filaments from one model.
5. **Stay annotation-free** — supervision from the generator only; that is the paper's headline.

---

## 3. Architecture: dense local shape patches (PatchPerPix)

Reference: Hirsch, Mais & Kainmueller, *PatchPerPix for Instance Segmentation*
(arXiv:2001.07626), SOTA on BBBC010 — overlapping, crossing, thin curvilinear worms in 2D,
which is the closest published setting to microtubules.

### 3.1 The representation

The network predicts, for every foreground pixel `x`, a **local binary patch**: for each offset
`dx` in a fixed neighbourhood, whether `x + dx` belongs to *the same instance as x*. This is a
learned local shape, not an affinity.

Why this solves the two standing bottlenecks:

- **Parallels.** Two pixels 3 px apart on two different filaments have *different* patches — one
  contains filament A only, the other filament B only. The separation survives even when the
  semantic mask fuses them into one band, which is exactly the information §1's screen showed is
  otherwise destroyed.
- **Crossings.** Overlap pixels are **excluded from patch prediction** (their patch is genuinely
  ill-defined), and are instead recovered because patches centred on *non-overlap* pixels of each
  filament reach across the junction and claim them. Both filaments claim the shared pixels, so
  a pixel ends up in two instances. Verified in the paper's own words: *"overlapping instance
  patches can be selected, potentially resulting in pixels covered by multiple instances."*

### 3.2 Why this is not a rerun of a closed dead end

The project has three recorded negatives in this area, and the failure mode of each is precisely
what this representation lacks:

| closed route | why it failed | why patches differ |
|---|---|---|
| per-pixel embeddings / Object Condensation | parallel-merge: pixels **compete** for one label | a pixel may belong to several patches; no competition |
| affinities + Mutex Watershed | plateaued at 0.49; affinities carry no **shape** | patches are learned shapes, not pairwise scores |
| DETR-style curve set prediction | curves **diverged** globally (p50 10.4 px at tol 5) | nothing is predicted globally; instances are assembled from local pieces |

### 3.3 Network

`ppp+dec` from the paper: a U-Net predicts a low-dimensional **code** per pixel; a jointly
trained decoder expands the code into the binary patch. Direct patch prediction (`ppp`) needs one
output channel per offset — a 25x25 patch is 625 channels at full resolution — which does not
fit, and the paper's own ablation prefers the end-to-end decoder over a separately trained
autoencoder.

Three heads, all from one trunk:
- **code** (per pixel) → decoder → patch,
- **foreground** (binary), which the assembly needs and which is also the metric's semantic term,
- **number of instances** at each pixel (categorical), which is how overlap pixels are found at
  inference and masked out of the patch loss during training.

The trunk is v5H's nnU-Net ResEnc-M, warm-started (1350/1364 tensors transferred cleanly for the
set-prediction attempt, so the plumbing is known to work).

**Patch size is the one parameter that must be derived, not guessed.** A patch has to reach
across a crossing from a pixel outside it. Two 3-px filaments meeting at angle `a` overlap over
roughly `3 / sin(a)` px: ~6 px at 30 degrees, ~17 px at 10 degrees. MT-34's crossing-angle
distribution has median 53 degrees with 19.5 % below 30, so a patch of ~25 px covers the great
majority and ~40 px covers nearly all. **Derive it from the measured crossing-angle distribution
and state the residual fraction it cannot span** — do not tune it on VAL.

### 3.4 Assembly

Consensus affinities over patch pairs, greedy set cover to select high-scoring patches, then
connected components on the positive subgraph (mutex watershed is the paper's alternative).
Surviving hyperparameters: the patch probability threshold `t` and the partitioning choice.

That is **two** knobs against instancer A's seventeen, and no length filter anywhere. Honest
framing for the paper: this is *less* post-processing, not *none* — the polylines-out alternative
(a recurrent tracer, option B in the review) is the only genuinely zero-post-processing design,
and it was not chosen.

### 3.5 Resolution

Predict at the 1.5x eval frame rather than native: two parallels 3 px apart at native are 4.5 px
apart there, which is more room for two patches to differ. Cost is 2.25x the pixels on a
foreground that is only 1.5–2 % occupied, and the paper's sparse restriction of `patch(p_x)` to
the foreground applies. **A/B both resolutions on synth before committing.**

---

## 4. Phase 1: the generator

Needed under every architecture, annotation-free, and never checked.

- **Length prior.** Replace the `(80, 950)` clamp and the single log-normal with a prior that
  covers 15–950 px. The degenerate pile-up at 950 is a defect in its own right.
- **Density.** `target_fg = 1.6` was fitted with long filaments. Short microtubules at HTW's
  ~24 per 512² frame change the foreground fraction at fixed count, so `n_mt_range` must be
  re-calibrated **in the same pass**, not inherited.
- **Do not fit the prior to the benchmark.** Per the standing rule against MMD-fitting morphology
  to a biased corpus, widen the prior to *span* both regimes rather than matching either.

---

## 5. Plan, with pre-declared gates

| phase | what | gate | cost |
|---|---|---|---|
| ~~0a~~ | ~~false-positive contrast audit~~ | **DONE** — ceiling 0.88–0.93, §1.4 | — |
| ~~0b~~ | ~~no-filter baseline~~ | **DONE** — filter worth +0.057, debt 434 FP, §1.4 | — |
| **1** | generator length + density | synth GT length distribution spans all three sources | CPU |
| **2a** | patch autoencoder on synthetic patches | reconstruction IoU > 0.95 at the chosen code size | minutes |
| **2b** | `ppp+dec` trained on synth, scored on synth | **beat instancer A's 0.4166 on the same 40 synthetic frames, or drop the direction** | ~hours, 1 GPU |
| **2c** | real combined VAL, per-arm threshold and per-arm parameters | beat 0.3847 pooled with no source regressing | CPU |
| **3** | combined TEST, once, pre-registered | — | — |

Phase 2b's gate is copied verbatim from the set-prediction attempt, where it correctly killed a
direction in one run instead of three months.

---

## 6. Risks, stated rather than assumed away

1. **The foreground problem does not go away.** PatchPerPix replaces the assembler; §1.1 says the
   foreground still costs 0.283 pooled. Phase 1 and the standing `pos_weight` lead address it, but
   this design alone cannot reach 0.90.
2. **Oracle screens bound what is possible, not what is trainable.** The thin-target lever
   promised +0.030 from an oracle screen and delivered +0.0095 trained. The 0.918 ceiling is an
   upper bound on the *assembler*, not a forecast.
3. **Assembly cost.** The consensus/set-cover stage is expensive enough that the authors wrote it
   in CUDA. Sparse foreground helps; measure it on one frame before scaling.
4. **Overlap pixels are excluded from the patch loss**, so shallow crossings — where the overlap
   region is longest — lose the most supervision. This is the case the current system is already
   weakest on (junction identity 0.656 shallow vs 0.808 steep).
5. **The benchmark ceiling may be below 0.90** (§1.4). Phase 0a settles it.

---

## 7. Open decisions for the user

1. **One universal configuration, or per-lab annotation-free calibration?** (calibrate the
   generator on a lab's unlabelled frames, then tune on that synthetic data). The second fits
   "a model good on mine and on HTW" and stays annotation-free, but it changes what the paper
   claims.
2. **Where the "no post-processing" line sits.** This design keeps two knobs; a recurrent tracer
   would keep none, at higher risk.
3. **What happens if Phase 0a puts the ceiling below 0.90.**

---

## 8. Related work to position against

- **CurvSegFlow** (arXiv:2606.21608, Braun & Lánský, BIOCEV) — microtubules, iterative refinement
  by flow matching. **Semantic masks only, and supervised**, so it does not pre-empt the instance
  claim, but it is a mandatory citation and a foreground baseline from a neighbouring lab.
- **FISBe** (arXiv:2404.00130) — benchmark for instance segmentation of long thin filamentous
  structures; PatchPerPix is one of its baselines.
- **Trexplorer Super** (arXiv:2507.10881) / **RefTr** (arXiv:2511.20823) — recurrent centerline
  tracking; the option-B fallback if the patch route fails its gate.
