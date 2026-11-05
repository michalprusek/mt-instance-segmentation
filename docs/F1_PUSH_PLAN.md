# Raising strict centerline-F1 — the measured diagnosis and the plan (2026-08-13)

> Supersedes the working diagnosis in `docs/TODO.md` §3 ("STILL #1 BOTTLENECK = the SEMANTIC
> FOREGROUND") and the reading of §20/§20a/§21 that the mask is *shattered*. Both were partly
> right about the symptom and wrong about the cause. Everything below is measured; the scripts
> that produced each number are named.

## 1. Where the F1 actually goes

Two benchmarks, and they fail for **different reasons**. Merging them is what produced six
consecutive null results.

### Real MT-34 VAL — the mask contains the microtubules; assembly loses them

Per-microtubule coverage of the predicted foreground, at the metric's own 5 px tolerance
(`scratchpad/real_cov.py`): **94.2 % of 463 microtubules are ≥95 % covered.** The attainable
true-positive ceiling is therefore 93.5 %, and the pipeline delivers 51.6 %.

The mask factorial (`scratchpad/factorial_real.py`, instancer A, 17 VAL frames) isolates why.
Read **true positives**, not F1 — deleting far-from-GT components removes false positives
mechanically, so only a rise in TP shows that assembly improved:

| mask variant | F1 (model params) | TP | FP | isolates |
|---|---|---|---|---|
| B — model, as shipped | 0.490 | 229 | 355 | — |
| C — `model ∪ oracle` (holes filled, excess kept) | 0.519 | 244 | 302 | breaks: **+15 TP** |
| E — `model ∩ dilate(oracle,4)` (excess removed, breaks kept) | 0.640 | 270 | 225 | excess: **+41 TP** |
| A — oracle | 0.861 | 358 | 89 | near-GT mask *shape*: **+88 TP** |

Removing structure far from any microtubule **creates 41 true positives**: the excess was not
just false-positive mass, it was actively preventing correct assembly. The largest single term
is the last one — even with all far-from-GT junk deleted, 88 microtubules are still lost to
defects *within 4 px of the ground truth*. `diag_failure_geometry.py` (D2) names them: with the
oracle mask 100 % of coverage breaks sit on a crossing or a close-parallel pair, whereas with
the model mask **28 % of breaks are mid-filament with no geometric excuse, and only ~30 % of
those are actual holes**. The mask fails by **shape** — spurs, blobs and width irregularity
that shatter the skeleton arc graph while the pixels are present.

**Filling holes is worth 15 TP.** That is the entire prize that morphological closing,
anisotropic closing, gap-linking, the temporal model and cbDice were all competing for.

### Synthetic in-domain — one third of the labels are below the detection floor

The premise was audited first, because v4b was trained 2026-07-08, the generator was touched
2026-07-09, and `data/synth_eval` was rendered 2026-08-10. Re-scoring with `control` — a model
trained 2026-08-12 on exactly the generator that rendered those frames — gives **0.095**, worse
than stale v4b's 0.153, against an oracle-mask 0.734. The in-domain claim survives its own audit.

But the synthetic ground truth is not attainable. Measuring per-microtubule SNR with one
estimator on both sets (`scratchpad/real_snr_ref.py`):

| set | SNR p25 / p50 / p75 | below SNR 1 |
|---|---|---|
| real MT-34 VAL (human-annotated) | 0.92 / **1.45** / 2.14 | 28.1 % |
| synthetic VAL (generator-labelled) | 0.37 / **0.50** / 0.73 | **88.2 %** |

Coverage is 97.5 % for synthetic microtubules above SNR 1 and 63 % below it. **The generator
labels filaments it renders below the noise floor** — 37 % of synthetic ground truth carries a
median predicted probability of 0.001 along its own centerline (D4). The same configuration
generates the *training* data, so roughly a third of training positives are unlearnable.

## 2. The causal chain

1. The generator paints labelled microtubules at ~1/3 the contrast-to-noise of any real
   annotated one.
2. The network is therefore trained to declare foreground on evidence three times weaker than
   a real microtubule ever presents, and to do so for targets that carry no signal at all.
3. On real data that low-evidence trigger fires on background texture and on the field-stop
   rim, producing far-from-GT excess (**worth +41 TP**) and ragged near-GT shape (**+88 TP**).
4. The shallow full-resolution `hr` branch amplifies it: ablation (D4) shows it contributes
   **no measurable F1** (+0.017, 95 % CI [−0.032, +0.068], p=0.31) while supplying **68 % of
   the mask's connected components**. Its receptive field is 29 px — it cannot integrate
   evidence along a 375 px filament, which is exactly what a faint filament requires.

This explains the six nulls in one stroke: every one of them repaired **short breaks**, and
short breaks are the smallest term in the budget.

## 3. Plan

Ordered by measured expected value. Selection on synth VAL and real VAL only; real TEST scored
once, at the end, with a bootstrap CI and the exposure count recorded.

| # | intervention | evidence | expected |
|---|---|---|---|
| 1 | **Generator SNR fix**: contrast ×3.0 so labelled-MT SNR matches the real annotated distribution (`scratchpad/snr_calib.py` sweep: ×3 → median 1.36 vs real 1.45) | 88.2 % of synth GT below SNR 1 vs real 28.1 %; coverage 97.5 % above SNR 1 | removes ~1/3 unlearnable labels; predicted to cut the low-evidence trigger that causes the excess |
| 2 | **Drop the `hr` branch**, keep DINO + ASPP; give the decoder a large receptive field | hr = 68 % of components, F1 contribution is a null; DINO-only reaches 1.61 components/GT ≈ nnU-Net's 1.49 | targets the +88 TP shape term |
| 3 | **Recipe**: online infinite generation, cosine LR + warmup, AMP, EMA, rotation augmentation, and **checkpoint selection on instance-F1** (`scripts/val_instance_f1.py`) rather than a proxy | loss still descending at cutoff; last epoch is never the best (prodA gate picked 27/45, control 10/30); §17p is the cost of proxy selection | removes a known lottery |
| 4 | **Separate binary foreground head** alongside the K orientation channels | union target is 1 component/MT while per-channel targets are 4.35; nnU-Net's binary head gives 40 % fewer components at higher foreground | shape |
| 5 | Soft two-bin orientation assignment + sub-pixel stamping; crossing multi-label term | amodality realised at only 14.3 % of applicable crossings; weaker bin p=0.145 vs 0.718 | crossings |
| 6 | Generator crossing-angle distribution | synth 44.1 % below 30° (median 33.6°) vs real 19.5 % (median 53.1°) | crossings |

**Not to be run again**: any sixth intervention that repairs short breaks. Bounded above by
+15 TP on real and +0.07 F1 on synth.

## 3b. Levers tested and closed (2026-08-13, same session)

**Learned arm-pairing cost — NEGATIVE, and informative.** `src/instance/linker.py` replaces the
hand-crafted `pair_cost` with a 14-feature MLP trained on synthetic data only, where the label
is exact (two skeleton arcs are a positive pair iff they belong to the same generated
microtubule — supervision real data cannot provide). Trained on a *separate* synthetic set
(`data/synth_link`, disjoint from the eval split) over a mixture of mask qualities: predicted,
rasterised ground truth, and ground truth degraded with measured spur/hole/width defects.

On held-out synthetic candidates it clearly beats the geometric cost — **precision@k 0.862 vs
0.776**, holdout accuracy 0.921. On real MT-34 VAL, end to end:

| configuration | F1 | TP | FP |
|---|---|---|---|
| geometric cost (shipped) | 0.4899 | 229 | 355 |
| learned, junctions **and** gap links | 0.4636 | 174 | 302 |
| learned, **junctions only** | 0.4928 | 229 | 354 |

Two conclusions. (i) All of the damage came from replacing the *gap-linking* cost, whose
`c_open_link` / `link_max_gap` / `bridge_thr` were tuned together — swapping the scorer under
them changes two things at once. That path now defaults to the geometric cost
(`params["linker_gap"]`). (ii) At junctions the learned cost is a **null**: identical true
positives, 229 either way. A better junction decision is not what is missing.

That is a genuinely useful negative. It rules out the intervention the external literature most
strongly suggested (learned arc-level edge costs, after Vesselformer's 17.9 → 78.1 edge mAP on
synthetic vessels) and it corroborates the factorial: the loss is not in *deciding* which arms
continue, it is that the mask's local shape around the ground truth corrupts the arms
themselves. Fix the mask; the existing matcher is already good enough given a clean one.

*First version of this experiment was wrong and the error is worth recording*: the bridge
features were sampled from a binary mask during training and from a continuous probability
field at inference, which cost 111 true positives (F1 0.379). The symptom that exposed it was
`p_open` having no effect at all — a threshold that does nothing means the scores are degenerate,
not that the threshold is optimal.

**Decoding — closed.** A flat-threshold sweep peaks at 0.469 (thr 0.30) against the shipped
0.466, and Frangi ridge enhancement scores 0.435. Worth ~0.004; not a lever.

## 3c. Training runs in flight, and the baseline they must be read against

`scripts/train_v5.py` validates on the first 8 non-empty VAL frames, which is **6 Alice frames
plus the two easiest 586 frames** (31 and 4 microtubules); every crossing-dense frame in VAL
(101, 57, 57, 54, 36 microtubules) is in the other half. So the training curves are NOT
comparable to the 0.490 / 0.457 figures, which are 17-frame numbers. Measured on the identical
8-frame subset, **v4b scores 0.5815** — that is the number these runs have to beat.

Two consequences, both handled rather than noted: the per-epoch selection is biased toward the
easy half, so the final checkpoint must be re-scored on the full VAL before it is chosen; and
any comparison to historical numbers has to state which subset it used.

| run | architecture | generator | 8-frame VAL |
|---|---|---|---|
| baseline v4b | frozen DINOv2 + hr branch | original | **0.5815** |
| A | ResU-Net (from scratch) | SNR-corrected | 0.280 @ ep20 → 0.302 @ ep24 |
| C | ResU-Net (from scratch) | original | **0.016** @ ep20, gate never passed |
| B | frozen DINOv2 + hr (v4b's trunk) | SNR-corrected | running |

**RETRACTED: "A versus C shows the generator fix works."** That comparison scored both models
at ``prob_thr=0.4418`` — a threshold tuned for v4b, for neither of them. Swept per model on the
same 8 frames at the same epoch 16:

| run | generator | thr 0.45 | thr 0.75 | thr 0.90 | thr 0.96 | thr 0.99 |
|---|---|---|---|---|---|---|
| A | SNR-corrected | 0.195 | 0.370 | 0.406 | **0.505** | 0.500 |
| C | original | 0.059 | 0.195 | 0.371 | 0.480 | **0.489** |

At its own operating point C reaches **0.489** against A's **0.505** — a difference of 0.016 on
8 frames, which is nothing. The "0.280 versus 0.016" and "C fails to converge" reported earlier
were **entirely an artefact of a mis-calibrated shared threshold**. C converges fine; it simply
puts its probability mass higher, and at 0.4418 its mask floods the frame.

So the generator SNR fix, whose *defect* is solidly measured (synth labelled-MT median SNR 0.50
against real 1.45), currently shows **no measurable downstream benefit**. It joins the list of
interventions that moved a diagnosed quantity without moving F1.

**The methodological finding is the durable one, and it is bigger than the run.** A model
comparison at a fixed shared threshold measures probability calibration, not model quality —
here it manufactured a 0.47 F1 difference out of a 0.016 one. Every model-versus-model number
in this repository that did not sweep the threshold per model should be re-derived.

**Scope, checked rather than asserted.** The obvious suspect was cbDice, whose "collapse to
25 % foreground at w=2.0" (§20b) was read off a fixed threshold. Swept on MT-34 VAL, the
collapse is **real**: foreground 28.1 % at thr 0.35, still **17.3 % at 0.90**, and 0.0 % by
0.99, with F1 at 0.008 / 0.000 / 0.000 — there is no operating point at which that checkpoint
produces a usable mask. Its probability field is saturated, not displaced. §20b stands.

The remaining exposure is smaller than feared: the gated retrain, the temporal model and
prodA/B all share v4b's recipe and land at 1.85–1.98 % foreground at the same threshold, so
their calibrations match and those comparisons are probably sound. The confound bites when the
data or the objective changed a lot — which is exactly the v5 runs and nnU-Net.

### The architecture swap failed, and the measurement says why

Full VAL (16 frames), each model swept to its own best threshold through the identical code
path (`scripts/eval_v5.py`): **v4b 0.458, v5A 0.365.** Mask topology at matched foreground
explains it — the from-scratch network is not producing a cleaner mask, it is producing a
markedly worse one:

| mask | foreground | components / MT | endpoints / MT | branch points |
|---|---|---|---|---|
| oracle (target) | 1.59 % | **0.70** | **1.96** | 148 |
| v4b — frozen DINOv2 + hr branch | 1.90 % | 5.27 | 10.53 | 168 |
| v5A — ResU-Net, thr 0.98 | 1.90 % | **9.52** | **18.69** | 155 |

This **contradicts the D4 audit's central architectural recommendation**. That audit measured
the frozen-DINOv2 pathway alone at 1.61 components per microtubule against the full model's
5.09, and concluded the shallow high-resolution branch supplies 68 % of the shattering for no
benefit — so replacing the whole trunk with a deep encoder-decoder should de-shatter the mask.
Trained rather than ablated, this one does the opposite. Ablating a branch at inference and
training without it are different experiments, and D4 flagged exactly that caveat.

**What this does and does not license.** It does NOT show that from-scratch encoder-decoders
are the wrong family: **nnU-Net, trained from scratch on this same synthetic data, reaches 1.49
components per microtubule** — cleaner than v4b's 5.27 and far cleaner than this ResU-Net's
9.52 — and its instances already beat v4b downstream (0.418 vs 0.379, junction identity 0.720
vs 0.511, `docs/TODO.md` §3). So run A indicts **this recipe**, not the family, and the frozen
ViT-L is not uniquely able to supply long-range coherence. Intervention #2 as specified (my
ResU-Net) is withdrawn; the architecture question is reopened toward nnU-Net's topology, which
is the strongest untested candidate the project already has on disk.

**A's training-curve numbers under-measure it by ~0.2, and the reason is itself evidence.**
Those validations decode at ``prob_thr=0.4418``, tuned for v4b's probability calibration. A
sweep on A's epoch-28 checkpoint (`scripts/eval_v5.py`, same 8 frames):

| threshold | 0.45 | 0.60 | 0.75 | 0.85 | 0.92 | **0.96** | 0.98 | 0.99 |
|---|---|---|---|---|---|---|---|---|
| F1 | 0.352 | 0.410 | 0.450 | 0.468 | 0.489 | **0.517** | 0.510 | 0.487 |
| foreground % | 3.77 | 3.02 | 2.50 | 2.18 | 1.92 | 1.69 | 1.47 | 1.24 |

So A at epoch 28 is **0.517** against the baseline's 0.5815, not 0.320 — with 12 epochs still
to run. The optimum sits where the foreground fraction lands at 1.69 %, next to v4b's 1.85 %
and the oracle's 1.51 %: the mask is the right size, it was being cut at the wrong place.

That the new model needs a threshold of 0.96 where v4b needs 0.44 is not a nuisance, it is the
diagnosis confirmed from the other side. A network trained on filaments it can actually see
becomes confident and separates foreground from background sharply; a network trained to mark
filaments at SNR 0.3 must hedge, and a hedging network is one whose mask fragments at any
threshold. Checkpoint selection during training inherits the old threshold and is therefore
also mis-calibrated — the final choice is made by this sweep on the full VAL split instead.

### Run B — the single-variable experiment, and its verdict: NULL

v4b's own trunk, only the training data changed (SNR-corrected generator), 30 epochs, decoded
from the binary union head with the threshold swept per checkpoint on the **full** 16-frame VAL:

| checkpoint | 0.75 | 0.85 | 0.90 | 0.95 |
|---|---|---|---|---|
| ep24 | 0.438 | **0.454** | 0.452 | 0.430 |
| ep27 | 0.449 | 0.446 | 0.451 | 0.433 |
| ep30 | 0.432 | 0.431 | 0.439 | **0.455** |

Best 0.455 against v4b's 0.458. **Paired bootstrap, B − v4b = −0.004 [−0.057, +0.058],
p = 0.868.** Split by benchmark half, both also null: sparse Alice +0.024 [−0.076, +0.130]
(p=0.67), crossing-dense −0.020 [−0.074, +0.056] (p=0.51).

On the 8-frame training-validation subset B had reached **0.648 against v4b's 0.590**. That
subset is 6 Alice frames plus the two sparsest 586 frames, and the gain did not survive contact
with the crossing-dense half — a second demonstration, within one session, that the subset a
number is measured on can invent a 0.06 effect. Both traps this session (shared threshold, easy
subset) inflate in the same direction: toward believing an intervention worked.

**Verdict: the generator SNR correction is a null.** The defect is solidly measured — synthetic
labelled microtubules sit at median SNR 0.50 against real 1.45, and 88 % of synthetic ground
truth is below SNR 1 — but correcting it does not move F1 on real data. It joins the list.

## 3d. Re-deriving the model comparison under per-model thresholds

The calibration confound above says every fixed-threshold model comparison should be re-run.
The first one re-run is the most promising: nnU-Net, trained from scratch on this same
synthetic data, whose predictions have been sitting on disk scored once at 0.418.

Full VAL (16 frames), each model swept to its own operating point through one code path:

| model | best thr | F1 | TP | FP | FN | foreground |
|---|---|---|---|---|---|---|
| v4b | 0.44 | 0.458 | 229 | 355 | 234 | 1.90 % |
| **nnU-Net** | 0.995 | **0.501** | 231 | 292 | 232 | **1.32 %** |
| v5A ResU-Net, max-over-K | 0.96 | 0.365 | 170 | 396 | 293 | 2.25 % |
| v5A ResU-Net, **binary union head** | 0.85 | 0.387 | 187 | 393 | 276 | 2.12 % |

nnU-Net's sweep is 0.482 / **0.501** / 0.484 across thresholds 0.99 / 0.995 / 0.998, and it was
previously reported at 0.418 — the whole difference is where its probability mass sits.

**But the margin does not survive its interval.** Paired bootstrap over the 16 VAL frames,
nnU-Net − v4b = **+0.043 [−0.008, +0.102], p = 0.109**. Suggestive, not demonstrated. Nothing
in this session has produced a statistically supported improvement on VAL, so **TEST was not
scored** — spending that exposure on a result this size would buy a number the interval cannot
support, and the project already carries four TEST exposures.

Two smaller results worth keeping: the **binary union head is worth +0.022** over max-over-K on
the same checkpoint (0.387 vs 0.365), the first measured support for D2's most-backed
recommendation; and nnU-Net reaches its optimum at 1.32 % foreground, closest of any model to
the oracle's 1.51 %.

## 3e. Final standing, full MT-34 VAL, every model swept to its own operating point

| system | best thr | F1 | vs v4b (paired bootstrap) |
|---|---|---|---|
| nnU-Net (existing predictions, re-scored) | 0.995 | **0.501** | +0.043 [−0.008, +0.102], p=0.109 |
| **v4b (baseline)** | 0.44 | **0.458** | — |
| B — v4b trunk + SNR-corrected generator | 0.95 | 0.455 | −0.004 [−0.057, +0.058], p=0.868 |
| A — ResU-Net + SNR-corrected generator | 0.85 | 0.387 | not run (clearly below) |

**Nothing in this session produced a statistically supported improvement, so MT-34 TEST was not
scored.** The exposure count stays at four. The nnU-Net lead is the one worth pursuing, and it
needs a proper retrain with the K=6 amodal head before it can be claimed.

## 3f. nnU-Net with the amodal head (runs D and E, in flight)

nnU-Net's own framework cannot train the target this project needs: its head assigns each pixel
exactly one class, so a crossing pixel cannot belong to two microtubules. That is why the
existing nnU-Net run is binary-only, and why its 0.501 comes with a mask instancer B cannot
consume. The fix is to take the **architecture** and supervise it with per-channel sigmoids.

The topology is read out of nnU-Net's own `plans.json` for `Dataset501_MTSynth` rather than
guessed — `ResidualEncoderUNet`, 8 stages, features 32→512, `n_blocks_per_stage
[1,3,4,6,6,6,6,6]`, InstanceNorm + LeakyReLU, patch 512 — together with its optimiser (SGD,
momentum 0.99 nesterov, weight decay 3e-5, poly LR `(1-t)^0.9`) and deep supervision over the
top four decoder outputs. Targets for the deep heads are **max**-pooled, not averaged: a 2 px
filament averaged to quarter resolution falls below every threshold and would train the deep
heads on a blank target.

The contrast with the failed run A is the point: A was a 5-stage plain U-Net capped at 256
channels with AdamW, and its seven /2 stages fewer are exactly the long-range integration the
segmenter audit said was missing.

| run | architecture | generator | isolates |
|---|---|---|---|
| D | nnU-Net ResEnc-M + K=6 amodal + union head | original | the HEAD change vs the 0.501 binary nnU-Net |
| E | same | SNR-corrected | the generator fix on a trunk that is not frozen DINOv2 |

Validation scored 12 VAL frames rather than 8, reaching `training_img_11` (101 microtubules) so
the selection signal was not blind to the crossing-dense half; the final selection was still
redone on the full split with the threshold swept per checkpoint (both peak at 0.97, confirmed
by extending the sweep to 0.998 where both collapse).

### Results — and one of them overturns the earlier verdict

| system, full VAL, own operating point | thr | F1 |
|---|---|---|
| nnU-Net, binary head, nnU-Net's own framework | 0.995 | **0.5046** |
| **E — nnU-Net topology + amodal head, SNR-corrected generator** | 0.97 | 0.4715 |
| v4b | 0.4418 | 0.4655 |
| D — nnU-Net topology + amodal head, original generator | 0.97 | 0.3866 |

| paired bootstrap (n=16) | diff | 95 % CI | p | |
|---|---|---|---|---|
| **E − D (the generator fix)** | **+0.085** | [+0.039, +0.124] | **0.001** | **significant** |
| E − v4b | +0.006 | [−0.047, +0.068] | 0.881 | ns |
| **E − nnU-Net (binary head)** | **−0.033** | [−0.059, −0.008] | **0.012** | significant |
| nnU-Net (binary) − v4b | +0.039 | [−0.008, +0.095] | 0.120 | ns |

**The generator SNR correction works — but only on this architecture.** On the frozen-DINOv2
trunk it was a clean null (run B, −0.004 [−0.057, +0.058], p=0.868); on the nnU-Net topology it
is **+0.085, p=0.001**, the first statistically supported positive result in this work. That is
an interaction, not a contradiction: corrected labels can only be exploited by a network with
the receptive field and capacity to exploit them. The earlier conclusion "the SNR fix is a null"
was therefore true only of the trunk it was tested on — a lesson about what a data intervention
is tested against, which is exactly the mistake §3c's retraction was about, in a new costume.

**The amodal head, as trained here, loses to the binary head** by 0.033 (p=0.012). That
comparison is NOT clean: the binary run had nnU-Net's own framework, 250 epochs and its full
augmentation pipeline, against this one's 40 epochs and a hand-written augmentation set. The
same-data arm is D against the binary run — −0.118 — of which most is plausibly budget and
recipe rather than the head. **Separating "the amodal head costs F1" from "my training of it is
weaker" has not been done and should not be asserted either way.**

### 3f-bis. The factorial that settles it (runs F and G, in flight)

`--head binary` trains the union channel alone, completing a 2x2 in which everything except the
two factors is identical — one trainer, one budget, one augmentation set, one optimiser:

| | original generator | SNR-corrected |
|---|---|---|
| **amodal head** | D = 0.3866 | E = 0.4715 |
| **binary head** | G (running) | F (running) |

It answers two questions at once and gives each of them two independent estimates: the head
costs `F − E` and `G − D`; the generator fix is worth `F − G` against the measured
`E − D = +0.085`. `scripts/factorial_report.py` consumes the swept JSONs and prints every
pairwise interval plus the main effects and the interaction.

### Factorial results — the generator fix confirmed twice, and the head question answered

Full VAL, every arm swept to its own operating point, one trainer and one budget throughout:

| | original generator | SNR-corrected |
|---|---|---|
| **amodal head** | D = 0.3866 | E = 0.4715 |
| **binary head** | G = 0.4518 | **F = 0.4920** |

| effect | diff | 95 % CI | p | |
|---|---|---|---|---|
| **generator, amodal head** (E−D) | +0.085 | [+0.039, +0.124] | 0.001 | **significant** |
| **generator, binary head** (F−G) | +0.040 | [+0.013, +0.069] | 0.004 | **significant** |
| head, on **noisy** labels (G−D) | +0.065 | [+0.024, +0.109] | 0.001 | **significant** |
| head, on **corrected** labels (F−E) | +0.021 | [−0.032, +0.073] | 0.443 | ns |

Main effects: generator +0.063, head +0.043, **interaction −0.045**.

**The generator SNR correction is confirmed on both heads independently.** That settles it: the
defect is real, correcting it pays, and the earlier "it is a null" verdict was a property of the
frozen-DINOv2 trunk it was tested on, not of the intervention.

**The interaction is the answer to the head question.** A binary union head is worth +0.065 when
a third of the labels carry no signal, and nothing measurable (+0.021, ns) once they do. The
amodal channels are simply harder to learn from unlearnable labels; on corrected data they cost
nothing. So the K=6 representation can be kept for instancer B and for the amodal crossing story
**without paying F1 for it** — which is a better outcome than the earlier −0.033 suggested, and
the earlier number is now explained as budget plus label noise rather than the head.

**None of it beats the best baseline, though.** F − nnU-Net (framework-trained, binary) =
−0.013 [−0.053, +0.030] p=0.540; F − v4b = +0.027 p=0.500. The best system on VAL remains the
framework-trained nnU-Net at 0.5046. Worth noting that F matches it on **one sixth of the
training budget** (40 epochs against 250, without nnU-Net's augmentation pipeline), which is
where the unexploited headroom most plausibly sits.

**TEST remains unscored**; exposure count still four.

*Process note:* the first attempt at these two sweeps died silently five hours earlier.
`eval_v5.py` detects a checkpoint's head width from key names, and nnU-Net's ResEnc calls its
output layers `decoder.seg_layers.N.weight` rather than `head.weight` — so the detection fell
through to its default of K+1=7, which **happened to be correct for the amodal arms** and was
therefore never exercised until a 1-channel checkpoint hit it. A default that silently matches
the cases you test first is worse than no default.

## 3g. External baselines, measured on MT-34 for the first time

Every baseline number this project quotes — TARDIS 0.326, ORION 0.519, v19 0.696, SAM 3 0.712 —
is on the OLD 12-frame Alice set. MT-34 has 32.1 crossings per frame against Alice's 2.2, so
those numbers describe a different problem. `scripts/baseline_mt34.py` runs the two end-to-end
competitors (image in, instances out, nothing of ours in between) through the identical strict
metric, at 1.5x, with the same skeletonise-and-dilate conversion, each swept to its own
operating point on VAL.

**Both reproduce their published Alice numbers and then collapse on the crossing-dense half** —
which is simultaneously the check that they are wired up correctly:

| system (MT-34 VAL, own best operating point) | pooled | Alice half (n=6) | dense half (n=10) |
|---|---|---|---|
| **nnU-Net (binary head), thr 0.995** | **0.505** | — | — |
| **v4b (our production baseline)** | **0.466** | 0.766 | **0.273** |
| B — v4b trunk + SNR-corrected generator | 0.455* | 0.790 | 0.253 |
| **SAM 3 "thin line", conf 0.60** | 0.432 | **0.743** | 0.245 |
| A — ResU-Net from scratch | 0.387* | — | — |
| **TARDIS zero-shot (real-trained)** | 0.117 | 0.311 | **0.000** |

`nnU-Net − v4b = +0.039 [−0.008, +0.095], p = 0.120` — still not separable at n=16.
(*) A and B were scored before the harness fix below and are ~0.008 pessimistic; their ranking
is unaffected.

**A harness bug found while assembling this table, and it is the kind that quietly moves
numbers.** `eval_v5.py` eroded the field-stop mask by 4 px and left the orientation channels
unmasked, while `run_oracle_eval.py` — which produced every historical number in this project,
including TEST 0.457 — does neither. The erosion deleted real microtubules near the rim and
cost ~0.008. Since the nnU-Net numbers came from one harness and the instancer-axis numbers
from the other, the cross-table comparisons carried a hidden term about the size of the margins
being reported. `eval_v5.py` now reproduces `run_oracle_eval.py` exactly (v4b: 0.4655 from
both). The rule this earns: **a second harness must be validated against the first on a known
system before any number it produces is quoted.**

TARDIS recovers 31 of the 104 Alice-half microtubules and **0 of the 359** on the dense half.
SAM 3 degrades more gracefully but still loses two thirds. Their published numbers reproduce
almost exactly on the Alice half (TARDIS 0.311 vs 0.326; SAM 3 0.743 vs 0.712), so this is a
property of the benchmark, not of the harness.

**TARDIS's zero is a detection failure, not the crossing-collapse this project's history would
predict.** Measuring coverage of the ground-truth centerlines by *any* of its instances:

| TARDIS | GT | instances emitted | GT-length coverage (median) | ≥95 % covered |
|---|---|---|---|---|
| Alice half | 104 | 106 | **83.2 %** | 33.7 % |
| dense half | 359 | 123 | **0.0 %** | 0.0 % |

On the Alice half it genuinely finds the filaments (median 83 % covered, fragmentation 1.16)
and loses them only to the strict 95 % rule. On the dense half the 123 instances it emits lie
essentially nowhere near an annotated microtubule. So the sentence is "the real-trained
baseline does not transfer to these acquisitions at all", not "it finds them and cannot keep
their identity through crossings" — a claim worth getting right before it appears in a paper.

**Two recorded claims need rescoping.** `docs/TODO.md` W10 says "we do NOT beat SAM3" — true on
Alice (0.712 vs our 0.697) and false on MT-34, where we lead 0.458 to 0.432 pooled and 0.273 to
0.245 on the half that matters. And CLAUDE.md's "bar to beat: TARDIS zero-shot 0.326" is an
Alice bar; on MT-34 that bar is 0.117.

### The instancer axis, same v4b mask, MT-34 VAL

The table above compares whole systems. This one holds the foreground fixed and varies only the
assembly step, so the two axes stay separable (`run_oracle_eval.py --masks model`):

| instancer | pooled | Alice | dense | junction-id (dense) | max curvature |
|---|---|---|---|---|---|
| **A — curvature-bounded matching (ours)** | **0.466** | — | — | 0.356 | 0.191 |
| B — lifted beam tracing | 0.367 | — | 0.172 | — | — |
| PySOAX — classical filament tracer | 0.261 | 0.505 | 0.115 | 0.034 | **0.528** |

Paired bootstrap, stratified by task: `B − A = −0.099 [−0.143, −0.050]` and
`PySOAX − A = −0.205 [−0.262, −0.147]`, both p < 0.001.

**Instancer parameters are fitted to v4b's mask and penalise every other foreground.** Sweeping
three knobs on nnU-Net's mask, `merge_radius` 9.0 (shipped) → 3.0 is worth **+0.018**, monotone
across all six combinations of the other two: a large contraction radius helps a shattered mask
and destroys a clean one by merging two genuinely separate crossings. Treat that number as a
**bias diagnostic, not as nnU-Net's score** — it was selected on the same 16 real VAL frames it
is measured on, which is precisely the setup §17q showed rewards permissive settings against
incomplete human ground truth. The legitimate version is an Optuna re-tune on SYNTHETIC VAL
writing a new named params file, and every architecture arm needs the same budget.

PySOAX emits instances whose maximum curvature is **0.528 rad/px** against the 0.239 maximum
over all 957 annotated microtubules — shapes a microtubule physically cannot take — and its
junction identity on the dense half is 0.034 against A's 0.356. That is the failure instancer
A's hard curvature bound was built for, now measured on MT-34 with an interval.

Not run here: ORION and v19, which are internal heuristics rather than external baselines.

**One inconsistency to keep in view:** `run_oracle_eval.py` scores instancer A at 0.466 on this
split while `eval_v5.py` scores the same model at 0.458. The 0.008 is most likely the
field-stop mask being applied slightly differently. Small, but it means numbers from the two
scripts must not be mixed in one table without saying which produced them.

## 3g-bis. The other two generator defects, calibrated (2026-08-16)

### Crossing statistics — two wrong hypotheses before the right knob

Measured per unit AREA, because crossings per FRAME are not comparable between a 768 px
synthetic crop (0.59 Mpx) and a 1344x1416 real frame (1.90 Mpx) — an error worth naming, since
it briefly made the generator look 3x too sparse when it was not:

| per Mpx | MT | crossings | close-parallel pairs | median angle | below 30 deg |
|---|---|---|---|---|---|
| **real MT-34** | 24.0 | **20.4** | **3.6** | **53.2** | **19.0 %** |
| synth, before | 14.2 | 15.8 | 4.7 | 44.3 | 33.6 % |
| **synth, after** (`density_center` 8→12, `bundle_prob` 0.20→0.05) | 18.3 | **20.9** | **3.6** | **55.1** | **19.3 %** |

Four of the five statistics now match; only microtubules per Mpx stays low, and crossings and
parallels — the quantities the task is about — are hit almost exactly.

**Two candidate knobs were tested first and are measurably NOT the cause.** `orient_kappa`, the
nematic-alignment concentration and the obvious suspect, moves the median crossing angle only
between 42 and 53 degrees across its whole 0–3 range with no trend. `cluster_frac` is likewise
flat (35.7 % shallow at 0.20 against 30.6 % at 0.55). The knob that does control it is
`bundle_prob`, and the useful part is that lowering it fixes **both** deviations at once:
bundles were the source of the excess shallow crossings *and* of the excess close parallels
(4.7 against reality's 3.6). Reducing them therefore does not trade the parallel-bundle
training signal away — the generator had simply been over-producing bundles relative to
reality. Config: `calib_reg418_morph_snr3_xstat.json`, with the measurement recorded inside it.

### Field stop — synthesised into the training data, and round-trip tested

The generator composites onto real EMPTY-FIELD backgrounds, which carry no aperture, so the
model never had a reason to learn that a field boundary is not a filament — and 102 of 367
false positives on TEST are exactly that rim. `apply_field_stop` in `scripts/train_v5.py`
(`--fov-prob`) synthesises an octagonal aperture with geometry and photometry measured off the
**19 of 34 MT-34 frames that actually have one**: interior ~91 % of the frame, surround at
**2.5–5x the interior median** (37→187, 4563→11783), a short ramp so the transition gradient is
present rather than idealised away, and labels outside the aperture dropped.

Two details that were wrong on the first attempt and are worth keeping written down. The stop is
applied to the **raw render, before the percentile stretch**, because that is where it sits in
reality: a bright surround drags the frame's percentiles and squashes the imaged interior, which
is a distortion the model meets at inference and would never see if the aperture were pasted on
afterwards. And the surround is *bright* — `fov.py` says so and the measurement confirms it,
while an early reading of `predict_v4b_mt34.py` suggested dark (that is the Alice frames' dark
border, a different thing: 0.845 inside → 0.424 outside).

`tests/test_field_stop.py` includes the check that matters: the synthetic aperture must be found
by `mt_bench.fov.fov_mask`, the detector written against the real one. If the detector cannot
see it, the model is being trained on a boundary that does not resemble the boundary it will
meet.

## 3h. How to tune the instancer — researched, and the answer is "almost not at all"

A five-agent sweep (HPO methodology, objective design, and an empirical knob-sensitivity study
on 511 configurations x 25 exact-GT synthetic frames) plus an adversarial review. The headline
is a negative that saves most of the compute:

**Of the 14 assembly knobs, exactly ONE has headroom that survives to held-out frames:**
`merge_radius`, +0.042 [+0.004, +0.087], p=0.026. The next three largest-spread knobs gain on
the tuning split and **lose** on held-out. Composing the top-15 one-at-a-time optima is *worse*
than moving `merge_radius` alone (0.406 against 0.472 held-out). This reproduces, independently
and at much larger scale, the +0.018 I measured by hand for nnU-Net — and it says a 17-knob
Optuna study is the wrong instrument.

The correct framing is **algorithm configuration** (irace / SMAC / ParamILS), not neural HPO:
a non-differentiable combinatorial pipeline scored as a mean over problem instances. The
multi-fidelity dial is therefore a *frame subset*, and Optuna's `WilcoxonPruner` implements
frame-racing directly.

### What the adversarial review caught, before any compute

1. **A fatal bug in the tooling the protocol was to run on.** `tune_diag_common.score_frame`
   did `p.pop("prob_thr", None)` and instanced a mask that `load_frames` had thresholded ONCE,
   so the flagship 12x12 grid over (`merge_radius` x `prob_thr`) would have degenerated to
   1x12 and reported a **fabricated** per-model operating point. That knob is the whole point
   of per-model tuning — the measured optimum differs by 2x between models (0.44 for v4b,
   0.995 for nnU-Net). **Fixed.**
2. **The proposed tuning set covers 6 % of the benchmark.** Real MT-34 crossings are
   *bimodal*: 10 of 34 frames carry 685 of 733 crossings (93 %), and only 2 of 34 sit in the
   5-15 crossing range that 768 px synthetic crops produce. A tuning set must be **stratified**
   to bracket both modes — ~60 frames at crop 1344 (the dense mode) and ~60 at the Alice mode.
3. **The objective should stay the reported metric.** A multi-threshold surrogate was proposed
   for a smoother landscape; it is free but buys nothing on a tuning set where every frame sits
   in one mode, and it adds a surrogate-mismatch risk class.
4. **Real VAL was about to become a 3-way model selection** on human annotations. The
   foreground must be pre-declared on a synth-only criterion, with real VAL reported as a
   diagnostic carrying no decision.
5. **A fifth TEST exposure was pre-committed unconditionally**, against this project's own
   rule. It stays conditional on a real-VAL paired bootstrap whose interval excludes zero.
6. Budget: **~150k frame-evaluations (~5 h) instead of ~546k (~10.5 h plus ~30 h of
   unbudgeted risk)** — and a zero-compute triage on the already-stored 511-config rows can
   cancel the expensive stage before a single frame is rendered.

Two hazards in the shipped tuner also surfaced: `--tune-kappa` still exposes `kappa_max` to the
optimiser, which would destroy the project's central physics claim if anyone used it (it should
move to a separate ablation script and be blocked by a test), and `min_length=44.7` sits at 89 %
of its [6, 50] search range, so its optimum may be outside the range that was searched.

**Protocol, short form:** fix the tooling; render a stratified synth TUNE / SELECT / TEST-synth
split with disjoint seed bases (the generator makes fresh exact-GT frames for free, so the
tuning set never has to be reused — an escape from overtuning that no published HPO study has);
grid `merge_radius` x `prob_thr` exhaustively **per foreground model**; report TUNE / SELECT /
TEST-synth together so the winner's curse is a printed number rather than an assumption; and
treat anything beyond those two knobs as a null until a triage says otherwise.

## 3i. Run-to-run variance, measured at last — and it moves the blame

Three seeds of the best arm (F: nnU-Net topology, binary head, SNR-corrected generator),
identical except `--seed`, each swept to its own operating point on full VAL:

| seed | 0 | 1 | 2 | mean | SD |
|---|---|---|---|---|---|
| F1 | 0.4920 | 0.4851 | 0.4849 | **0.4873** | **0.0040** |

**Training noise is ~0.004, not the ±0.02 this project has been assuming.** No arm had ever
been repeated, so that number was a guess; it is now measured, and it is five times smaller.

The consequence matters more than the number. Every effect reported here sits far above the
seed noise — the generator fix (+0.085 and +0.040), the head effect on noisy labels (+0.065) —
yet several of them are still *not separable* by the paired bootstrap, whose intervals run to
±0.05. Those two facts together locate the uncertainty precisely: **it is the benchmark, not
the training.** Seventeen frames, not the optimiser, is what stops this project from resolving
a 0.03 effect. That is an argument for reviewing more of CVAT's 281 pre-annotated frames
(see the `benchmark-size-mt34` memory), and it is a stronger argument than any model change on
the list.

All three seeds land above v4b's 0.4655 and all three remain within its bootstrap interval.

## 3j. The two-knob tune, executed (`scripts/tune_two_knobs.py`)

Run for model H on stratified synthetic frames with exact GT — 30 dense (crop 1344) plus 30
sparse (768) for TUNE, and 30 further dense frames, rendered from a disjoint seed base, for
SELECT. `merge_radius` x `prob_thr`, 24 configurations, nothing else touched, `kappa_max`
untouchable by construction.

| `merge_radius` | 1.5 | 2.5 | 3.5 | **5.0** | 7.0 | 9.0 (shipped) |
|---|---|---|---|---|---|---|
| best TUNE F1 | 0.329 | 0.336 | 0.364 | **0.371** | 0.356 | 0.336 |

A clear peak, not a plateau — and it is not where the same knob's optimum sat for nnU-Net on
real VAL (3.0). Different foreground, different optimum: that is the whole case for tuning per
model rather than shipping one vector.

**The disjoint re-ranking earned its keep.** The TUNE argmax (5.0, 0.85) is only fourth-best on
fresh frames; (5.0, 0.97) wins there:

| config | TUNE | SELECT (fresh frames) |
|---|---|---|
| merge 5.0, thr 0.85 | **0.3714** (1st) | 0.2174 |
| merge 5.0, thr 0.97 | 0.3558 (4th) | **0.2431** (1st) |
| shipped (merge 8.98) | — | 0.1531 |

**On held-out synthetic frames the tune is worth +0.090** over the shipped parameters.

One number in that output must not be over-read. The script prints "optimism (TUNE − SELECT) =
+0.113", but SELECT is dense-only while TUNE is half dense and half sparse, so that figure
mixes the winner's curse with the fact that dense frames are simply harder. The valid
comparison is the **+0.090 measured on the same frames**; a clean optimism estimate needs a
SELECT set with TUNE's composition, which is a cheap fix for the next run.

Written write-once to `src/instance/params_a_model_v5H.json` (`merge_radius` 8.98 → 5.0,
`prob_thr` 0.44 → 0.97), so §17n's failure — a tuning run overwriting a canonical file and
destroying the reproducibility of earlier numbers — cannot recur.

## 3k. Where it closes

The tuned parameters transfer to real data at **+0.006, not the +0.090 they were worth on
held-out synthetic frames** — and only `merge_radius` transfers at all; the synthetic optimum
for `prob_thr` (0.97) is not the real one (0.99). That is a second, independent data point on
§17q's claim that synth tuning beats real-VAL tuning: it does not hurt, but on a different
foreground and a different architecture its benefit is an order of magnitude smaller than the
synthetic number suggests. **Quote the transferred number, never the tuning-set one.**

Final standing, full real MT-34 VAL, every system swept to its own operating point:

| system | thr | F1 |
|---|---|---|
| nnU-Net binary, framework-trained (250 epochs, old generator) | 0.995 | **0.5046** |
| **H tuned** — nnU-Net topology, binary head, all three generator fixes, `params_a_model_v5H` | 0.99 | **0.4953** |
| F — same, SNR fix only, 3 seeds | 0.93 | 0.4920 (mean 0.4873 ± 0.0040) |
| v4b — previous production | 0.44 | 0.4655 |
| SAM 3 "thin line" | 0.60 | 0.4318 |
| TARDIS zero-shot | — | 0.1166 |

| paired bootstrap (n=16 frames) | diff | 95 % CI | p |
|---|---|---|---|
| H tuned − v4b | +0.030 | [−0.024, +0.097] | 0.331 |
| H tuned − nnU-Net framework | −0.009 | [−0.055, +0.036] | 0.697 |
| H tuned − F | +0.003 | [−0.044, +0.054] | 0.917 |

**MT-34 TEST was NOT scored.** The pre-declared rule was that a fifth exposure requires the
real-VAL improvement over the shipped system to have an interval excluding zero. It does not
(+0.030, p=0.331). Exposure count stays at four.

The four leading systems are mutually indistinguishable on 17 frames while their point estimates
span 0.466 to 0.505. With training noise measured at SD 0.0040, that spread is not run-to-run
variation — the benchmark simply cannot resolve it. Everything downstream of that follows.

## 4. Honest ceiling

0.920 is instancer A's score *with a perfect mask*, and even there it loses 63 false positives
and 59 microtubules on TEST. Reaching 0.92 requires the predicted mask to be indistinguishable
from rasterised ground truth **and** the assembler to hold its oracle-mask performance. The
measured intermediate targets are what the factorial licenses: 0.640 is what excess removal
alone is worth today, 0.861 what near-GT shape repair adds. A result in the **0.60–0.75** band
on real TEST would be a large, defensible improvement over 0.457; 0.92 remains the north star,
not a commitment.
