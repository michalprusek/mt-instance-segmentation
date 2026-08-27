# TODO — consolidated: NC submission checklist (§0) + the measured record (§1–§4)

> Single source of truth (persists across sessions). **TARGET = NATURE COMMUNICATIONS** (user decision
> 2026-08-25; revised down from Nature Methods, which the 2026-07-11 honest verdict already said the
> evidence does not support — see [[nature-comms-target]]). Legend: [ ] todo · [~] in progress · [x] done.
> 🔴 = **NC-BLOCKING** · 🟣 = deferred to a possible later NM paper, explicitly NOT on the NC critical path.
> Peer-review IDs W1–W11 mapped in. **§0 is the submission checklist and is the operative plan; §1–§3b are
> the measured record that feeds it; §4 is paper-wide rigor.** PAPER_PLAN.md §"Claims → experiments" is
> superseded by §0 wherever the two disagree.

---
## 0 · NC SUBMISSION — the critical path (written 2026-08-25)

### 0.0-GOAL 🔴 THE TARGET IS NOW A NUMBER: strict centerline-F1 >= 0.90 (user 2026-08-25)

**User directive:** *"potrebuju pro NC maximalizovat F1 - chci se dostat alespon pres 90%... zapomen
na soucasnou moji best metodu - je to jen baseline, ktery musim prekonat."* v5H is therefore a
**baseline to beat, not a system to defend**, and architecture changes are authorized.

**Measured that same day, and it bounds everything below:**

| measurement | value | source |
|---|---|---|
| v5H shipped, MT-34 VAL | **0.480** (17 frames incl. the zero-GT probe) | lever 0 |
| **oracle mask + current instancer, VAL** | **0.876** | lever 0, `A_oracle` |
| attainable TP with the CURRENT mask | 310 / 463 | lever 0, `attainTP` |
| achieved TP | **227** | lever 0 |
| FP that are fragments of ANNOTATED MTs | **84.3 %** (258/306) | lever 3 |
| FP on unannotated real filaments | 15.7 % (48), at 57.7x null contrast | lever 3 |

- [ ] 🔴 **0.0-G1 · A perfect foreground is NOT enough — 0.90 exceeds the oracle-mask ceiling of
      0.876.** The assembler must change too; no amount of segmentation work alone reaches the target.
- [ ] 🔴 **0.0-G2 · The dominant error is ASSEMBLY, not detection.** 83 microtubules are lost between
      `attainTP` 310 and `TP` 227 with the mask held fixed (73 % conversion; ~95 % is needed), and
      84.3 % of "false positives" are fragments of real annotated microtubules. **FP and fragmentation
      are one problem.** `cc/MT` is 2.10 against the oracle's 0.53.
- [ ] 🔴 **0.0-G3 · The benchmark ceiling is ~0.92 and 0.90 sits AT it.** 48 detections lie on real
      filaments the annotator never drew and can never score as correct. Report progress against
      ~0.92, never against 1.0. See [[f1-90-target]].
- [x] 🔴 **0.0-G4 · The "never repair short breaks again" rule is SUPERSEDED for v5H.** The v4b-era
      budget (holes +15, excess +41, shape +88) does not hold on this foreground: **holes +45,
      excess +51, residual shape +53**. Holes tripled and are now co-equal with excess. The six nulls
      that rule was built on were measured on v4b's mask. Re-open closing/gap-linking on v5H.
- [ ] 🔴 **0.0-G5 · Main line for the target: end-to-end curve SET PREDICTION** (research 2026-08-25:
      Relationformer arXiv:2203.10202 · RefTr arXiv:2511.20823 · TopoMask/TopoBDA/CenterLineDet ·
      closest precedent Mathys et al. arXiv:2504.11858). It has **no assembly step**, so 0.0-G2's
      failure mode cannot occur; overlapping curves do not compete for pixels, which is why it escapes
      BOTH prior failures (embedding parallel-merge, DIST degree-2 crossing collapse); and the
      generator supplies unlimited exact per-instance polyline GT, which is exactly its supervision.
      **Two risks to state, not hide:** it forfeits the instancer's immunity to the domain gap, and it
      turns the derived hard kappa_max into a soft learned prior. Cheapest screen: train the head on
      synth, score on synth, and require it to beat instancer A's 0.695 there before touching real data.
- [x] 🔴 **0.0-G6 · HTW IS NOW IN THE BENCHMARK — deliberate user decision, 2026-08-25.**
      I argued against it and was overruled on an explicit goal change: *"to mi nevadi, ze mi to snizi
      uspesnost. chci mit model, co bude dobry na mem i htw."* **The target is a model that works on
      BOTH labs**, which is a stronger claim than one tuned to a single one — the cost is simply that
      the headline number is lower. Recorded so it reads as a choice, not an accident:
  - **HTW is no longer a sealed held-out set, and that is irreversible.** It can never again serve as
    independent cross-lab evidence (NC-C4 is void). Cross-lab generality now has to come from TIRF
    (NC-C2/C3) or a genuinely new lab.
  - **What it costs in the number:** HTW instance-F1 was 0.234 against MT-34's ~0.50, and it is
    *semantic-limited* — foreground tol2 0.620, ~38 % of centerline uncovered, so there are no arcs to
    group. Four interventions (DR-wide, ASPP, deep-sup, scale-aug) failed to move it. **Reaching >=0.90
    on the combined set therefore requires fixing HTW's foreground, which is a harder problem than
    anything on this list.**
- [x] 🔴 **0.0-G7 · The combined benchmark is BUILT: `data/real/mt34_htw_eval/`, 100 frames /
      2526 polylines**, split 50 VAL / 50 TEST.

      | source | frames (val/test) | polylines |
      |---|---|---|
      | 585 Alice | 6 / 6 | 229 |
      | 586 crossing-dense | 11 / 11 | 728 |
      | htw | 33 / 33 | 1569 |

  - **The scale trap was handled by normalising the DATA, not the metric.** HTW's `.npy` GT is stored
    at 1.5x the tif resolution while the h5 layout is defined to hold native 1x
    (`write_frame_h5`: *"the x1.5 eval upscale is applied by the evaluation pipeline, not stored
    here"*). The converter divides by 1.5 on the way in, so after conversion **one convention holds
    across the whole benchmark and no code path needs to know which lab a frame came from.**
  - **Verified empirically, not just arithmetically** (`scratchpad/verify_htw_scale.py`): mean image
    contrast under the GT is **78.3 at the correct scale, 21.1 at the wrong one, 22.8 for a null**
    (same curves dropped at random). At the wrong scale the GT sits *below chance* — which is exactly
    what "silently collapses the metric" looks like. Max coordinate 766.5 -> 511.0 on a 512 px frame.
  - HTW was split 33/33 val/test with the project's own `assign_split` (alternating over sorted
    names), so **a held-out TEST still exists**; tune on the 50-frame VAL, score TEST once.

### 0.0-SETPRED · Attempt 1 at end-to-end curve set prediction — **NEGATIVE** (2026-08-26)

Built and run against a **pre-declared** decision rule: *beat instancer A on synthetic data, where
GT is exact and n is unlimited, or the direction is dropped.* It did not, by a factor of 32.

| method, 40 fresh synth frames, identical images | F1 | TP / 355 | FP |
|---|---|---|---|
| curve set prediction (best obj_thr 0.7) | **0.0128** | 4 | 605 |
| **instancer A on the same frames** | **0.4166** | 117 | 182 |

**What it was:** DETR-style. nnU-Net ResEnc-M trunk warm-started from v5H (1350/1364 tensors),
stride-8 memory, 6-layer transformer decoder, 300 queries x 16 control points, Hungarian matching in
the loss only, direction-invariant L1. 147M params, 30 epochs, ~40 min on an A100. Loss converged
cleanly (0.673 -> 0.161, plateau by epoch 25). Code: `scripts/setpred_model.py`,
`train_setpred.py`, `eval_setpred.py`; 7 tests, mutation-tested.

**Three hypotheses for the failure, measured rather than argued:**

| hypothesis | verdict |
|---|---|
| representation: 16 control points cannot express a microtubule at tol=5 | ❌ **REFUTED** — GT resampled to 16 points scores **F1 = 1.000** |
| coordinate order `(x=col,y=row)` flipped | ❌ **REFUTED** — 6.87 px as-is vs 399.84 px flipped |
| predicted curves too short for 95 % length coverage | ❌ **REFUTED** — median length 211.6 px vs GT 216.1 (ratio 0.979) |

**What is actually wrong:** the curves have the right count, the right length distribution and sit in
the right region, but they **do not follow individual microtubules**. Per-point distance to the best
matching prediction is p25 4.62 / **p50 10.39** / p75 20.52 / p90 36.59 px — a long tail, not an
offset. Only 30 % of GT length falls within the 5 px tolerance and **2.1 % of curves reach the
required 95 % coverage**; relaxing the tolerance to 12 px still only gets 19 % there. **This is a
divergence failure, not a near miss.**

- [x] 🔴 **A real bug was found and fixed on the way, and it invalidated the first eval entirely.**
      `generate_frame` returns the FULL geometric centerline including the part outside the field of
      view — coordinates reach col = -382 on a 512 px crop. Training clipped to the crop; the first
      eval did not, so the model was scored against curves that are mostly off-frame and scored
      0.0000. Both now use the project's own `build_synth_eval.clip_polyline` (longest contiguous
      run inside the frame). The training path had a second, smaller version of the same bug: a
      boolean mask spliced two separate visible runs into one curve that jumps across the frame.
- [ ] **What a second attempt would have to change** — every published system in this family uses
      machinery this one omitted: auxiliary losses at each decoder layer, deformable attention,
      higher-resolution memory (stride 8 quantises position at 8 px, comparable to the error), and
      above all **iterative refinement** — RefTr (arXiv:2511.20823) is *entirely* about refining
      whole trajectories, which is precisely the fix for curves that start right and diverge.
      **Do NOT re-run this configuration.** A second attempt is a new build, not a longer run, and it
      needs its own budget decision.

### 0.0-REPAIR · Oracle-free mask repair recovered **+11 of the +90** on offer (2026-08-26)

**First, the correction that matters:** the factorial's `C`/`E`/`CE4` variants define what to fill
and what to delete **by reference to the oracle mask**. "Far from any ground-truth microtubule" is
not computable without ground truth, so **+93 TP is a budget, not a method**. This screen tested
operations using only the image and the predicted probability map, same 17 real VAL frames, same
instancer, TP read rather than F1.

| variant | F1 | TP | dTP | fg % |
|---|---|---|---|---|
| B_model (shipped) | 0.4519 | 227 | — | 1.77 |
| **H_hyst_0.9** — hysteresis, seed 0.97 / grow 0.9 | **0.4775** | **238** | **+11** | 1.95 |
| CL_close_r2 | 0.4521 | 230 | +3 | 1.79 |
| CL_close_r3 | 0.4416 | 219 | −8 | 1.83 |
| H_hyst_0.7 | 0.4294 | 218 | −9 | 2.18 |
| H_hyst_0.5 | 0.4134 | 216 | −11 | 2.39 |
| HP / HPC (combinations) | 0.42–0.43 | 211–218 | −9…−16 | 2.2 |
| *ref_CE4 (ORACLE, not achievable)* | *0.7199* | *317* | *+90* | *2.05* |
| *ref_A (ORACLE)* | *0.8833* | *373* | *+146* | *1.51* |

- [x] **Hysteresis thresholding is a real but small win: +11 TP.** Separating the seed decision from
      the grow decision is the right idea — a component is kept only if it contains strong evidence,
      but may then extend through weak evidence. Verified in isolation: a filament with a faint
      stretch splits into 2 components under a flat threshold and stays 1 under hysteresis, while an
      isolated weak blob with no strong seed is correctly not resurrected.
- [x] 🔴 **And it only works at a HIGH grow threshold.** At `lo` 0.7 and 0.5 the TP count *falls*
      (−9, −11) while `fg` rises 1.95 → 2.18 → 2.39 %. **Direct confirmation of the bridging
      diagnosis:** added weak pixels create more spurious fusions than they fill holes.
- [ ] 🔴 **`PR_prune` is UNTESTED, not negative — a defect in my own screen.** It returned exactly
      identical TP, FP and fg. Cause: the mask is `prob > 0.97`, so every pixel in it has p > 0.97
      and the mean probability along any branch can never fall below a 0.3 or 0.5 threshold. **The
      criterion could not fire.** This is the lever aimed at the bridging term (+51), so it is the
      one worth redoing: score branches against the probability map at a LOWER level, or by shape
      (elongation, width consistency) rather than by a threshold the mask already guarantees.
- [ ] **Anything winning here needs its own instancer tuning before it is compared.** Every variant
      changes the mask statistics while `params_a_model_v5H` stays fitted to the current one, so
      +11 TP is a LOWER bound (cf. `merge_radius` 9.0 → 3.0 moving nnU-Net by +0.018).
- [x] **Verdict on lever (b): the +93 budget is largely NOT reachable by post-processing.** Simple
      morphological and probabilistic operations recovered 12 % of it. The information that separates
      "excess" from "real filament" is not present in the probability map at these thresholds — which
      is an argument for fixing the foreground upstream, not for more post-hoc repair.

### 0.0-OVERLAP · Where the system stands on overlapping microtubules (measured 2026-08-26)

User asked whether the model can segment overlapping microtubules. It largely already does — at the
INSTANCE level, geometrically, without amodal channels — but the two halves of "overlap" are in very
different states. v5H, MT-34 VAL, `scripts/crossing_report.py`:

| mask | task | crossings | **junction identity** | shallow <30° | oblique | steep >60° |
|---|---|---|---|---|---|---|
| model | 585 | 7 | 0.714 | 0.750 | 0.500 | 1.000 |
| **model** | **586 dense** | **302** | **0.752** | 0.656 | 0.744 | 0.808 |
| oracle | 585 | 7 | 1.000 | 1.000 | 1.000 | 1.000 |
| **oracle** | **586 dense** | **302** | **0.917** | 0.885 | 0.909 | 0.942 |

| capability | model | oracle | headroom |
|---|---|---|---|
| crossings (junction identity) | **0.752** | 0.917 | 0.165 |
| close parallels (bundle recovery) | **0.232** | 0.717 | **0.485** |

- [x] **Crossings are largely solved, and solved GEOMETRICALLY.** Three of four crossings keep their
      identity with the model's own binary mask (PySOAX: 0.000 on non-shallow crossings). v5H has
      K=1, so this comes from the junction matching under the hard kappa_max, not from amodal
      channels — the earlier measurement that `w_ori` contributes exactly 0.000 on predicted masks
      is consistent with that. Identity degrades toward shallow crossings (0.656 vs 0.808), which is
      the geometrically hardest case.
- [ ] 🔴 **Close parallels are NOT solved, and the headroom is 3x larger.** Only 23 % of pairs
      running 2–6 px apart come back as two instances, against 72 % with an oracle mask.
      **This is a FOREGROUND problem, not an instancer one:** once the mask fuses two filaments into
      one band the gap information is physically absent and no amount of geometric pairing recovers
      it. Levers, cheapest first: (1) soft two-bin orientation assignment + a multi-label crossing
      term — returns amodality to the network without the bin-boundary fragmentation that hurt v4b's
      hard assignment; (2) higher foreground resolution — two filaments 3 px apart at
      `mask_half_width=1.0` are at the edge of what the representation can express; (3) a loss term
      that explicitly penalises fusing parallels, which BCE+Dice on the union does not.
- [ ] **Instance-level amodality already works and should be stated as such in the paper:** each
      predicted polyline is rasterised separately, so two instances may share pixels. What v5H does
      NOT have is amodality inside the network (v4b's K=6 channels).
- [x] 🔴 **Three integration bugs in `crossing_report.py`, one of them the dangerous kind.**
      Missing `pairs` argument; unscaled tolerances (`count_crossings` and `arm_offset` must carry
      the x1.5 upscale as `run_oracle_eval.py` does); and reading `ji.get("n_used", 0)` from a
      contract whose keys are `n_crossings` / `n_preserved` / `rate` / `by_angle` / `by_angle_n`.
      **The third turned a wrong key into a plausible ZERO instead of a crash**, and the report
      confidently stated there were no crossings in a benchmark built to be crossing-dense. Fixed by
      indexing the contract rather than `.get()`-ing it with defaults.

### 0.0-PARALLEL 🔴 The training TARGET is spending the gap between close parallels (2026-08-26)

Training-free screen (`scripts/parallel_screen.py`): rasterise the ground truth at a range of
half-widths, instance it, and read the result. This is an upper bound on what a perfectly trained
foreground of that width could deliver. Real MT-34 VAL, 16 frames, 50 close-parallel pairs, all rows
on the SAME frames:

| target half_width | fg % | **bundle** | **junction** | frag | **F1** |
|---|---|---|---|---|---|
| **<= 0.75** (quantises to a 1 px line) | 0.59 | **0.950** | **0.964** | **1.022** | **0.9061** |
| 1.00 — what the trainer uses today | 1.61 | 0.820 | 0.919 | 1.034 | 0.8760 |
| 1.50 | 1.85 | 0.859 | 0.896 | 1.047 | 0.8511 |

- [x] 🔴 **A thin target dominates on EVERY metric — this is not a trade-off.** bundle +0.130,
      junction +0.045, fragmentation better, F1 +0.030, at a third of the foreground.
- [x] 🔴 **Mechanism, and why it is irreversible.** The instancer skeletonises the mask anyway, so
      thickness looks irrelevant — but skeletonising a **fused band** of two parallels yields ONE
      centerline while two separate thin bands yield TWO. The information is destroyed at the mask
      level, before the instancer ever sees it, so no downstream geometry can recover it.
- [x] 🔴 **This corrects 0.0-G1.** "0.90 is above the oracle-mask ceiling of 0.876" held only for
      `half_width=1.0`. With a thin target the oracle ceiling is **0.9061** — the target is just
      under the ceiling, not above it. **The ceiling was a property of the training target, not of
      the task.**
- [x] **And it explains the historical choice.** `mask_half_width=1.0` was recorded as "best
      localization" — chosen on a SEMANTIC metric over single filaments, where a thicker target
      helps. On the INSTANCE metric with close parallels it is the opposite. Same shape of error as
      the threshold-calibration confound: the metric decisions were made on is not the metric the
      result is reported on.
- [x] **Correction to my own earlier advice.** I recommended soft two-bin ORIENTATION assignment as
      the parallel lever. That is wrong: two parallel filaments share an orientation, so they land in
      the SAME bin and stay fused. Orientation binning separates crossings, not parallels — which is
      what `PAPER_PLAN` said all along (orientation channels -> crossings, mutex repulsion ->
      parallels).
- [ ] 🔴 **OWED: the training A/B.** `scripts/train_v5.py` now takes `--target-half-width`
      (default 1.0, so unflagged runs are bit-identical). Two arms, identical but for that flag,
      same config as run H (`--arch nnunet --head binary --epochs 40 --epoch-len 6000 --batch 8`).
      **Not launched: both GPUs are running the user's own active-segmenter jobs.** Cost estimate
      ~1.5 h per arm on a free A100.
- [ ] **Risks to check in that A/B, not assume away:** a 1 px target has a third of the positive
      pixels, so class imbalance worsens (`pos_weight` is untuned at 8.0) and a thin prediction is
      more fragile to a one-pixel localisation error — the oracle screen cannot see either. Also
      re-tune the instancer for the new mask statistics before quoting any comparison.

### 0.0-THIN · Thin target: **+0.071 at a shared threshold, +0.0095 swept — i.e. a NULL** (2026-08-26)

> [!warning] The first version of this entry claimed a +0.071 win. That number came from a
> SINGLE-THRESHOLD comparison and is wrong. Swept per model, the difference is +0.0095 — inside the
> benchmark's ±0.05 resolution. Corrected the same session, before it reached anything downstream.

**Swept evaluation, full 17-frame real VAL, each arm at its OWN optimum:**

| threshold | T10 (thick target) | T05 (thin target) |
|---|---|---|
| 0.50 | 0.4066 | **0.4783** ← T05 optimum |
| 0.85 | 0.4504 | 0.4375 |
| 0.97 | 0.4604 | 0.2087 |
| 0.99 | **0.4688** ← T10 optimum | 0.1049 |

**0.4783 vs 0.4688 = +0.0095**, TP 249 vs 246. Not a meaningful win.

- [x] 🔴 **I walked straight into this project's own documented trap.** [[threshold-calibration-confound]]
      says *"sweep prob_thr PER MODEL before comparing; a shared threshold faked a 0.47 F1 gap."*
      The training-time validation scores both arms at one threshold, which is near-optimal for T10
      and catastrophic for T05 (0.2087 at 0.97) — hence the phantom +0.071. **The optimum MOVED from
      0.99 to 0.50:** a thin target makes the network less confident, so high thresholds destroy it.
      Any future arm must be swept before it is compared, including against itself at an earlier epoch.
- [x] 🔴 **The oracle screen's promise did NOT transfer.** Oracle masks predicted +0.030 F1 and
      bundle 0.820 -> 0.950 for a thin target; the trained model delivers +0.0095. A perfectly thin
      GT mask instances well, but a NETWORK trained toward one produces a blurrier, less confident
      map whose thresholding does not recover the crisp 1 px line. **Oracle-mask screens bound what
      is possible, not what is trainable** — worth remembering before the next one is used to
      justify a run.
- [x] 🔴 **Bundle recovery DOES improve — and the gain is cancelled by fragmentation.** Both arms
      at their own optimum (T10 thr 0.99, T05 thr 0.50):

      | source | metric | T10 thick | T05 thin | Δ |
      |---|---|---|---|---|
      | 585 | junction identity | 0.571 | **0.857** | +0.286 |
      | 585 | **bundle recovery** | 0.500 | **0.750** | **+0.250** |
      | 585 | fragmentation | 1.075 | 1.049 | better |
      | 586 dense | junction identity | 0.699 | **0.735** | +0.036 |
      | 586 dense | **bundle recovery** | 0.286 | **0.340** | **+0.054** |
      | 586 dense | fragmentation | 1.211 | **1.253** | **worse** |

      **The intervention works on the metric it was built for**, and on crossings too — but the thin
      mask also breaks more easily, and in F1 the two cancel. This is the §17p pattern exactly:
      two real, opposed effects summing to a null.
- [ ] **Size caveat, stated rather than buried:** the dense half carries ~50 parallel pairs in
      total, so 0.286 -> 0.340 is **two or three pairs**. Not separable from noise at this n. The
      direction agrees with the Alice half (+0.250) and with the oracle screen, but agreement is
      not significance.
- [ ] 🔴 **The blocker is now FRAGMENTATION, not separation.** That reframes the next lever: keep
      the thin target (it demonstrably separates better) and attack what breaks it. Candidates that
      have NOT been tried on a thin target: connectivity-aware loss (soft-clDice at low weight —
      the earlier negative was measured on a THICK target, so it does not transfer), and re-tuning
      the instancer's gap-linking for a thin mask's statistics, which are wholly different
      (`fg` 0.82 % vs 1.54 %, optimum threshold 0.50 vs 0.99).
- [ ] **Second silent-failure of the day, same shape as the first.** The bundle measurement first
      returned an EMPTY result because the wrapper piped python through
      `grep -E "source|model|shallow"` — a traceback matches none of those, so a crash rendered as
      "no output" under `set -eu` with exit code 0. Earlier the same class of error was
      `ji.get("n_used", 0)` returning a plausible zero. **Never filter a command's stderr through a
      whitelist grep**; keep failures loud.

**Original A/B record (training-time validation, 12 frames, one threshold — kept because the
pos_weight arm is still informative):**

| arm | target half_width | pos_weight | VAL F1 (single thr) | fg % | cc/gt |
|---|---|---|---|---|---|
| T10 — control | 1.0 | 8 | 0.4249 | 2.03 | 10.00 |
| T05 — thin | 0.5 | 8 | 0.4963 | 0.74 | 11.09 |
| T05W — thin + reweighted | 0.5 | 24 | 0.4230 | 1.05 | 13.92 |

- [x] **The class-imbalance risk was measured, and the answer is "do nothing".** Raising
      `pos_weight` 8 -> 24 for the thinner target HURT even at the shared threshold, and pushed
      `cc/gt` to 13.92. Keep 8.
- [x] **`cc/gt` rises 10.00 -> 11.09 with the thin target** — it IS more fragmented, the risk the
      oracle screen could not see.

Training-time validation, 12 real VAL frames, single threshold, final epoch. Control and treatment
share trainer, generator calibration, budget, augmentation, optimiser and seed:

| arm | target half_width | pos_weight | VAL F1 | TP | FP | FN | fg % | cc/gt |
|---|---|---|---|---|---|---|---|---|
| T10 — control (= run H recipe) | 1.0 | 8 | 0.4249 | 123 | 221 | 136 | 2.03 | 10.00 |
| **T05 — thin target** | **0.5** | 8 | **0.4963** | **142** | **195** | **117** | 0.74 | 11.09 |
| T05W — thin + reweighted | 0.5 | 24 | 0.4230 | 122 | 238 | 137 | 1.05 | 13.92 |

- [x] 🔴 **THE FIRST A/B MEASURED NOTHING, AND THE GUARD IS WHY.** `ori_channels_soft` accepted
      `half_width` and never read it — the target's width came entirely from a hard-coded 3x3
      dilation. Two arms therefore compared a configuration with itself for 3 hours on two GPUs.
      My guard had asserted that the STRING `half_width=self.target_half_width` appeared in the
      source: true, and worthless. **A guard must test the EFFECT, not the plumbing.** The
      replacement builds the target both ways on the same instances and compares foreground
      fractions (4.716 % vs 1.632 %, ratio 2.89x, `/tmp/test_hw_effect.py`).
      Two things were salvaged: the fix touches only the `< 1.0` branch, so the control arm stayed
      valid and did not need re-running; and two independent runs on different GPUs came out
      **bit-identical**, so with a fixed seed run-to-run variance is exactly zero and any A/B delta
      is attributable to the intervention rather than to noise.
- [x] **Portability defects fixed on the way** (all the same shape — hard-coded host-local paths):
      `train_v5.py` now takes `--bg-dir`, because `/home/prusek/BIOCEV/datasets` is a symlink from
      shared NFS home to a tulen-LOCAL disk and dangles everywhere else; and the validation report
      no longer dies with `KeyError: 'total_tp'` when a model predicts nothing, which would have
      killed a 3 h run at its first checkpoint.

### 0.0-BASELINE 🔴 The number to beat: **0.3847** (measured 2026-08-25)

v5H on the combined MT-34 + HTW **VAL** (50 frames), each threshold swept, best at `prob_thr=0.99`:

| source | n | TP | FP | FN | micro-F1 |
|---|---|---|---|---|---|
| 585 Alice | 6 | 83 | 27 | 21 | **0.776** |
| 586 crossing-dense | 11 | 157 | 260 | 202 | 0.405 |
| htw | 33 | 237 | 230 | **561** | 0.375 |
| **pooled mean-F1** | 50 | 477 | 517 | 784 | **0.3847** |

- [ ] 🔴 **0.0-B1 · The combined benchmark holds TWO different problems, not one harder one.**
      On **htw the dominant error is FN — 561 of 798 GT never found (70 %)**, i.e. *detection*, which
      matches its "semantic-limited" diagnosis (foreground tol2 0.620, ~38 % of centerline uncovered).
      On **586-dense FP ~ FN** (260 vs 202), i.e. *assembly*. **Reaching 0.90 needs recall on HTW and
      assembly on MT-34, and those are different levers** — a model tuned for one will not rescue the
      other. Report both halves on every future number; a pooled figure hides which problem moved.
- [x] **0.0-B2 · Baseline is VAL, not TEST.** The combined TEST half (50 frames) stays unscored, so a
      final number still exists. Every development decision is made on VAL.

### 0.0 What the paper claims, after the venue decision

**Headline (N2 + annotation-free end to end):** microtubule *instance* segmentation in the dense-crossing
regime, trained and tuned with **zero human annotations** — the semantic network on synthetic frames only,
and the instancer's 17 hyperparameters on synthetic ground truth (§17q: TEST 0.416 → **0.457**,
+0.041 [+0.018, +0.065], p<0.001 — synthetic tuning *beats* real-VAL tuning, because incomplete human GT
rewards permissive settings).

**N1 (calibrate-what-you-segment) is DEMOTED to an ablation.** It is not causally supported and three
attempts to support it failed: W2 (region/residual-MMD did not beat global-cosine downstream), C2
(tightening the coupling *hurt*, 0.627 → 0.559), DR (widening priors hurt held-out HTW monotonically).
**C1.5 / W3 identifiability was RUN and came out NEGATIVE** — `PAPER_PLAN.md` still lists it as "unrun,
cheap, decisive" and that entry is stale. Do not spend on N1; report it honestly as a design choice with
its negatives, which is a genuine rigor asset.

- [x] 🔴 **0.0a Stale claims in `PAPER_PLAN.md` FIXED (2026-08-25).** Venue note → NC decided + pointer to
      §0 as operative; N1 → ablation with the spine restated; **C1.5/W3 marked RUN and NEGATIVE and closed**
      (it had stood as "unrun, cheap, decisive" for weeks while the result was already in). Also recorded:
      C2ST saturates — synth and every real lab separate at ~0.98 in DINOv2, so that test says little.

### 0.1 🔴 NC-A · The capability delta — and it is a REGIME, not a pooled number

The strongest available NC claim is **not** "higher F1". It is that the methods a biologist would reach for
**collapse where microtubules cross**, and this one does not:

| crossing-dense half of MT-34 | pooled F1 | junction identity |
|---|---|---|
| ours (v4b / H) | 0.273 | 0.511 (model mask) · **0.965** (oracle) |
| SAM 3 "thin line", tuned | 0.245 | not measured |
| TARDIS zero-shot (real-trained) | **0.000** (0 TP of 359) | not measured |
| PySOAX, equal Optuna budget | — | **0.000** on all 288 non-shallow crossings (oracle) |

The F1 column is *within benchmark noise* (n=11 on that half). The identity column is a factor-of-∞
separation — and it has **never been measured on model masks for the baselines**. That single experiment
is the highest-value item on this list.

- [ ] 🔴 **NC-A1 · Junction-identity + bundle-recovery head-to-head on MODEL masks**, all four systems
      (ours / SAM3 / TARDIS / PySOAX), crossing-dense half, paired bootstrap CIs, stratified by source task.
      *Acceptance:* a separation whose paired interval excludes zero. *Cost:* no training — all four
      predictors already run through `scripts/baseline_mt34.py`; needs the metric applied, not new models.
      **If this lands, it is the paper's Figure 2 and the pooled-F1 parity stops being the story.**
- [ ] 🔴 **NC-A2 · Pooled F1 toward 0.60–0.75 on real TEST** (`F1_PUSH_PLAN` §4 calls this band "large and
      defensible"; the factorial licenses 0.640 from excess removal alone, 0.861 from near-GT shape repair).
      Ordered by measured expected value — **do not run c/d before NC-B1**, they are 0.03-size effects and
      the current benchmark cannot resolve them:
  - [ ] (a) **Instance-level temporal fusion** (F1 lever #1) — no training at all. `rec2` is **0.971**
        while F1 is 0.457: the centerline length is found and lost in *assembly*. A microtubule is followed
        by 2.7–3.5 tracks (oracle 1.59); merging co-moving fragments converts fragmentation into F1
        directly. Video only. **Biggest measured prize on the list.**
        *Re-measured on v5H (2026-08-26, protocol §22): the prize is unchanged and now sized on the
        DEPLOYED model — 2.86–3.18 tracks per **detected** microtubule against an oracle 1.74–2.05, with
        the identity rate at 0.014 saying the tracker already knows which fragments belong together.
        `deploy_v5H/track.py` produces the tracks this step would consume.*
  - [ ] (b) **Cut false positives WITHOUT adding fragmentation.** §17p moved both and they cancelled
        (FP 404→327 p<0.001, fragmentation 1.182→1.293 p=0.006). Candidates: soft-clDice at weight 0.1
        under the NEW recipe (the 0.5 collapse was the old one); field-stop augmentation at training time,
        since the measured FP win came from the rim, not from filaments.
  - [ ] (c) **Fair architecture comparison** — equal Optuna budget per arm, **per-arm threshold AND per-arm
        instancer params** (measured: `merge_radius` 9.0→3.0 moves nnU-Net 0.501→0.519; every architecture
        comparison so far ran through a stack fitted to v4b). Includes the nnU-Net K=6 retrain and the
        binary union head (+0.022, cheap, untested on v4b's trunk).
  - [ ] (d) **Segmenter HPO — never done, at all.** `pos_weight=8` is inherited and every new run over-fires
        (fg 2.5–5 % against a 1.9 % target). Minimal sweep: `pos_weight` {2,4,8,16} × LR, equal budget.
- [ ] **NOT to be run again:** any intervention that repairs short breaks. Bounded above at +15 TP on real
      and +0.07 F1 on synth, and six consecutive nulls have already paid for that lesson.

### 0.2 🔴 NC-B · Benchmark power — the gate on everything above

```
measured training noise (3 seeds)   SD 0.0040
paired bootstrap at n=16            ±0.05        ← 12× larger
spread of the 4 leading systems     0.466–0.505  ← mutually indistinguishable
```

**The limiting uncertainty is the benchmark, not the model.** Every further GPU-hour spent on the method
has an expected measurable return of ~0 until n grows. This is the reason NC-A2(c) and (d) are gated.

- [ ] 🔴 **NC-B1 · Expand the benchmark: review CVAT task 586's 281 pre-annotated frames.** Annotations are
      pre-drawn, so this is **correction, not drawing**. *Acceptance:* n ≥ 100 eval frames with a declared
      VAL/TEST split, and the recomputed paired-bootstrap width on a known pair (H − v4b) below 0.03.
      *Cost:* human time, no compute. **Nothing else on this list can resolve a 0.03 effect.**
      ⚠️ **The bias this deepens, and how to keep it measurable:** those 281 frames are untouched
      v7+PySOAX **model output** (verified 2026-08-16: only 538 of 18 059 polylines, 3 %, are
      `source="manual"`, all in frames 0–21), and even inside MT-34's task-586 block **~26 % of polylines
      are unreviewed model output**. Correcting model output scales the benchmark but scales its
      agreement bias with it — a PySOAX-family instancer is graded against its own ancestor. So NC-B1
      is only valid **paired with NC-B2(b)'s from-scratch subset**: the corrected frames buy statistical
      power, the independent ones bound the bias. Report the two blocks separately, always.
      [[benchmark-size-mt34]]
- [ ] 🔴 **NC-B2 · What does 0.457 even mean? Two cheap measurements, both unrun:**
  - [ ] (a) **How many of the 326 false positives are real microtubules?** MT-34 GT is human-corrected
        model output and is demonstrably incomplete on sparse frames; unannotated detections already
        measured at **2.06× the residual contrast of background** (§17j machinery does this). *Acceptance:*
        a stated fraction plus a corrected upper bound on attainable F1. **If that bound is well below 1.0,
        0.457 reads completely differently and it changes what is worth optimising.** No training.
  - [ ] (b) **Inter-annotator agreement.** SynthMT anchored itself to human 0.81; there is **no human–human
        strict centerline-F1 on MT-34 at all**. *Acceptance:* ≥2 annotators × ~10 frames drawn **from
        scratch** (never corrected from model output), reported with a CI, and that number becomes the
        ceiling row in every results table. Without it no reviewer — and no author — can judge 0.457.

### 0.3 🔴 NC-C · Generality — DECISION OWED FROM THE USER

NC wants the method to work beyond one lab and one modality. Two candidates, both expensive, and the
choice changes what the paper claims:

- **(i) Cross-lab.** HTW (66 frames) is **SEALED and one-shot** — spend it at paper-freeze as a held-out
  block, never during development. Caveat that must be stated: the DR coverage check found HTW is the
  **closest** lab to synth, so on its own it is weak far-domain evidence. The *"better real eval dataset
  is coming"* (user) sits on this critical path.
- **(ii) TIRF, second modality.** `render_tirf` already exists and morphology is shared, so the work is
  bounded: acquire real dark TIRF backgrounds + frames, **freeze morphology, calibrate appearance only**,
  show transfer. This is the stronger NC claim — *one morphology model, swap the render* — and it is the
  cleaner demonstration of the annotation-free principle.

- [x] 🔴 **NC-C0 · DECIDED 2026-08-25 (user): BOTH.** TIRF is the primary generality axis; HTW is added at
      paper-freeze as a held-out block. Consequences, in order:
  - [ ] 🔴 **NC-C1 · Confirm real TIRF data with the lab** — dark TIRF backgrounds + annotated frames.
        **This is the gating dependency and nothing else in NC-C can start without it.** Ask early.
  - [ ] 🔴 **NC-C2 · TIRF appearance calibration only — morphology FROZEN.** That constraint *is* the
        claim ("one morphology model, swap the render"); letting morphology move would dissolve it.
        Re-use the IRM calibration machinery with `CALIB_OBJECTIVE` unchanged.
  - [ ] 🔴 **NC-C3 · TIRF end-to-end eval** through the identical pipeline and the identical strict metric,
        with its own threshold and its own instancer params (per §"instancer params disadvantage every
        non-v4b foreground" — a new modality needs its own tuning run exactly as a new architecture does).
  - [ ] **NC-C4 · HTW spent ONCE, at freeze, as a held-out block used in NO lever decision** (§0 NC-G4).
        **State its caveat in the paper rather than letting a reviewer find it:** the DR coverage check
        measured HTW as the *closest* lab to synth, so it is near-domain evidence of transfer, not
        far-domain. Reported that way it is honest and still worth having; reported as "cross-lab
        generalisation" it over-claims.
  - [ ] **Budget note:** this is the most expensive branch on the list and it now has two data
        dependencies (TIRF acquisition, HTW seal). Neither blocks NC-A, NC-B or NC-E — keep those moving.

### 0.4 NC-D · The biological readout (what separates NC from a methods journal)

- [ ] **NC-D1 · One figure in which the tool measures a biological quantity that could not be obtained by
      hand at this scale**, with its uncertainty. The tracker (§18/§19) is the material that exists.
      Candidates: per-MT growth/shrinkage dynamics, bundle statistics over time, crossing-angle
      distributions across conditions. *Blocked on:* which experimental conditions the corpus actually
      spans — **ask the lab.**

### 0.5 🔴 NC-E · Integrity blockers — none of these may survive into a submission

- [ ] 🔴 **NC-E1 · MT-34 leakage carve-out (W9-followup).** All 22 new frames are in the generator's
      calibration corpus (`training_img_114.tif` = `mt_corpus/images/0015_local_irm_tif_img_114.tif`).
      Not label leakage, but a reviewer who finds it stops reading. Carve out, recalibrate, retrain,
      re-derive the headline numbers and **report the delta** — exactly as was already done for HTW.
      Numbers may drop; that is a finding, not a failure.
- [ ] 🔴 **NC-E2 · Agreement bias (W9-followup).** MT-34 GT is human-corrected v7+PySOAX output, which
      flatters PySOAX-family instancers — and ours is one. Report error rates split by `manual` vs `file`
      polylines. Cheap; the split already exists in the h5 attrs.
- [ ] 🔴 **NC-E3 · Reproducibility (§17n).** `tune_instancer.py` overwrote a canonical params file, so the
      **v1→v2 deltas (0.893→0.920, 0.379→0.416) are NOT reproducible** and must appear as development
      history, never as measured improvements. Going forward: every paper number = named script + named
      write-once params file + commit hash, and every table generated by script, never by hand.
- [ ] 🔴 **NC-E4 · Oracle params are still real-VAL-tuned** (`params_a_v2.json`). The 0.920 oracle number
      must not stand next to an annotation-free claim until it has had the same synth-tuning treatment
      the shipped config got in §17q.
- [ ] 🔴 **NC-E5 · TEST multiplicity.** Four exposures so far, each pre-declared and each carrying its
      interval. Disclose the count in the paper and **pre-register the final TEST run** before it happens.

### 0.6 🔴 NC-F · Baselines — the part reviewers attack

- [ ] 🔴 **NC-F1 · There is NO paired test against SAM 3.** The recorded "we lead 0.458 to 0.432 on MT-34"
      is a bare point estimate at n=16, on a benchmark where four systems spanning 0.466–0.505 are
      mutually indistinguishable. **Never write "we beat SAM3."** Owed: paired bootstrap, plus SAM3's
      confidence tuned on a proper VAL split (the Alice-tuned 0.60 is optimistic). The honest and stronger
      claim is **annotation-free parity with a 3.3 GB real-trained foundation model**, which does not
      depend on an interval.
- [ ] **NC-F2 · TARDIS and ORION through the identical pipeline**, each swept to its own operating point.
- [ ] **NC-F3 · Reproduce every baseline's published anchor before trusting any number from it** — on the
      dataset its own paper reports. Without the anchor, "baseline is weak here" and "we broke the
      baseline" are indistinguishable, and a reviewer assumes the latter. Applies to SynthMT, TARDIS, SAM3.
- [ ] **NC-F4 · Report every low baseline WITH its structural reason.** TARDIS's 0.000 on the dense half is
      only credible when paired with the mechanism (degree-2 graph cut collapses 2D X-crossings) — bare,
      it reads as a baseline we broke.

### 0.7 🔴 NC-G · Statistics

- [ ] 🔴 **NC-G1 · Unit of analysis is the frame.** Average over seeds per frame, *then* run the paired
      test. Testing (seed, frame) pairs as independent is pseudoreplication and inflates every p-value.
- [ ] 🔴 **NC-G2 · Holm–Bonferroni over the whole family of comparisons in the paper**, reporting the
      adjusted column; the raw one is descriptive only. Count *every* attempted comparison — quietly
      dropping an uncomputable one shrinks the family and flatters the survivors.
- [ ] **NC-G3 · Seeds:** ≥3 everywhere (SD 0.0040 measured), ≥6 for instance metrics.
- [ ] **NC-G4 · A held-out block used in NO lever decision** (HTW, and TIRF if NC-C0 goes that way). It is
      the strongest available answer to the panel-fitting objection and costs no extra compute.

### 0.8 Ordering — what to do first, and why

1. **NC-B1** (benchmark expansion) — starts now, runs in parallel with everything, gates NC-A2(c)/(d).
2. **NC-A1** + **NC-B2** — no training, high information, they decide what the paper's headline is.
3. **NC-E1** (leakage) — long pole: recalibration + retrain, and every frozen number depends on it.
4. **NC-A2(a)** (temporal fusion) — biggest measured prize that needs no training.
5. **NC-C0** — user decision, then acquire data; the long pole for generality.
6. Everything else, once the benchmark can actually measure it.

---
## 1 · DATASET — synthetic generator

**Done / in place:** morphology sampler ↔ render split (`render_irm`/`render_tirf`, `generate_frame(...,modality=)`);
length-coupled stiffness (Pampaloni) + new morph params (curve_base, orient_kappa/nematic, kink_prob, short_prob,
tip taper, along-intensity); region_mmd + |residual| polarity-invariant calibration; inversion→training
augmentation (generator always-dark); real backgrounds composited; calibrated (region_mmd MMD 0.0174→0.0069).
PHYSICS render (two-beam interference) + multi-regime morphology (static/gliding/dynamic) + 2 peer-review rounds (§15).

- [x] 🔴 **W1-DR — DOMAIN RANDOMIZATION ablation (task #17) DONE → v4b stays primary, DR = evidence.** Built
      `scripts/dr_cfg.build_cfg_dr` (env DR=1; nuisances widened beyond calib + per-regime floors), trained v7-DR + ran a
      DINOv2 coverage check. **Widening MONOTONICALLY HURT held-out HTW** (sem 0.620→0.504→0.422; inst 0.234→0.199→0.169;
      Alice held 0.940→0.920, inst 0.653→0.636). Coverage check: HTW is the CLOSEST lab to synth (near-domain) → tight
      calibration wins; **benchmark has no truly-far experiment → sees only DR's COST, not benefit.** USER DECISION: keep
      v4b, retain DR machinery+ablation+coverage as HONEST paper evidence of the calibration↔randomization tradeoff; DEFER
      DR verdict to a real OOD test set. **Do NOT re-tune DR width vs this biased benchmark.** protocol §15, wf w2ez5ernh.
- [~] **DR validation (task #18):** DINOv2 feature-COVERAGE check DONE (`dr_coverage2.py`, residual space: DR raises synth
      diversity 0.087→0.103≈real 0.108, but centroid gap ~0.51 persists, HTW closest). **Leave-one-experiment-out worst-group
      F1 needs a genuinely-far annotated experiment — BLOCKED on the incoming better real eval dataset.**

- [ ] 🟣 **W1 (was 🔴 under the NM legend) — calibration target ≠ eval distribution.** N1-family, so not
      NC-blocking; the part that IS NC-blocking is the leakage carve-out (§0 NC-E1), which forces a
      recalibration anyway and will re-open this measurement for free. Original entry: The wavier "improved" generator HURT Alice instance-F1
      (0.62→0.41) ⇒ the 320-frame corpus may not represent the eval set. Characterize corpus provenance/diversity;
      test an Alice-like (straighter) calibration target; quantify corpus↔Alice shift. Don't over-tune to corpus artifacts.
- [x] 🟣 **W2 — no CAUSAL proof calibration improves TRANSFER. CLOSED NEGATIVE by the §0.0 venue decision.**
      C1: region/residual-MMD did NOT beat global-cosine on semantic tol2 (only over-firing); the proxy gap
      (MMD ≠ transfer) was never closed. **N1 is now an ablation, so this is no longer NC-blocking** — it is
      reported honestly as a negative. Re-opening it belongs to a later NM attempt, not to this paper.
- [x] 🟣 **W3 — identifiability / variance-recovery (C1.5): RUN, NEGATIVE, CLOSED (see §0.0a).** The design was
      to show region_mmd has a well at the true std while frame-pooled residual_mmd is flat. **It did not
      reproduce on test (2026-07-11)**, and C2ST additionally saturated — synth and every real lab all separate
      at ~0.98 in DINOv2. This item read "currently asserted, never demonstrated" for weeks *after* the result
      was already in; that staleness is exactly what §0.0a was raised to fix. **Do not re-attempt without new evidence.**
- [ ] **Param review + add relevant knobs** (user directive): review ALL GenConfig params (each Normal(mean,std));
      add missing physical ones — waviness freq/amp (review), MT bending rate/stiffness, MT whitening/bleaching
      frequency (detach params; TIRF photobleaching), property VARIANCES (now maybe identifiable via region_mmd),
      curvature-dependent breaks. Decide calibratable vs fixed prior; document. Recalibrate with >300 (=400) trials.
- [ ] 🔴 **W4 — TIRF quantitatively + cross-lab. → OWNED BY §0 NC-C0(ii), and it is my recommended generality axis.**
      TIRF uncalibrated (no real corpus); all data single-source IRM. Get real DARK TIRF backgrounds + frames,
      **freeze morphology, calibrate ONLY TIRF appearance**, show transfer. Blocked on data availability — ask the lab.
- [x] 🟣 **W5 — physics-based appearance: DONE (§15 built the two-beam interference renderer), and not NC-blocking.**
      The physics render exists and shipped; whether it beats the parametric one is an N1-family question, which
      §0.0 demoted. No further work for this paper.
- [ ] Minor realism: real set has MORE dirt specks than synth (raise `spot_rate`); appearance render width slightly
      thin (2.26 vs real 2.86 px — mask fixed, render not); fit the cross-section profile from bg-subtracted real signal.
- [ ] (Fundamental limit) parametric EXPRESSIVENESS ceiling — calibration finds best point WITHIN the family; SimGAN
      refinement was a WASH; some appearance gap is model-class-bound. Mitigated by real backgrounds.

---
## 2 · SEGMENTATION MODEL — semantic foreground (DINOv2-hybrid)

**Done / in place:** DINOv2-hybrid (frozen ViT-L blocks 5/11/17/23 + high-res branch); orientation-keyed "overpass"
head (K=6, amodal); mask_half_width=1.0 (best localization); raw input (residual HURT); inversion augmentation;
over-firing clean 0.6–0.7×; semantic tol2 **0.947** (val/test 6/6 split). Foreground is essentially solved.

- [~] **W8 — don't over-claim on semantic. → still live; §0.0 settles what the contribution IS.** tol2 0.947 is on the EASY part ("semantic foreground largely solved" —
      own knowledge doc). Frame honestly; the contribution must be the calibration principle or the HARD instance side.
- [x] 🔴 **W11 — novelty framing: DECIDED by §0.0 — do NOT lead with it.** DINOv2-hybrid is standard
      (SegDINO/DPT), and "calibrate-what-you-segment" stayed UNPROVEN (C1 ambiguous; tightening the synergy
      HURT 0.627→0.559; W2 and W3 both closed negative). The dichotomy this item posed — "either prove it or
      don't lead with it" — is resolved on the second branch. **The paper leads with N2 + annotation-free
      end to end (§0.0); N1 appears as an ablation with its negatives.**
- [ ] tol1/Dice plateau (0.81/0.71) is partly segmenter-resolution-bound (2px filaments, DINOv2 /14 + high-res branch);
      low priority — diminishing returns vs a bigger/higher-res backbone.

---
## 3 · INSTANCE SEGMENTATION — the wall

**Scoreboard (synth-only, strict centerline-F1 / bundle-recovery):** heuristic ORION-lite 0.41–0.62 · affinity+MWS
0.49 (2 iterations plateaued) · per-layer orientation 0.11 (wavy MTs fragment across bins) · pretrained-DIST hybrid
0.33 (over-segments, doesn't fit our pc stats) · **fine-tuned DIST = running (pipeline13)**. Oracle affinity single-layer
= 0.86 (mechanism sound; learned affinity quality is the gap). Pattern CONFIRMS knowledge-doc: synth-only instance caps out.

- [x] 🔴 **W7 — instance step: BROKEN OPEN on the instancer side (TEST, 2026-08-11).** A reaches
      **0.893 pooled on MT-34 TEST with an oracle foreground** (PySOAX tuned to the same budget: 0.590) and
      **preserves junction identity 0.916** where PySOAX scores **0.000 on all 288 non-shallow crossings**.
      Confirmed on synthetic data with EXACT GT (A 0.695 vs PySOAX 0.397, same angle profile). protocol §17j.
- [x] **v2 instancer levers DONE (protocol §17k).** Displacement-aware pairing cost + gap linking
      + (negative) orientation term. **TEST: oracle 0.893 → 0.920, model 0.379 → 0.416, junction
      identity 0.916 → 0.965 and now FLAT across crossing angles, bundle recovery 0.544 → 0.634.**
      The "shallow crossings are mask-width-bound" claim from §17c is SUPERSEDED — a long fused
      stretch makes the gap direction reliable, which is what the displacement term needs.
- [ ] **Orientation term (`w_ori`) = NEGATIVE.** +0.010 oracle VAL, exactly 0.000 on model masks.
      Once displacement is in the cost, geometry already resolves what the amodal channels would.
      Kept (free, off by default) but it is not the hybrid win that was hypothesised.
- [x] **Foreground metric found (§17k) → SHIPPED as `mt_bench.fg_quality` (§17m).** `cc_per_gt` 0.82 /
      `gaps_per_mt` 0.80 / `endp_per_kpx` 0.79 pairwise ranking accuracy over 198 same-frame model pairs;
      `prec2` — half of what the segmenter is tuned on — ranks at **0.58** despite ρ=+0.87. Now an
      instancer-free, per-epoch-cheap module with `select_checkpoint` doing CONSTRAINED minimisation:
      all three improve monotonically under dilation (raw `fg` itself ranks 0.75!), so selection is
      bounded by an over-firing ceiling (fg ≤ 3× in-domain synth) and a collapse floor (rec2 ≥ 0.90).
      13 tests incl. a `(x=col,y=row)` transpose regression. **Gate any foreground retrain on this.**
- [x] 🔴 **Foreground RETRAIN gated on `fg_quality` — DONE, NEGATIVE (§17p).** 30 epochs, v4b recipe,
      selection on real VAL. The gate's own metric improved 25 % (continuity 0.841 → 0.634; `cc_per_gt`
      7.86 → 3.71, `endp_per_kpx` 21.0 → 14.9) and **downstream F1 did not move**: TEST 0.416 → 0.393,
      −0.023 [−0.061, +0.013], p=0.230. Two effects DO survive on the crossing-dense half — false
      positives **404 → 327** (−77 [−116,−43], p<0.001, visibly the field-stop firing gone) and
      fragmentation **significantly WORSE** (1.182 → 1.293, +0.111 [+0.018,+0.239], p=0.006); they cancel.
      Junction identity +0.113 and bundle recovery +0.109 looked like headlines and are NOT separable
      from noise at n=11 — the intervals earned their keep. **Mechanism: the proxy moved without the
      target.** `cc_per_gt` counts mask components; `fragmentation` counts predicted instances — the mask
      got more connected while the instancer still cut it into more pieces. **Scope correction: the
      0.79–0.82 ranking accuracy was measured ACROSS four foregrounds and does NOT transfer to selecting
      checkpoints WITHIN one training run.** Also measured: coverage F1 spans only 0.825–0.873 across the
      15 checkpoints while continuity spans 0.634–1.289 — but both rules picked epoch 28, so the gate's
      marginal value over coverage F1 was untestable here. **DECISION: v4b stays primary.**
- [x] 🔴 **ANNOTATION-FREE END TO END — the instancer's hyperparameters no longer need real GT (§17q).**
      The 17 knobs were fitted on human polylines (MT-34 real VAL); refitted on SYNTHETIC VAL at the
      same 100-trial budget they **beat** the real-tuned ones: TEST pooled **0.416 → 0.457**,
      +0.041 [+0.018,+0.065] p<0.001; crossing-dense half **0.265 → 0.327**, +0.062 [+0.029,+0.095].
      Both error modes improve together (FP 404→326 p<0.001, fragmentation 1.182→1.145 p=0.028) —
      unlike §17p, which traded them. Mechanism: exact GT rewards CONSERVATIVE settings (`w_kappa`
      8.99→16.11, `min_length` 33.8→44.7, `window` 20.2→28.4); human GT is incomplete on sparse frames
      so it rewards permissive ones, i.e. **real-VAL tuning was fitting annotation noise.**
      SHIPPED CONFIG IS NOW `params_a_model_synthtuned.json`.
- [ ] 🔴 **The synthetic SEQUENCE generator does not reproduce v5H's training appearance** (found
      2026-08-26, protocol §22). Single-frame strict F1 on `data/synth_seq_*` is **0.067–0.152** for the
      same weights that score 0.495 on real MT-34 VAL, and threshold is ruled out: a sweep from 0.50 to
      0.99 is flat (F1 0.066–0.078) while the foreground fraction passes straight through its 1.6 % target.
      Cause is structural, not tuning: `build_synth_seq.py` builds its config from a calibration JSON via
      `gen_train.build_cfg`, while v5H was trained through `train_v5.py`'s own dataset path, which carries
      the SNR/contrast correction and the field stop. **Every tracking number is therefore a lower bound**,
      and any future temporal training would be trained on video the segmenter has never seen the like of.
      *Fix:* build sequences through `train_v5.py`'s dataset path (or lift that path into the generator),
      then re-run §22's arms. *Cost:* no GPU for the diagnosis; one prediction pass to re-measure.

- [ ] **Same treatment owed to the ORACLE-mask params** (`params_a_v2.json`, still real-VAL-tuned).
      It is a diagnostic ceiling rather than the shipped system, but the 0.920 oracle number should
      not go in a paper next to an annotation-free claim without it.
- [ ] **TEST multiplicity.** This session scored MT-34 TEST four times (v1/v2 re-instrumentation,
      gated retrain, synth-tuned). Each was declared before it ran and each carries its interval, but
      the accumulated exposure belongs in the write-up.
- [ ] 🔴 **F1 LEVER #1 — temporal fusion at the INSTANCE level. → OWNED BY §0 NC-A2(a)** (designed, measured, NOT built).
      The tracker already knows which fragments belong together: a microtubule is followed by
      **2.7–3.5 tracks** (oracle 1.59). Merging co-moving fragments into one instance per frame
      converts fragmentation directly into F1 and needs **no training at all**. The size of the prize
      comes from the metric: `rec2` is **0.971** while F1 is **0.457** — the model finds 97 % of the
      centerline length and loses it in ASSEMBLY, because strict F1 wants 95 % coverage by ONE object.
      Video only; the single-frame equivalent is lever #3 below.
- [ ] 🔴 **F1 LEVER #3 — how many of the 326 false positives are real microtubules? → OWNED BY §0 NC-B2(a).**
      On new-22 the model produces 326 FP against 193 TP, and MT-34's GT is human-corrected model
      output that is demonstrably incomplete on sparse frames. Measure the residual image contrast
      under each FP (the §17j machinery already does this: unannotated detections carried 2.06× the
      contrast of background). If a large share sit on real filaments, **the measurable F1 ceiling is
      below 1.0** and that changes what is worth optimising. Cheap, no training, and it must be done
      BEFORE any precision-targeted work.
- [ ] **Next lever for the foreground is NOT another gated retrain.** The two things that actually moved
      are opposed, so the target is explicit: cut false positives further WITHOUT adding fragmentation.
      Candidates worth costing: a connectivity-aware loss (soft-clDice weight was 0.1 — the run that
      collapsed at 0.5 was with the OLD recipe), and training-time field-stop augmentation, since the
      measured FP win came from the rim rather than from filaments.
- [~] **Uncertainty on every head-to-head number (§17m).** `instance.metrics.paired_bootstrap` /
      `bootstrap_ci`: frame-resampling CIs, **paired** (one frame multiset scored by both methods, CI on
      the difference — two marginal intervals overlap for almost any pair at n=17) and **stratified by
      source task** (pooled MT-34 is a fixed 6 Alice + 11 new-22 design). `run_oracle_eval.py` now dumps
      per-frame rows under `_frames` so `scripts/bootstrap_report.py` re-derives any interval offline —
      adding a CI to a TEST number must never mean re-running TEST. **DONE for the current system (§17n):**
      oracle TEST A 0.920 [0.870, 0.966] · B 0.813 [0.717, 0.903] · PySOAX 0.590 [0.470, 0.724];
      **A − B = +0.107 [+0.047, +0.182]** and A − PySOAX = +0.330 [+0.219, +0.432], both p < 0.001. A and
      B's marginal intervals OVERLAP — the paired form is what resolves them. On model masks A 0.416 vs
      B 0.308, A − B = +0.108 [+0.063, +0.160]: the instancer ranking does not depend on foreground quality.
- [ ] 🔴 **REPRODUCIBILITY HOLE found by that gate (§17n).** `tune_instancer.py` writes `params_a.json`
      IN PLACE and `tulen_v2_chain.sh` copies it to `params_a_v2.json` only afterwards ⇒ **v2 overwrote v1's
      parameters**; `params_a.json` ≡ `params_a_v2.json` and `params_a_model.json` ≡ `params_a_model_v2.json`,
      locally and on tulen. **The v1→v2 deltas (0.893→0.920, 0.379→0.416) are NOT reproducible** and must be
      reported as development history, not measured improvements. v1's 9 tuned keys recovered from
      `tune_a.log` trial 107 into `params_a_v1_recovered.json` (provenance only — the cost function changed
      too). FIXES OWED: tuning must write a NEW NAMED file, never mutate a canonical one; **and this repo
      needs version control before any paper number is frozen** — *(update 2026-08-25: the repo HAS been under
      git since 2026-08-11; what remains owed is the named-script + named-params + commit-hash discipline,
      now tracked as §0 NC-E3. Commits only on request.)*
- [x] 🔴 **(SUPERSEDED by §3b's first item — kept for the record.) "STILL #1 BOTTLENECK = the SEMANTIC FOREGROUND".** With v4b's
      foreground the instancer drops 0.893 → 0.379 on real, and 0.695 → **0.180 in-domain on synthetic**,
      where there is no domain gap and no annotation error. Measured with the same metric, v4b's semantic
      tol2 is 0.749 on synth vs 0.779 on real — it does not do better on its own training distribution.
      Tested and REFUTED: "the missed microtubules are invisible" (they carry 0.77× the contrast of found
      ones and are the same length). ⇒ next effort goes into the foreground, not the instancer.
- [ ] **nnU-Net's foreground instances BETTER than v4b's** (0.418 vs 0.379 pooled, junction identity 0.720
      vs 0.511) at near-identical semantic F1 ⇒ **semantic coverage F1 does not rank foregrounds by their
      downstream instancing value.** Find a foreground metric that does; it would be a paper contribution.
- [~] (superseded) instance step. NEW DIRECTION (2026-08-10, user): the failure is PySOAX-style greedy minimum-angle
      junction handling; microtubules cannot kink, so bound the curvature. Built `src/instance/`:
      **A** = junction-cluster contraction + window-fitted tangents + per-junction MIN-COST MATCHING under a hard
      kappa_max + priced "leave open"; **B** = curvature-constrained beam tracing in the orientation-lifted
      (x, y, theta) graph (amodal: consuming a path retires only the orientation slices it used).
      **ORACLE MT-34 VAL: PySOAX 0.684 -> A 0.862 pooled; junction-identity 0.045 -> 0.78; PySOAX max curvature
      0.512 rad/px vs the 0.239 maximum over 957 annotated MTs (physically impossible output).**
      OPEN: model-mask gap is large (A pooled 0.338) and is dominated by over-segmentation on thicker/noisier
      predicted foreground + the field stop — being re-tuned on model-mask VAL. protocol §17.
- [ ] **Same-budget baseline parity (W10 lesson applied).** All three instancers get identical Optuna budgets on the
      same oracle VAL objective; PySOAX's shipped params were fitted months ago at 1x on different data. Running.
- [x] Learned affinity + Mutex-Watershed built + tested (0.49; upweight+Dice+8-offsets didn't close the gap).
- [~] TARDIS **DIST**: pretrained hybrid done (0.33); coordinate-level FINE-TUNE on our synth point clouds running
      (research-safe, doesn't over-fire; scripts save_alice_fg / dist_instance_eval --ckpt / gen_dist_data / dist_finetune).
- [ ] DIST degree-2 graph cut collapses 2D X-crossings — may need orientation-keyed multi-point input feeding DIST.
- [ ] Curved MTs fragment across orientation bins; 3+ MTs sharing a pixel exceed the 2-per-pixel amodal ceiling (dense mesh).
- [ ] Learned instance OVER-FIRING risk on real — verify the point cloud (from the clean foreground) doesn't over-fire.

---
## 3b · SESSION 2026-08-13/14 — diagnosis rewritten, five levers closed (see `docs/F1_PUSH_PLAN.md`)

- [x] 🔴 **THE DIAGNOSIS IN §3 ABOVE IS WRONG IN ITS CENTRAL CLAIM.** On real MT-34 VAL the
      predicted mask covers ≥95 % of **94.2 %** of the 463 microtubules at the metric's own 5 px
      tolerance. The semantic foreground is NOT the bottleneck on real data; the loss is in
      assembly. Mask factorial (read as TP, not F1): filling holes +15 TP, removing far-from-GT
      excess **+41 TP**, near-GT mask *shape* **+88 TP**. Six prior nulls all attacked the +15.
- [x] 🔴 **THRESHOLD-CALIBRATION CONFOUND — invalidates fixed-threshold model comparisons.**
      Two models compared at a shared `prob_thr` differ by their probability calibration, not
      their quality: the same pair scored 0.280 vs 0.016 at v4b's 0.4418 and **0.505 vs 0.489**
      at their own optima. nnU-Net was reported at 0.418 and scores **0.501** swept. **Owed:
      re-derive every model-vs-model number that used a shared threshold.** The genuinely
      suspect one is cbDice's "collapse to 25 % foreground" (§20b) — one sweep of the cbB
      checkpoint confirms or overturns it. Gated-retrain/temporal/prodA-B all landed at
      1.85–1.98 % fg at the same threshold, so those calibrations probably match.
- [x] **Generator defect measured: labelled microtubules are rendered below the noise floor.**
      Synth labelled-MT median SNR **0.50** vs real annotated **1.45** (same estimator);
      88.2 % of synth GT below SNR 1 vs 28.1 % real; coverage 97.5 % above SNR 1, 63 % below.
      Corrected config `calib_reg418_morph_snr3.json` (contrast ×3).
- [x] 🔴 **FACTORIAL (head x generator, one trainer, one budget) — the generator fix is
      CONFIRMED TWICE and the amodal head is exonerated.** Full VAL, each arm swept:
      D amodal+orig 0.387 · E amodal+snr3 0.472 · G binary+orig 0.452 · **F binary+snr3 0.492**.
      Generator: **+0.085 [+0.039,+0.124] p=0.001** (amodal) and **+0.040 [+0.013,+0.069]
      p=0.004** (binary). Head: **+0.065 p=0.001 on noisy labels, +0.021 ns on corrected ones**
      — interaction −0.045. **The K=6 amodal head costs nothing once the labels are learnable**,
      so it can be kept for instancer B; the earlier −0.033 was budget plus label noise.
      F matches the framework-trained nnU-Net (−0.013, p=0.540) at 1/6 the training budget.
      Still nothing beats 0.5046 significantly → TEST untouched.
- [x] 🔴 **...and correcting it WORKS, but only on a large-receptive-field architecture.**
      nnU-Net ResEnc topology + K=6 amodal head, same recipe, only the generator differs:
      **E 0.4715 vs D 0.3866, paired +0.085 [+0.039, +0.124], p=0.001 SIGNIFICANT** (full VAL,
      each swept to its own optimum at thr 0.97). On the frozen-DINOv2 trunk the same
      intervention was a clean null (below). **Interaction, not contradiction:** corrected
      labels need a network able to exploit them. Do not quote either result without its
      architecture. E matches v4b (+0.006, p=0.881) and still loses to the binary-head
      nnU-Net (−0.033, p=0.012) — but that last comparison confounds head, budget and
      augmentation, and is settled only by a binary-head run through the SAME trainer.
- [x] 🔴 **The same fix is a NULL on the frozen-DINOv2 trunk (run B).** v4b's own trunk,
      only the data changed, 30 epochs, union-head decode, threshold swept on full VAL:
      **0.455 vs v4b 0.458, paired −0.004 [−0.057, +0.058], p=0.868.** Both halves null too.
      On the 8-frame validation subset it had looked like +0.058 (0.648 vs 0.590) — the subset
      is 6 Alice + the 2 sparsest 586 frames and the gain vanished on the crossing-dense half.
      **Two subset/threshold traps in one session, both inflating toward "it worked".**
- [x] **Learned arm-pairing cost (`src/instance/linker.py`) — NEGATIVE.** Beats the geometric
      cost on held-out synthetic candidates (precision@k 0.862 vs 0.776) and produces
      **identical TP (229)** end to end on real VAL. Junction decisions are not the limit.
- [x] **Decoding — closed.** Flat-threshold sweep and Frangi ridge enhancement worth ~0.004.
- [x] **From-scratch ResU-Net — NEGATIVE.** 0.365 vs v4b 0.458 on full VAL; its mask carries
      9.52 components/MT at 1.90 % fg against v4b's 5.27. Indicts the recipe, not the family:
      nnU-Net reaches 1.49 at 3.2 %.
- [ ] 🔴 **nnU-Net re-eval is the live lead:** 0.501 vs v4b 0.458 swept, but paired bootstrap
      **+0.043 [−0.008, +0.102], p=0.109** — not separable at n=16. Needs a proper nnU-Net
      retrain (K=6 head for the amodal channels) before it can be claimed.
- [ ] **Binary union head is worth +0.022** over max-over-K on the same checkpoint. Cheap,
      supported, and untested on v4b's architecture — fold it in.
- [ ] **Instancer params are still fitted to v4b's mask statistics** (`merge_radius` 8.98,
      `window` 28.4, `min_length` 44.7). Any new foreground needs its own tuning run before
      being compared, exactly as it needs its own threshold.
- [x] 🔴 **EXTERNAL BASELINES MEASURED ON MT-34 for the first time** (`scripts/baseline_mt34.py`,
      end-to-end, strict metric, each swept to its own operating point on VAL):
      **SAM 3 "thin line" 0.432** (Alice half 0.743 / dense half 0.245) and
      **TARDIS zero-shot 0.117** (Alice half 0.311 / dense half **0.000**, 0 TP of 359).
      Both reproduce their published Alice numbers, so the harness is right and the collapse is
      the benchmark. **Rescopes W10: "we do NOT beat SAM3" is Alice-only** — on MT-34 we lead
      0.458 vs 0.432 pooled and 0.273 vs 0.245 on the crossing-dense half. CLAUDE.md's
      "bar to beat: TARDIS 0.326" is likewise an Alice bar; on MT-34 it is 0.117.
- [x] **Instancer parameters demonstrably disadvantage every non-v4b foreground.** Sweeping
      three knobs for nnU-Net's mask: `merge_radius` 9.0 (shipped, fitted to v4b) → 3.0 gives
      **0.501 → 0.519**, monotone across all six combinations of the other two. A large
      contraction radius helps a shattered mask and destroys a clean one. **Every architecture
      comparison so far ran through a downstream stack fitted to v4b** — the fix is an equal
      tuning budget per arm, on SYNTHETIC VAL (§17q), writing a new named params file.
- [ ] **Segmenter hyperparameters have never been tuned, at all.** `pos_weight=8` is inherited
      from v4b and every new run over-fires (fg 2.5–5 % against a 1.9 % target); loss weights,
      deep-supervision depth and the augmentation set are my choices. Runs D/E take their
      architecture and optimiser from nnU-Net's own `plans.json`, which is the only part with
      provenance. Minimal sweep owed: `pos_weight` {2,4,8,16} and LR, equal budget per arm.
- [x] **Instancer tuned per model, two knobs only** (`scripts/tune_two_knobs.py`,
      `params_a_model_v5H.json`, write-once). On held-out SYNTH +0.090 over the shipped vector;
      on real VAL **+0.006**, and only `merge_radius` transfers (`prob_thr` optimum is 0.97 on
      synth, 0.99 on real). Second data point on §17q: synth tuning does not hurt, but its
      benefit shrinks by an order of magnitude across foregrounds. Quote the transferred number.
- [x] **Generator fixes 2 and 3 (crossing statistics, field stop) are a NULL on real VAL.**
      Run H (all three fixes) 0.4898 against run F (SNR fix only) 0.4873 ± 0.0040 — inside one
      seed SD. Plausible reason for the field stop specifically: inference already applies
      `fov_mask`, so the rim false positives are masked away regardless (FP 314 → 298).
- [x] **Run-to-run variance measured: SD 0.0040** over three seeds, not the assumed ±0.02.
      Every effect reported sits far above it, yet the paired-bootstrap intervals are ±0.05 —
      **the limiting uncertainty is the 17-frame benchmark, not the training.**
- [ ] 🔴 **TEST NOT SCORED, by the pre-declared rule.** Best system H tuned 0.4953 vs v4b
      0.4655 = +0.030 [−0.024, +0.097] p=0.331. The four leading systems (0.466–0.505) are
      mutually indistinguishable at n=16. Exposure count unchanged at four.
- [ ] 🔴 **The highest-value next action is NOT a model change: review CVAT task 586's 281 pre-annotated
      frames. → OWNED BY §0 NC-B1**, which adds the acceptance criterion (n ≥ 100, bootstrap width < 0.03)
      and the agreement-bias caveat that makes the expansion valid.

---

## 4 · PAPER-WIDE — benchmark, baselines, rigor, generality

> Re-scoped 2026-08-25 for the **Nature Communications** target. The actionable form of everything here
> now lives in **§0**; this section keeps the *measured record* and the peer-review IDs. Where an item has
> a §0 owner it is named, so nothing is tracked in two places.

- [x] 🔴/🟠 **W9 — proper benchmark stressing CROSSINGS/PARALLELS — DONE 2026-08-10: `data/real/mt34_eval/` (MT-34).**
      34 frames / 957 GT polylines = refreshed Alice (CVAT 585, which IS the Alice set) + 22 human-reviewed frames of
      CVAT 586. **32.1 crossings/frame vs Alice 2.2**; 5.4 vs 0.9 close-parallel pairs. Junction-identity,
      fragmentation and bundle-recovery metrics implemented in `src/instance/metrics.py`. Split 6/6 + 11/11, tune on VAL.
      protocol §17, [[mt34-benchmark]]. **Superseded in scale by §0 NC-B1** — 34 frames cannot resolve a 0.03 effect.
- [ ] 🔴 **W9-followup — MT-34 leakage carve-out. → OWNED BY §0 NC-E1.** All 22 new frames are in the generator
      calibration corpus (`training_img_114.tif` = `mt_corpus/images/0015_local_irm_tif_img_114.tif`). Not label
      leakage, but for publication they must be carved out and the generator recalibrated (as was done for HTW).
- [ ] **W9-followup — agreement bias. → OWNED BY §0 NC-E2.** MT-34 GT is human-corrected v7+PySOAX output →
      PySOAX-family instancers are flattered. Report error rates split by `manual` vs `file` polylines.
- [ ] **FIELD-OF-VIEW STOP (2026-08-10).** Task-586 frames were acquired through an octagonal field stop that
      Alice lacks; v4b fires on its edge (spurious detections carry 2.06x the residual contrast of an annotated MT).
      `mt_bench.fov` masks it but recovers only ~16% of the spurious mass at a GT-safe 4 px erosion.
      **Real fix = teach the generator about field stops** — but note the measured null: run H's field-stop fix
      changed nothing on real VAL, plausibly because inference already applies `fov_mask` (FP 314 → 298).
      Training-time field-stop augmentation is still open as **§0 NC-A2(b)**.
- [~] **W10 — apples-to-apples baselines. → OWNED BY §0 NC-F.** SAM 3 `thin line` (SynthMT's engine) reproduced on
      Alice, identical pipeline. **CONF SWEEP on Alice: 0.10=0.390, 0.25=0.588, 0.40=0.674, 0.60=0.712 vs OURS 0.697
      → on Alice we DO NOT beat it.** On MT-34 the point estimates reverse (0.458 vs 0.432 pooled, 0.273 vs 0.245 on
      the dense half) but **no paired test exists** and at n=16 four systems spanning 0.466–0.505 are indistinguishable
      — so the reversal is NOT a claim (§0 NC-F1). **The supportable framing is annotation-free PARITY vs a 3.3 GB
      real-trained foundation model, plus the crossing-regime separation (§0 NC-A1).**
      Env `/home/prusek/sam3_env` (py3.11, ultralytics 8.4.92), weights `/disk1/prusek/sam3/sam3.pt` (gated),
      scripts `scripts/sam3_alice.py`, `scripts/baseline_mt34.py`. STILL TODO: TARDIS + ORION on the same pipeline;
      SAM3 conf tuned on a VAL split (Alice-tuned = optimistic).
- [ ] **W6 — statistical rigor. → OWNED BY §0 NC-G.** Confidence intervals, multi-seed training (SD measured at
      0.0040, not the assumed ±0.02), per-frame distributions rather than single-seed point estimates,
      Holm correction over the comparison family, frame as the unit of analysis.
- [x] **NATURE METHODS BAR — assessed and NOT MET; venue revised to NC (2026-08-25).** The former bar was
      (D1) solve instance synth-only OR (D2) a general cross-lab/modality annotation-free pipeline with causal
      calibration proof (W2) + identifiability (W3) + benchmark (W9) + statistics (W6), plus (D3) fair vs SynthMT,
      (D4) TIRF quantitative, (D5) resolve W1. **W2 and W3 both came back negative**, so D2 is not reachable on the
      current evidence, and the 2026-07-11 honest verdict already said so. 🟣 **The genuinely NM-sized gap nobody has
      costed is GT PROVENANCE:** MT-34's ground truth is human-corrected *model output*, and an NM eval set would need
      an independently drawn subset with inter-annotator variability — weeks of human time. §0 NC-B2(b) buys the
      cheap half of that (the agreement number) because NC needs it too; the full independent re-annotation is
      deferred with the NM target.
- [ ] Maintain living docs: PAPER_PLAN.md (**stale — see §0.0a**), protocol.md, INSTANCE_SEGMENTATION_RESEARCH.md
      (update, don't just append).
