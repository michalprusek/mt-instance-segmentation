# TODO — consolidated, grouped by DATASET / SEGMENTATION / INSTANCE (+ paper-wide)

> Single source of truth (persists across sessions). Target = **Nature Methods** — everything must be
> reviewer-proof (see docs/PAPER_PLAN.md, nature-methods-target memory, peer review). Legend: [ ] todo ·
> [~] in progress · [x] done. 🔴 = Nature-Methods KILLER. Peer-review IDs W1–W11 mapped in.
> A **better real eval dataset than the 12 Alice frames is COMING** (user) — re-run all real evals on it.

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

- [ ] 🔴 **W1 — calibration target ≠ eval distribution.** The wavier "improved" generator HURT Alice instance-F1
      (0.62→0.41) ⇒ the 320-frame corpus may not represent the eval set. Characterize corpus provenance/diversity;
      test an Alice-like (straighter) calibration target; quantify corpus↔Alice shift. Don't over-tune to corpus artifacts.
- [ ] 🔴 **W2 — no CAUSAL proof calibration improves TRANSFER.** C1: region/residual-MMD did NOT beat global-cosine
      on semantic tol2 (only over-firing). Show the objective causally improves downstream INSTANCE-F1 vs baselines
      (not just tol2). Proxy gap (MMD≠transfer) must be addressed or the N1 claim weakens.
- [ ] **W3 — identifiability / variance-recovery (C1.5).** On a synthetic target with KNOWN std, show region_mmd
      has a well at the true std while frame-pooled residual_mmd is flat. Currently asserted, never demonstrated.
      (Also validate: does region_mmd's density=25 give better downstream than residual_mmd?)
- [ ] **Param review + add relevant knobs** (user directive): review ALL GenConfig params (each Normal(mean,std));
      add missing physical ones — waviness freq/amp (review), MT bending rate/stiffness, MT whitening/bleaching
      frequency (detach params; TIRF photobleaching), property VARIANCES (now maybe identifiable via region_mmd),
      curvature-dependent breaks. Decide calibratable vs fixed prior; document. Recalibrate with >300 (=400) trials.
- [ ] **W4 — TIRF quantitatively + cross-lab.** TIRF uncalibrated (no real corpus); all data single-source IRM.
      Get real DARK TIRF backgrounds + frames, freeze morphology, calibrate ONLY TIRF appearance, show cross-lab transfer.
- [ ] **W5 — physics-based appearance (consider).** Appearance parametric (Gaussian core + halo), not a physics
      forward model (two-reflection interference). Evaluate if a physics renderer improves realism / reviewer-defensibility.
- [ ] Minor realism: real set has MORE dirt specks than synth (raise `spot_rate`); appearance render width slightly
      thin (2.26 vs real 2.86 px — mask fixed, render not); fit the cross-section profile from bg-subtracted real signal.
- [ ] (Fundamental limit) parametric EXPRESSIVENESS ceiling — calibration finds best point WITHIN the family; SimGAN
      refinement was a WASH; some appearance gap is model-class-bound. Mitigated by real backgrounds.

---
## 2 · SEGMENTATION MODEL — semantic foreground (DINOv2-hybrid)

**Done / in place:** DINOv2-hybrid (frozen ViT-L blocks 5/11/17/23 + high-res branch); orientation-keyed "overpass"
head (K=6, amodal); mask_half_width=1.0 (best localization); raw input (residual HURT); inversion augmentation;
over-firing clean 0.6–0.7×; semantic tol2 **0.947** (val/test 6/6 split). Foreground is essentially solved.

- [ ] **W8 — don't over-claim on semantic.** tol2 0.947 is on the EASY part ("semantic foreground largely solved" —
      own knowledge doc). Frame honestly; the contribution must be the calibration principle or the HARD instance side.
- [ ] **W11 — novelty framing.** DINOv2-hybrid is standard (SegDINO/DPT). The genuinely-new idea is
      "calibrate-what-you-segment" (shared encoder for calibration + segmentation) — but its benefit is UNPROVEN
      (C1 ambiguous; tightening the synergy HURT 0.627→0.559). Either prove it (W2) or don't lead with it.
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
- [ ] **Same treatment owed to the ORACLE-mask params** (`params_a_v2.json`, still real-VAL-tuned).
      It is a diagnostic ceiling rather than the shipped system, but the 0.920 oracle number should
      not go in a paper next to an annotation-free claim without it.
- [ ] **TEST multiplicity.** This session scored MT-34 TEST four times (v1/v2 re-instrumentation,
      gated retrain, synth-tuned). Each was declared before it ran and each carries its interval, but
      the accumulated exposure belongs in the write-up.
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
      needs version control before any paper number is frozen** (ask the user first — commits only on request).
- [ ] 🔴 **STILL #1 BOTTLENECK = the SEMANTIC FOREGROUND, and it is NOT the domain gap.** With v4b's
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
## 4 · PAPER-WIDE — benchmark, baselines, rigor, generality

- [x] 🔴/🟠 **W9 — proper benchmark stressing CROSSINGS/PARALLELS — DONE 2026-08-10: `data/real/mt34_eval/` (MT-34).**
      34 frames / 957 GT polylines = refreshed Alice (CVAT 585, which IS the Alice set) + 22 human-reviewed frames of
      CVAT 586. **32.1 crossings/frame vs Alice 2.2**; 5.4 vs 0.9 close-parallel pairs. Junction-identity,
      fragmentation and bundle-recovery metrics implemented in `src/instance/metrics.py`. Split 6/6 + 11/11, tune on VAL.
      protocol §17, [[mt34-benchmark]].
- [ ] **W9-followup — MT-34 leakage carve-out.** All 22 new frames are IN the generator calibration corpus
      (`training_img_114.tif` = `mt_corpus/images/0015_local_irm_tif_img_114.tif`). Not label leakage, but for
      publication they must be carved out and the generator recalibrated (exactly as was done for HTW).
- [ ] **W9-followup — agreement bias.** MT-34 GT is human-corrected v7+PySOAX output → PySOAX-family instancers are
      flattered. Report error rates split by `manual` vs `file` polylines.
- [ ] **FIELD-OF-VIEW STOP (new, 2026-08-10).** Task-586 frames were acquired through an octagonal field stop that
      Alice lacks; v4b fires on its edge (spurious detections carry 2.06x the residual contrast of an annotated MT).
      `mt_bench.fov` masks it but recovers only ~16% of the spurious mass at a GT-safe 4 px erosion.
      **Real fix = teach the generator about field stops** (composite onto backgrounds that have one, or synthesise one).
- [~] **W10 — apples-to-apples baselines.** SAM 3 `thin line` (SynthMT's engine) reproduced on Alice, identical pipeline
      (1.5× input → skeleton→instance → centerline-F1 → same GT). **CONF SWEEP: 0.10=0.390, 0.25=0.588, 0.40=0.674,
      0.60=0.712 vs OURS 0.697. → TUNED SAM3 (0.712) is COMPARABLE to / marginally above ours (0.697), within n=12 noise —
      WE DO NOT BEAT IT.** The zero-shot 0.588 is under-tuned; don't cite it as a win. **⇒ "we beat SynthMT on instance-F1"
      is UNSUPPORTED; reframe to annotation-free PARITY vs a 3.3 GB real-trained foundation model + the calibration novelty.**
      Env `/home/prusek/sam3_env` (py3.11, ultralytics 8.4.92), weights `/disk1/prusek/sam3/sam3.pt` (gated), script
      `scripts/sam3_alice.py`. STILL TODO: TARDIS + ORION on same pipeline; SAM3 conf tuned on a VAL split (Alice-tuned = optimistic).
- [ ] **W6 — statistical rigor.** Confidence intervals (n=12 is tiny); multi-seed training (±0.02 noise); per-frame
      distributions, not single-seed point estimates.
- [ ] **Nature Methods bar:** either (D1) SOLVE instance synth-only, OR (D2) a general cross-lab/modality annotation-free
      pipeline with CAUSAL calibration proof (W2) + identifiability (W3) + proper benchmark (W9) + statistics (W6);
      plus (D3) fair vs SynthMT (W10), (D4) TIRF quantitative (W4), (D5) resolve W1.
- [ ] Maintain living docs: PAPER_PLAN.md, protocol.md, INSTANCE_SEGMENTATION_RESEARCH.md (update, don't just append).
