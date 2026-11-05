# Generator / training research, 2026-09-16: five lanes, one survivor

Step 4 of the 2026-09-15 directive ("identifikuj slabé části, udělej research a iteruj").
Inputs: the generator inventory (what SPARSE35 trained on), the ledger of closed levers
(`closed_levers.md`, audit 2026-09-16), the price list (measured ceilings, 2026-09-16), and the
adversarially verified output of five research lanes — irm-physics, sim-to-real, label-convention,
diagnostics, data-side. **13 proposals entered verification; 1 survived, 12 were killed** (11
refuted, 1 unverified). Every number below is quoted from the price list, the ledger, the inventory
or the verification output; none is derived arithmetic. Nothing in this file has been written into
`docs/TODO.md` yet — the falsifiers in §2 are the text to pre-declare there before any read.

## 1. Executive summary

1. SPARSE35 on the declared instrument: roi303 TEST **0.641 ± 0.006** (P 0.59 / R 0.71, oracle 0.935), htw TEST **0.601 ± 0.012** (0.62 / 0.58, oracle 0.871). Dev baseline roi303 VAL 0.6558 @0.98; gap 0.244 to the same-path oracle 0.900.
2. Of that 0.244 the model owns **≈0.06–0.07**; **≈0.12–0.13** is annotation convention and single-frame evidence (evidence-free stretches ≈0.07, unannotated filaments and seeds ≈0.05). The single-frame bound on roi303 VAL is 0.74–0.76.
3. The largest model-owned row any generator or training lever can buy is **crossing erosion, ≤0.03** (GT band painted ≤40 px of crossings: +0.029). Every other model row is ≤0.02: field-stop edge ≈0.02 by length share (re-priced by the verifiers at +0.007 by prediction count; ARM 50 negative), merging 0.01–0.02 (hw 2.0 +0.017 / hw 3.3 +0.012), particles + ambiguous ≈0.02.
4. On htw the reachable row is detection (+0.11 band ceiling at hw 2.0), but merging is dead (−0.009 / −0.017), crossing erosion ≈0.00, and single-seed resolution is ~0.04 — htw cannot license an arm on its own; it is the mandatory second read.
5. Resolution: 3-seed sd 0.006 (roi303) / 0.012 (htw); the bar is ≥0.01 on roi303 VAL micro per 6 GPU-h arm. Only the crossing-erosion row clears that bar with room; the others straddle it.
6. The ledger holds one significant generator gain ever (SNR ×3: +0.085 [+0.039, +0.124], p 0.001); every image-statistic lever since — noise texture, noise amplitude, wide physics priors, density, photometric and scale augmentation, field stop, ridge distractors — is null or negative on the instrument it was read on.
7. The five lanes' 12 kills fell on ledger collision, on price, or both: the aperture-edge family (4 proposals, incl. diagnostics P3) is bounded at +0.002–0.007 by prediction count, the distractor family (2) is closed by ARM 41's catastrophe and TODO 4154's dichotomy, the optical halo (1) prices ≤0.01 with a pre-screen biased toward launching, the two other diagnostics (2) decide nothing that can license an arm, the ensemble/SWA (1) is inference-side, the two-frame input (1) has no co-registered sibling in roi303. ZERO-BAND (1) is unverified (verifier failed), not refuted.
8. The one survivor is a **renderer artefact**, not a prior: `_paint` stamps filaments at rounded pixel positions, so rendered peak contrast depends on orientation relative to the pixel grid by 21–39 % across the shipped PSF range — larger than the per-MT contrast spread (rel_std 0.2) — and the rot90/flip augmentation preserves rather than averages it. It survives as a 15-CPU-minute pre-screen with a fireable kill line plus renderer hygiene; the GPU arm is conditional on the pre-screen.
9. **No GPU arm is approved unconditionally by this round.** The survivor's own modal outcome (null-to-+0.01) sits below the bar; it earns 6 GPU-h only if the per-orientation weak-share profile on real is elevated in the axis-aligned bins by more than the in-domain bin-to-bin spread and by enough to be worth ≥0.01.
10. Honest ceiling for "improve the model" on this benchmark as annotated: ≈0.06–0.07 on roi303 VAL (to ≈0.74–0.76), of which the priced, actionable part today is ≤0.03. **The 0.90 goal is not reachable by any model lever on the benchmark as annotated**; the ≈0.12–0.13 convention/single-frame share is a benchmark decision (§5), not a training problem.

## 2. Ranked survivors

One proposal survived. It decomposes into three stages with separate costs, and the verification
approved them separately; the ranking below is over those stages, by (ceiling × plausibility) / cost.
Padding this section with unverified levers would defeat its purpose — see "Out of scope" at the end.

### Rank 1 — Orientation-isotropic filament rendering (sim-to-real lane; survives)

**Full title:** anti-aliased `_paint` with contrast renormalised to the shipped orientation-averaged
peak. **Status:** pre-screen + renderer hygiene approved; 6 GPU-h arm conditional on the pre-screen.

**Mechanism.** `_paint` (`synth/mt_generator.py:531-543`) stamps the Gaussian cross-profile at
`np.round`-ed pixel positions with 0.5-px normal offsets, so the rendered profile depends on the
filament's angle to the pixel grid. Measured with the repo's own code at training-time sampling
(step 1.0 px, render half-width 6, width σ 1.78, mass conserved at 8.92 per unit length): an
axis-aligned filament's raw column profile is [2.92, 1.56, 0.90, 0.39, …] against the ideal
continuous [2.00, 1.71, 1.06, 0.48, …] and a 45° filament's [2.10, 1.62, 1.09, 0.51, …] — a +39 % peak
for axis-aligned filaments. After the frame PSF the peak still swings: PSF 0.49: 2.65 at 0°/90° vs
1.91–2.16 oblique (max/min 1.39); PSF 0.705 (shipped centre): 2.32 vs 1.81–1.95 (1.28); PSF 0.92:
2.09 vs 1.73–1.81 (1.21). The 45° line additionally carries an along-line ripple (peak sd ±0.30 at
PSF 0.49 vs ±0.10 at 7.5°). The 3×3-square band target (`scripts/train_v5.py:277-287`) is
anisotropic the same way (3.03 px per unit length at 0°/90°, 3.43 at 10°, 3.96 at 30°/60°, 3.56 at
45°). Both artefacts are exactly symmetric under the trainer's rot90/flip group
(`train_v5.py:459-470`; 0↔90, 15↔75), so augmentation preserves rather than averages them, while
the real instrument is isotropic. The network can therefore learn an orientation-conditional
contrast/width expectation that real filaments violate — a sim-only shortcut cue
(arXiv:2004.07780) that would surface as confidence dips wherever a real filament runs at the
"wrong" angles, feeding the plain-stretch deficit (15 % of on-filament samples below 0.98 on
roi303 vs 1 % in domain) and its doubling near crossings (0.28 vs 0.15).

**Fix.** Render the signed field on a 4× supersampled grid (or evaluate the Gaussian at the exact
perpendicular distance) and box-downsample; then renormalise `amp` so the ORIENTATION-AVERAGED
rendered peak equals the shipped one. The renormalisation is mandatory: an anti-aliased stamp lowers
axis-aligned peaks ~25–30 % and raises oblique ones ~10 %, and SNR ×3 is the only significant lever
in the ledger — without it this is a contrast arm in disguise. The Euclidean-disk band fix
(`train_v5.py:277-287`) is a second lever, a THIN-shaped narrowing at 30–60°, and is declared as
the follow-up, NOT part of this arm.

**Target item + ceiling.** Crossing erosion of the map (≤0.03, model-owned; Step 2c cross40
painted band +0.029) and the with-evidence part of the along-filament deficit that rides under it.
Ceiling ≤0.03 on roi303 VAL; **modal outcome null-to-+0.01** (proposer's own). htw priced at 0
unless the per-orientation read fires there too. The verifier's caveat: the plain-stretch deficit is
largely evidence-free (Step 2g: weak share 4.4 % at the metric's own tolerance; AUC weak-vs-displaced
0.51–0.58), so no rendering fix recovers that part, and the link from the orientation cue to the
crossing-specific doubling is unmeasured — hence the arm is conditional.

**Code touch points.**
- `synth/mt_generator.py:531-543` (`_paint`: `np.round` stamping, 0.5-px offsets) — the fix.
- `synth/mt_generator.py:406-410` (`cross_profile`, frame width σ N(1.78, 0.35)) — the profile being stamped.
- `synth/mt_generator.py:672-704` (IRM photometry; `amp`, contrast floor at `:690`, composite + PSF at `:696-700`) — the renormalisation site.
- `scripts/train_v5.py:459-470` (rot90/flip group) — unchanged; the reason the artefact is not averaged out.
- `scripts/train_v5.py:277-287` (3×3 square band) — follow-up only, not this arm.
- `scripts/diag_line_integral.py` — sampling for pre-screen (1) (plain-stretch samples ≥40 px from crossings, p<0.98 as "weak").
- Deployment: `GenConfig` is built by `gen_train.build_cfg` on tulen (`train_v5.py:1129-1132`), local mirror `scripts/photometry_screen.py:191-216`; the flag must be threaded through both trees (memory `tulen-two-trees`: deploy to both, verify md5).
- Data: cached SPARSE35 ep040 roi303 VAL dump (139 frames); `data/synth_eval_s35` (48 exact-label synthetic frames); htw VAL (33).

**Training-free pre-screen and kill line.**
- **(1) Per-orientation weak-share profile, ~15 CPU-minutes, no rendering.** On the cached SPARSE35 ep040 roi303 VAL dump and on the 48 exact-label synth frames: plain-stretch p<0.98 share (`diag_line_integral.py` sampling, ≥40 px from crossings) binned by the local GT tangent angle mod 90° in ±7.5° bins (13 bins), computed WITHIN frame (so nematic frames do not confound) and then pooled; repeat on htw VAL. The in-domain profile is the null; the axis bins {0, 90}±7.5° are the contrast of record; 30°/60° (band-anisotropy bins) read as a secondary. **Kill line:** the axis-aligned bins on real are not elevated relative to the oblique bins beyond the bin-to-bin spread the in-domain profile shows, or the real profile is flat within that spread → the model does not exploit the cue; the fix is rendering hygiene only, NO GPU arm. **Launch gate (verifier's re-pricing):** the axis-bin excess must be at least a third of the with-evidence deficit, so that removing it is worth ≥0.01.
- **(2) Render-side effect test, seconds.** Supersampled `_paint` must bring the max/min peak over the 13 angle bins to ≤1.05 at PSF 0.49 (shipped: 1.39) and leave the orientation-averaged peak within 2 % of the shipped renderer. The guard tests the EFFECT, not the plumbing (the `half_width` lesson).

**Cost.** Pre-screen (1) ~15 CPU-minutes; (2) seconds. Arm: 6 GPU-h + 1 CPU-h threshold sweeps; the
4× supersample roughly doubles generator CPU per 640-px frame (dataloader workers absorb it).

**Falsifiers to pre-declare in `docs/TODO.md` before any read** (arm stage only):
1. Paired roi303 VAL pooled micro ≥ +0.01 vs SPARSE35 ep040 at each block's own VAL threshold (seed sd 0.006).
2. Mechanism gate: the per-orientation below-threshold profile on real must FLATTEN (axis-bin excess falls to within the in-domain spread). A positive F1 without a flattened profile is a contrast arm in disguise → not claimable.
3. Rescale falsifier: precision at matched coverage must rise; a threshold shift alone is a fail (the ARM 48/49 lesson).
4. htw VAL recall read as the along-filament discriminator (mandatory second reading; priced at 0, so a null there does not kill, a fall beyond seed spread does).
5. Bit-identical-marginals check for the hygiene ship (criterion proposed HERE, reusing the effect test's tolerance — not a verified number): with renormalisation on, the per-frame contrast, width and mask-width marginals of the rendered set must match the shipped renderer within 2 % — otherwise the "hygiene" is a photometry change and must be read as an arm.

**Collision check (verifier).** No ledger footprint: grep of `docs/TODO.md` for anti-alias /
supersampl / sub-pixel / np.round / stamp / isotrop / axis-aligned / aliasing hits only NOISE's
"isotropic kernel" (2357) and rot90/flip mapping tests (2886, 3059). Every closed relative — SCALE
(1452–1475, null at native), AUG_PHOTO (4962–4975, negative at native), THIN / prior bank (171–180),
noise amplitude / texture (915–926, 4082–4100) — is isotropic and could not have moved an artefact
that is invariant under the augmentation group. No real statistic enters (not a prior fit; the
8/8-replicated rule is respected), no post-processing, no self-training. The inventory lists
sub-pixel stamping as a known simplification ("crude: rounded offsets, 0.5 px steps"), not a closed route.

**Ranking of the stages.**
- **Stage A — pre-screen (1). Run this FIRST.** Cost ≈ 0, its kill line can actually fire on existing dumps, and it is the only reading in this round that can either close the lane or re-price the arm above the bar. Nothing else on the table has that property.
- **Stage B — effect test (2) + flagged renderer.** Cost seconds; ceiling 0 by itself (hygiene); implement behind a `GenConfig` flag, default OFF.
- **Stage C — the 6 GPU-h arm.** Ceiling ≤0.03 × plausibility low (modal null-to-+0.01) / 6 GPU-h → below the bar today; runs only if Stage A passes its launch gate.

**One caution the verdict undersells.** Shipping the anti-aliased `_paint` "as hygiene" changes every
training image: it re-baselines SPARSE35 and confounds every subsequent arm. The ledger precedent is
the photometry zero-crossing fix (PHOTO_V6): a genuine defect fix (flipped label 12.0 → 2.1 %) that
read −0.0016 [−0.037, +0.031], p 0.95. Keep the renderer behind a flag until Stage C or the
bit-identical-marginals check has been read; do not let it become the default silently.

### Out of scope of this round (not ranked, not closed)

The ledger's named OPEN / virgin items — R3 path-connectivity losses (CAPE, maximin, MALIS),
continuous orientation-field head, instance-gap valley head, gap-weighted background loss between
close parallels, segmenter HPO (`pos_weight=8` inherited), third NOISE seed, `--target-half-width ≥ 2`,
LOWRES isolated from SCALE, the 9–11× close-parallel over-production, Screen B, real-background-
negative-beside-filament (§7.2(b)) — did not enter the five lanes. Their absence from §2 is not
closure; they were simply not proposed.

## 3. Killed proposals (do not re-propose as written)

| # | lane | proposal | one-line reason |
|---|---|---|---|
| 1 | irm-physics | Layered aperture-edge model (in-focus rim ringing + defocused stray-light foot + camera rail) | Price: ceiling 0.010–0.015 requires 100 % capture of the aperture-only share; two prior null/negative stops at the same site (run H, ARM 50); the un-varied third candidate is placement (stop apothem 0.50–0.54 on a 640 render = corner wedge, `train_v5.py:328-336`), not shape. |
| 2 | irm-physics | Defocused scatterers with an interferometric ring PSF (physics upgrade of `add_spots`) | Both grounds: TODO 4154 closes distractors as a dichotomy; ring outer arcs are faint single-signed curved ridges at MT width/amplitude overlapping the gliding-ring prior = ARM 41's cause (0.1454 vs 0.5203, recall 0.159) at another geometry; author's modal is null. |
| 3 | irm-physics | Partially coherent line-spread profile with opposite-sign flanks (optical halo) | Price ≤0.01 after decomposing cross40; merging term dead on htw; kill line has no sign test and acceptance (3) fits a real statistic. The halo was in the generator until 2026-07-08 (Optuna amplitude) and removed on review — show the never-run isolated A/B on cached v4b-era artifacts first. |
| 4 | sim-to-real | Label-consistent compositing on real field-stop backgrounds (clip render + label to the illuminated field) | Rims were already real negatives in every arm (both polarities); target is ARM 50's row at +0.002–0.005; same crops looked at three times (TODO 371–378, 342–349, 425–428); removes ~a third of positives in stop tiles (ARM 42's recall-falsifier direction). Modal null-to-negative. |
| 5 | sim-to-real | Physically rendered debris (two-beam rings of spheres + defocused blobs), gated on empty-frame response | Distractor family closed with an explicit "no arm 41b" ruling about the concept; ceiling at the resolution floor after Step 2d′ removed half the target; declared modal null. Empty-frame interior-firing rate is a diagnostic for the record, not a lever. |
| 6 | label-convention | ZERO-BAND: contrast floor 0.22 → 0, filaments pass through the interference zero; bounded interior height excursions to 56.4 nm; label kept | **Correction 20:25:** the verifier DID refute it (survives=false); the workflow script mis-matched its truncated title. Price: the wrong-polarity convention is already shipped (mt_generator.py:472-477, :690) and the model still breaks 73 % of real flipped samples; share-based ceiling ≈ 0.006; the SNR-axis arm was null (−0.0016). |
| 7 | diagnostics | P1 — declared-instrument re-read of eight cached arms (NOISE, NOISE_S2, NOISE2, DENSE, ARM45, ARM47, AUG_PHOTO, SCALE) | Every arm already has a 0.80/0.80 or micro reading ≤ +0.004; price list §6 keeps the FP charge it assumes dropped; NOISE htw anatomy (TP 258→256) contradicts the +0.11 target. Ledger hygiene, not a lever. |
| 8 | diagnostics | P2 — htw detection deficit by covariate support + matched-covariate twin recall | Its only licensed action (widen a curve prior) is the closed wide-priors family (ARM 47, TODO 3675–3699), measured to hurt htw (W1-DR, TODO 6055–6062); length already shipped and null after Holm; ≤0.01 on roi303, htw unresolvable at 33 frames. |
| 9 | diagnostics | P3 — real aperture-edge profile on 76 roi303 VAL stop frames + edge-only delete ceiling | Edge-only ceiling ≈ +0.007 by prediction count (ARM 50 pre-read +0.002–0.005); lever null twice; most of the profile is already in `apply_field_stop`'s docstring. Named prerequisite of ARM 50, harmless CPU-minutes, cannot license an arm above the seed sd. |
| 10 | data-side | P1 — three-seed probability ensemble on cached dumps, folded into one checkpoint by SWA | Inference-side ensemble rejected by the user by name (diagnostic-only since 2026-09-04); checkpoints are already EMA (`train_v5.py:1239-1468`) so single-seed SWA is a near no-op and cannot inherit the ensemble's read; rows it claims are seed-consensus; modal +0.00–0.01. |
| 11 | data-side | P2 — aperture-aware use of the 148 closed-iris backgrounds (mask labels/field to illuminated field, edge-straddling crops, RAW-measured ramp) | Own gate (edge-class ceiling ≥0.010) cannot be met: deleting all 171 edge FP = +0.007 (25 px), +0.004 (10 px), 0 on htw; surround is partially lit (0.29–0.52×) so masking labels is a convention, not physics; part (iii) is ARM 50's retry measured on 8-bit copies. Bookkeeping survives: measure RAW transition width on tulen; correct TODO 2538. |
| 12 | data-side | P3 — two-frame input from real time-lapse siblings (pre-screen run and killed by the lane) | No co-registered sibling exists in roi303 (motile assay, filaments leave position; shared component is the background); TEST frames' siblings are VAL in 94 cases; h5 has no stack/index/dt; sequence generator does not reproduce the trainer's appearance (TODO 6162). Reopen only with source stacks + dt, ≥80 % of curves stable within 5 px, sequence-parity fix, and a user ruling on time-integrated evidence. |

Killed-proposal by-products that are still owed (bookkeeping, not arms): RAW aperture transition
width on tulen (ARM 50's recorded prerequisite; rows 1, 9, 11); TODO 2538 correction — the
background pool already holds 148 real apertures (row 11); confirm the 8 % sibling-stability figure
on tulen's RAW block before writing the P3 closure into `docs/TODO.md` (row 12).

## 4. Sources

External, carried by the survivor:
- arXiv:1810.10093 — Structured Domain Randomization: the structure of what is randomised, not only its range, decides transfer.
- arXiv:2004.07780 — Geirhos et al., Shortcut learning in deep neural networks: a cue present only in training data is learned in preference to the intended one.
- arXiv:2306.10941 — Kreitner et al.: rendering fidelity of thin structures decides annotation-free transfer.

Internal:
- `docs/research/2026-09-02-scale-and-gap-audit.md` §A (synthetic per-sample FWHM 4.03 vs config expectation 4.53 — consistent with an over-peaked stamp; real width falls with SNR while synthetic rises, A.2).
- `synth/mt_generator.py:531-543, :406-410, :672-704`; `scripts/train_v5.py:277-287, :459-470, :1129-1132`; `scripts/photometry_screen.py:191-216`; `scripts/diag_line_integral.py`.
- The session's own numpy reproduction of `_paint` at training-time sampling (0°/45° profiles, PSF 0.49/0.705/0.92 peaks, band px per unit length) — reproducible from the two generator files above.
- Generator inventory (this round); `closed_levers.md` (ledger audit 2026-09-16); price list 2026-09-16; `docs/research/2026-09-16-research-brief.md`; `docs/research/2026-09-16-step1-results.md` (Steps 2c, 2g); `docs/TODO.md` line ranges as cited in §3.

The killed lanes' external sources were not carried into the verification output and are not
reproduced here; none is needed to keep them killed.

## 5. What is NOT a model problem and belongs to the benchmark

Roughly half of the roi303 VAL gap — ≈0.12–0.13 of 0.244 — is owned by the annotation and by the
single-frame nature of the evidence, not by the network: **evidence-free stretches ≈0.07** (the
annotator drew through image with no ridge within the metric's own tolerance; weak-vs-displaced AUC
0.51–0.58, kill line ≥0.20 recall of weak at 80 % blob rejection missed at 0.038–0.051 (±2 px) / 0.052 (±5 px); the only model-side route would be information
from outside the frame, i.e. time, which killed proposal P3 shows roi303 cannot supply), and
**unannotated filaments + seeds ≈0.05** (on_gt<0.10 = 0.63 of off-annotation length; delete-far 5
lifts 0.656 → 0.749), which on htw is 81 % of the over-long excess. These are decided by ruling, not
by training: the **annotation-completion pass** requested 2026-09-09 and never done; an **IGNORE
rule** for unannotated filaments (model-free, pre-declared, secondary — micro removes the macro
weighting but not the FP/FN charge for following an unannotated stretch); and the **time-series
provenance** items the data-side lane surfaced — byte-identical roi_img_208/209 VAL–TEST pair,
206/207, 172/173, and 94 cross-split siblings, plus an h5 with no stack index or dt. Until those are
ruled on, the reachable target for "improve the model" on this benchmark is the single-frame bound
0.74–0.76, not 0.90.
