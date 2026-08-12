# Cross-frame microtubule tracking — design

**Date:** 2026-08-12
**Status:** approved (architecture), implementation starting

## Goal

Track individual microtubules across frames of IRM video, and do it **without giving up
single-frame operation**. Four readouts, all of which the user asked for and which turn out to
be layers on one correspondence:

1. **identity** — the same microtubule keeps one label for as long as it is visible;
2. **gliding velocity** — the shift along the filament's own contour, per microtubule per frame;
3. **tip dynamics** — growth and shrinkage of the two ends;
4. **fragmentation healing** — pieces that move together are one microtubule, so temporal
   agreement repairs the single-frame segmentation.

## What the data actually is (measured, 2026-08-12)

- CVAT **project 31 "Wiggly_MT"**: 80 tasks, one per `.mp4`, recorded 2026-02-19, conditions
  spanning pH 5.8–8.8 with MES/HEPES buffers, 4 channels × 4 positions.
- Each task holds **2–5 consecutive frames**, not the whole video. So real short sequences do
  exist — 80 of them — as unannotated images.
- **There is NO cross-frame microtubule ground truth.** The task metadata reports "tracks"
  (5, 5, 13 on sampled tasks) and that reading is a trap: exporting them shows the only label
  is `crop` and the shapes are **boxes that do not move** — task 438 has 5 boxes identical
  between frames 0 and 1. They are crop regions for cutting sub-images, not filament tracks.
  CVAT counts them as tracks because they are track-type objects; they track nothing.
- 252 of the 584 corpus frames were drawn from these videos and then deduplicated down to 2–5
  frames each, which is why the generator has only ever seen stills.
- **Displacement between consecutive frames is single-digit pixels** (user). With no GT this
  cannot be verified against annotations; it can be sanity-checked by running the current model
  on consecutive frames and measuring how far matched filaments move. That is self-referential
  and only good for order of magnitude — a large displacement would still be obvious.

Small displacement is the favourable case: centerline overlap between frames is large, so
association is geometrically easy and the arclength shift is measurable sub-pixel.

**Consequence for every claim in this document:** all quantitative tracking results will rest
on **synthetic sequences**. Real video supports qualitative inspection and an order-of-magnitude
displacement check, nothing more, unless someone annotates identity across frames. That is a
real limitation and must be stated wherever tracking numbers are reported.

## Architecture

One model, one generator, one instancer. The single frame is the **degenerate case of the
sequence**, never a separate code path — the deployed hub already shows what happens otherwise
(the v7 wrapper carries a comment about two copies that "silently drifted apart").

| stage | single frame | video |
|---|---|---|
| semantic | stack (t, t, t) | stack (t−1, t, t+1) |
| instancer | unchanged | unchanged, per frame |
| tracker | does not run (empty track set, not an error) | min-cost matching t → t+1 |
| readout | instances | identity, velocity, tips, temporal fusion |

**The instancer does not change.** It was frozen yesterday with intervals attached
(`params_a_model_synthtuned.json`, MT-34 TEST 0.457 [0.379, 0.533]); reopening it for an
unrelated reason would forfeit that.

### Hard gate on single-frame quality

The temporal semantic model is trained with **random temporal-context dropout**: some batches
get (t−1, t, t+1), some get (t, t, t). Single-frame performance is then measured directly on
MT-34 TEST against the current **0.457**, with a paired interval.

**If single-frame quality regresses beyond noise, the temporal model is not shipped** and the
delivery is the tracker alone (approach A), with the semantic stage untouched. This is a
stated pass/fail criterion, not an intention.

## Generator: sequences

`generate_sequence(background, rng, cfg, n_frames, dt)` → per-frame images plus per-frame
instance lists carrying **stable identities**. `n_frames=1` must reproduce `generate_frame`
exactly, so the still and video paths cannot drift.

Morphology is sampled once by the existing `sample_scene`; each subsequent frame **evolves the
centerlines** and re-renders. The generator already separates morphology from appearance
(`sample_scene` → `render_irm`), so this is an extension at an existing seam rather than a
rewrite.

### Motion model, per regime

The regimes already exist in `GenConfig` and carry their own physics:

- **static** (~44 % of frames) — surface-immobilised. Sub-pixel positional jitter only; shape
  fixed. Tests that the tracker does not invent motion.
- **gliding** (38 %) — motor-propelled. The filament **advances along its own contour**: shift
  the centerline by `v·dt` in arclength, extend the leading tip along the extrapolated
  direction and retract the trailing one, so length is preserved. Velocity is sampled per
  microtubule and held for the sequence. This is the regime the real videos are in.
- **dynamic** (18 %) — dynamic instability. Body anchored; each tip independently grows or
  shrinks under two-state switching with catastrophe and rescue rates.

Plus, in every regime, a **frame-global stage drift** shared by all microtubules. It is real,
and it is a confound the tracker must not mistake for motility — a tracker that reports drift
as gliding velocity is wrong in exactly the way that matters here.

Appearance is re-rendered per frame with the same background but **fresh noise**, so intensity
flicker and dropout are present frame to frame. That is what makes temporal fusion worth
anything: a noise-driven foreground dropout in frame t is not there in t±1.

### Correspondence ground truth

Free and exact: instance identity is the list index carried through evolution. Each frame
records centerline, per-instance arclength offset since the previous frame, and tip positions —
so velocity error and tip error are measurable directly, not inferred.

## Tracker

Structurally the instancer's junction matcher, applied between frames instead of within one:
**min-cost matching** between instances at t and t+1, with a priced "leave unmatched" option
that becomes track birth and death.

Cost between two centerlines from centerline overlap, arclength offset, and endpoint
continuity. Roughly five coefficients, fitted **on synthetic sequences** — the same choice that
beat real-annotation fitting by +0.041 [+0.018, +0.065] yesterday, and the one that keeps the
pipeline annotation-free.

No learned weights. Geometry first: at single-digit-pixel displacement the geometry is likely
to saturate, and a learned association would be a component whose proxy has not been validated
— the failure mode that cost 8 GPU-hours yesterday. Learned association is deferred until
geometry is measured to fail.

## Metrics

- **identity**: track fragmentation and identity switches against synthetic GT. There is no
  real equivalent — see the data section;
- **velocity**: error in px/frame against the sampled synthetic velocity, and specifically
  whether frame-global drift leaks into it;
- **tips**: endpoint position error and growth-rate error in the dynamic regime;
- **fragmentation healing**: instance F1 before and after temporal fusion, on synthetic
  sequences. MT-34 is stills, so it cannot measure this; the real 2–5-frame tasks can show it
  qualitatively only;
- **single-frame non-regression**: MT-34 TEST F1 vs 0.457, paired interval. The gate.

## Risks

- **There is no real track GT at all** (established, not feared — see the data section). Every
  quantitative tracking claim is synthetic-only. The honest options are to accept that and say
  so, or to have someone annotate identity across frames on a handful of the 80 real sequences.
  Annotating even 5 sequences would turn "synthetic-only" into "validated on real data", and is
  the single highest-value thing a human could contribute here.
- **The temporal model may cost single-frame quality.** Covered by the gate above.
- **The generator's motion priors are invented until calibrated.** Step 1 measures real
  displacement from the annotated tracks; until then every velocity number is synthetic-only.
- **Scope.** Four readouts is a lot. Order: sequences + tracker + identity first, because
  velocity, tips and fusion are all read off the same correspondence.
