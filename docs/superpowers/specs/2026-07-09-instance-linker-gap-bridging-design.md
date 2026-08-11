# Design — Phase 1: enrich the amodal instance linker with gap-bridging evidence

**Goal:** maximize strict centerline **instance-F1** on the Alice+HTW benchmark. Current best (v4b):
Alice 0.653 / HTW 0.234; oracle single-layer affinity = 0.86 → the linker/grouping is the recoverable gap.
**HTW is fully sealed** (no images, no GT) until one final scoring; tune only on Alice + synthetic proxies.

## Diagnosis
The learned linker (`learn_amodal.py`, `amodal_eval2.py`) decides "same MT?" for each per-orientation-channel
arc pair from **endpoint-local geometry only**: 7 features = [endpoint-overlap, gap/MAXGAP, arm-dir dot,
|Δchannel|, log len_i, log len_j, len-ratio] → MLP → greedy union-find by P. It never inspects whether signal
BRIDGES the two arcs, so it cannot resolve the two bottlenecks:
- **Close parallels:** near endpoints + parallel direction look linkable, but nothing connects them laterally.
- **Crossings:** one MT split across a junction needs linking *through* the crossing, where foreground bridges the gap.
The missing cue — evidence along the gap — is the SAME signal for both.

## Changes
1. **Enriched linker features (core lever)** — extend `pair_features` (7 → ~12), threading the
   max-over-channels probability map `pmax = channels.max(0)` through `frame_pairs`/`passemble`:
   - **Bridge continuity:** sample `pmax` along the straight segment between the two closest endpoints →
     `bridge_mean`, `bridge_min` (does signal actually connect them?).
   - **Collinearity offset:** perpendicular distance of each endpoint from the *other* arc's extrapolated
     tangent (normalized by MAXGAP) → separates parallels (high offset) from true continuations.
   - **Gap-vector alignment:** connecting-vector angle vs each arm direction → both ≈1 for a real continuation.
2. **Arc de-fragmentation (as needed):** merge collinear arcs within a channel + prune short spurs before
   pairwise linking (a wavy MT currently shatters across orientation bins).
3. **Gated grouping (as needed):** block a merge that gives one instance a branchy/3-way endpoint (an MT is a
   single path); otherwise keep greedy union-find by descending P.
4. **Retrain linker on the diverse multi-regime pool + hard-negative mining** (close parallels labeled negative).

## Validation (sealed-HTW-safe)
- Primary: **Alice instance-F1** (0.653 → target ~0.75), semantic no-regression.
- **Synthetic crossing/parallel STRESS set** (N parallels @ 2/4/6 px, X-crossings): directly measures both
  bottlenecks in-domain AND is the generalization proxy before the one-shot HTW confirm.
- **One-shot HTW** only at the end.

## Non-goals / YAGNI
No graph-cut/CRF (start with gated greedy). No generator/appearance changes in Phase 1 (DR-wide already shown to
hurt near-domain HTW). Phase 2 (semantic thin-filament decoder for the HTW-semantic gap) designed AFTER Phase 1 results.

## Artifacts
`learn_amodal.py` (enriched features + retrain), `amodal_eval2.py` (matching features), new
`gen_stress.py` (synthetic stress set) + `stress_eval.py`. Linker weights → `amodal_mlp_v7.pt`.
