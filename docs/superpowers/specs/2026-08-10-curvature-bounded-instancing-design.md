# Curvature-bounded instance segmentation of microtubules — design

> Date: 2026-08-10. Status: APPROVED by user (2026-08-10), autonomous execution authorized.
> Companion docs: `docs/INSTANCE_SEGMENTATION_RESEARCH.md`, `docs/protocol.md`, `docs/TODO.md`.

## 1. Problem

Turn a (very strong) **semantic** microtubule foreground into **instances**. The current
tooling — PySOAX (`~/BIOCEV/code/microtubules/convnext_instance_seg/pysoax.py`, a pure-Python
reimplementation of SOAX) — works well on isolated filaments but **breaks microtubules at
crossings**. Broken/kinked microtubules do not occur physically: MT persistence length is
millimetre-scale and the documented *breaking* curvature is ~0.43 µm⁻¹, so a corner in a traced
centerline is always a tracer artifact.

**Governing constraint (user):** the derivative along a predicted polyline must be bounded —
`κ(s) = dθ/ds ≤ κ_max`. No sharp kinks, ever.

**Standing constraint (user, 2026-08-10):** synth-only still holds. The instancer and the
semantic model train on synthetic data only; the new human annotations are **test data only**.
Hyperparameters may be tuned on a real VAL split (existing project convention), never on TEST.

## 2. Root-cause analysis of PySOAX (from reading the source)

Four structural causes, none of which is a hyperparameter:

1. **Single-layer skeleton with first-come-first-served junction ownership.**
   `_extract_paths_from_skeleton` traces from endpoints marking `used[]`. Junction pixels are
   consumed by whichever path arrives first; the second MT through an X finds them used and
   `break`s → fragments. A 2D skeleton cannot represent an amodal crossing.
2. **45°-quantized direction estimate.** In `_trace_path` the "straightest continuation" is
   computed from `direction = current - prev`, a single 8-connected step, compared against
   candidate single steps. Both are quantized to 8 directions → shallow-angle crossings
   (20–30°) are decided essentially at random.
3. **Greedy, order-dependent, no global consistency.** Nothing forces the 4 arms of an X to be
   paired into 2 consistent through-paths; a wrong choice is never revisited.
4. **Skeletonization deforms junctions.** `skeletonize` turns an X into two degree-3 nodes
   joined by a short bridge (Y–Y), which biases the pairing and injects a spurious kink.

The re-link step (`_link_and_output`, `grouping_distance`, `direction_threshold=0.87 ≈ 29°`) is
an endpoint-local *soft* filter: it bounds neither curvature along the resulting polyline nor
junction-level consistency.

## 3. Literature grounding

The correct home for a bounded-curvature constraint is the **orientation-lifted space** ℝ²×S¹,
not the 2D skeleton.

- **Curvature-penalized minimal paths / Finsler elastica** — Chen & Cohen, IJCV 2017
  (arXiv:1612.00343), CVPR 2016. Lift to (x,y,θ); the metric penalizes |dθ/ds|; global minimum
  via fast marching. Designed for tubular structures.
- **Sub-Riemannian geodesics in SE(2)** — Bekkers & Duits, CIARP 2015 (arXiv:1508.02553),
  arXiv:1704.04192. Sideways motion forbidden, turning costs. Developed for retinal vessel
  **crossings and bifurcations**.
- **Invertible orientation scores** — Franken & Duits. Lifting the *image* to (x,y,θ)
  "un-crosses" crossings because the two filaments occupy different θ.
  **Our existing K=6 orientation-keyed head is a learned, discretized orientation score.**
- **SIFNE** — excises a 2×2 region around each intersection, then reassembles fragments by
  orientation continuity (60° threshold) + gap-direction agreement. Explicitly cannot reassemble
  high-curvature fragments.
- **SOAX/TSOAX** (Sci Rep 2019) — snakes stop at intersections, then link by **minimum angle**.
  Same greedy logic as PySOAX.
- **KnotResolver** (Bioinformatics 2024, arXiv:2404.12029) — directed graph, nodes = cross-overs,
  edges = paths; minimum-angle pairing but decided over the graph.
- **Integer programming over path graphs** — Türetken & Fua, CVPR 2013 / TPAMI 2016. Global
  subset selection with topological constraints; handles loops.
- **ORION Intersection→Overpass** (CVPRW 2019) — orientation layers; the project's 0.519 baseline.

**Gap identified:** no standard filament tracer imposes a *hard curvature bound* or performs
*global arm matching* at a junction. All use local minimum-angle heuristics. The user's
observation targets a real methodological gap.

## 4. Benchmark — "MT-34" (Phase 0)

| source | frames | polylines | note |
|---|---|---|---|
| CVAT task 585 (`v7_pysoax_alice_2026-05-07`, job 557), re-export | 12 | 229 | supersedes the stale local h5 (231) |
| CVAT task 586 (`v7_pysoax_general_mt_2026-05-07`), frame ids 0–21 | 22 | 728 | ~33 MT/frame vs Alice ~19 |
| **total** | **34** | **957** | |

**Established by inspection, not assumption:**
- Task 585 *is* the Alice eval set (same 12 files, per-frame polyline counts match the local h5).
  CVAT was updated 2026-06-04 with human fixes → the CVAT version is authoritative.
- In task 586, human edits (`source="manual"`) stop exactly at frame id 21. "First 22" therefore
  coincides with the fully human-reviewed block. From id 22 onward everything is raw
  `source="file"` model output.
- Coordinate scale: **CVAT exports at native 1×**, like Alice (Alice GT max x = 1118.1 on a
  1120-wide image). HTW's 1.5× is the outlier. `CLAUDE.md`'s "both GTs at 1.5×" is wrong for
  Alice; the 1.5× is an *eval-pipeline* upscale convention, not a property of the Alice data.
- Mixed frame sizes: 1024×1024, 1024×1022, 1383×1341, 1192×1192 (586) and 1120×938, 1416×1344
  (Alice). The pipeline must handle per-frame sizes.

**Edge-case frames:** id 4 (`training_img_102`) has 0 annotations → kept, scored in micro-F1
only (counts false positives), excluded from the macro mean (per-frame F1 undefined). ids 7 and 8
(`training_img_105`, `training_img_106`) have 2 and 5 polylines, all `source="file"` — never
touched by a human. Kept (user asked for the first 22), with a **sensitivity number reported on
the 19 fully-reviewed frames**.

**Format:** `data/real/mt34_eval/` mirroring `alice_eval`: per-frame `.h5` with `image` +
`polylines/pl_XXXX` in `(x=col, y=row)`, plus `tif/`. Native 1× coordinates.

**Verification gate (blocking):** render a polyline-on-image overlay for all 34 frames into
`data/enc_sensitivity_testset/mt34_overlays/`. No metric is trusted before this is inspected —
it catches both the `(x=col,y=row)` transpose and any scale error.

**Split:** extend the existing convention. Alice 6 val / 6 test (existing deterministic
alternating split), new-22 → 11 val / 11 test, alternating by sorted filename. All tuning
(κ_max, cost weights, Optuna) on VAL only; TEST scored once at the end.

**κ_max estimation:** resample GT polylines at constant Δs, compute the |Δθ|/Δs distribution over
all 957 MTs, take a high quantile (~99.5%). Cross-check against the literature breaking curvature
0.43 µm⁻¹. The histogram is a paper figure ("microtubules physically cannot kink").

**Benchmark characterization (addresses TODO W9):** count GT crossings and their angles, and
parallel bundles (pairs within 2–6 px over ≥20 px), separately for Alice and the new 22. This is
the first dataset on which the N2 claim (crossings/parallels) can be measured at all.

**Documented caveats (not silently absorbed):**
- **Leakage:** all 22 frames are in the generator calibration corpus
  (`training_img_114.tif` = `data/real/mt_corpus/images/0015_local_irm_tif_img_114.tif`).
  Harmless for instancer tuning; for the paper they must be carved out of calibration and the
  generator recalibrated, exactly as was done for HTW. Tracked as a TODO, non-blocking.
- **Agreement bias:** the GT is human-corrected v7+PySOAX output, so PySOAX-family instancers
  are flattered. Mitigation: report error rates separately on `manual` vs `file` polylines.
- **Coarse GT polylines:** median 5 vertices/polyline in task 586. Fine at tol=5; do not tune the
  tolerance down.

## 5. Metrics and oracle setup (Phase 1)

**Primary metric unchanged:** strict centerline-F1 (`centerline_f1.py`, tol=5,
length/precision coverage 0.95) at 1.5× upscale, per-set (Alice / new-22) + pooled, clean vs
chimera. Changing it would break comparability with 0.697 / 0.519 / 0.326.

**New diagnostic metrics** (what makes this a contribution rather than tuning):
1. **Junction identity preservation** — for each GT crossing, does each of the two MTs keep a
   single predicted id through it? Broken down by crossing angle (shallow vs perpendicular).
2. **Fragmentation rate** — predicted instances per GT MT.
3. **Bundle recovery** — N parallels at 2–6 px gap recovered as N instances?
4. **Max |Δθ|/Δs of predicted polylines** — proves the constraint holds by construction.

**Oracle masks:** rasterize GT polylines at half-width 1.0 (matching the `mask_hw=1.0` training
convention) at 1.5× → binary. Also synthesize **oracle K=6 orientation channels** from GT tangents
(amodal: a crossing writes into two channels). This isolates the algorithm from the segmenter for
both instancers.

**Baseline run:** current PySOAX on oracle masks, with **error attribution** into the three causes
(junction fragmentation / wrong pairing / gap). This decides the relative weight of A vs B and is
the "before" number for the paper.

## 6. Instancer A — curvature-bounded junction matching

1. Skeletonize the mask, build the pixel graph.
2. **Junction-cluster contraction:** degree-≥3 nodes within ~3 px collapse into one junction node,
   eliminating the Y–Y bridge artifact (cause 4).
3. **Arcs** = maximal degree-2 chains between junction/endpoint nodes; smoothed and resampled at
   constant Δs.
4. Per arc-end at a junction: tangent by **PCA/total-least-squares over a window of L px**
   (L ≈ 8–16, tuned) and curvature κ_in by circle fit over the same window (fixes cause 2).
5. **Per-junction min-cost perfect matching** over the arm-ends
   (`networkx.max_weight_matching`, blossom):
   `cost(i,j) = w₁·|Δθ| + w₂·|κᵢ − κⱼ| + w₃·(through-junction gap length)`,
   with a **hard forbid** when the implied |Δθ|/Δs > κ_max, plus a "leave open" option at fixed
   cost `c_open` so genuine T-junctions and MT ends survive (fixes cause 3).
6. Chain matched arcs into instances (union-find over the pairings).
7. **Gap linking** across foreground holes: same cost + hard κ bound + a foreground-continuity
   bridge check (the feature that gave +0.044 on Alice in the existing linker).

Differences vs PySOAX: window tangent instead of 1 px · junction as a unit with global matching
instead of greedy · hard κ bound instead of a soft 29° threshold · no `used[]` race.
Tuned with Optuna on oracle VAL.

## 7. Instancer B — orientation lift (x, y, θ)

- **Nodes** = (pixel, θ-bin), from oracle GT channels or the v4b K=6 head. K=6 (30°) is coarse →
  refine θ by circular interpolation in the doubled-angle representation to ~12–18 bins.
- **Edges:** (p,θ) → (p′,θ′) where p′ is a spatial neighbour in direction ≈θ and
  |Δθ| ≤ κ_max·Δs.
- **Cost:** `−log p_fg(p,θ) + λ(Δθ)²/Δs` — the discrete analogue of the Finsler-elastica metric.
- **Extraction:** seeds at high-confidence tips; curvature-penalized shortest path (Dijkstra);
  take the best path, **remove only its (p,θ) nodes — not the whole pixel** — and repeat. This is
  where the amodal win comes from: removing (p,θ₁) leaves (p,θ₂) free for the crossing MT
  (fixes cause 1).

**Why this differs from the earlier per-layer attempt that scored 0.11:** that segmented each
orientation bin *independently*, so a wavy MT sweeping its tangent through all bins shattered into
~25 arcs. Here it is **one joint graph** in which a bin transition is a legal, priced edge, so a
wavy MT traverses it as a single path. The cut moves from *between bins* to *between instances*.
This must be verified explicitly on wavy synthetic frames, not assumed.

## 8. Model masks and nnU-Net (Phase 3)

- Run A and B on v4b predictions (`dino_seg_ori_v4b.pth`, semantic union + K channels). The
  oracle→model drop attributes residual error to segmentation vs instancing.
- **nnU-Net trained on the same synthetic data as v4b** (preserving synth-only), as a **candidate
  for the primary semantic model** (user decision, 2026-08-10). 2D configuration, residual-encoder
  preset, full resolution, same `mask_hw=1.0` GT. Specific risk to watch: nnU-Net auto-configures
  patch size and downsampling, which can coarsen 2 px filaments — the exact mechanism that sank
  v8/ASPP (Alice tol2 0.940→0.914, fg% rose). Therefore track tol1/tol2 and the over-firing fg%
  ratio, not Dice. If it wins on the extended benchmark it becomes primary; if it loses it is the
  nnU-Net baseline reviewers will ask for either way.

## 9. Code, tooling, deliverables

- `src/instance/` — the instancer (curvature-bounded matching + lifted tracer). CPU-side, belongs
  in the repo since it is the paper's method.
- The repo currently has **no build/test harness**; add a minimal `requirements.txt` + `pytest`
  setup and document the commands in `CLAUDE.md`, per its own instruction.
- GPU work (v4b inference, nnU-Net training, Optuna sweeps) runs on **tulen**; code is mirrored to
  `~/mt_enc_exp/scripts/`.
- `data/real/mt34_eval/` + README documenting format, split, scale, and caveats.
- Living-doc updates: `docs/TODO.md`, `docs/protocol.md`,
  `docs/INSTANCE_SEGMENTATION_RESEARCH.md`, `docs/PAPER_PLAN.md`.

## 10. Risks

| risk | mitigation |
|---|---|
| GT agreement bias (PySOAX-seeded) | report `manual` vs `file` error rates separately |
| Calibration-corpus leakage | documented; carve-out + recalibration tracked as TODO for the paper |
| K=6 too coarse for shallow crossings | circular interpolation to 12–18 θ bins; measure identity preservation vs crossing angle |
| Lifted graph too large / slow | restrict nodes to foreground pixels; sparse Dijkstra; per-frame timing budget |
| Wavy MTs fragmenting across θ bins (prior 0.11 failure) | joint graph with priced bin transitions; explicit verification on wavy synth frames |
| Mixed frame sizes + 1.5× convention | per-frame handling; blocking overlay verification gate |
| nnU-Net coarsening thin filaments | track tol1/tol2 + fg% over-firing ratio, not Dice |

## 11. Scope boundaries

Not in scope: retraining or re-calibrating the generator; changing the primary metric; touching
the HTW set (sealed); self-training or any training on real annotations.
