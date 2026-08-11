# Instance segmentation research — point-cloud vs orientation for MT parallels & junctions

> Deep-research synthesis (2026-07-05, 100 agents, 18 primary sources, 20 verified / 5 refuted
> claims). Question: is a TARDIS/DIST point-cloud instance head better than orientation-field +
> arm-pairing or Object-Condensation for (a) separating CLOSE PARALLEL MTs and (b) resolving
> X-JUNCTIONS amodally — and how to train it synthetic-only without over-firing on real?
> The two bottlenecks (parallels, junctions) are the project's instance-seg priorities — see CLAUDE.md.

## Headline verdict
**A pure point-cloud / DIST head is NOT, by itself, better for 2D IRM microtubules.** In filament
mode DIST hard-codes a **linear chain with a max-degree-2 greedy graph cut** (keeps the 2 highest-prob
edges per node). A degree-2 node **cannot carry the 4 arms of a shared X-junction** on a single 2D
skeleton → every crossing collapses. **This is exactly the WEAVE failure** already hit (oracle GT
edges fragment to 0.16–0.27). It works in 3D cryo-ET only because projected crossings are z-separated.
**The fix is at the point-sampling / representation level, not the decoder.**

## What actually resolves crossings amodally (bottleneck #2 = junctions)
A **multi-layer / orientation-keyed** representation where a shared pixel belongs to >1 instance:
- **ORION "Intersection→Overpass"** (Liu et al., CVPRW 2019, PMC8046259): 6 duplicate hourglass
  branches, each a 30° orientation bin ([0,30]…[150,180]); the two crossing filaments (different
  propagation directions) sort into different output layers → the X becomes an "overpass". Trained
  **purely on synthetic** (180k straight lines, control-point deformed, 6 buckets). **This is the
  project's ORION 0.519 baseline.** Reaches F1 0.7161 on its own 10-image real MT set (SKIoU-F,
  thresholds 0.5–0.95 — NOT directly comparable to the strict 0.326/0.519/0.696 Alice bar).
  Limit: only crossings whose filaments fall in DIFFERENT bins split; shallow-angle same-bin
  crossings need a terminus/arm-pairing rejoin step (so arm-pairing is intrinsic, not a competitor).
- **Layered embeddings / bilayer masks** (Layered Embeddings ICIAR 2019 arXiv:2002.06264; BR-Net
  arXiv:2411.17557): assign occluded pixels to a dedicated 2nd layer. **HARD two-per-pixel cap** —
  where 3+ filaments share a pixel it recovers at most 2. A real ceiling for dense IRM mesh regions.

## Separating close parallels (bottleneck #1)
- **AVOID fixed-radius push-pull embedding** (De Brabandere discriminative loss; Object-Condensation
  family): a fixed inter-cluster margin below which adjacent parallels cannot be resolved → intrinsic
  parallel-merge (the project's OC F1 ~0.01). Confirmed mechanism, arXiv:2507.23359.
- **PPGNet path-conditioned connectivity** (Zhang et al., CVPR 2019): decide "same instance?" from
  features sampled ALONG the candidate path (64 equidistant bilinear samples → conv → connection
  prob), not from embedding distance. Domain-robust. Caveat: if along-path sampling is too coarse it
  "ignores the gaps and predicts all junctions connected" → sampling must be fine enough to see the
  physical inter-filament gap; documented failure is co-linear, and it samples STRAIGHT segments (needs
  adaptation to curved MT centerlines).
- **Mutex Watershed** (Wolf et al., TPAMI 2020, arXiv:1904.12654): partition an affinity graph with
  attractive + long-range **repulsive ("mutex") edges**; two nodes joined by an active mutex edge can
  **NEVER merge** (constraint inherited on cluster merge). **Hard separation, not a soft margin** —
  the structurally correct way to keep close parallels apart. For MTs: short-range attractive affinity
  along each centerline + long-range repulsive affinity across the gap between adjacent parallels.

## Why our DIST fine-tune over-fired 6× on real
Native TARDIS injects synthetic data at the **point-cloud / coordinate level** (50% simulated,
training-only); the SO(n)-invariant graph head **never sees images**. We fine-tuned the **image FNet**
on synth images → the domain gap lives in the image encoder → 6× over-fire → DIST NaN. **Fix: derive
points from the already-clean DINOv2 semantic foreground (which does NOT over-fire) and train ONLY the
coordinate-level connectivity head on synthetic point clouds.** (Confidence: medium — causal
attribution is synthesis, not a cited experiment.)

## Recommended build (ranked by expected instance-F1 gain per effort; engineering synthesis, unquantified)
1. **Extend the 0.519 ORION overpass:** from the frozen DINOv2 foreground derive an **orientation-keyed,
   MULTI-point-per-pixel** point cloud (at each pixel emit one point per locally-supported tangent/
   orientation bin) → a crossing yields two points routed to two orientation layers → **defeats the
   degree-2 collapse** without abandoning a point-cloud head.
2. **Replace collinear arm-pairing heuristic with LEARNED per-orientation-layer connectivity** —
   DIST-style pairwise edges run independently within each orientation layer (chains stay degree-2, but
   crossings survive as two chains in different layers), OR PPGNet-style path-conditioned connectivity
   sampling DINOv2 features along the candidate centerline (fine sampling to resolve gaps).
3. **Close parallels: hard separation via Mutex-Watershed long-range repulsive / mutex edges** between
   adjacent chains — NOT a fixed-radius embedding margin.
4. **Train the connectivity head purely on free synthetic polyline GT at the POINT-CLOUD level; keep
   DINOv2 frozen** so nothing over-fires on real.
5. **Dedicated metrics:** bundle-recovery (are N parallels at 2–6 px gaps recovered as N instances?) +
   per-junction identity-preservation (fraction of X-crossings where both filaments keep continuous
   identity), alongside the strict centerline-F1 clean/chimera breakdown.

## Direct precedent found
**SynthMT** — "Synthetic data enables human-grade microtubule analysis with foundation models for
segmentation" (PLOS Comput Biol, 2026): 6,600 synthetic IRM MT images, params tuned to real, foundation
models. The closest published analogue to this project's pipeline.

## Refuted (voted down 0-3 — do NOT rely on these)
- DIST's dimensionlessness injects NO appearance info (refuted — it's more nuanced).
- "2D micrograph MT is unimplemented in TARDIS" (refuted — it is supported; re-check live repo).
- "PPGNet junction-graph encodes arbitrary crossings losslessly" (refuted).
- "TARDIS separates parallels via a distance-transform threshold at fixed nm sizes" (refuted).
- "Tubular-neurite method clusters micro-segments locally then links" (refuted).

## Open questions (verify empirically before committing)
1. Does an orientation-keyed multi-point-per-pixel cloud actually keep two crossing MTs as two
   separate degree-2 chains through the X on real Alice? How many bins (ORION=6) before shallow-angle
   same-bin crossings stop collapsing?
2. What along-centerline sampling density / repulsive-edge range resolves real IRM parallels 2–6 px
   apart at 1.5× upscale, given the DINOv2 patch-token resolution, without re-merging?
3. Is our DIST over-firing genuinely from the image FNet (fixable by training only a coordinate-level
   head on the clean DINOv2 foreground), or does the point cloud over-fire even from a clean foreground?
4. For 3+ MTs sharing a pixel, does any amodal representation scale beyond the two-per-pixel cap?

## Key sources
TARDIS/DIST: MLSB 2022 (mlsb.io) + biorxiv 2024.12.19.629196 + github SMLC-NYSBC/TARDIS. ORION:
PMC8046259. Pick-and-Trace (synthetic-only recurrent tracing, MICCAI 2023): link.springer 978-3-031-
43993-3_61. PPGNet: CVPR 2019 (Zhang). Mutex Watershed: arXiv:1904.12654. Layered Embeddings:
arXiv:2002.06264. BR-Net: arXiv:2411.17557. SynthMT: PLOS Comput Biol 10.1371/journal.pcbi.1013901.


---

## UPDATE 2026-08-10 — curvature-bounded instancing (implemented, `src/instance/`)

New question from the user: PySOAX works but **breaks microtubules at crossings**; the derivative
along a predicted polyline must be bounded. Literature search + source reading of PySOAX.

**The gap is real.** No standard filament tracer imposes a hard curvature bound or solves the
junction as a unit: SOAX/TSOAX stop snakes at intersections and link by **minimum angle**; SIFNE
excises a 2x2 region around each intersection and reassembles by orientation continuity (60 deg),
explicitly failing on high curvature; KnotResolver builds a cross-over graph but still pairs by
minimum angle. All are local, greedy, soft-threshold rules.

**The right formalism is the orientation lift.** Bounding ``kappa = dtheta/ds`` lives naturally in
R^2 x S^1, not on a 2D skeleton:
- Curvature-penalised minimal paths / Finsler elastica -- Chen & Cohen, IJCV 2017 (arXiv:1612.00343),
  CVPR 2016. Orientation-lifted metric, global minimum via fast marching, built for tubular structures.
- Sub-Riemannian geodesics in SE(2) -- Bekkers & Duits, CIARP 2015 (arXiv:1508.02553), arXiv:1704.04192.
  Sideways motion forbidden, turning costs; developed for retinal vessel CROSSINGS.
- Invertible orientation scores (Franken & Duits) -- lifting the image to (x, y, theta) "un-crosses"
  crossings. **Our K=6 orientation head is exactly a learned, discretised orientation score**, which
  is the connection that makes instancer B natural rather than novel-for-its-own-sake.
- Global path selection by integer programming -- Turetken & Fua, CVPR 2013 / TPAMI 2016 -- remains the
  escalation if per-junction matching proves insufficient.

**What was built and what it settles.** See `docs/protocol.md §17` for the full path, the four
structural causes of PySOAX's crossing breakage, the derivation of kappa_max = 0.25 rad/px from the
957 MT-34 ground-truth microtubules, and the oracle-mask results (PySOAX 0.684 -> A 0.862 pooled;
junction identity 0.045 -> 0.78).

**Two limits that are now quantified rather than asserted:**
1. *Shallow crossings are mask-width-bound, not algorithm-bound.* Two bands of half-width r crossing
   at angle alpha stay skeletally fused over ``L ~= 4r/sin(alpha)`` (predicted 15.5 px at 15 deg,
   measured 14.4). Instancer A's `bridge_max_len` follows from this, and raising it also absorbs
   genuine short segments between nearby crossings.
2. *Instancer B's angular consumption width sets the same wall from the other side*: two filaments
   crossing at less than ``consume_deg`` are absorbed into one. Report junction identity BY CROSSING
   ANGLE (the metric does) rather than pooled, or this limit hides.

**Open question 3 from the original research is answered.** "Is the over-firing from the image FNet,
or does the point cloud over-fire even from a clean foreground?" On MT-34 the v4b foreground does NOT
over-fire (predicted fg% 1.93 vs ~1.6 in-domain synth) and semantic recall is 0.92-0.999; the
model-mask instance gap comes from over-segmentation of a thicker/noisier foreground plus a field
stop the generator never modelled -- not from over-firing.
