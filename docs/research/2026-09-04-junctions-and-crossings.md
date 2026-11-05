# Junctions, crossings and close parallels — literature research (2026-09-04)

Scope: how methods from 2018–2026 keep a thin curvilinear structure's identity through an
X-crossing, and how they keep two close parallels apart; what of that transfers to THIS
project (synthetic-only nnU-Net ResEnc-M, binary 3 px centerline target, skeleton → arc graph →
per-junction min-cost matching under a hard `kappa_max = 0.25 rad/px`, strict centerline-F1).
Every recommendation is mapped against the closed dead-end list in the brief and against the
PatchPerPix design that is already specified (`docs/superpowers/specs/2026-08-27-patchperpix-instancer-design.md`,
designed but not run). All external text was treated as data; numbers are quoted from the
papers' own tables or from this project's diagnostics, and every claim carries an arXiv ID / DOI.

Read §0 (what is actually broken) before the table — several published "crossing solutions"
solve a problem this project does not have.

---

## 0. The measured mechanism the recommendations must address

The brief's headline numbers (roi303 VAL, 141 human-drawn frames, 2329 crossings): junction
identity **0.323** with the model mask vs **0.749** with an oracle mask through the *same*
instancer; bundle recovery **0.352** vs **0.706**. Two project diagnostics sharpen this into a
mechanism (`scripts/diag_crossing_map.py` → `data/enc_sensitivity_testset/crossing_diag_native/crossing_diag.json`,
SPARSE35 at native scale, thr 0.98, n = 1083 GT crossings; and
`docs/research/2026-09-02-scale-and-gap-audit.md`):

| quantity (roi303 VAL, native, thr 0.98) | value |
|---|---|
| per-crossing identity, model mask / oracle mask | **0.218** / 0.791 (n = 1083) |
| both arms of both filaments in ONE connected component of the thresholded mask ("mask-level identity") | **0.173** |
| failures where the mask is DISCONNECTED across the crossing | **97.6 %** (847 failures) |
| failures where the min probability within ±4.5 px of the crossing falls BELOW the decode threshold on at least one filament | **82.8 %** |
| failures where an arm is not found in the mask at all (±15 px) | 62.1 % |
| failures the oracle mask gets right | 75.1 % |
| mask area in a 7 px disc at the crossing, relative to the oracle mask: identity-OK / failed | 1.00 / **0.49** |
| identity, same-polarity crossings / OPPOSITE-polarity crossings (IRM dark×bright) | 0.233 (n = 973) / **0.082** (n = 110) |
| identity by crossing angle: shallow < 30° / oblique / steep > 60° | 0.360 / 0.166 / **0.154** |
| identity by probability dip at the crossing: < 0.02 / 0.02–0.2 / 0.2–0.5 / > 0.5 | 0.31 / 0.13 / 0.11 / 0.05 |

Two estimators, not one: the brief's 0.323 / 0.749 are `instance.metrics.junction_identity` on the
1.5× eval frame at thr 0.99 (one predicted instance must cover both arms of a GT filament); the
0.218 / 0.791 above are `diag_crossing_map.py`'s per-crossing statistic at NATIVE scale, thr 0.98,
with ±15 px arm samples and a ±2 px ridge tolerance. They are not comparable with each other and
neither says native inference is worse at crossings; the diagnostic's value is the failure split.

Reading: on the human-drawn block the crossing is not a *blob* that confuses the skeleton
(that was the MT-34 picture, where the 586 half already sits at 0.752 model vs 0.917 oracle);
it is an **erosion**. The probability ridge is present (semantic coverage of the GT centerline
at tol 2 has p50 = 1.00 at thr 0.5) but it is not ≥ 0.98 through the crossing and along fainter
arms, so the thresholded mask breaks into four free ends (or three, when an arm is missing), the
junction never forms in `build_arc_graph`, and the instancer is left to gap-link across a hole
where the image itself carries no ridge (opposite-polarity cancellation) or a weak one. The
failure montage (`crossing_failures.png`) shows exactly this: clear ridges in the probability
map, fragments in the 0.98 mask, half-empty instance panels.

Two consequences for ranking:

1. **The lever is what the network emits at and near crossings, and how the decoder treats the
   crossing neighbourhood** — not a better arm-pairing rule. With an oracle mask the existing
   matcher already keeps identity in 75–79 % of crossings; the pairing rule is not the
   bottleneck (consistent with the closed "learned MLP arc-pairing" null).
2. **On this diagnostic steep crossings fail MORE than shallow ones (0.15 vs 0.36)** — the
   opposite of the MT-34 profile (0.808 steep vs 0.656 shallow). Caveat before building on it:
   at a shallow X the two filaments' ±15 px arm samples nearly coincide, so one fused instance can
   cover both and the shallow rate may be tolerance-inflated (the same caveat `docs/protocol.md` §17
   raised for PySOAX's shallow 0.258); recompute with a larger arm-sample distance before quoting
   the angle profile as a mechanism. Steep X's are where the union target is
   a compact "+", where cancellation of opposite polarities is spatially concentrated, and where
   the eroded mask leaves the shortest arms to skeletonise. Any orientation-based trick helps
   shallow crossings; it does nothing for this majority.

For close parallels the audit found the ridge confidence itself is depressed at 4.5–6 px gaps
(pair ridge max 0.964 < thr 0.99, while two ridges do exist at 0.5 in 86 % of stretches), and
that separable pairs (< 7.5 eval px) are rare on roi303 — 19 stretches in 141 frames — against
26 near-coincident annotation duplicates that no map can separate. Bundle recovery 0.35 → 0.7
therefore buys little F1; it is ranked accordingly below.

---

## 1. Ranked recommendations

Costs are implementation hours on this codebase (target generation in `scripts/train_v5.py`
`ori_channels_soft` / `OnlineDS`, head in the ResEnc-M wrapper, consumption in
`src/instance/skeleton_graph.py` / `matching.py` / `instancer_a.py`), not training time. Gains
are what the source papers report; the "why it addresses" column is an argument, not a
forecast — every row must be screened under the pre-registered campaign protocol
(`docs/TODO.md` §0.0-CAMPAIGN: per-arm threshold sweep, paired bootstrap, Holm, judged on
junction identity + bundle recovery + coverage-only counts, not F1 alone).

| # | Change (what the generator supplies for free → what the network emits → what the decoder does) | Source papers | Reported gain there | Why it targets junction identity 0.32 → ~0.75 / bundle 0.35 → ~0.7 | Cost (h) | Main risk |
|---|---|---|---|---|---|---|
| **1** | **Crossing-region supervision**: per-pixel multiplicity/overlap class (0 / 1 / ≥ 2 instances) as a second head, plus a region weight ×(4–8) on the centerline loss inside r = 6 px of every GT crossing point. Target: `n(x) = #instances whose 3 px raster covers x`; overlap := centerline-to-centerline distance ≤ 2 px (not raster overlap, so 3–9 px parallels are NOT overlap). | PatchPerPix "overlap" class (arXiv:2001.07626 §2.3); MultiStar overlap probability (arXiv:2011.13228); overlapping-chromosome 4-class net, overlap-region IoU 94.7 % (arXiv:1712.07639); OASIS overlap-region loss weighting (arXiv:2510.06194); Layering paper's overlap-completion phase (arXiv:2210.03551) | chromosomes: overlap IoU 0.947; OASIS: overlap-region errors cut ~3× (−41 % → −13 %); MultiStar/ppp: enables amodal output at all | 83 % of failures are a sub-threshold dip within 4.5 px of the crossing and 62 % a missing arm; the union target gives crossing pixels no special status and the BCE is dominated by background and easy ridge pixels (foreground is ~2 % of a frame, crossing neighbourhoods a small fraction of that). An explicit overlap class + region weight is the direct fix for "the network under-fires exactly where two filaments meet", including opposite-polarity cancellation (identity 0.08), which the network can only learn to bridge if the loss makes those pixels matter. Also yields the junction map that #2 consumes. | 6–10 | Over-firing of the ≥ 2 class on close parallels (looks like overlap at 1.5×) → guard with the centerline-distance definition and the bundle-recovery guardrail; region weight raises FP fragments at crossings if set > 8. |
| **2** | **Junction-conditioned decode**: use the predicted overlap/junction map J (from #1, or a standalone Gaussian junction heatmap σ = 2 px) to (a) seed junction contraction in `build_arc_graph` at local maxima of J with `p_J > 0.5`, (b) apply a LOCAL low threshold (0.5) inside a disc of radius R = max(6, 4r/sin α) around each seed so the four arms connect, (c) skip gap-linking and go straight to `match_junction` for arms whose tips lie inside a seed disc. | Sat2Graph vertexness + edge-slot decode (arXiv:2007.09547 §3.1: vertex local maxima above `p_thr`, edges by distance + cosine cost, w = 100); L-CNN junction heatmap + line verification (arXiv:1905.03246); Uslu & Bharath multi-task junction detection re-verified on the centerline map (arXiv:1806.03175); ORION terminus pairing (DOI 10.1109/CVPRW.2019.00021) | Sat2Graph: stacked-road crossing precision 83.1 %, false-overpass rate 0.37 %; Uslu: outperforms a plain CNN junction classifier on DRIVE/IOSTAR (no crossing-specific number) | 97.6 % of failures are a disconnected mask at the crossing; the existing junction contraction only fires when the skeleton itself has a degree ≥ 3 node. A predicted junction is evidence the mask does not carry. Local hysteresis differs from the closed global hysteresis (null, +11 TP on MT-34; −32 TP at 0.9 on roi303) because it is applied only where a second head says two filaments meet, so it cannot fuse parallels elsewhere. | 4–6 (+ #1) | If J is wrong at parallels the disc fuses them → cap R, require ≥ 3 arms inside the disc before lowering the threshold; evaluate on junction identity BY ANGLE and on bundle recovery, never F1 alone. |
| **3** | **Soft ridge / distance-regression target with max-over-instances** instead of the hard 3 px union: `y(x) = max_i exp(−d_i(x)² / 2σ²)`, σ = 1.0 px at training scale, L1/L2 (or BCE against the soft target), decode by ridge non-max suppression perpendicular to the local tangent rather than a global threshold. | Sironi et al. centerline distance regression (CVPR 2014, DOI 10.1109/CVPR.2014.351; TPAMI 2016, DOI 10.1109/TPAMI.2015.2462363); DeepFlux context flux (arXiv:1811.12608); TextField direction field (arXiv:1812.01393); FlowSDF / CurvSegFlow distance-transform flow matching (arXiv:2606.21608) | Sironi: regression beats classification because "locations on centerlines and immediately next to them are extremely difficult to distinguish"; DeepFlux +0.8 (SK-LARGE), +3.5 (WH-SYMMAX), +4.8 (SYM-PASCAL) F-points over the best binary-classification skeleton detectors | The audit shows the ridge is 2.1 px wide at p > 0.99 while the target is 3 px and the filament 5.5 px; a hard threshold converts a 0.04 confidence dip into a break. A regression target makes the ridge LOCATION the supervised quantity and its height a smooth function of distance; `max` over instances (not union) keeps y = 1 through crossings on both filaments. Ridge NMS is threshold-free → removes the "broken at 0.99 / fused at 0.5" dilemma the TODO names. Interacts with the THIN arm already in the factorial (1 px target: bundle 0.82 → 0.95 in an ORACLE-mask screen on synth; the trained THIN arm delivered only +0.0095 F1). | 6–8 | Changes calibration (the pre-declared thresholds no longer apply — sweep NMS height instead); regression targets can blur tips; must be run at both inference scales. |
| **4** | **Continuous orientation field head** `(cos 2θ, sin 2θ)` per foreground pixel (2 channels, unit-norm, loss = 1 − cos(2(θ̂−θ)) or L2 on the doubled-angle vector), masked out where multiplicity ≥ 2; consumed as GEOMETRY: (a) replace `window_tangent`'s heading/curvature at each `ArmEnd` by the field average over the arm's last `window` px (robust to skeleton wobble near an eroded junction), (b) optional orientation-lifted flood: from each arm tip, integrate the field through the junction disc and record which opposite arm it lands on as an `allow` predicate. | Batra et al. joint orientation + segmentation for road connectivity (CVPR 2019, CVF open access); orientation scores / lifting to R²×S¹ un-cross crossings (Bekkers et al. JMIV 2014, arXiv:1212.3530; Franken & Duits IJCV 2009 DOI 10.1007/s11263-009-0213-5; SE(2) group CNN lifting layer arXiv:1804.03393); ORION's synthetic-only orientation branches transfer to real MTs (DOI 10.1109/CVPRW.2019.00021) | Bekkers 2014: 109/114 crossings (95.6 %) correctly tracked on HRF; ORION: F1 0.716 vs SOAX 0.269 / SIFNE 0.288 on its 10 real MT images (SKIoU-F, thresholds 0.5–0.95, NOT the strict metric) | Not a K=6 rerun: no bins (nothing to shatter on wavy filaments), regression not classification, and it is consumed as the arm heading, not as a channel-agreement cost (the `w_ori` term that measured 0.000). Helps the shallow-crossing tail (on MT-34 the shallow/steep gap is 0.656 vs 0.808) and every gap-link across an eroded crossing, where the heading currently comes from a 4–8 px skeleton stub. | 8–12 | Expected effect is bounded: on the roi303 diagnostic steep crossings (which already have unambiguous headings) fail at least as often as shallow ones, subject to the tolerance caveat in §0; the field is undefined within ±r of a crossing and must be masked; if #1/#2 fix the erosion this row may add little. Screen it AFTER #1–#3. |
| **5** | **Instance-gap ("valley") supervision for parallels**: a third head `g(x) = 1` on the medial line between two DIFFERENT GT centerlines whose gap is 3–12 px (target scale), plus a U-Net-style background weight `w(x) = w0·exp(−(d1+d2)²/2σ²)` (w0 = 10, σ = 5) on the centerline BCE so painting foreground into the gap is expensive; decode `p' = p·(1 − g)` before skeletonisation. | U-Net separation weight map (arXiv:1505.04597 §3); DCAN contour channel (arXiv:1604.02677); HoVer-Net horizontal/vertical distance maps (arXiv:1812.06499); Scherr et al. neighbour-distance target (arXiv:2004.01486); TextField boundary-pointing field separates adjacent instances (arXiv:1812.01393) | U-Net: touching HeLa cells separated without post-processing; HoVer-Net SOTA on clustered nuclei; Scherr: top-3 in CTC 2020 with one model | The audit says the pair's own ridges drop to 0.96 at 4.5–6 px and the mid-gap sits at 0.35–0.62; a valley head gives the decoder an explicit "there are two" signal at 0.5 without lowering the ridge threshold globally. | 4–6 | Small population: 19 separable stretches / 141 frames on roi303, so the F1 payoff is bounded (bundle 0.35 → 0.7 ≈ tens of TP); annotation duplicates (26 coincident runs) must be excluded from the metric before any claim. |
| **6** | **Crossing-preserving geodesic linking** for junctions whose mask is disconnected: replace the straight-gap turn cost by a data-driven sub-Riemannian / Finsler-elastica geodesic on `(x, y, θ)` between two arm tips, cost `∫ C(p) ds + ξ κ²` with `C = 1/(δ + p)` and the hard bound `κ ≤ kappa_max`; the geodesic length is the pair cost passed through `match_junction`'s `scorer`/`allow` hooks. | Bekkers–Duits SR geodesics in SE(2) (arXiv:1503.01433, SIAM J Imaging Sci 2015; fast marching arXiv:1508.02553; projective line bundle arXiv:1704.04192, DOI 10.1007/978-3-319-68445-1_89); van den Berg et al. crossing-preserving data-driven Cartan tracking (arXiv:2208.11004); spherical variant (arXiv:2504.03388); Chen–Mirebeau–Cohen Finsler elastica (arXiv:1612.00343) | van den Berg 2022: whole vascular trees tracked in two runs, crossings preserved by construction; failures remain "at crossings in extreme cases" (their Fig. 11); no per-crossing percentage reported | Uses the probability MAP (which carries the ridge) instead of the thresholded mask, exactly where the mask fails; curvature-penalised by construction so kappa_max stays a hard constraint. | 10–15 | Compute (a 3-D eikonal solve per junction; restrict to a 40 px window and 16 orientations); with opposite-polarity cancellation p ≈ 0 at the crossing pixel, so δ must be ≥ 0.05 or the geodesic detours; the gain over #2's simpler local threshold is unproven. |
| **7** | **PatchPerPix dense local shape patches** (the already-specified design): per-pixel patch code + decoder, overlap class, consensus set-cover assembly. Position: the ESCALATION if #1–#3 leave junction identity < 0.6, not a first screen. | PatchPerPix (arXiv:2001.07626); FISBe baselines (arXiv:2404.00130) | BBBC010 worms (crossing, overlapping), binary foreground masks as INPUT: avAP[0.5:0.95] 0.925 vs 0.724 prior SOTA (raw-image block is a separate, lower table); FISBe thin neurons: avF1 0.34 ± 0.01 (best of three baselines), "many touching neurons are falsely merged" | Amodal at crossings by construction (both filaments' patches claim the shared pixels); parallels separated because a patch centred on filament A contains only A. | 30–50 | Overlap pixels are excluded from the patch loss, so shallow crossings lose the most supervision; assembly is CUDA-heavy; FISBe shows the method merges thin touching instances — i.e. the parallel case is its weak side, not its strong one. |
| **8** | **Recurrent tip-to-tip tracer** (option B): detect tips, trace each instance point-by-point with a sequence model conditioned on the image; crossings resolved by the recurrent state. | Pick-and-Trace (MICCAI 2023, DOI 10.1007/978-3-031-43993-3_61); Trexplorer Super (arXiv:2507.10881); RefTr (arXiv:2511.20823); DeepBranchTracer (arXiv:2402.01187); RNGDet (arXiv:2202.07824) | Pick-and-Trace: full text not retrievable (publisher-blocked; 2 citations); Trexplorer Super: "strong performance on synthetic data does not necessarily translate to real datasets"; DBT: Length-F1 0.828 vs 0.831 for FR-Net on DRIVE, i.e. parity | Only route with zero post-processing; keeps identity through crossings as a sequence, not a pixel decision. | 40–80 | The DETR-style set predictor already diverged (0.013); recurrent tracers are the same family with a different decoder; no per-crossing evidence in the literature; the synthetic→real warning is explicit in Trexplorer Super. |

Summary of the ranking logic: #1–#3 change what the network is *asked* to emit at crossings and
along ridges (the measured failure), #2 lets the decoder use that; #4–#6 are geometric
refinements that help the shallow/eroded tail; #7–#8 replace the assembler and are gated on
#1–#3 failing.

---

## 2. Implementation notes per recommendation

### 2.1 Crossing-region supervision (multiplicity head + region weight)

**Targets (generator side, free).** For a frame with instances `{c_i}` (polylines, x = col,
y = row) rasterised at the training scale with the current `half_width`:

- `n(x) = Σ_i 1[d(x, c_i) ≤ half_width + 0.5]` — the number of centerline rasters covering x.
- **Overlap definition that does not confuse parallels:** mark `x` as class 2 only if
  `n(x) ≥ 2` AND the two nearest centerlines are within 2 px of each other at their closest
  approach *inside a 5 px window around x* (i.e. they genuinely cross or coincide there). Two
  parallels 3–9 px apart then stay class 1 on each ridge with class 0 between them — otherwise
  the head would label a bundle as an overlap and #2 would fuse it. PatchPerPix and Layering
  use raster overlap because their objects are wide; MTs are 3 px targets on 5.5 px filaments,
  so the raster overlap of a 4 px gap pair is not zero.
- Junction points `J = {crossing points of pairs of GT polylines}` (segment–segment
  intersections; the generator's `sample_scene` gives the polylines, `mt_bench.gt_stats` already
  computes crossings and angles for MT-34). Region weight
  `w(x) = 1 + (w_J − 1)·1[dist(x, J) ≤ r_J]`, `w_J = 6`, `r_J = 6 px` (training scale;
  scale with the 1.5× frame if training there). The 6 px radius covers the fused length
  `4r/sin α` for α ≥ 30° at r = 1.5; for shallower crossings the disc is elongated along the
  bisector to `4r/sin α` (cap 18 px, the current `bridge_max_len`).

**Head.** Add 3 logits (softmax over {0, 1, ≥ 2}) next to the existing binary logit; the
multiplicity target is derived from the same `ori_channels_soft` machinery (it already knows
which pixels are amodal: "it tells the crossing term which pixels are genuinely amodal",
`train_v5.py` l. 197). Loss: `L = w(x)·BCE_pos_weight(p, y_union) + λ_n·CE(n̂, n)`, `λ_n = 0.5`,
class weights for CE proportional to inverse frequency clipped at 20 (class 2 is ~0.1 % of
pixels). Deep supervision as now.

**What changes at inference.** Nothing in the instancer is required for the loss part; the
expected effect is on the probability map itself (fewer sub-threshold dips at crossings). The
`n̂ ≥ 2` map is the junction evidence for §2.2. Evaluate with `diag_crossing_map.py`: the
pre-declared success criterion is `fail_dip_gt_thr` and `fail_arm_missing` falling, and
`identity_opposite_polarity` rising from 0.08 — those are the mechanism numbers, F1 comes last.

**Evidence base.** PatchPerPix adds "an 'overlap' class ... trained jointly ... by means of
added cross-entropy loss" and uses it to exclude overlap pixels from assembly
(arXiv:2001.07626 §2.3). MultiStar predicts an "Overlap Probability" map and masks overlap
areas in the object-probability and star-distance losses (arXiv:2011.13228 Fig. 1). Hu et al.
train a 4-class net (background / chromosome A / chromosome B / overlap) and reach IoU 94.7 % on
the overlap region (arXiv:1712.07639 §5). OASIS adds a region weight `W_region(x, y)` on overlap
regions in a segmentation-regression loss and reduces reconstruction error in overlap-dominated
regions from −41.1 % to −13.3 % (arXiv:2510.06194). The Layering paper trains overlap regions in
a second phase so that "more than one layer will give foreground prediction" at overlaps
(arXiv:2210.03551 §2.1). None of these is a thin-filament result; the transfer claim is that
crossing pixels need their own supervised status, which the union target denies them.

### 2.2 Junction-conditioned decode

**Inputs.** `p` (H, W) probability; `J` = `n̂[≥ 2]` from §2.1 (or a standalone junction heatmap
head trained with a Gaussian σ = 2 px at each GT crossing, MSE, as in L-CNN's junction map
arXiv:1905.03246 §3 and Sat2Graph's vertexness arXiv:2007.09547 §3.1).

**Decode.**
1. Seeds `S = {local maxima of J with J > 0.5, NMS radius 4 px}`.
2. `mask = p > thr` as now; additionally `mask |= (p > thr_low) & disc(S, R)` with
   `thr_low = 0.5`, `R = clip(4·half_width/sin α̂, 6, 18)` where α̂ is estimated from the two
   dominant orientations of `p` in the disc (structure-tensor eigen-analysis at σ = 2 px, or from
   the §2.4 field if present).
3. `build_arc_graph(mask, merge_radius, ...)`: pass `S` as forced junction pixels so the
   contraction happens at the seed even if the skeleton has no degree-3 node there (the arms end
   inside the disc as free ends; treat free ends within `R` of a seed as arms of that junction).
   This is the Sat2Graph decode step "vertex extraction ... local maxima with vertexness greater
   than `p_thr`; edge connection by `||(v + d) − u|| + w·cos_dist`, w = 100, max 15 m"
   (arXiv:2007.09547 §3.1) transplanted onto the arc graph: the existing `match_junction` is the
   edge-connection step.
4. `match_junction` runs unchanged (hard `kappa_max`, `c_open`), with `gap_len` = tip distance.

**Why this is not the closed hysteresis.** The closed result is *global* hysteresis (grow the
0.98 mask into the 0.9/0.7/0.5 mask everywhere: +11 TP on MT-34, −32/−248/−418 TP on roi303),
which fuses parallels and grows spurs everywhere. Step 2 lowers the threshold only inside discs
where a second head asserts a crossing; the number of affected pixels is ~17 crossings × π R²
per frame. If `J` is wrong at a bundle, the guard "≥ 3 free ends or ≥ 1 degree-3 node inside the
disc" prevents fusing two parallels (they contribute only 2 ends when they run through).

**Pre-declared criterion.** Junction identity by angle on roi303 VAL and `mask_connected`
(0.173 → target > 0.6), with bundle recovery as a guardrail (must not drop below 0.35).

### 2.3 Soft ridge target with max-over-instances, ridge-NMS decode

**Target.** `y(x) = max_i exp(−d_i(x)²/(2σ²))`, `d_i` = Euclidean distance to polyline `i`
(sub-pixel, via the existing sub-pixel stamping in `ori_channels_soft`), σ = 1.0 px at the
training scale (FWHM 2.35 px ≈ the p > 0.9 ridge width the network already produces). `max`, not
sum and not union: at a crossing both filaments contribute y = 1 along their own centerlines and
the crossing pixel is 1, so nothing in the target says "here is a blob". Loss: L1 on `y` over
foreground-weighted pixels (Sironi's regression) or BCE against the soft target (keeps the
current `pos_weight` machinery). Sironi et al. motivate exactly this: classification is
"inaccurate because locations on centerlines and locations immediately next to them are
extremely difficult to distinguish" (CVPR 2014, DOI 10.1109/CVPR.2014.351; TPAMI 2016,
DOI 10.1109/TPAMI.2015.2462363). DeepFlux goes one step further and regresses a 2-D flux vector
toward the nearest skeleton pixel, defining the skeleton as flux sinks, and reports F 0.732 vs
0.724 (SK-LARGE) and 0.502 vs 0.454 (SYM-PASCAL) over the best binary-classification skeletons
(arXiv:1811.12608 Table 1). TextField's post-processing is the same idea for instances: bin the
predicted direction into 8 neighbours, follow parents to roots, group roots by a 3×3 dilation
(arXiv:1812.01393 §III-E).

**Decode.** Ridge pixels = `{x : ŷ(x) ≥ h, ŷ(x) ≥ ŷ(x ± n̂)}` with `n̂` the normal from the
structure tensor (or §2.4's field), `h` swept per arm like `prob_thr` today (expect the optimum
near 0.3–0.5 as the THIN arm's did). Then skeletonise the 1–2 px ridge set as now. The
ridge test is local, so two parallels separated by a valley of depth > 0 stay two ridges
regardless of the absolute level — this is what the audit asked for ("two ridges exist at
0.5 in 86 % of 4.5–6 px stretches, but the ridge maxima are below 0.99").

**Interaction with the factorial.** The THIN arm (1 px hard target) is a special case of this
with σ → 0; in the oracle-mask screen on synth it lifted bundle 0.820 → 0.950 and junction
identity 0.919 → 0.964 (`train_v5.py` l. 720), but the TRAINED thin arm delivered +0.0095 F1 and
fragments more. A soft ridge keeps the thin geometry without the
hard-threshold fragility; A/B it against THIN, not only against SPARSE35.

### 2.4 Continuous orientation field (regression, masked at overlaps)

**Target.** For each foreground pixel with `n(x) = 1`: `θ(x)` = tangent of the nearest
polyline at its closest point; `t(x) = (cos 2θ, sin 2θ)` (doubled angle removes the ±π
ambiguity; the standard structure-tensor representation). For `n(x) ≥ 2` or background:
no loss. Loss `L_ori = Σ_x 1[n=1]·(1 − t̂·t)` with `t̂` unit-normalised, or plain L2 on the
2-vector; weight 0.5 relative to the centerline term. Head: 2 channels. This is what Batra et
al. call orientation learning (they quantise; CVPR 2019 open access) and what ORION learned with
six hard branches — the difference from the closed K=6 head is that no pixel is ever assigned
to a bin, so no filament is ever split at a bin boundary, and the output is never skeletonised
per channel.

**Consumption (the part that decides whether this is a rerun).**
- (a) In `_arm_at` (instancer_a.py l. 127): replace `theta` from `window_tangent(poly, ...)` by
  `θ_f = ½·atan2(Σ sin 2θ̂, Σ cos 2θ̂)` over the arm's last `window` px, sign-resolved by the
  skeleton direction; keep `kappa` from the skeleton. The current heading is estimated from a
  4–8 px stub of an eroded skeleton (the failure montage shows stubs), which is exactly where a
  learned field is more reliable than geometry. `pair_cost` and `kappa_max` are unchanged.
- (b) Optional `allow` predicate: from arm tip `a`, integrate `x' = x + s·(±u(θ_f))` through the
  junction disc (step 1 px, 30 steps, direction updated from the field at each step, turn capped
  at `kappa_max`), and allow the pair `(a, b)` only if the flow lands within 3 px of `b`'s tip.
  This is a one-junction version of Bekkers' orientation-score tracking (arXiv:1212.3530 §3),
  where lifting to (x, y, θ) "unwraps" crossing lines (their Fig. 1) and 95.6 % of crossings
  were tracked correctly (Table 2).

**Expected effect and its limit.** The MT-34 measurement of `w_ori = 0.000` was for a
channel-agreement COST on a single-layer skeleton; the heading itself was never learned. On
roi303 the diagnostic puts most failures at steep, eroded crossings (§0, with its tolerance
caveat), where headings are unambiguous but arms are missing; the field helps only once #1/#2 restore the arms. Screen it third, report by angle.

### 2.5 Instance-gap ("valley") supervision

**Targets.** `d1(x), d2(x)` = distances to the two nearest DISTINCT GT centerlines. Valley
channel `g(x) = 1[|d1 − d2| ≤ 0.7 px ∧ d1 + d2 ≤ 12 px ∧ min(d1, d2) > half_width]`, i.e. the
medial line of a gap narrower than 12 px, off both rasters. Background weight on the union BCE:
`w(x) = 1 + w0·exp(−(d1 + d2)²/(2σ²))·1[y = 0]`, `w0 = 10`, `σ = 5 px` (U-Net's separation term,
arXiv:1505.04597 §3, with d1 + d2 in place of their d1 + d2 to touching cells). HoVer-Net's
horizontal/vertical distance maps (arXiv:1812.06499) and Scherr's "neighbour distance" (arXiv:2004.01486)
are the same signal regressed rather than weighted; for 1–2 px ridges the binary valley is
simpler and the gradient of a distance map would be dominated by the filament width.

**Decode.** `p' = p·(1 − ĝ)` before thresholding/ridge extraction. Nothing else changes.

**Population caveat (why it is ranked 5th).** roi303 has 19 stretches with gap < 7.5 eval px and
26 near-coincident runs that are annotation duplicates (audit §4). The metric's pair window is
[2, 6] px; any claim must first exclude the duplicates (`build_roi_dataset.py` dedupe, declared
before use). A gain here is worth tens of TP on this benchmark, not hundreds.

### 2.6 Crossing-preserving geodesic linking

For a junction whose arms are free ends (no degree-3 node after §2.2, or without §2.2), compute
for each candidate pair `(a, b)` the sub-Riemannian geodesic distance on
`M2 = R² ⋊ S¹` with the Reeds–Shepp metric of Bekkers et al. (arXiv:1503.01433; the
data-driven Cartan version arXiv:2208.11004 §3, ξ² = g11 = 0.01, g33 = 1, sideways cost
ζ² = 0.01) and an external cost `C(x, θ) = 1/(δ + p(x)·ρ(x, θ))`, `ρ` = a 16-orientation
line-filter response (cake-wavelet or steerable second derivative) so that the cost is low on
the ridge in the ridge's orientation only. Solve by anisotropic fast marching (arXiv:1508.02553)
on a 40 × 40 × 16 window per junction; `cost(a, b)` = geodesic length, `kappa_implied` = max
|dθ/ds| along the back-tracked geodesic, rejected if > `kappa_max`. Pass through `match_junction`
as `scorer = 2·c_open − cost` scaled to the geometric cost's range.

**Evidence and limit.** Lifting "manifestly disentangles" crossings (arXiv:2208.11004 §1, Fig. 1)
and the 2014 tracker resolved 109/114 crossings; the 2022 model still "may steer the tracking
away from the actual vessel at crossings in extreme cases" (their Fig. 11). For IRM the
crossing pixel of opposite-polarity filaments has p ≈ 0 (cancellation), so `δ ≥ 0.05` is needed
to let the geodesic cross a 3–6 px hole at a bounded price. This is the principled version of
what §2.2 does with a disc and a low threshold; it is worth building only if §2.2's simple
version leaves identity < 0.6 at steep crossings.

### 2.7 PatchPerPix — where it sits now

The design is specified and unrun. What this research adds: (i) FISBe (arXiv:2404.00130 §4)
reports that on thin filamentous neurons PatchPerPix "achieves best results for all metrics
except for false merges" but "many touching neurons are falsely merged even with different
color", avF1 0.34 ± 0.01 — the parallel case is its weakness, not its strength, and its overlap
handling "can only bridge small overlaps up to the used patch size"; (ii) on BBBC010 (wide
crossing worms) it reaches avAP 0.925 vs 0.724 for the previous best when binary foreground
masks are the input (arXiv:2001.07626 Table 3.1, top block; the raw-image block is reported
separately and is lower).
Recommendations #1 and #2 are strictly cheaper (1–2 days vs weeks) and reuse the current
instancer; #1's multiplicity head is also PatchPerPix's overlap head, so nothing is thrown away
if the escalation happens.

### 2.8 Recurrent tracer — evidence status

Pick-and-Trace (Liu, Peng, Caplan, Kambhamettu, MICCAI 2023, LNCS pp. 635–645,
DOI 10.1007/978-3-031-43993-3_61) "identifies filaments' tip points and traces each instance
from tips with a sequential encoder-decoder"; the full text and its numbers could not be
retrieved (Springer and the MICCAI page were unreachable; Semantic Scholar/CrossRef carry no
abstract; 2 citations). The same group's earlier ORION code base (github VimsLab/filaments_quantification)
does not include it. Trexplorer Super's own conclusion, "strong performance on synthetic data
does not necessarily translate to real datasets" (arXiv:2507.10881), and this project's DETR
set-prediction null (0.013) argue for leaving this as the last option.

---

## 3. Literature survey

### 3.1 Classical cytoskeletal / fibre tracers — what each does at a crossing

Extracted from the primary texts (publisher full text, PMC, bioRxiv, shipped source code) by a
delegated reader; provenance per row: *raw* = quoted from the held PDF/code, *verified* = fetched
then re-checked by exact-string query, *unverified* = summariser output only. Reference
corrections relative to the brief: SIFNE is Zhang, Nishimura & Kanchanawong, *Mol Biol Cell*
2017 (not Cytoskeleton); DeFiNe is *Sci Rep* 2015 (not PLOS CB); KnotResolver is published in
*Bioinformatics* 2024; the systematic review is Østerlund, Persson & Nikoloski, *CSBJ* 2023.

| Tool · venue · DOI | Crossing rule (quoted / paraphrased) | Parallel separation | Quantitative crossing result |
|---|---|---|---|
| **SOAX core** — Xu et al., *Med Image Anal* 2014, DOI 10.1016/j.media.2013.10.015, PMC3945675 (verified) | Snakes are cut at junctions; branches are re-paired by collinearity with *Normalized Cuts*, edge weight `w_ij = (1/π)·arccos(\|v_i·v_j\|/(‖v_i‖‖v_j‖))`; "junctions in cytoskeletal filament networks typically involve straight fibers crossing one another … we use this smoothness constraint"; during evolution a colliding snake merges with a converged one only if the angle exceeds θ = 2π/3, else a T-junction forms; higher-degree junctions = connected-component clusters of nearby T-junctions (no radius given). | overlap stop `D_min = 2` voxels; no stated limit | none per junction; "vertex error is around one voxel when the SNR is greater than 4" |
| **SOAX** — Xu et al., *Sci Rep* 2015, DOI 10.1038/srep09081 (paper unverified; code raw) | Paper: "clustering nearby T-junctions into a single junction followed by configuring the local connectivity of SOACs by cutting and splicing them such that they do not end or bend sharply at junctions". **Shipped code** (`snake_tip_set.cc`) is NOT Normalized Cuts: greedy `FindSmoothestPair` (max `acos(v1·v2)` over tip directions from the last `grouping_delta = 8` points), link if angle > `direction_threshold = 2.1 rad` (≈ 120°), remove, recurse; `grouping_distance_threshold = 4.0`, `overlap_threshold = 1.0`. | "works best … such that the PSF width is narrower than their average separation" | synthetic vertex error 1.08 px, Hausdorff 7.15 px; nothing per junction |
| **TSOAX** — Xu et al., *Sci Rep* 2019, DOI 10.1038/s41598-018-37182-6, PMC6368602 (verified) | Curves "dissected at colliding points ('T-junctions') which are clustered into detected network junctions"; re-linked by bipartite matching with `D_ori = 1 − θ/π` (θ = angle between tip tangents; opposite-facing tips ⇒ low cost) plus a temporal term; global minimum path cover over the k-partite time graph, `η = 20 px`. | none stated | actin tracks P 97.9 % / R 93.1 % (80.3 / 73.4 without local matching); "tracking errors … correspond to failures to track a filament past an intersection" |
| **SIFNE** — Zhang, Nishimura, Kanchanawong, *Mol Biol Cell* 2017, DOI 10.1091/mbc.E16-06-0421, PMC5231901 (raw) | Crossing pixels = skeleton pixels whose 8-neighbour sum > 3; "the crossing point pixel, along with a 3 × 3 neighborhood (or larger) … are removed" (manual: 7 × 7); fragments re-joined only at termini inside a fan (`R_fan`, `θ_fan`) under three constraints (direction mismatch ≤ φ_max, gap ≤ d_max, fragment-vs-gap direction ≤ ψ_max), derived from a max curvature "κ ≈ 1 rad/mm [sic]" → 57.3° and 1 µm at 20 nm/px; cost `C_i = C_angle·Δθ_i/max + C_gap·Δθ_gap,i/max`. | "filament spacing < ∼120 nm results in progressively lower accuracy" (≈ 6 px at 20 nm/px); "the identity of individual filaments is not always resolvable at sites of intersection or bundling" | none per crossing; the authors' own compound-error argument: "even with 95 % accuracy at each intersection, the overall tracing accuracy for a filament passing through 10 intersections is 0.95^10 or ∼60 % … linkage accuracy must be much greater (> 99 %) or global-scale constraints may be required" |
| **FiNTA** — Flormann et al., *FASEB J* 2021, DOI 10.1096/fj.202100048R (verified, bioRxiv text) | Node graph with arbitrary connectivity; junctions merged within a "unification distance"; a filament continues through a node while the turn stays below a "break-off angle θ" (value in Table S1, not retrieved). No sentence on crossings/overlap. | none stated | connectivity 3–6 recovered on artificial images; real FN 8.55 % / FP 1.54 % (EM); "almost all of the filaments that are not traced arise where the angle between two filaments is small" |
| **DeFiNe** — Breuer & Nikoloski, *Sci Rep* 2015, DOI 10.1038/srep18267, arXiv:1601.00847 (verified) | Global *Filament Cover Problem*: choose candidate paths covering every edge (at least once = overlaps allowed, or exactly once) minimising intensity/thickness "roughness"; angle enters only as a hard cut-off when generating candidates ("stop the search for a path when it violates the straightness criterion r_p,angle < 60°"); solved as a (fractional) ILP. | not discussed | contrived network "with crossings and overlapping filaments as well as a loop": JI = 1; neuron JI1 0.937; plant actin JI1 0.655 (7 manual decompositions; automated "as good as any manual") |
| **CytoSeg 2.0** — Nowak et al., *Bioinformatics* 2020, DOI 10.1093/bioinformatics/btaa035 (raw) | none — "crossings or endpoints of the skeleton [are] nodes"; outputs network statistics, no filament identity | none | none |
| **Steger ridge detector** — Steger, *IEEE TPAMI* 1998;20(2):113–125, DOI 10.1109/34.659930 (raw: TUM report FGBV-96-03 text); Fiji *Ridge Detection* plugin (imagej.net/plugins/ridge-detection, fetched) | Lines are grown from the pixel of maximum second directional derivative by adding, among the three neighbours compatible with the current direction, the one that minimises `d + cβ` (d = sub-pixel distance, β = \|α₂ − α₁\| ∈ [0, π/2]), "in the current implementation, c = 1"; "At junction points, it will select one branch to follow without detecting the junction, which will be detected later on … until the best matching candidate is a point that has already been added to another line. If this happens, the point is marked as a junction, and the line that contains the point is split into two lines at the junction point." ⇒ every line ENDS at a junction; identity across a crossing is never decided. Fiji plugin: overlap resolution NONE = "any point of potential intersection will be treated as an end point for the ridges involved"; SLOPE = "the assumption that when two ridges overlap, it is more likely that they will continue on their path than make turns" (recommended only for brief overlaps). | none stated; width estimation is extrapolated at junctions "where the line width usually grows beyond the range of 2.5σ" | none |
| **Filament Sensor** — Eltzner et al., *PLOS ONE* 2015, DOI 10.1371/journal.pone.0126346 (verified) | Width-aware line sensor over 180 orientations, long segments first; each accepted segment writes its orientation into a field 𝓞 within diameter `w_j + 2`; "if less than 30 % of its pixels have a conflicting orientation entry in 𝓞 — i.e. the entry differs by less than … 20 degrees from the segment's orientation — the segment is accepted"; otherwise conflicting endpoints are removed iteratively. ⇒ true X-crossings (large angle) pass, near-parallel overlaps are truncated. | "close parallel lines of three pixel distance … are well identified … for smaller distances (one and two pixels) … the FS fails" | forks "up to 6.5° … well resolved … 3.5° … not resolved by any of the methods"; no crossing statistics |
| **FiberApp** — Usov & Mezzenga, *Macromolecules* 2015, DOI 10.1021/ma502264c (raw) | semi-automatic: A* lowest-cost path between two user-clicked points, then an active contour (curvature-penalised); no crossing rule — identity is fixed by the user's clicks | none | none |
| **FIRE / CT-FIRE** — Stein et al., *J Microsc* 2008, DOI 10.1111/j.1365-2818.2008.02141.x (closed access; secondary only); Bredfeldt et al., *J Biomed Opt* 2014, DOI 10.1117/1.JBO.19.1.016007 (verified) | nucleation on distance-map ridges, extension with direction memory, branches linked "if they share the same nucleation point and the directions of the branch tips … within the threshold of angle differences … the one that is more aligned … is selected"; CT-FIRE names "the ability to properly segment crossing or cross-linked fibers" an unfixed FIRE limitation | none | none per crossing (F-measure vs 3 observers over 9290 fibre evaluations) |
| **KnotResolver** — Khatri, Yadav, Athale, *Bioinformatics* 2024, DOI 10.1093/bioinformatics/btae538, arXiv:2404.12029 (verified) | branch points = Moore-neighbourhood sum ≥ 4; directed graph of branch/segment vertices; ≤ 10 candidate paths per combination; the path with minimum DTW distance to the PREVIOUS frame's contour wins (temporal, not geometric); manual initialisation if frame 1 self-intersects | none | sub-pixel contour accuracy; Dice vs SNR as figures only; degrades after ~60 frames for MTs > 20 µm |
| **TARDIS** — Kiewisz, Bepler et al., bioRxiv 2024 v2, DOI 10.1101/2024.12.19.629196, PMC11702698 (verified) | DIST predicts a probability adjacency matrix; filament prior "each node has at most two neighbors"; instances grown by "iteratively find[ing] up to two edges with the highest probability" (greedy, no cutoff for filaments). **No sentence containing "crossing", "intersect" or "overlapping filaments" exists in the paper.** | "12.5 nm for microtubules … to reduce noise and separate closely packed structures" | 3-D MT tomograms mCov 0.72 vs Amira 0.42; 112 TIRF images "successfully segmented" — **no 2-D instance metric reported** |
| **SynthMT** — Koddenbrock et al., *PLOS Comput Biol* 2026, DOI 10.1371/journal.pcbi.1013901 (raw) | instance segmentation by foundation models (SAM3 with the text prompt "thin line" best); amodal per-slice GT; no crossing rule; qualitative only: "SAM3Text captures all visible MTs … including most of the complex intersections", "SAM2 merges intersecting filaments", "FIESTA and TARDIS … can resolve intersections, but they miss a substantial number of filaments" | none | synthetic SKIoU/AP/F1@.5/F1@.75: SAM3Text 0.85/0.86/0.91/0.90 (+HPO 0.93/0.95/0.97/0.96), TARDIS 0.45/0.48/0.59/0.45, FIESTA 0.12/0.12/0.22/0.18; real IRM (= this project's `htw_eval`, 66 crops, 2 annotators): mean SKIoU 0.74 ± 0.18 vs human 0.80 ± 0.14 |
| **Review** — Østerlund, Persson, Nikoloski, *Comput Struct Biotechnol J* 2023, DOI 10.1016/j.csbj.2022.12.023, PMC9804014 (unverified) | "there is currently no consensus on … how to handle identification of individual FSs by defining junctions and overlap correctly"; "all existing rules regarding how to stitch pieces together beyond junctions rely on angles and distances" | — | — |

Cross-cutting: **not one classical source reports a per-crossing accuracy or an error-vs-angle
curve.** The rules are of three kinds — none (Steger, CytoSeg: lines end at junctions), pairwise
collinearity at the junction (SOAX/TSOAX/SIFNE/FiNTA/FIRE; Fiji Ridge Detection's SLOPE option) and
global path cover (DeFiNe; TSOAX over time) — and SIFNE's own discussion argues
the first cannot exceed ~60 % over ten crossings. This project's per-junction min-cost matching
under a hard `kappa_max` is already the strongest member of the first family (on an oracle mask
it keeps identity in 0.79–0.92 of crossings where PySOAX scores 0.000 on non-shallow
crossings, `docs/protocol.md` §17); DeFiNe's exact-cover vs at-least-once switch is the
formal version of the amodal question and is the only classical escalation worth keeping in
view (a path-cover ILP over the arc graph with `kappa_max` generating candidates). The stated
parallel-resolution floors — 3 px (Filament Sensor, synthetic), ~6 px at 20 nm/px (SIFNE) — are
the only published numbers for the 3–9 px regime.

### 3.2 Deep-learning approaches to overlapping / crossing thin structures

**(a) Multi-channel "overpass" outputs.** ORION (Liu et al., CVPRW 2019, DOI 10.1109/CVPRW.2019.00021,
PMC8046259): six orientation branches covering [0°,30°) … [150°,180°), each detecting
filaments of that orientation so "intersections [become] overpasses"; a terminus-pairing
algorithm regroups pieces across branches by (i) similarity of propagation directions,
(ii) distance, (iii) alignment of the tip-to-tip vector — thresholds "will be set for each
criteria" (no values given). Trained on 120 000 synthetic 64 × 64 × 7 tiles built from 180 000
deformed straight lines; tested on 10 real MT images: F1 0.7161 / precision 0.6456 / recall
0.8189 / SKIoU 0.8166 vs SOAX 0.2691 and SIFNE 0.2875 (SKIoU-F at thresholds 0.5–0.95 — not
the strict centerline-F1). **No per-crossing analysis**; stated limitations: "six side outputs
merge together may not fully recover the original input, which causes large gaps … and leads to
failure in regroup step", and pairing "may fail to find the right pairs". This is exactly the
failure the project measured with its K=6 head (per-channel binning shatters wavy filaments;
same-bin shallow crossings still need pairing). *Lesson kept:* synthetic-only orientation
supervision transfers to real MTs; *lesson dropped:* hard bins.

**(b) Layered / amodal representations.** Layered Embeddings (arXiv:2002.06264): pixel
embeddings consistent across a visible layer and an occluded layer; their ablation with
ground-truth clustering shows "three or more layers of mask output … instead of just
'foreground and occlusion' could improve performance" — i.e. a two-layer cap is real.
Instance Segmentation via Layering (Chen, Wu, Merhof, arXiv:2210.03551): N sigmoid layers,
a layering loss that pushes ADJACENT objects into different layers (overlap regions ignored in
phase 1), then an overlap-completion phase so that "more than one layer will give foreground
prediction" at overlaps; instances = connected components per layer; evaluated on BBBC010
(crossing worms), OCC, CCDB, where "the two parallel and close C. elegans" are the case that
breaks Mask R-CNN's NMS. BCNet bilayer occlusion (arXiv:2103.12340) is the same idea for
natural images. Relevance: these are learned, adaptive versions of the K=6 idea — the layer is
assigned by adjacency, not by orientation — but they still need a pixel-competition loss
between neighbours, which is the mechanism that merged parallels in the closed embedding
route; not recommended as a first move.

**(c) Explicit overlap / multiplicity heads.** PatchPerPix (arXiv:2001.07626 §2.3): "we
predict the number of instances per pixel by extending the foreground classification task by an
'overlap' class"; overlap pixels are excluded from the patch loss and assembly, hence "only
overlapping regions with a maximum diameter smaller than the size of the patches can be
covered". MultiStar (arXiv:2011.13228): an overlap-probability head that masks overlap areas out
of the object-probability and star-distance losses and un-suppresses truly overlapping proposals
in NMS. Overlapping chromosomes (Hu et al., arXiv:1712.07639): four classes {background, A, B,
overlap}, overlap IoU 94.7 %, A/B 88.2 % / 94.4 % — the simplest per-pixel multiplicity
precedent, though only for pairs. DoNet (arXiv:2303.14373) decomposes cell clusters into
intersection and complement regions and recombines them. OASIS (arXiv:2510.06194) adds a region
weight on overlap pixels to a segmentation-regression loss and reduces reconstruction error in
overlap-dominated regions from −41.1 % to −13.3 %. These are the basis of recommendation #1.

**(d) Junction heads and graph decoders.** Sat2Graph (arXiv:2007.09547 §3.1): a
`(1 + 3·D_max)`-channel tensor per pixel — vertexness `p_v`, then `D_max = 6` slots of
(edgeness, dx, dy), one slot per 60° sector; decode = local maxima of `p_v` above `p_thr`, then
for each slot connect to the vertex minimising `‖(v + d) − u‖ + 100·cos_dist(d, u − v)` within
15 m. Because slots are per pixel and per sector, two roads crossing without a junction
(overpasses) are representable: "precision of 83.11 % … recall of 49.81 % on stacked roads …
only 0.37 % of intersections are incorrectly predicted as overpasses". Losses on edge slots are
computed only at ground-truth vertices because the encoding is inconsistent along straight
segments. RNGDet / RNGDet++ (arXiv:2202.07824, arXiv:2209.10150): transformer that grows the
graph vertex-by-vertex and "can handle complicated intersection points with various numbers of
incident road segments" — iterative, real-data, no crossing metric. L-CNN (arXiv:1905.03246):
junction heatmap `J` + offset `O` on a 128 × 128 grid, K = 300 junction proposals, then a line
verification network on features pooled along the candidate line (LoI pooling) — the
image-feature line verifier is the precedent for "pair arms from image evidence along the
candidate path" (also PPGNet's path-conditioned connectivity, arXiv:1905.03415; HAWP's
holistic attraction field, arXiv:2003.01663). Uslu & Bharath (arXiv:1806.03175): multi-task
net (interior, centerline, edges, junction patterns) whose junction candidates are
"re-examined in centerline probability maps to verify if they connect at least 3 branches";
Abbasi-Sureshjani et al. detect bifurcations/crossings from orientation scores (ISBI 2016, DOI
10.1109/ISBI.2016.7493241). TopoNet (arXiv:2304.05277) is lane-topology reasoning in driving
scenes (lane–lane connectivity via a scene graph), not a thin-structure crossing method.
Basis of recommendation #2; the arm-pairing part is deliberately left to the existing
min-cost matcher (the learned arc-pairing null), with image-feature verification only as the
optional `allow` predicate.

**(e) Orientation lifting (R² × S¹).** Bekkers, Duits et al. (JMIV 2014, arXiv:1212.3530, DOI
10.1007/s10851-013-0488-6): invertible orientation scores via cake wavelets; "crossing lines in
the image domain are clearly unwrapped in the orientation score domain"; edge tracking in
SE(2) with junction detection by clustering candidate points on position then orientation;
**109/114 crossings (95.61 %) correctly tracked** on 9 HRF images, bifurcations 76.1 % correct
with 18.6 % "bifurcation was actually part of a crossing" errors (Table 2). Sub-Riemannian
geodesics in SE(2) (arXiv:1503.01433; fast marching arXiv:1508.02553; projective line bundle
arXiv:1704.04192): curvature-penalised, sideways motion forbidden. van den Berg et al.
(arXiv:2208.11004): data-driven Cartan connection, crossing-preserving vesselness cost, whole
vascular trees in two runs, but "the data-driven term may steer the tracking away from the
actual vessel at crossings in extreme cases" (Fig. 11); spherical variant arXiv:2504.03388.
Chen–Mirebeau–Cohen Finsler elastica (arXiv:1612.00343) is the minimal-path counterpart with
an explicit curvature penalty. SE(2) group CNNs (arXiv:1804.03393) put the lifting layer inside
the network; Franken & Duits' crossing-preserving diffusion (IJCV 2009, DOI
10.1007/s11263-009-0213-5) is the enhancement version. This family is the formal home of the
project's `kappa_max` (bounding dθ/ds lives naturally on the lifted space) and the basis of
recommendations #4 and #6. No paper in it reports thin-filament (≤ 6 px) crossing accuracy.

**(f) Direction / flux fields.** TextField (arXiv:1812.01393): per-pixel unit vector pointing
away from the nearest instance boundary, trained with an instance-balanced Euclidean loss,
decoded by binning directions into 8 neighbours, following parents to roots on the symmetry
axis, grouping roots by a 3 × 3 dilation — the field "separates adjacent text instances, which
is challenging for classical segmentation-based approaches"; note their own remark that pixels
on the symmetry axis have low magnitude "due to the cancellation of opposite direction". DeepFlux
(arXiv:1811.12608): 2-D flux toward the nearest skeleton pixel within a fixed neighbourhood,
skeleton = flux sinks, F 0.732 / 0.695 / 0.840 / 0.502 / 0.491 on SK-LARGE / SK506 / WH-SYMMAX /
SYM-PASCAL / SYMMAX300 vs 0.724 / 0.681 / 0.805 / 0.454 / 0.480 for the best prior method.
Cellpose (Stringer et al., *Nat Methods* 2021, DOI 10.1038/s41592-020-01018-x; bioRxiv
10.1101/2020.02.02.931238): horizontal/vertical gradients of a simulated heat diffusion from
each cell's median pixel, L2 loss on the flows + weighted cross-entropy on the probability map,
inference by 200 Euler steps into fixed points; touching cells separate because their flows
converge to different sinks — a solution for compact objects that does not carry over to a
1-D centerline (the sink of a filament is its whole axis). HoVer-Net (arXiv:1812.06499)
regresses horizontal/vertical distances to the instance centre of mass to separate clustered
nuclei. For centerlines the usable member of this family is the flux/distance regression
(recommendation #3) and the direction field's separation role (recommendation #5).

**(g) Dense local shape patches.** PatchPerPix (arXiv:2001.07626) and its FISBe evaluation
(arXiv:2404.00130) — see §2.7. FISBe also states why most proposal-free methods fail on thin
filaments: "three-label models degenerate … because their interior equals their boundary;
models predicting pixel affinities become inappropriate if they rarely encounter foreground in
their fixed pixel neighborhoods; metric learning models lack …" (§1) — a published statement of
the project's affinity+MWS plateau (0.49) and embedding null.

**(h) Recurrent / iterative tracers.** Pick-and-Trace (MICCAI 2023, DOI
10.1007/978-3-031-43993-3_61; abstract and numbers unavailable through any open channel);
Trexplorer Super (arXiv:2507.10881; 3-D CT trees; "strong performance on synthetic data does not
necessarily translate to real datasets"); RefTr (arXiv:2511.20823; confluent-trajectory
refinement, tree topology enforced — crossings do not exist in its output space);
DeepBranchTracer (arXiv:2402.01187; joint centerline/boundary segmentation + direction/radius
regression + iterative tracing; motivated by "two adjacent or tangled branches … misinterpreted as
a single entity"; Length-F1 0.828 vs 0.831 for the best segmentation baseline on DRIVE, i.e. no
gain on 2-D vessels; crossover branches shown only qualitatively). Recommendation #8.

**(i) Semantic MT segmenters (context, no instance step).** CurvSegFlow (Braun, Lánský et al.,
arXiv:2606.21608): time-conditioned flow matching with a U-Net, evaluated on MicSim_FluoMT,
SynthMT, a new `IRM_InVitroMT` set (Beber, Braun, Lánský, Zenodo) and retinal/coronary sets;
Dice 0.9604 on SynthMT from 100 training images, "effectively segmented dense and overlapping
filaments while minimizing false positives" — semantic masks only, supervised, no instance or
crossing metric; mandatory neighbouring-lab citation. MTFlow (arXiv:2601.14841) is its MT-only
precursor. ASE_Res_UNet (arXiv:2507.07800; bioRxiv 10.1101/2025.10.23.684152): noise-adaptive SE
attention, Dice 0.9807 on SynthMT, semantic only. Liu & Zhang cGAN augmentation for thin
structures (arXiv:2512.10334): Pix2Pix from binary masks with a filament-aware structural loss,
SKIoU gains on the Liu/Caplan MT + actin sets — semantic. Sheka et al. (USBEREIT 2025,
"Instance Segmentation Neural Network Trained on Synthetic Dataset of Actin Filaments",
ResearchGate 393340589): full text not accessible; title-level evidence only. WireSeg-32K
(arXiv:2609.03102, 2026-09-02): physics-grounded synthetic wires with heavy crossings and
self-occlusion; SAM3 baselines F1@75 0.179 (hard synthetic) / 0.296 (real) — a fresh datapoint
that thin crossing deformable objects remain hard even for foundation models.

### 3.3 Close-parallel separation for sub-resolution gaps

*Instance-aware losses.* U-Net's separation weight `w(x) = w_c(x) + w_0·exp(−(d_1(x) + d_2(x))²/2σ²)`
with `w_0 = 10, σ ≈ 5 px` (arXiv:1505.04597 §3) forces the background between touching
instances to be learned; DCAN's contour channel (arXiv:1604.02677) and HoVer-Net's distance maps
(arXiv:1812.06499) make the boundary/centre an explicit target; Scherr et al.'s neighbour-distance
transform (arXiv:2004.01486) uses "not only touching cells but also close cells in the training
process" and is "notably robust to annotation errors"; TextField's boundary-pointing field
(arXiv:1812.01393) is the same signal as a vector. All are for objects with an interior; for a
1–2 px ridge the equivalent is the valley/gap target of §2.5. *Regression instead of
classification* (Sironi 2014/2016; DeepFlux) is the other lever: it keeps ridge location and
ridge height separate, which is what the audit says fails at 4.5–6 px (ridge height 0.96 < thr).

*Super-resolved output heads.* SuperVessel (arXiv:2207.13882) trains a super-resolution
auxiliary branch (removable at test time) and reports IoU +6 % with more tiny vessels from
low-resolution input; no gap-vs-width analysis. No paper found gives the minimum separable gap
as a function of filament width for a learned segmenter; the only published floors are Filament
Sensor's 3 px (synthetic line pairs) and SIFNE's ~6 px at 20 nm/px (SMLM). This project's own
measurement is more specific than anything published: two ridges survive at thr 0.99 from
~6 eval px (4 native) upward in 72–90 % of stretches, at 4.5–6 px in 43 % (86 % at 0.5), and
the raw image resolves the two closest bins WORSE than the network (intensity dip 0.26 / 0.43 of
ridge contrast). A Rayleigh-type estimate (two equal Gaussian ridges only develop a central dip for
d > 2σ ≈ 0.85 FWHM) puts the floor for a 5.5 px FWHM pair at ≈ 4.7 px; the network is already near it, which is why #5 is ranked low and #3 (ridge NMS at a lower height)
is the cheaper way to harvest the 4.5–6 px bin.

### 3.4 Synthetic supervision — which auxiliary targets are known to transfer

- **Orientation** (ORION, DOI 10.1109/CVPRW.2019.00021): six orientation branches trained on
  synthetic tiles only, F1 0.716 on real MTs; Batra et al. (CVPR 2019, CVF open access) show
  joint orientation + segmentation learning improves road connectivity on real data (exact
  APLS numbers not retrievable here). Project evidence: the K=6 head's `w_ori` = 0.000 on model
  masks, but the July 2026 amodal-channel experiment reached Alice 0.653 vs heuristic 0.565 —
  the orientation *representation* transferred, the *binning* did not.
- **Overlap / multiplicity class** (PatchPerPix, MultiStar, chromosomes): trained on real
  annotations in those papers; WireSeg-32K and SynthMT show synthetic amodal instance GT
  trains foundation-model instance heads that transfer (SynthMT real SKIoU 0.74 vs human 0.80).
  No paper trains a multiplicity head on synthetic data and tests it on real filaments — that
  transfer is an open question #1 must answer on the tirf12/roi303 guardrails.
- **Junction / vertex heatmaps** (Sat2Graph, L-CNN, Uslu): real data only; their transfer from
  synthetic is unmeasured, but a junction is the least appearance-dependent target of all (it
  is geometry), which is the argument for trying it.
- **Distance / flux regression** (Sironi, DeepFlux, Cellpose): all real-trained; Cellpose's
  generalist models were later trained partly on synthetic-like augmentation but the paper does
  not separate this.
- **Local shape patches** (PatchPerPix): real-trained; the project's spec already argues the
  transfer.
- **Project-internal evidence that matters more than any of the above**: (i) the generator's
  contrast fix (×3) is the only intervention that ever produced a significant gain (+0.085,
  p = 0.001); (ii) three consecutive levers (thin target, length prior, learned linker) looked
  positive before per-arm tuning and null after — so every auxiliary target must be screened
  with the campaign protocol, on junction identity and the `diag_crossing_map` mechanism
  numbers, before it is believed.

---

## 4. What NOT to do (with evidence)

1. **Hard orientation-binned amodal channels (K = 6 or any K).** Closed by measurement
   (`w_ori` = 0.000 on model masks; per-layer binning shatters wavy filaments) and by ORION's own
   limitation list ("six side outputs merge together may not fully recover the original input,
   which causes large gaps", DOI 10.1109/CVPRW.2019.00021). Continuous regression (#4) is the
   replacement; it must be consumed as geometry, not as a channel-agreement cost.
2. **Fixed-margin pixel embeddings / object condensation / pixel-competition layering** for
   parallels: structural parallel-merge (project null); FISBe states "metric learning models
   lack …" the properties needed for thin filaments (arXiv:2404.00130 §1); the Layering paper's
   loss "penalizes adjacent objects having the same embedding vector" — the same competition.
3. **Degree-2 point-cloud cuts (TARDIS DIST in 2-D).** The paper's own rule is "up to two edges
   with the highest probability" per node and it contains no crossing sentence and no 2-D
   instance metric (bioRxiv 10.1101/2024.12.19.629196); SynthMT measures TARDIS at AP 0.48 on
   synthetic IRM vs 0.86 for SAM3Text (DOI 10.1371/journal.pcbi.1013901).
4. **Global set prediction / iterative graph transformers as the next attempt** (DETR-style,
   RNGDet, LETR arXiv:2101.01909): the project's set predictor diverged (0.013 vs 0.417); those
   architectures target straight segments or road graphs with dense real supervision; no
   thin-filament crossing evidence; Trexplorer Super warns explicitly about synthetic→real
   (arXiv:2507.10881).
5. **Global hysteresis / closing / pruning of the mask** — measured null on MT-34 (+11 TP) and
   negative on roi303 (−32 … −418 TP). Only the junction-*conditioned* local version (#2)
   escapes this, and it must be A/B'd on junction identity, not F1.
6. **Soft-clDice or topology losses on a ≤ 3 px target** — structurally inert (project
   measurement: at 1 px the skeleton IS the mask); DTU-Net-style topological similarity
   (arXiv:2205.11115) has the same degeneracy on 1-px structures.
7. **Two-layer amodal caps** as a design target (Layered Embeddings' own ablation asks for
   ≥ 3 layers, arXiv:2002.06264; chromosome nets handle pairs only, arXiv:1712.07639): dense IRM
   fields have 3+ filaments sharing pixels; #1's multiplicity class is "≥ 2", and the amodal
   rasterisation of instances already allows any count.
8. **Minimum-angle greedy pairing at junctions** (SOAX code's `FindSmoothestPair` with a 120°
   cut-off; SIFNE fan; FIRE most-aligned branch): strictly weaker than the current min-cost
   perfect matching under `kappa_max` (PySOAX 0.000 on 288 non-shallow crossings vs A 0.78 on
   the oracle mask, `docs/protocol.md` §17), and SIFNE's authors show why (0.95^10).
9. **Quoting SKIoU-F1 or Dice crossing claims as evidence for the strict metric.** ORION's
   0.716, SynthMT's 0.74, CurvSegFlow's Dice 0.96 are all tolerant-overlap numbers; none
   measures identity through a crossing. The only per-crossing figures among the sources reviewed here
   are Bekkers 2014 (95.6 % on retinal crossings) and Sat2Graph's stacked
   roads (P 83 % / R 50 %).
10. **Letting an optimiser touch `kappa_max`, or fitting the generator's crossing/parallel
    statistics to the benchmark.** `kappa_max` is derived (0.239 max over 957 MT-34 GT
    filaments at an 8 px baseline; transferred unchanged to HTW, max 0.242) — every classical
    tool that instead *tuned* an angle cut-off (SOAX 120°, DeFiNe 60°, SIFNE 57.3°) reports no
    crossing accuracy. Crossing density and angles in the generator should span the real range
    (domain randomisation), not match a biased corpus.
11. **Expecting orientation tricks to fix the roi303 majority failure.** On the per-crossing
    diagnostic steep crossings fail at least as often as shallow ones (0.154 vs 0.360, shallow
    possibly tolerance-inflated); 97.6 % of failures are a disconnected mask and 83 % a
    sub-threshold dip — erosion/cancellation, which orientation cannot repair — #1/#2/#3 first.

---

## 5. Open questions to settle by screening (pre-register before running)

1. Does a multiplicity/overlap head trained on synthetic IRM fire on real crossings without
   firing on real 3–9 px bundles? (Screen: `diag_crossing_map.py` mechanism numbers +
   bundle recovery on roi303 VAL; tirf12 as guardrail.)
2. Does crossing-region weighting alone (no extra head) already lift
   `identity_opposite_polarity` above 0.08? (Cheapest possible arm — one line in the loss.)
3. Junction-conditioned local threshold: how much of the 0.218 → 0.791 gap closes with the
   ORACLE junction map (upper bound) before any head is trained? This is a CPU-only screen
   with GT crossing points and the cached SPARSE35 maps, and it bounds #2 the way the oracle
   mask bounds the instancer.
4. Soft-ridge target vs THIN: same trainer, same seed, both swept at both scales; report
   fragmentation (`cc/gt`) alongside junction identity.
5. If #1–#3 leave steep-crossing identity < 0.6 on roi303: PatchPerPix (spec) vs geodesic
   linking (#6) — decide by the oracle-junction-map screen in (3): if a perfect junction map
   with the current matcher reaches ≥ 0.7, the representation is sufficient and #6 is
   unnecessary; if not, the assembler must change.

---

## 6. References

Cytoskeletal / fibre tracers
- Xu T, Vavylonis D, Huang X. 3D actin network centerline extraction with multiple active contours. *Med Image Anal* 2014. DOI 10.1016/j.media.2013.10.015. PMC3945675.
- Xu T, Vavylonis D, Tsai F-C, et al. SOAX: a software for quantification of 3D biopolymer networks. *Sci Rep* 2015;5:9081. DOI 10.1038/srep09081. Code: github.com/tix209/SOAX (`snake_tip_set.cc`, `snake.cc`).
- Xu T, et al. Automated tracking of biopolymer growth and network deformation with TSOAX. *Sci Rep* 2019. DOI 10.1038/s41598-018-37182-6. PMC6368602.
- Zhang Z, Nishimura Y, Kanchanawong P. Extracting microtubule networks from superresolution single-molecule localization microscopy data. *Mol Biol Cell* 2017;28(2):333–345. DOI 10.1091/mbc.E16-06-0421. PMC5231901.
- Flormann D, et al. A novel universal algorithm for filament network tracing and cytoskeleton analysis. *FASEB J* 2021. DOI 10.1096/fj.202100048R. bioRxiv 10.1101/2021.01.04.425230.
- Breuer D, Nikoloski Z. DeFiNe: an optimisation-based method for robust disentangling of filamentous networks. *Sci Rep* 2015;5:18267. DOI 10.1038/srep18267. arXiv:1601.00847.
- Nowak J, Gennermann K, Persson S, Nikoloski Z. CytoSeg 2.0: automated extraction of actin filaments. *Bioinformatics* 2020;36(9):2950–2951. DOI 10.1093/bioinformatics/btaa035.
- Steger C. An unbiased detector of curvilinear structures. *IEEE TPAMI* 1998;20(2):113–125. DOI 10.1109/34.659930 (text from TUM report FGBV-96-03, mv.in.tum.de). Fiji Ridge Detection plugin documentation: imagej.net/plugins/ridge-detection (github.com/thorstenwagner/ij-ridgedetection).
- Eltzner B, Wollnik C, Gottschlich C, Huckemann S, Rehfeldt F. The Filament Sensor for near real-time detection of cytoskeletal fiber structures. *PLOS ONE* 2015;10(5):e0126346. DOI 10.1371/journal.pone.0126346.
- Usov I, Mezzenga R. FiberApp: an open-source software for tracking and analyzing polymers, filaments, biomacromolecules, and fibrous objects. *Macromolecules* 2015. DOI 10.1021/ma502264c.
- Stein AM, Vader DA, Jawerth LM, Weitz DA, Sander LM. An algorithm for extracting the network geometry of three-dimensional collagen gels. *J Microsc* 2008;232(3):463–475. DOI 10.1111/j.1365-2818.2008.02141.x (closed access; cited via CT-FIRE).
- Bredfeldt JS, Liu Y, Pehlke CA, et al. Computational segmentation of collagen fibers from second-harmonic generation images of breast cancer. *J Biomed Opt* 2014;19(1):016007. DOI 10.1117/1.JBO.19.1.016007. PMC3886580.
- Khatri D, Yadav SA, Athale CA. KnotResolver: tracking self-intersecting filaments in microscopy using directed graphs. *Bioinformatics* 2024;40(9):btae538. DOI 10.1093/bioinformatics/btae538. arXiv:2404.12029.
- Kiewisz R, et al. Accurate and fast segmentation of filaments and membranes in micrographs and tomograms with TARDIS. bioRxiv 2024 (v2). DOI 10.1101/2024.12.19.629196. PMC11702698. Code: github.com/SMLC-NYSBC/TARDIS.
- Koddenbrock et al. Synthetic data enables human-grade microtubule analysis with foundation models for segmentation. *PLOS Comput Biol* 2026. DOI 10.1371/journal.pcbi.1013901.
- Østerlund I, Persson S, Nikoloski Z. Tracing and tracking filamentous structures across scales: a systematic review. *Comput Struct Biotechnol J* 2023;21:452–462. DOI 10.1016/j.csbj.2022.12.023. PMC9804014.

Deep learning — crossings, overlaps, junctions
- Liu Y, Kolagunda A, Treible W, Nedo A, Caplan J, Kambhamettu C. Intersection to Overpass: instance segmentation on filamentous structures with an orientation-aware neural network and terminus pairing algorithm. *CVPRW* 2019, pp. 125–133. DOI 10.1109/CVPRW.2019.00021. PMC8046259. Code: github.com/VimsLab/filaments_quantification.
- Liu Y, Peng S, Caplan J, Kambhamettu C. Pick and Trace: instance segmentation for filamentous objects with a recurrent neural network. *MICCAI* 2023, LNCS, pp. 635–645. DOI 10.1007/978-3-031-43993-3_61.
- Hirsch P, Mais L, Kainmueller D. PatchPerPix for instance segmentation. *ECCV* 2020. arXiv:2001.07626.
- Mais L, Hirsch P, Managan C, et al. FISBe: a real-world benchmark dataset for instance segmentation of long-range thin filamentous structures. *CVPR* 2024. arXiv:2404.00130.
- Liu Y, Psota E, Pérez L. Layered embeddings for amodal instance segmentation. arXiv:2002.06264 (2020).
- Chen L, Wu Y, Merhof D. Instance segmentation of dense and overlapping objects via layering. arXiv:2210.03551 (2022).
- Ke L, Tai Y-W, Tang C-K. Deep occlusion-aware instance segmentation with overlapping bilayers (BCNet). *CVPR* 2021. arXiv:2103.12340.
- Walter FC, Damrich S, Hamprecht FA. MultiStar: instance segmentation of overlapping objects with star-convex polygons. IEEE 2021. arXiv:2011.13228.
- Hu RL, Karnowski J, Fadely R, Pommier J-P. Image segmentation to distinguish between overlapping human chromosomes. arXiv:1712.07639.
- Jiang H, Zhang R, Zhou Y, Wang Y, Chen H. DoNet: deep de-overlapping network for cytology instance segmentation. *CVPR* 2023. arXiv:2303.14373.
- Schueler J, et al. Overlap-aware segmentation for topological reconstruction of obscured objects (OASIS). arXiv:2510.06194.
- He S, Bastani F, Jagwani S, et al. Sat2Graph: road graph extraction through graph-tensor encoding. *ECCV* 2020. arXiv:2007.09547.
- Xu Z, Liu Y, Gan L, et al. RNGDet: road network graph detection by transformer in aerial images. arXiv:2202.07824; RNGDet++ arXiv:2209.10150.
- Zhou Y, Qi H, Ma Y. End-to-end wireframe parsing (L-CNN). *ICCV* 2019. arXiv:1905.03246.
- Zhang Z, Li Z, Bi N, et al. PPGNet: learning point-pair graph for line segment detection. *CVPR* 2019. arXiv:1905.03415.
- Xue N, Wu T, Bai S, et al. Holistically-attracted wireframe parsing (HAWP). *CVPR* 2020. arXiv:2003.01663.
- Xu Y, Xu W, Cheung D, Tu Z. Line segment detection using transformers without edges (LETR). *CVPR* 2021. arXiv:2101.01909.
- Li T, Chen L, Wang H, et al. Graph-based topology reasoning for driving scenes (TopoNet). arXiv:2304.05277.
- Uslu F, Bharath AA. A multi-task network to detect junctions in retinal vasculature. arXiv:1806.03175 (2018).
- Abbasi-Sureshjani S, Smit-Ockeloen I, Bekkers E, Dashtbozorg B, ter Haar Romeny B. Automatic detection of vascular bifurcations and crossings in retinal images using orientation scores. *ISBI* 2016. DOI 10.1109/ISBI.2016.7493241.
- Batra A, Singh S, Pang G, Basu S, Jawahar CV, Paluri M. Improved road connectivity by joint learning of orientation and segmentation. *CVPR* 2019 (CVF open access; code github.com/anilbatra2185/road_connectivity).
- Bekkers E, Duits R, Berendschot T, ter Haar Romeny B. A multi-orientation analysis approach to retinal vessel tracking. *J Math Imaging Vis* 2014;49:583–610. DOI 10.1007/s10851-013-0488-6. arXiv:1212.3530.
- Hannink J, Duits R, Bekkers E. Vesselness via multiple scale orientation scores. arXiv:1402.4963.
- Bekkers EJ, Duits R, Mashtakov A, Sanguinetti GR. A PDE approach to data-driven sub-Riemannian geodesics in SE(2). *SIAM J Imaging Sci* 2015;8(4):2740–2770. arXiv:1503.01433.
- Sanguinetti G, Bekkers E, Duits R, Janssen M, Mashtakov A, Mirebeau J-M. Sub-Riemannian fast marching in SE(2). arXiv:1508.02553 (2015).
- Bekkers EJ, Duits R, Mashtakov A, Sachkov Y. Vessel tracking via sub-Riemannian geodesics on R² × P¹. *GSI* 2017, LNCS pp. 773–781. DOI 10.1007/978-3-319-68445-1_89. arXiv:1704.04192.
- van den Berg N, Smets B, Pai G, Mirebeau J-M, Duits R. Geodesic tracking via new data-driven connections of Cartan type for vascular tree tracking. arXiv:2208.11004.
- van den Berg NJ, Sherry FM, Berendschot TTJM, Duits R. Crossing-preserving geodesic tracking on spherical images. arXiv:2504.03388.
- Chen D, Mirebeau J-M, Cohen LD. Global minimum for a Finsler elastica minimal path approach. *IJCV* 2017. arXiv:1612.00343.
- Franken E, Duits R. Crossing-preserving coherence-enhancing diffusion on invertible orientation scores. *IJCV* 2009;85(3):253–278. DOI 10.1007/s11263-009-0213-5.
- Duits R, Franken E. Left-invariant stochastic evolution equations on SE(2) … via invertible orientation scores. arXiv:0711.0951.
- Bekkers EJ, Lafarge MW, Veta M, Eppenhof KAJ, Pluim JPW, Duits R. Roto-translation covariant convolutional networks for medical image analysis. arXiv:1804.03393 (2018).
- Xu Y, Wang Y, Zhou W, Wang Y, Yang Z, Bai X. TextField: learning a deep direction field for irregular scene text detection. *IEEE TIP* 2019. arXiv:1812.01393.
- Wang Y, Xu Y, Tsogkas S, Bai X, Dickinson S, Siddiqi K. DeepFlux for skeletons in the wild. *CVPR* 2019. arXiv:1811.12608.
- Stringer C, Wang T, Michaelos M, Pachitariu M. Cellpose: a generalist algorithm for cellular segmentation. *Nat Methods* 2021;18:100–106. DOI 10.1038/s41592-020-01018-x. bioRxiv 10.1101/2020.02.02.931238.
- Graham S, Vu QD, Raza SEA, et al. HoVer-Net: simultaneous segmentation and classification of nuclei in multi-tissue histology images. *Med Image Anal* 2019. arXiv:1812.06499.
- Chen H, Qi X, Yu L, Heng P-A. DCAN: deep contour-aware networks for accurate gland segmentation. *CVPR* 2016. arXiv:1604.02677.
- Ronneberger O, Fischer P, Brox T. U-Net: convolutional networks for biomedical image segmentation. *MICCAI* 2015. arXiv:1505.04597.
- Scherr T, Löffler K, Böhland M, Mikut R. Cell segmentation and tracking using CNN-based distance predictions and a graph-based matching strategy. *PLOS ONE* 2020. arXiv:2004.01486.
- Sironi A, Lepetit V, Fua P. Multiscale centerline detection by learning a scale-space distance transform. *CVPR* 2014, pp. 2697–2704. DOI 10.1109/CVPR.2014.351. Journal version: Sironi A, et al. Multiscale centerline detection. *IEEE TPAMI* 2016;38:1327–1341. DOI 10.1109/TPAMI.2015.2462363.
- Hu Y, Qiu Z, Zeng D, Jiang L, Lin C, Liu J. SuperVessel: segmenting high-resolution vessel from low-resolution retinal image. arXiv:2207.13882.
- Wolf S, Bailoni A, Pape C, et al. The Mutex Watershed and its objective. *IEEE TPAMI* 2020. arXiv:1904.12654.
- Naeem R, Hagerman D, Alvén J, Svensson L, Kahl F. Trexplorer Super: topologically correct centerline tree tracking of tubular objects in CT volumes. arXiv:2507.10881.
- Naeem R, Hagerman D, Alvén J, Kahl F. RefTr: recurrent refinement of confluent trajectories for 3D vascular tree centerlines. arXiv:2511.20823.
- Liu C, Zhao T, Zheng N. DeepBranchTracer: a generally-applicable approach to curvilinear structure reconstruction using multi-feature learning. arXiv:2402.01187 (2024).
- Lin M, Bashir Z, Tolsgaard MG, Christensen AN, Feragen A. DTU-Net: learning topological similarity for curvilinear structure segmentation. arXiv:2205.11115.

Microtubule / filament segmenters and synthetic data (context)
- Sid'El Moctar SM, Ait Laydi A, Beber A, Braun M, Lánský Z, El Mourabit Y, Bouvrais H. CurvSegFlow: time-conditioned flow matching for robust segmentation of curvilinear structures in noisy biomedical images. arXiv:2606.21608 (dataset `IRM_InVitroMT`, Beber/Braun/Lánský, Zenodo).
- Sid El Moctar SM, Ait Laydi A, El Mourabit Y, Bouvrais H. MTFlow: time-conditioned flow matching for microtubule segmentation in noisy microscopy images. arXiv:2601.14841.
- Ait Laydi A, Cueff L, et al. A novel attention mechanism for noise-adaptive and robust segmentation of microtubules in microscopy images (ASE_Res_UNet). arXiv:2507.07800; bioRxiv 10.1101/2025.10.23.684152.
- Liu Y, Zhang Y. A conditional generative framework for synthetic data augmentation in segmenting thin and elongated structures in biological images. arXiv:2512.10334.
- Sheka A, et al. Instance segmentation neural network trained on synthetic dataset of actin filaments. *USBEREIT* 2025 (IEEE; full text not accessed; ResearchGate 393340589).
- Dai Z, Wang L, Yang Y, Fei X. WireSeg-32K: a physics-grounded synthetic dataset for wire instance segmentation. arXiv:2609.03102.

Project documents cited
- `docs/research/2026-09-02-scale-and-gap-audit.md` (filament width, ridge width, gap-vs-probability audit).
- `data/enc_sensitivity_testset/crossing_diag_native/crossing_diag.json` + `crossing_failures.png` (`scripts/diag_crossing_map.py`, SPARSE35 native, thr 0.98).
- `docs/protocol.md` §17 (curvature-bounded instancer, `kappa_max` derivation, PySOAX comparison, `4r/sin α` fusion length).
- `docs/superpowers/specs/2026-08-27-patchperpix-instancer-design.md` (PatchPerPix design, ceilings).
- `docs/TODO.md` §0.0-CAMPAIGN (pre-registered protocol), §0.0-OVERLAP (MT-34 junction identity by angle).
- `docs/INSTANCE_SEGMENTATION_RESEARCH.md` (2026-07 research + 2026-08-10 update).

Not retrievable in this pass (stated so the reader does not assume they were checked): Pick-and-Trace full text; SIFNE Supplemental Table S1 (φ_max, d_max, ψ_max); FiNTA Table S1 (break-off angle); Stein 2008 full text; Batra 2019 ablation numbers; Sheka 2025 full text; the Layering paper's BBBC010 AP table.
