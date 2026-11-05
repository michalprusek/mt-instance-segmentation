# Continuous centerlines from a network output: what the thin-structure literature (2019–2026) says, and what it implies for the MT segmenter

Date: 2026-09-04. Literature review only; nothing was trained or run. Every claim carries an arXiv ID or DOI; the
numbers quoted were read in the papers' method/ablation sections (not titles or abstracts) unless marked
"abstract only". Downloaded paper text was treated as data.

## 0. Scope and how to read this

**The question.** Our probability map already contains the filaments (semantic coverage of GT centerlines ~100 % at
p ≥ 0.5, per the 2026-09-02 scale-and-gap audit, `docs/research/2026-09-02-scale-and-gap-audit.md`), but no single
threshold turns it into an instance-ready mask: at the shipped 0.98–0.99 the mask BREAKS (crossings, faint
stretches, tips; 49 % of GT fail on coverage vs 9 % on precision, i.e. BREAK:FUSE ≈ 5:1), at ≤ 0.9 close parallels
FUSE. The instancer is not the bottleneck (0.95 with an oracle mask). So the review asks one thing of every paper:
*how does it decide WHERE the ridge is separately from WHETHER it exists, and what evidence shows that this
preserves continuity through low-contrast stretches better than thresholding a probability map?*

**Audit numbers every recommendation is anchored to** (SPARSE35, roi303 VAL, eval frame = 1.5× zoom):

| quantity | value | source |
|---|---|---|
| width of the p > 0.99 band on isolated real filaments | **2.08 px** (p25–p75 1.82–2.50); 7.2 % of samples never reach 0.99 | audit §A.4 |
| width of the p > 0.5 band / image FWHM / training target | 4.14 px / 5.51 px / 3 px | audit §A.4 |
| p at 2 px and 3 px from the centerline (isolated) | 0.51 / 0.18 | audit §B.3 |
| close parallels, gap 4.5–6 eval px: median min-between / ridge max | **0.35 / 0.964** (isolated ridge 0.998) | audit §B.1 |
| two ridges survive at thr 0.99 / 0.9 / 0.5, gap 4.5–6 px | 43 % / 71 % / **86 %** | audit §B.1 |
| two ridges survive at thr 0.99, gap 6–7.5 / 7.5–9 / 9–13.5 px | 90 % / 74 % / 72 % | audit §B.1 |
| real FWHM native / 1.5× vs synth native | 3.51 / 5.51 px vs 4.03–4.81 px (config 4.53) | audit §A |
| GT that is never found at any threshold | 13 % | task statement |

Two consequences drive the ranking. (i) The map is a *step*, not a *peak*: BCE on a 3-px band trains the network to
saturate inside the band and collapse outside; the ridge maximum still sits on the centerline (median p at exact GT
points 0.999), but its *value* is what dips at crossings, faint stretches and between close parallels. A decoder
that localises by "is this pixel a local maximum across the ridge?" instead of "is p above T?" is immune to the
value dip as long as the maximum survives — which the audit says it does in 86 % of the hardest parallel stretches at
p ≥ 0.5 and in ~100 % of isolated filaments. (ii) The 13 % never-found GT cannot be recovered by any decoder or loss;
that is synthetic-contrast realism and is outside this review except for §2.4.

**Closed dead ends were respected.** Per-pixel embeddings, affinities + mutex watershed, DETR-style set prediction,
soft-clDice/cbDice, 2-D hysteresis / closing / pruning, 1-px targets, Frangi/LoG prior banks, pos_weight sweeps,
TARDIS DIST, learned arc-pairing costs are not recommended again. Where a recommendation *resembles* a closed item
(R1 vs "hysteresis", R2 vs ORION's distance head, R6 vs ORION's orientation head, R8 vs DETR set prediction) the
difference is stated explicitly and the closed result is used as evidence.

---

## 1. Ranked recommendations

Costs are engineering hours before the first training run; retrain cost is the same as one SPARSE35 run unless
stated. "Break/Fuse" says which half of the dilemma the item attacks.

| # | Change | Papers | Reported gain (structure) | Why it addresses BREAK vs FUSE here | Cost | Main risk |
|---|---|---|---|---|---|---|
| **R1** | **Decoder: ridge NMS along the Hessian normal on the smoothed p-map, sub-pixel Steger rule, then hysteresis *on the 1-px ridge set* (seed p ≥ 0.9, extend p ≥ 0.3 along the ridge tangent)**. No retrain; run on cached SPARSE35 maps. | Steger 1998 (DOI 10.1109/34.659930); Sironi et al. CVPR 2014 / TPAMI 2016; DeepFlux 1811.12608 (sink rule) | Sironi: regression + NMS beats classification on PR curves for dendrites/roads (Fig. 1, qualitative + PR); DeepFlux: sink decoding +2.0/+4.9 % F over per-pixel classification with the same backbone (SK-LARGE/SYM-PASCAL) | BREAK: a local maximum survives a value dip (ridge max 0.964 at 4.5–6 px gaps, 0.999 on isolated GT points) so faint stretches stay linked at p ≥ 0.3 where coverage is ~100 %. FUSE: NMS keeps two maxima whenever a dip exists between them, and the audit shows one in essentially every stretch (median min-between vs ridge max: 0.35 vs 0.964 at 4.5–6 px; 0.06 / 0.067 / 0.029 vs ≈ 0.999 at 6–7.5 / 7.5–9 / 9–13.5 px) — a 1-px ridge set cannot merge, unlike a 2-D hysteresis mask whose low threshold floods the 0.35 valley. | 4–8 h | Spurious maxima in flat low-p background (gate on ridge curvature, not only p); saddle gap at X-crossings (≈ one FWHM, inside the instancer's gap-linking range). |
| **R2** | **Target: two heads — an existence head (BCE, decoded at 0.3–0.5) and a localisation head regressing a peaked distance-to-centerline heatmap** (Sironi's `d = exp(a(1−D/d_M))−1`, a = 6, or a Gaussian σ = 1.2 px), decoded by R1's NMS on the regression map. ORION had a distance head but decoded by skeletonising the thresholded semantic mask, so this is untested, not closed. | Sironi CVPR 2014; Oner et al. 2009.07011; CAPE 2504.00753; DeepCenterline 1903.10481; DDT 1912.03383 | Oner Tab. I (RoadTracer): U-Net trained with MSE on a truncated distance map APLS 66.3 vs segmentation 62.5; CAPE's stated reason for distance maps: "probability maps saturate near zero in disconnected regions, reducing loss sensitivity" | BREAK: a regression target has a maximum *by construction* on the centerline and a non-saturating gradient across gaps; BCE saturates (the 2.08-px band). FUSE: with a = 6, d_M = 4 px the target at the midpoint of a 4.5-px gap is 3 % of the peak — the dip is built into the label. | 1–2 d + retrain | Regression under nnU-Net's Dice/deep-supervision assumptions (compute per-scale targets from polylines, not by downsampling); exp target needs normalisation to [0,1]; L1 safer than MSE against generator label noise. |
| **R3** | **Loss: path-connectivity term on the localisation head** — CAPE (Dijkstra on the predicted distance map inside a 10-px dilation of each GT path, cost Σ ŷ²) or Oner's windowed maximin region-separation loss (α = 1e-4, β = 0.1, 64×64 windows). The generator gives exact polylines, so GT graphs and shortest paths are free. | CAPE 2504.00753; Oner 2009.07011; WPRF 2607.07123 (2026, Max-Min variant) | CAPE: DRIVE APLS 68.2 (MSE) → 80.2, clDice-loss 74.5, InvMALIS 77.7; CREMI 81.9 → 93.6; Brain 83.9 → 89.8. Oner: RoadTracer APLS 66.3 → 75.8, DeepGlobe 62.3 → 75.2, Canals 70.9 → 76.3; WPRF: up to +7.2 pp clDice on fragile datasets, DRIVE U-Net 0.782 → 0.793 | BREAK only: both losses put gradient exactly on the pixels that sever a GT path (the crossing / faint-stretch dips). They do not see fusion at < 10 px gaps (the sliver between parallels is inside the dilated GT region), so FUSE stays with R1/R2. | 1–2 d + retrain (CAPE ~200 lines; Oner needs a maximin/MALIS routine) | Over-weighting creates false positives (Oner Tab. IV: α 1e-3 → APLS 72.3 vs 75.8 at 1e-4); CAPE's α is not given in the excerpt read — tune on synth VAL. |
| **R4** | **Train at the evaluation scale with zoom randomisation using the same interpolation as inference (z ~ U(1.1, 1.5) applied after noise), plus Kreitner's local contrast field and random down-up-sampling blur.** | Kreitner et al. 2306.10941 (IEEE TMI); Amarnath 2604.02977; IMN 2110.13428; nnU-Net DA defaults | Kreitner Tab. V (nnU-Net on synthetic OCTA → real OCTA-500 DSC): 0.734 → 0.805 (structured background) → 0.819 (+speckle) → 0.836 (+contrast field) → 0.841 (+down-up sampling). Resolution study: thin-vessel sensitivity swings by up to 15.8 pp with resizing while Dice barely moves. | BREAK: the network was trained to put a 3-px target on a 4.5-px filament and is tested on a 5.5-px one (audit: 37 % of real samples wider than the synth p95); width mismatch is a direct cause of low-confidence stretches. Neutral on FUSE. | 2–4 h + retrain | Width is only part of the native-vs-1.5× effect (noise texture after bilinear zoom was not measured); nnU-Net's own scale DA (0.7–1.4, p = 0.2) may not be active in `OnlineDS`. |
| **R5** | **Target (alternative to R2): DeepFlux context flux** — 2-channel unit vector from each pixel within r px of the centerline to its nearest centerline pixel, (0,0) on the centerline; decode = flux sinks. | DeepFlux 1811.12608; AFM 1812.02122 | Ablation: flux vs binary classification, same net: +2.0 % (SK-LARGE) / +4.9 % (SYM-PASCAL); BCE on the same *dilated* GT: −6 % / −8 % — evidence against the current 3-px band; r insensitive over 3–11 (best 7) | FUSE: nearest-centerline assignment splits the gap between parallels at the midline, producing two sinks with opposite flux; fusion would require the network to invent one sink. BREAK: a sink exists wherever neighbouring flux converges, independent of confidence magnitude. | 1–2 d + retrain | Flux is undefined at the X-centre (single sink → junction node, acceptable); 2-channel regression under nnU-Net; decode threshold λ on |F̂| is bimodal so less fragile than p. |
| **R6** | **Auxiliary orientation head (double-angle `(cos 2θ, sin 2θ)` or 16–36 multi-label bins on a 3-px band) used by the decoder** for NMS normals, tangent-consistent hysteresis linking and two-orientation NMS through crossings. ORION's orientation head existed (K = 6, used for arm-pairing, null at junctions); the new use is localisation. | Batra et al. CVPR 2019 (IEEE 8953380); Kluvanec et al. 2203.13004; Sat2Graph 2007.09547; Curvi-Tracker (PR 2026) | Batra: +9 % / +7.5 % road topology metric (SpaceNet / DeepGlobe) from joint orientation learning + refinement (abstract; per-component ablation not read — CVF PDF blocked). Repo: 36 bins × 10° + background = 37 classes, band ±10 px (config) / ±5 px (code default), task weights 1:1 | BREAK: linking through dips uses tangent continuity rather than p. FUSE/crossings: a multi-label pixel at an X carries two bins → NMS per active orientation keeps two ridge tracks through the crossing. | 1 d + retrain | ORION precedent; gains are indirect (only through the decoder); adds a head to a model whose foreground is the bottleneck. |
| **R7** | **Loss (cheap topological alternative to R3): homotopy-warping loss** on the existence head, λ_warp = 1e-4, CE on the identified critical pixels. | Hu 2112.07812 (NeurIPS 2022) | RoadTracer: DICE 0.587 → 0.603, Betti error 1.591 → 1.251, warping error 10.41 → 8.85 ×10⁻³ (Tab. 2, λ = 1e-4); CE better than MSE/Dice on the critical pixels (Tab. 3) | BREAK: critical pixels are exactly the break pixels; FUSE: also flags false merges of GT-separate components (parallels ≥ 1 px apart in the 3-px raster are separate). | 0.5–1 d | Operates on binary masks (amodal crossings are one component anyway); CPU distance-transform cost per batch; weaker evidence on thin biomedical data than CAPE/Oner. |
| **R8** | **Representation (next generation): Sat2Graph graph-tensor encoding** — dense vertexness + D_max = 6 edge slots (p_e, dx, dy) per pixel, vertices every d = 20 px, decoded by local maxima + slot matching. Dense per-pixel regression, *not* DETR set prediction (no Hungarian matching, no global queries). | Sat2Graph 2007.09547 (ECCV 2020); DeH4R 2508.13669; GraphMorph 2502.11731 | TOPO-F1 70.36 → 76.40 with the *same* U-Net backbone (city-scale); qualitative: segmentation approaches fail "along closeby parallel roads", GTE handles them; stacked roads (identity through an overpass) precision 83.1 %, recall 49.8 %, 0.37 % false overpasses | FUSE and crossing identity directly: an edge slot points *through* the crossing to the next vertex of the same filament; parallels get separate vertices. BREAK: edges are predicted, not thresholded. | 1–2 wk + retrain | Vertex spacing 20 px vs 30-px filaments; GTE label inconsistency along long segments (the paper admits it); big change to the decoder. |
| **R9** | **Learned tracer refinement seeded at arc endpoints** (Curvi-Tracker Direction-Net multi-label + Forward-Net step/width; NETracer node-edge estimation + geodesic search; Wolterink entropy stop θ_H = 0.9 with 3-step moving average). | Curvi-Tracker (PR 173:112797, 2026); NETracer (ICCV 2025); DeepBranchTracer 2402.01187; Wolterink 1810.03143 | DBT Roads: Length-F1 0.816 vs U-Net 0.794, ABL (average branch length) 200 vs 148; DRIVE SSD-F1 0.893 vs 0.841 but precision 0.827 vs 0.870. DBT ablation (SSD-F1): geometric tracing alone 0.50 (M1); centerline map alone 0.845 (M3); both 0.893 (M4); + LSTM 0.91 (M5) | BREAK: tracer continues through dips using direction + step prediction. | 3–5 d | Tracers jump to adjacent branches (NETracer's motivation) — the exact close-parallel failure; lower precision in DBT's tables; the instancer already gap-links at 0.95 with an oracle mask. |

Recommended order: **R1 (diagnostic, days) → R4 + R2 (one retrain) → R3 on top (second retrain) → R6 if the
crossing-identity term still lags → R5 as the alternative to R2 if regression under nnU-Net misbehaves → R8 only
if R1–R3 stall.**

---

## 2. Specifics per recommendation

### R1. Ridge NMS + ridge-set hysteresis (Steger / Sironi / DeepFlux decoders)

Why this is not the closed "hysteresis" item. 2-D hysteresis thresholds the *map*: seeds at p ≥ hi grow into every
8-connected pixel with p ≥ lo. Between two parallels at 4.5–6 px the median valley is 0.35, so any lo < 0.35 floods
the valley and fuses them, while lo ≥ 0.35 leaves the faint stretches (p 0.2–0.5) broken — that is the null result.
Ridge NMS first reduces the map to the set of pixels that are local maxima *across* the ridge; two parallels give two
maxima whenever a dip of any depth exists between them. The audit's per-threshold percentages (43/71/86 % at
0.99/0.9/0.5 for 4.5–6 px; 90/74/72 % at 0.99 above 6 px) are confounded by ridge *loss* at the threshold, which is
exactly what NMS sidesteps; the relevant statistic is the dip itself — median min-between 0.35 against a ridge max
of 0.964 at 4.5–6 px, and 0.06 / 0.067 / 0.029 against ≈ 0.999 in the three bins above 6 px — i.e. a dip exists in
essentially every stretch, so NMS separability is bounded only by dip survival under smoothing (step 1 below), not
by any threshold. Hysteresis then runs only *along* the 1-px ridge set, which cannot fuse because the valley pixels
are not in the set.

Algorithm (Steger 1998, §3; the same NMS Sironi used on regression maps):

1. Compute Gaussian derivatives of the eval-frame probability map, `r_x, r_y, r_xx, r_xy, r_yy`, with a *small*
   σ ≤ 1.0 px (start at 0.7). Steger's σ ≥ w/√3 rule is for a raw bar profile; the network output is already a
   smooth ridge (p > 0.5 band 4.14 px ≈ FWHM → σ_ridge ≈ 1.76 px), and every extra smoothing eats the dip between
   parallels. Separability condition for two equal Gaussian ridges at gap g after smoothing with σ:
   `g > 2·sqrt(σ_ridge² + σ²)` — 4.05 px at σ = 1.0, 4.6 px at σ = 1.5 (already above the 4.5-px floor of the
   hardest audit bin), 3.8 px at σ = 0.7. Check the bound against the measured ridge width before choosing σ.
2. At every pixel take the eigenvector `n = (n_x, n_y)` of the Hessian `[[r_xx, r_xy],[r_xy, r_yy]]` with the largest
   |eigenvalue| λ₁; a bright ridge in p has λ₁ < 0. Sub-pixel offset along n:
   `t = −(r_x n_x + r_y n_y) / (r_xx n_x² + 2 r_xy n_x n_y + r_yy n_y²)`; accept the pixel as a ridge point iff
   `|t n_x| ≤ 0.5 and |t n_y| ≤ 0.5`. The sub-pixel centre is `x + t n`; keep it (the instancer's polylines and the
   5-px metric both benefit from sub-pixel vertices).
3. Salience for hysteresis: use *two* quantities — the map value `p` at the ridge point and the ridge curvature
   `κ = −λ₁`. Seed a track if `p ≥ p_hi` (0.9) *and* `κ ≥ κ_hi`; extend from a seed to the neighbouring ridge point in
   the ±tangent direction `(−n_y, n_x)` that minimises Steger's linking cost (Euclidean distance + orientation
   difference, step ≤ 1.5 px, orientation change ≤ 30°) while `p ≥ p_lo` (0.3) and `κ ≥ κ_lo`. The κ gate is what
   stops low-p background plateaus from producing spurious maxima; set κ_lo from the isolated-filament κ
   distribution on synth VAL (e.g. its 5th percentile).
4. Crossings: the Hessian at an X is a saddle (λ₁ < 0 along one normal, λ₂ ≈ 0 or > 0), and NMS leaves a gap of
   about one FWHM (≈ 5–6 eval px) at the junction. Two options: (a) rely on the existing curvature-bounded gap
   linking (raise its gap tolerance to ≥ 8 eval px for endpoint pairs whose tangents are within 20° of collinear);
   (b) Steger's own junction rule — when a track's continuation hits another track within 1 px it is extended to
   touch it and a junction node is created. (a) reuses code that already exists.
5. Output the ridge set (1-px, sub-pixel offsets attached) to `skeleton_graph` in place of the thresholded mask.

Falsifiable predictions on the cached SPARSE35 roi303 VAL maps (no training): (i) at 4.5–6 px gaps the fraction of
stretches with two ridge tracks rises from 43 % (thr 0.99) to at least the 86 % measured at 0.5 — the NMS
condition (any dip, both ridges ≥ 0.3) is strictly weaker than the 0.5-threshold condition, provided σ respects the
separability bound of step 1; (ii) isolated filaments lose the 7.2 % of
samples that never reach 0.99 as break sources — breaks per 1000 px of isolated GT should fall by that order; (iii)
coverage-failures (49 %) should fall more than precision-failures (9 %) rise. If (iii) fails, the value dips are deeper
than the audit's medians suggest at crossings (not measured there) and R2/R3 are needed before R1 can pay off.

DeepFlux's decoder is the trained-representation analogue of the same idea: a pixel is skeleton iff `|F̂(p)| > λ` and
its flux-pointed neighbour has `|F̂| ≤ λ` (Algorithm 1 in 1811.12608), followed by a dilation-erosion pair. That is a
sink test, not a magnitude threshold, and it is why DeepFlux "excels in the high-precision regime" of the PR curve
(§4.3). See R5.

### R2. Existence head + peaked distance-heatmap localisation head

Targets (all computed at the *eval* scale from the generator's exact polylines, D = Euclidean distance to the
nearest centerline point, computed from the rasterised sub-pixel polyline or with a KD-tree on densified vertices):

- Sironi (CVPR 2014, eq. 3): `d(x) = exp(a(1 − D(x)/d_M)) − 1` for `D < d_M`, else 0; a = 6 in all their
  experiments, d_M = s/2 with s their 21-px feature window. For us d_M = 4 px (eval frame): the midpoint of a
  4.5-px gap (D = 2.25) gets `exp(2.625)−1 = 12.8` vs the peak `exp(6)−1 = 402`, i.e. 3 % — the dip needed by NMS is
  in the label. Normalise by `e^a − 1` so the head predicts in [0, 1]. Sironi's motivation is exactly ours: "points
  on the centerline itself ... and their immediate neighbours look very similar" so a classifier "does not guarantee
  that its value will be maximal at the centerline itself"; a regressor "whose values decrease monotonically as the
  distance ... increases" makes NMS well-posed (§3.1).
- Oner (2009.07011, eq. 3): `y = min(D, D_max)`, D_max = 20 px for roads; for us 6–8 px. Decode = local *minima*
  across the ridge. Linear range is friendlier to MSE than the exponential; the dip is shallower (at D = 2.25 the
  value is 2.25/6 = 37 % of D_max) but still a strict local minimum on the centerline.
- DeepCenterline (1903.10481, eq. 1–2): locally normalised distance `d_i = D_i / max_S D` on each cross-section,
  then `log(D + δ)` "to further highlight contrast at portions closer to centerline"; endpoints as a Gaussian
  heatmap `exp(−D_e²/Δ²)` thresholded at half its maximum. Their endpoint head is worth copying as-is: our
  "tips" break class is an endpoint-detection problem, and a Gaussian endpoint heatmap is the standard fix for its
  class imbalance.
- Gaussian ridge heatmap `h = exp(−D²/(2σ_h²))`, σ_h = 1.2 px: the keypoint-heatmap convention (DeepCenterline's
  endpoint head; pose estimation). Simplest to implement; at D = 2.25 the value is 17 % of the peak.

Loss: MSE (Oner, CAPE, DeepCenterline) or L1 with DeepFlux-style class balancing (eq. 3 of 1811.12608:
context+centerline pixels weighted by `|R_b|/N`, background by `(|R_c|+|R_s|)/N`). Keep the existence head on
BCE(+Dice) exactly as now but *decode it at 0.3–0.5*, where the audit says coverage is ~100 %; its only job is to
gate the NMS output (R1 step 3 uses `p_exist` for `p_lo/p_hi`) and to kill background ridges. Deep supervision:
compute the regression target per scale from the polylines (D scales with the zoom), never by max-pooling the
full-resolution heatmap.

Why the two-head split is the literal answer to the task's question. CAPE (2504.00753 §3): "distance map pixel
values rise linearly as disconnections increase, more effectively penalizing longer or severe gaps. Probability
maps, however, saturate near zero in disconnected regions, reducing loss sensitivity." Oner (§III-A): the regression
loss "penalize[s] the deviation of the predicted road center from its annotated position more gently than ... cross
entropy", which is what lets a connectivity term "force the network to predict uninterrupted roads ... even if they
do not coincide perfectly with the annotations". DDT (1912.03383) adds the classifier-side argument: "it is hard for
the classifier to distinguish voxels on the skeleton itself from those immediately next to it, as they have similar
features but different labels" (§1), which is a description of the 2.08-px band.

Relation to ORION. ORION's head set was semantic + K-bin orientation + distance + junction + endpoint
(`docs/MT_SEGMENTATION_KNOWLEDGE.md` l.72), but its decoder was "skeletonize semantic → orientation-binned field →
arm-pairing" (l.57): the distance head was never used to *localise*. R2's claim is about the decoder reading the
regression head with NMS, which ORION did not do.

### R3. Path-connectivity loss on the localisation head

CAPE (2504.00753 §3.1–3.2), verbatim procedure:
1. GT graph G = (V, E) from the polylines. For us: V = polyline endpoints + a vertex every 20 px along each polyline
   + crossing points; E = consecutive vertices along the same polyline (crossings do *not* connect the two
   filaments — the amodal identity is in the graph).
2. Pick a connected pair (v₁, v₂), take the GT path (trivially the polyline segment), render it, dilate by 10 px →
   mask M. Project v₁, v₂ onto the predicted distance map ŷ by taking the local minimum in a 7×7 window (v′₁, v′₂).
3. Dijkstra on `ŷ · M` between v′₁ and v′₂ with node cost `ŷ(n)²`; `L_CAPE += Σ_{n∈path} ŷ(n)²`. Remove the path's
   edges from E; repeat until E is empty.
4. `L = L_MSE(y, ŷ) + α L_CAPE`. α is not in the sections read; choose it so that at initialisation the two terms are
   within a factor of ~3 and sweep {0.1, 1} on synth VAL (Kirchhoff's SRL paper restricted its weight sweep to
   exactly {0.1, 1.0} for the same reason).
Reported (Tab. 1): DRIVE APLS 68.2 (MSE) / 76.8 (perceptual, Mosinska) / 74.5 (clDice loss) / 77.7 (InvMALIS) / **80.2
(CAPE)**; Dice 77.1 → 80.3; CREMI APLS 81.9 → 93.6; Brain (3-D LM) 83.9 → 89.8 with Dice 75.2 → 78.3.

Oner (2009.07011 §III-B), verbatim: `L = L_MSE + α L_TOPO`, `L_TOPO = L_disc + β L_conn`, α = 1e-4, β = 0.1
(dis-connection term ten times the false-connection term), loss computed independently in 64×64 windows and
summed. `L_disc = Σ_{p∈R} w_p ŷ[p]²` where R is the GT centerline dilated by 5 px and `w_p` counts the pairs of
background pixels on opposite sides of a GT road whose maximin path bottleneck is p (Kruskal maximum-spanning-tree
computation from MALIS, Turaga et al.). `L_conn = Σ_{A} Σ_{p∈A} v_p (ŷ[p] − y_D[p])²` penalises a predicted road
that splits one GT background region. Ablation (Tab. IV/V): α 1e-3 / 1e-4 / 1e-5 / 0 → APLS 72.3 / 75.8 / 71.4 /
66.3; β 1 / 0.1 / 0.01 → 71.4 / 75.8 / 74.3; windowing 64×64 beats global (75.8 vs 72.5) because "a single road
pixel gets a weight equal to the product of the size of the connected components" otherwise.

What neither loss fixes: fusion of parallels closer than the dilation radius (5–10 px) — the background sliver
between them lies inside R / M, so no term fires. That is by design here: BREAK is the 5:1 majority failure and
FUSE is handled by R1/R2's dip. If a FUSE-aware term is wanted later, WPRF (2607.07123) reports its Max-Min
reachability objective as bottleneck-focused (+7.2 pp clDice on "structurally fragile" datasets, 87 % of 54
architecture×dataset configurations improved, code released), but it is a July-2026 preprint with no
independent replication.

### R4. Train at the evaluation scale; randomise zoom, contrast field, blur

The mismatch (audit §A): synth filaments at native resolution are 4.03–4.81 px FWHM (config 4.53); real filaments
are 3.51 px native, **5.51 px after the eval 1.5× zoom**; 37 % of real eval-frame samples are wider than the synth
p95. nnU-Net's default spatial augmentation is isotropic scaling in (0.7, 1.4) with p = 0.2
(`nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py`, `scaling=(0.7, 1.4)`, `p_scaling=0.2`,
`p_synchronize_scaling_across_axes=1`) — if `OnlineDS` bypasses nnU-Net's transform pipeline, no scale augmentation
is applied at all; check `scripts/train_v5.py`.

Do: (1) render synth at native, add noise, *then* apply the same `zoom(img, z)` (same order/interpolation as
`eval_v5.py`) to image and polylines with z ~ U(1.1, 1.5): the audit's width-matching scale is 1.15–1.37× (config
expectation 1.29×), and the upper end is the eval zoom itself; synth 4.03–4.81 px then spans 4.4–7.2 px around
the real eval-frame 5.51 px (per-frame medians ×1.53 ≈ 4.85–5.83 px). Zooming after noise reproduces the
bilinear-smoothed noise texture the network sees at test. (2) Kreitner's contrast field: 9×9 control points
Γ ∈ [−1, 1], bicubic upsampled, applied as a multiplicative/gamma modulation of local contrast (2306.10941 §II-C-1,
eq. 8ff). (3) Random down-up sampling by s ~ U(0.5, 1) (Kreitner used U(0.25, 1) for OCTA) to emulate defocus.
Kreitner's ablation (Tab. V, nnU-Net trained on synthetic OCTA, tested on real OCTA-500, DSC): none 0.734;
+ structured background noise 0.805; + speckle 0.819; + contrast field 0.836; + down-up sampling 0.841. Their
strongest single lever — structured background — is the one the project already has (real empty-field
backgrounds); the contrast field and blur are the two not yet in `mt_generator.py` as far as the calib files show.
Menten et al. (2207.11102) found shearing/stretching/binomial noise "do not substantially boost" performance —
geometric warps are not the lever, photometric ones are. The resolution study (2604.02977) is the warning against
the opposite direction: aggressive downsampling cut thin-vessel sensitivity by up to 15.8 pp while Dice stayed flat.

### R5. Context-flux target (DeepFlux)

Target (1811.12608 eq. 1): dilate the 1-px centerline by a disk of radius r (r = 7 in the paper, F-measure insensitive
for r = 3…11; use r = 5–7 in the eval frame); for each context pixel p the flux `F(p) = (N_p − p)/|N_p − p|` where
N_p is the nearest centerline pixel; `F = (0,0)` on the centerline and in the background. Loss: L2 with the class
balancing of eq. 3. Decoder (Algorithm 1): bin the predicted flux direction into 8 neighbours; pixel p is skeleton
iff `|F̂(p)| > λ` and `|F̂(N_{∠F̂(p)}(p))| ≤ λ`; then dilate radius k₁, erode radius k₂. Ablation (Tab. 3–4, F-measure):
classification baseline 0.696 → flux alone 0.716 (+2.0 %, SK-LARGE), 0.409 → 0.458 (+4.9 %, SYM-PASCAL); ASPP alone
0.712 / 0.426; both 0.732 / 0.502 (the two gains are additive).
Retraining the baseline with BCE on the *dilated* GT drops to 0.673 (−6 %) / 0.425 (−8 %): a wider binary band
alone hurts, which is direct evidence against tuning our 3-px band width and for changing the representation.

Why it suits bundles: the nearest-centerline assignment partitions the gap between two parallels at the midline;
each side's flux points to its own filament, giving two sinks with a sign change between them. Why it suits faint
stretches: the sink test is a direction-consistency test, not a magnitude test; a low-contrast stretch whose
surrounding flux still converges is kept. At an X the flux is ambiguous in a ~r-px disc → one sink cluster → one
junction node, which the curvature-bounded matcher already expects. AFM (1812.02122) is the same idea for straight
segments (projection vector to the nearest line, "squeeze" decoder, +4.5 % on Wireframe) and is not needed
separately.

### R6. Orientation head for the decoder

Encodings. Batra (CVPR 2019, IEEE 8953380; repo `anilbatra2185/road_connectivity`): `getVectorMapsAngles` paints,
for every centerline segment, the pixels within θ of it (code default θ = 5 px; `config.json` uses
`angle_theta: 10`) with the segment angle in [0, 360), quantised into `angle_bin: 10`° bins → 36 classes + 1
background = `task2_classes: 37`; CE loss, `task1_weight = task2_weight = 1`; stacked-hourglass multi-branch.
Kluvanec (2203.13004): regress `(cos 2θ, sin 2θ)` — "maps any direction and its reverse to the same point", continuous,
no wrap discontinuity; used to split overlapping chromosomes in a second stage. For crossings use *multi-label*
bins (Curvi-Tracker's Direction-Net is explicitly multi-label; ORION's field was K = 6 multi-label): a pixel on an X
carries two active bins.

Use in the decoder (this is the part ORION never did): (i) replace the Hessian eigenvector in R1 step 2 by the
predicted normal — robust where the p-map is noisy; (ii) in R1 step 3 link only to ridge neighbours whose predicted
orientation differs by ≤ 20°; (iii) at pixels with two active bins run NMS twice, once per orientation, producing two
ridge tracks that pass *through* the crossing instead of one saddle gap. Sat2Graph's finding that the label is
"consistent ... at important places such as intersections and overpass/underpass roads" while inconsistent along long
segments (2007.09547 §5) is the same observation: orientation supervision is most informative exactly at junctions.

Caveat: the project already measured that orientation arm-pairing at junctions is not better than DIST-style
matching (closed item), and junction identity with an oracle mask is 0.75, i.e. the matcher is not the limiting
factor. R6 is therefore a localisation aid, expected to move the crossing gap and tangent continuity, not the
pairing rule.

### R7. Homotopy-warping loss (cheap topological term)

Hu (2112.07812): identify "critical pixels" by warping the prediction towards the GT (and GT towards the prediction)
under digital-topology constraints using distance transforms; apply a pixel loss only there:
`L = L_dice + λ_warp L_warp`. Ablation on RoadTracer (Tab. 2): λ = 0 / 2e-5 / 5e-5 / 1e-4 / 2e-4 → Betti error
1.591 / 1.307 / 1.412 / 1.251 / 1.354, DICE 0.587 → 0.603 at 1e-4; CE on the critical pixels beats MSE and Dice
(Tab. 3: Betti 1.251 vs 1.429 / 1.368); both warping directions together beat either alone (Tab. 4). Comparators in
the same table: clDice 0.591 DICE / Betti not shown in excerpt; TopoLoss (HuTopo) 0.584 / 1.378. Cost is a CPU
distance transform per sample. It is on the *binary* existence head, so it complements rather than replaces R3.

### R8. Sat2Graph graph-tensor encoding (next generation)

Encoding (2007.09547 §3.1): per pixel a vector `[p_v, (p_e1, dx1, dy1), …, (p_e6, dx6, dy6)]`; vertices are placed
every d = 20 px along GT edges (d < 5 "converts GTE back to road segmentation, making GTE unable to represent
stacking roads"; d = 50 "hard to approximate curvy roads"); overlapping edges (stacked roads) are shifted so
vertices from the two edges are ≥ 5 m apart. Losses: CE on p_v and p_e, L2 on (dx, dy), edge losses only at GT
vertex positions. Decoding: local maxima of p_v above p_thr; for each slot i of vertex v connect to the u
minimising `||v + (dx_i, dy_i) − u|| + w · cos_dist((dx_i, dy_i), u − v)`, w = 100 (robust for 5–100), max
distance 15 m; a higher edge threshold than vertex threshold helps. Results: TOPO-F1 70.36 → 76.40 with the same
U-Net backbone; with the same DLA backbone (Tab. 1, the fair row) APLS 57.22 → 63.14 (city-scale) and
56.36 → 64.43 (SpaceNet) — the segmentation U-Net rows (52.50 / 53.77) differ in backbone as well as encoding;
stacked-road crossing points precision 83.11 %, recall 49.81 %, false-overpass rate 0.37 %.

Why it is not the closed DETR item: it is dense, per-pixel, anchor-free regression trained with pixel losses — no
learned queries, no set matching, no global assignment. Why it fits the bottlenecks: the slot vector encodes
*which* next vertex a track continues to, so identity through a crossing and separation of parallels are
label-level facts, not decoder heuristics. Why it is last: 1–2 weeks of head + decoder work, the 20-px vertex spacing
vs 30-px filaments needs d ≈ 8–10 px in our frame (moving toward the "d < 5 = segmentation" regime), and DeH4R
(2508.13669) shows the field has since moved to decoupled vertex detection + adjacency prediction + graph growth
(+4.66 APLS, +10.18 IoU over RNGDet++ on CityScale, ~10× faster) — if R8 is ever built, build DeH4R's decoupling,
not RNGDet's autoregression.

### R9. Tracer refinement from arc endpoints

Wolterink (1810.03143): CNN classifies the local direction over a discretised sphere and regresses the radius
(`ℓ = ℓ_c + λ_r ℓ_r + λ_w||θ||²`, λ_r = 10); the tracker excludes directions ≥ 60° from the previous step, and stops when
the normalised entropy of the direction posterior exceeds θ_H = 0.9, computed as a moving average over three steps
"to encourage tracking through" low-contrast areas; it also stops when it comes too close to an already tracked
centerline. DeepBranchTracer (2402.01187): direction as K-bin classification per axis + cosine-similarity loss on
the softmax expectation (eq. 6–9), radius by MSE, an LSTM over the last l points, a weighted-BCE centerline map and
a *boundary* map that decides whether to continue; loss weights λ_d = 1, λ_r = 100, λ_c = 1; the next point is the
centerline pixel above T_c nearest to the geometric prediction (eq. 15). Its ablation (Tab. 1, fMOST-VTA, SSD-F1) is the
key number for us: geometric tracing alone (M1) 0.502, with LSTM (M2) 0.519; centerline map alone (M3) 0.845;
centerline + geometric (M4) 0.893; all three with LSTM (M5) 0.910. Tracing
without a dense image head is bad; tracing on top of one mostly improves *continuity* (ABL 128 → 198). NETracer
(ICCV 2025, CVF open access) adds edge prediction between successive nodes and a geodesic search precisely because
"existing iterative tracing methods tend to jump to adjacent branches" — the close-parallel failure. Curvi-Tracker
(Pattern Recognition 173:112797, 2026) deploys tracker agents on a coarse segmentation with a multi-label
Direction-Net and a Forward-Net for step size and width, reporting consistent gains in volumetric and topological
metrics on vessels, roads and cracks (abstract only).

For this project a tracer is a learned replacement for the fixed gap-linking step. Since the instancer reaches 0.95
with an oracle mask, the expected gain is bounded by how many breaks R1–R3 leave; build it only if endpoint pairs
with collinear tangents but > 8 px gaps remain a measurable failure class after R3.

---

## 3. What NOT to do (with evidence)

1. **Skeleton Recall Loss on its own.** Kirchhoff et al. (2404.03010, Tab. 2, nnU-Net): Roads clDice 88.79 → 89.06,
   DRIVE 80.26 → 80.83, Cracks 95.76 → 96.04 — gains of 0.3–0.6; the large gains are on 3-D ToothFairy (89.16 → 92.05)
   and on multi-class TopCoW where clDice runs out of memory. The independent replication (2508.11374, Tab. 3, nnU-Net,
   5 seeds) finds DRIVE Dice 84.05 → 84.01 and FPR 1.06 → 1.70 (significant), Roads FPR 1.35 → 1.83, Cracks 0.25 → 0.31
   — "the model starts over-predicting when SRL is utilized" — with only 3 of 15 metrics significantly improved.
   On our target the loss is a soft recall on a dilated skeleton, and our target *is* a dilated skeleton, so SRL
   degenerates to recall-weighted BCE — the pos_weight sweep (closed, null) already covered it.
2. **clDice / cbDice / soft-skeleton losses (closed).** Kirchhoff shows the soft skeleton is "perforated and jagged";
   Topograph (2411.03228, Tab. 1, Roads) clDice-loss costs Dice 0.819 → 0.803 and clDice 0.720 → 0.704 vs plain Dice.
3. **Persistent-homology losses (TopoLoss / HuTopo, Betti matching).** Berger et al. (2403.11001) and Topograph both
   report HuTopo "optimizes for the correct number of topological features but disregards spatial alignment"; on Roads
   Topograph's own gains are DIU 7.31 → 6.52 with clDice unchanged (0.720 → 0.711); TopoLoss's big wins are on EM
   neuron boundaries (ISBI12 Betti error 2.785 → 0.429, 1906.05404 Tab. 1), a closed-loop topology unlike open
   filaments. Cost is a persistence computation per patch per step. If a topological term is wanted, R7 is the cheap
   one; the Betti-matching efficient algorithm (2407.04683) is for 3-D.
4. **2-D hysteresis, closing, pruning on the map (closed).** The audit explains the null: valley 0.35 at 4.5–6 px
   means any low threshold that bridges faint stretches also floods the valley. Only a ridge-set (R1) or a
   representation with a built-in dip (R2/R5) escapes this.
5. **GAN-only domain adaptation of the synthetic frames.** Kreitner (2306.10941, Tab. VII): their GAN reaches 0.866
   DSC at epoch 50 but 0.836 / 0.826 at epochs 25 / 100, CycleGAN 0.852 / 0.818 / 0.826, NICE-GAN 0.34–0.39; the
   handcrafted ladder is a stable 0.841. The project's no-self-training rule also applies: a GAN trained on real
   frames is a form of real-data fitting.
6. **Tracing without a dense image head** (DBT M1: SSD-F1 0.50 vs 0.91) and **DETR/set prediction** (closed, 0.013).
7. **Simplified synthetic morphology / binary "DRIFT"-style structures.** SynthMT (PLOS Comput Biol 2026, DOI
   10.1371/journal.pcbi.1013901): tuning on DRIFT's simplified binaries "leads to substantial performance
   degradation" while tuning on SynthMT matches tuning on human-annotated real frames. Note SynthMT has no
   per-augmentation ablation and its metric (per-instance skeleton IoU, matched, unmatched = 0; SAM3Text 0.89 vs human
   0.88) is not comparable to strict centerline-F1 at 95 %/95 %.
8. **Downsampling or training coarser than test.** 2604.02977: thin-vessel sensitivity is the metric that moves
   (up to −15.8 pp) while Dice hides it; the audit's 0.4999 → 0.5202 native-vs-1.5× result is the local version.
9. **Widening the binary band instead of changing the representation.** DeepFlux's dilated-GT BCE control: −6 % / −8 %.
10. **Plug-and-play "learned reconnecting regularisation" (2408.12943).** Unsupervised, abstract-only evidence; skip.

---

## 4. Evidence digest by sub-question

### 4.1 Training targets other than a binary band

| target | paper | decoder | continuity evidence | notes for us |
|---|---|---|---|---|
| exp-peaked distance `exp(a(1−D/d_M))−1`, a = 6 | Sironi CVPR 2014 (EPFL Infoscience copy read; DOI not verified); TPAMI 2016 extension | NMS along the normal from the oriented-flux/Hessian eigenvector; scale = argmax over radial regressors | classification "does not guarantee maximal response at the centerline"; PR curves vs OOF/OFA/MDOF and a GradientBoost classifier on aerial roads, brightfield/2-photon dendrites | The template for R2. Their scale handling (hybrid: regressors at 2–4 radii + scale-space rescaling) is overkill for our 5–6 px width; one radius + zoom augmentation suffices. |
| truncated linear distance `min(D, 20)` | Oner 2009.07011 (TPAMI) | threshold on the distance map (their eval) | U-Net MSE 66.3 APLS vs seg 62.5; + TOPO 75.8 | Pairs with R3. |
| distance map + MSE | CAPE 2504.00753 | threshold | see R3 | Reason for distance maps quoted in R2. |
| locally normalised log-distance + endpoint Gaussian heatmap | DeepCenterline 1903.10481 (MICCAI 2019) | minimal path (Dijkstra with vertex weight `exp(ŷ)`) root → endpoints, merged | single-pixel centerlines "with no spurious branches" on 620 CTA coronary trees | Endpoint heatmap head is a cheap fix for tip breaks; the minimal-path decoder needs a root (tree assumption) — not for us. |
| surface-distance map, quantised, + geometry-aware refinement | DDT 1912.03383 (CVPR 2020) | reconstruct mask from skeleton + radius | PDAC vessels DSC 49.8 → 56.0 | The "classifier cannot separate skeleton from neighbour" argument. |
| context flux (unit vector to nearest skeleton pixel, r = 7) | DeepFlux 1811.12608 (CVPR 2019) | sink detection (Alg. 1) + closing | +2.0/+4.9 F over classification; dilated-GT BCE −6/−8 | R5. |
| attraction field (projection vector to nearest line) | AFM 1812.02122 (CVPR 2019) | squeeze module | +4.5 % on Wireframe | Straight segments; superseded by DeepFlux for curves. |
| orientation bins (36 + bg), band ±5–10 px | Batra CVPR 2019 | joint with segmentation; connectivity-refinement net trained on corrupted GT | +9 / +7.5 % topology (abstract) | R6. |
| double-angle `(cos 2θ, sin 2θ)` | Kluvanec 2203.13004 (CHIL 2022) | second-stage split of overlapping instances by orientation | qualitative + accuracy on synthetic chromosomes | Encoding for R6. |
| graph tensor: vertexness + 6 edge slots (p, dx, dy) | Sat2Graph 2007.09547 (ECCV 2020) | local maxima + slot matching | TOPO-F1 +6 with the same backbone; parallels and stacked roads | R8. |
| point sets per scatter region | PointScatter 2209.05774 (ECCV 2022) | greedy bipartite matching | tubular datasets; not read in depth | Set-prediction flavour; closed-list adjacent. |
| segmentation + skeleton + local discontinuity map (3 heads) | GLCP 2507.21328 (MICCAI 2025) | dual-attention refinement | "explicitly target local discontinuity regions" (abstract) | A supervised "where will I break" head; the generator can label it (pixels where p < T on synth VAL). Cheap add-on to R2 if breaks persist. |

### 4.2 Connectivity-preserving losses that are not clDice

| loss | mechanism | cost | published effect on thin structures | over-fires? |
|---|---|---|---|---|
| CAPE (2504.00753) | Dijkstra path cost Σ ŷ² on the predicted distance map within a 10-px dilation of each GT path | CPU Dijkstra per GT edge per sample | DRIVE APLS 68.2 → 80.2, Dice 77.1 → 80.3; CREMI 81.9 → 93.6 | not reported; masked search limits it |
| Oner region separation (2009.07011) | MALIS-style maximin between background pixels across a GT road, windowed 64×64, plus a false-connection term | Kruskal MST per window | RoadTracer APLS 66.3 → 75.8; DeepGlobe 62.3 → 75.2; Canals 70.9 → 76.3 | yes at α ≥ 1e-3 (72.3) and without L_conn ("many false positive road segments") |
| WPRF (2607.07123, 2026) | differentiable Max-Min (widest-path) reachability on a domain-restricted graph; bottleneck-aware weighting | DP per sample; no inference cost | +7.2 pp clDice max; DRIVE U-Net clDice 0.782 → 0.793, Mask2Former 0.805 → 0.831; Massachusetts Roads smallest gains | not reported |
| Skeleton Recall (2404.03010) | soft recall on a radius-2-dilated GT skeleton | negligible | ≤ +0.6 clDice on 2-D binary tasks | yes: FPR ↑ significantly on all three 2-D datasets in 2508.11374 |
| Homotopy warping (2112.07812) | CE on topologically critical pixels found by warping | distance transforms per sample | RoadTracer Betti 1.591 → 1.251 at λ = 1e-4 | not at λ ≤ 1e-4 |
| TopoLoss / HuTopo (1906.05404) | persistence-diagram matching | persistence per patch | ISBI12 Betti 2.785 → 0.429; weaker on Roads (Topograph Tab. 1: 7.50 DIU vs 7.31 Dice) | spatial mismatch (2403.11001, 2411.03228) |
| Betti matching (2211.15272; efficient 2407.04683) | spatially matched barcodes | high; 3-D algorithm in 2407.04683 | Roads DIU 6.73, B0 0.94 (best B0), clDice 0.707 | — |
| Topograph (2411.03228) | graph of union/intersection components; strict topology guarantee | moderate | Roads DIU 7.31 → 6.52, BM 6.62 → 5.64; clDice unchanged | — |
| Perceptual (Mosinska 1712.02190) | VGG feature loss | low | DRIVE APLS 76.8 in CAPE's table (2nd best) | — |

### 4.3 Threshold-free / NMS-style decoders and tracers, and what they do at crossings and gaps

- **Steger 1998**: Gaussian-derivative Hessian, sub-pixel maximum along the normal (`|t n| ≤ 0.5`), hysteresis on the
  second directional derivative, linking by distance + orientation; junctions handled by extending a line to touch
  the one it runs into. Gaps at X-crossings are the known weakness (the saddle); linking resolves them.
- **Sironi 2014/2016**: regress, take the max over radii, Canny-like NMS "along a line perpendicular to the local
  orientation" from the oriented-flux eigenvector, threshold the NMS image. No explicit junction step.
- **DeepFlux**: sink test on the flux field; ligature regions are the paper's stated strength ("less prone to gaps,
  discontinuities"); an X gives one sink cluster.
- **AFM squeeze**: pixels vote for their projection point; line segments are the accumulations.
- **DeepCenterline**: Dijkstra from a root to detected endpoints; paths merge when they meet (tree only).
- **Sat2Graph decode**: local maxima of vertexness + per-slot nearest vertex with a cosine penalty; crossings are
  vertices with 4 active slots; stacked roads are two vertices at the same location shifted ≥ 5 m in the label.
- **RoadTracer (1802.03680), RNGDet (2202.07824), RNGDet++ (2209.10150), DeH4R (2508.13669)**: iterative graph
  growth; at intersections the policy emits several next-vertex proposals; RoadTracer-style methods are the weakest
  in the Warping paper's table (DICE 0.547, Betti 2.218 vs U-Net 0.587 / 1.591) and in DBT's table (SSD-F1 0.313),
  which is why the field moved to hybrid detect-then-connect (Sat2Graph, DeH4R).
- **Wolterink tracker**: bidirectional from a seed, direction posterior over a sphere, entropy stop 0.9 with 3-step
  averaging, stop near an already-tracked centerline (prevents doubling but also prevents crossing a tracked
  filament — for MTs the "too close" rule must be replaced by a tangent-angle rule).
- **DeepBranchTracer / NETracer / Curvi-Tracker / TopoVST (2603.14909)**: direction + radius + continue/stop heads,
  with NETracer adding node-edge prediction and geodesic search to stop jumps to adjacent branches, and TopoVST
  (2026) using sphere graphs + GNN with space-occupancy filtering against spurious skeletons (abstract only).
- **GraphMorph (2502.11731)**: graph decoder + "SkeletonDijkstra" morphing the centerline probability map to the
  predicted graph; reports that using the resulting centerline mask "significantly reduces false positives"
  (abstract only).

### 4.4 Scale handling for 3–6 px structures with a ~1.3× train/test mismatch

- nnU-Net default: isotropic scaling (0.7, 1.4) with p = 0.2 — nominally covers 1.3× but only one sample in five is
  scaled at all, and only if the online generator goes through nnU-Net's transforms.
- Kreitner: random down-up sampling s ~ U(0.25, 1) is an explicit blur/scale augmentation with a measured +0.5 DSC on
  top of the noise ladder.
- Sironi: scale-space rescaling of filters lets one regressor serve neighbouring radii but "assumes that the aspect
  of tubular structures is scale invariant"; they use a small set of radii + rescaling. For a 1.3× mismatch, training
  at the test scale (R4) is simpler and exact.
- DeepFlux: context radius r insensitive over 3–11 → the flux representation tolerates width mismatch better than a
  fixed 3-px band.
- IMN (2110.13428): an up-sampling encoder ("magnifying glass") to keep thin capillaries, DSC 90.2 % on three
  OCTA sets; and 2604.02977 (2026): for low/mid-resolution datasets thin-vessel sensitivity peaks at native
  resolution and any downsampling hurts. Together: do not shrink; if anything, compute at ≥ test scale.
- Sub-pixel localisation: Steger's `t` offset (R1 step 2) gives sub-pixel centres from any smooth map; with a
  regression target (R2) the maximum is well defined and the offset is meaningful; with the saturated BCE map it is
  biased toward the band centre but still usable.

### 4.5 Synthetic-to-real transfer for filaments: what mattered

- **Kreitner 2306.10941 (OCTA, nnU-Net, synthetic-only training)**: ablation reproduced in R4; the ordering of
  levers is background structure ≫ contrast field ≈ speckle > blur; geometric warps (Menten 2207.11102) did not
  help; GAN adaptation can beat handcrafted (0.866) but is checkpoint-unstable.
- **SynthMT (PLOS Comput Biol 2026, DOI 10.1371/journal.pcbi.1013901; bioRxiv 10.64898/2026.01.09.698597)**: IRM
  microtubules; generator = polylines with segment length + correlated bend angles (persistence via flip
  probability), binary mask ⊛ PSF × contrast + background, additive Gaussian + Poisson + multiplicative speckle,
  spatially correlated background, vignetting, blur, distractor spots; parameters tuned by TPE (1000 iterations)
  maximising cosine similarity of DINOv2 layer-5 features to real frames. Evaluates nine existing methods
  (FIESTA, StarDist, TARDIS, SAM/SAM2/µSAM/CellSAM/Cellpose-SAM, SAM3/SAM3Text); no U-Net trained on SynthMT; no
  augmentation ablation; only "SynthMT-tuned = real-tuned ≫ DRIFT-tuned". The one transferable lesson is the
  calibration target (feature similarity), which the project already uses. Overlapping instances are stored as one
  slice per instance — the format, not the generator, keeps crossings amodal.
- **ASE_Res_UNet (2507.07800)**: trained on the MicSim_FluoMT synthetic fluorescence MT set (1192 images with
  "diversity in microtubule density, curvature, and crossing patterns"; an "easy" uniform-intensity and a
  "complex" fading-intensity variant), augmentation only rotations/flips/colour jitter; on SynthMT Dice 0.981;
  real transfer is shown qualitatively on 49 C. elegans frames (the model shown on real is the one trained on real
  MicReal, so no synthetic→real number). Confirms that intensity fading along filaments is the synthetic variant
  that makes extremities hard — a generator lever for our "tips" break class.
- **MTFlow (2601.14841)**: flow-matching refinement; on real MicReal_FluoMT Dice 0.697 vs U-Net++ 0.704, PR-AUC
  0.774 vs 0.744 — parity, not a lever.
- **cGAN augmentation for filaments (2512.10334)**: Pix2Pix from binary masks with a filament-aware skeleton loss;
  microtubule IoU 0.944 → 0.954, SKIoU 0.978 → 0.987; actin SKIoU 0.958 → 0.966 — small, and requires real annotated
  data to train the generator (conflicts with the no-real-training rule).
- **DeepVesselNet (1803.09340)** and **Menten (2207.11102)**: synthetic vessel trees as pre-training/annotation-free
  training; the recurring finding is that geometry realism is necessary but photometric artefacts decide transfer.

---

## 5. Suggested experiment order and pre-declared criteria

1. **R1 on cached maps** (no GPU): report on roi303 VAL the strict centerline-F1, coverage-failure and precision-
   failure shares, and the audit's two-ridge fraction per gap bin, for (a) thr 0.99 (current), (b) NMS + ridge
   hysteresis (0.9 / 0.3, κ-gated). Criterion: coverage failures fall by ≥ 10 pp with precision failures rising ≤ 3 pp;
   two-ridge fraction at 4.5–6 px ≥ 80 %. If precision failures rise more, the κ gate is too loose (raise κ_lo) before
   concluding.
2. **R4 + R2 retrain** (one run): existence head unchanged, add the regression head (Gaussian σ = 1.2 px or Sironi
   a = 6, d_M = 4), zoom U(1.1, 1.5), contrast field, down-up blur. Decode with R1 on the regression head, gated by
   existence ≥ 0.4. Criterion: pooled F1 on MT-34 + HTW (per source and pooled, per the reporting rule) ≥ current + 0.05;
   p > 0.99-band width is no longer the relevant statistic — instead report ridge-max distribution at 4.5–6 px gaps
   (target ≥ 0.98 median).
3. **R3 on top** (second run): CAPE with α ∈ {0.1, 1}. Criterion: coverage failures fall further with junction
   identity ≥ 0.45 (from 0.32) and no rise in chimera rate.
4. Decide R6 / R5 / R8 from the failure split after step 3; keep R9 in reserve.

---

## 6. References

Steger, C. An unbiased detector of curvilinear structures. IEEE TPAMI 20(2):113–125, 1998. DOI 10.1109/34.659930.

Sironi, A., Lepetit, V., Fua, P. Multiscale centerline detection by learning a scale-space distance transform. CVPR
2014 (the version read: EPFL Infoscience,
https://infoscience.epfl.ch/server/api/core/bitstreams/150aa079-eb63-4a4c-8817-79af4562f681/content; DOI not
verified in this session). Extended: Sironi, Türetken, Lepetit, Fua. Multiscale centerline detection. IEEE TPAMI
38(7):1327–1341, 2016 (PubMed 27295457).

Wang, Y., Xu, Y., Tsogkas, S., Bai, X., Dickinson, S., Siddiqi, K. DeepFlux for skeletons in the wild. CVPR 2019.
arXiv:1811.12608.

Xue, N., Bai, S., Wang, F., Xia, G.-S., Wu, T., Zhang, L. Learning attraction field representation for robust line
segment detection. CVPR 2019. arXiv:1812.02122.

Guo, Z., Bai, J., Lu, Y., Wang, X., Cao, K., Song, Q., Sonka, M., Yin, Y. DeepCenterline: a multi-task fully
convolutional network for centerline extraction. MICCAI 2019. arXiv:1903.10481.

Wang, Y., Wei, X., Liu, F., Chen, J., Zhou, Y., Shen, W., Fishman, E. K., Yuille, A. L. Deep distance transform for
tubular structure segmentation in CT scans. CVPR 2020. arXiv:1912.03383.

Oner, D., Koziński, M., Citraro, L., Dadap, N. C., Konings, A. G., Fua, P. Promoting connectivity of network-like
structures by enforcing region separation. IEEE TPAMI. arXiv:2009.07011.

Esmaeilzadeh, E., Garaaghaji, E., Hallaji Azad, F., Oner, D. CAPE: Connectivity-aware path enforcement loss for
curvilinear structure delineation. 2025. arXiv:2504.00753.

Zong, Y., Jia, R., Hu, M., Su, W., He, D. Widest-path reachability fields for connectivity-preserving slender
structure segmentation. 2026. arXiv:2607.07123.

Kirchhoff, Y., Rokuss, M. R., Roy, S., Kovacs, B., Ulrich, C., Wald, T., Zenk, M., Vollmuth, P., Kleesiek, J.,
Isensee, F., Maier-Hein, K. Skeleton recall loss for connectivity conserving and resource efficient segmentation of
thin tubular structures. ECCV 2024. arXiv:2404.03010.

Arora, D., Kumar, N., Gupta, S. Does the skeleton-recall loss really work? 2025. arXiv:2508.11374.

Shit, S., Paetzold, J. C., et al. clDice — a novel topology-preserving loss function for tubular structure
segmentation. CVPR 2021. arXiv:2003.07311.

Shi, P., Hu, J., Yang, Y., Gao, Z., Liu, W., Ma, T. Centerline boundary Dice loss for vascular segmentation. MICCAI
2024. arXiv:2407.01517.

Hu, X. Structure-aware image segmentation with homotopy warping. NeurIPS 2022. arXiv:2112.07812.

Hu, X., Li, F., Samaras, D., Chen, C. Topology-preserving deep image segmentation. NeurIPS 2019. arXiv:1906.05404.

Stucki, N., Paetzold, J. C., Shit, S., Menze, B., Bauer, U. Topologically faithful image segmentation via induced
matching of persistence barcodes. ICML 2023. arXiv:2211.15272. Efficient version: Stucki, Bürgin, Paetzold, Bauer.
Efficient Betti matching enables topology-aware 3D segmentation via persistent homology. 2024. arXiv:2407.04683.

Berger, A. H., Stucki, N., Lux, L., Bürgin, V., Shit, S., Banaszak, A., Rueckert, D., Bauer, U., Paetzold, J. C.
Topologically faithful multi-class segmentation in medical images. MICCAI 2024. arXiv:2403.11001.

Lux, L., Berger, A. H., Weers, A., Stucki, N., Rueckert, D., Bauer, U., Paetzold, J. C. Topograph: an efficient
graph-based framework for strictly topology preserving image segmentation. 2024/2025. arXiv:2411.03228.

Mosinska, A., Márquez-Neila, P., Koziński, M., Fua, P. Beyond the pixel-wise loss for topology-aware delineation.
CVPR 2018. arXiv:1712.02190.

Zhou, F., Gao, Z., Zhao, H., Xie, J., Meng, Y., Zhao, Y., Lip, G. Y. H., Zheng, Y. GLCP: global-to-local
connectivity preservation for tubular structure segmentation. MICCAI 2025. arXiv:2507.21328.

Batra, A., Singh, S., Pang, G., Basu, S., Jawahar, C. V., Paluri, M. Improved road connectivity by joint learning of
orientation and segmentation. CVPR 2019. IEEE document 8953380; code github.com/anilbatra2185/road_connectivity
(`data_utils/affinity_utils.py`, `config.json`).

Kluvanec, D., Phillips, T. B., McCaffrey, K. J. W., Al Moubayed, N. Using orientation to distinguish overlapping
chromosomes. CHIL 2022. arXiv:2203.13004.

He, S., Bastani, F., Jagwani, S., Alizadeh, M., Balakrishnan, H., Chawla, S., Elshrif, M. M., Madden, S., Sadeghi,
M. A. Sat2Graph: road graph extraction through graph-tensor encoding. ECCV 2020. arXiv:2007.09547.

Bastani, F., et al. RoadTracer: automatic extraction of road networks from aerial images. CVPR 2018.
arXiv:1802.03680.

Xu, Z., et al. RNGDet: road network graph detection by transformer in aerial images. 2022. arXiv:2202.07824;
RNGDet++. arXiv:2209.10150.

Gong, D., Ji, S. DeH4R: a decoupled and hybrid method for road network graph extraction. 2025. arXiv:2508.13669.

Zhang, Z., Zhao, Z., Wang, D., Wang, L. GraphMorph: tubular structure extraction by morphing predicted graphs. 2025.
arXiv:2502.11731.

Wang, D., Zhang, Z., Zhao, Z., Liu, Y., Chen, Y., Wang, L. PointScatter: point set representation for tubular
structure extraction. ECCV 2022. arXiv:2209.05774.

Liu, C., Zhao, T., Zheng, N. DeepBranchTracer: a generally-applicable approach to curvilinear structure
reconstruction using multi-feature learning. AAAI 2024. arXiv:2402.01187.

Liu, C., Jiang, Y., Zheng, N. NETracer: a topology-aware iterative tracing approach for tubular structure
extraction. ICCV 2025, pp. 20593–20602 (CVF open access); code github.com/CSDLLab/NETracer.

Heng, Z., Pagnucco, M., Meijering, E., Song, Y. Curvi-Tracker: curvilinear structure segmentation refinement by
iterative tracking. Pattern Recognition 173:112797, 2026 (ScienceDirect pii S0031320325014608).

Liu, Y., Zhang, M., He, J., Gu, Y. TopoVST: toward topology-fidelitous vessel skeleton tracking. 2026.
arXiv:2603.14909.

Wolterink, J. M., van Hamersvelt, R. W., Viergever, M. A., Leiner, T., Išgum, I. Coronary artery centerline
extraction in cardiac CT angiography using a CNN-based orientation classifier. Medical Image Analysis 2019.
arXiv:1810.03143.

Chen, L., Liu, W., Balu, N., Mossa-Basha, M., Hatsukami, T. S., Hwang, J.-N., Yuan, C. Deep open snake tracker for
vessel tracing. MICCAI 2021. arXiv:2107.09049 (abstract only).

Kreitner, L., Paetzold, J. C., Rauch, N., Chen, C., Hagag, A. M., Fayed, A. E., Sivaprasad, S., Rausch, S.,
Weichsel, J., Menze, B. H., Harders, M., Knier, B., Rueckert, D., Menten, M. J. Synthetic optical coherence
tomography angiographs for detailed retinal vessel segmentation without human annotations. IEEE TMI 2024.
arXiv:2306.10941.

Menten, M. J., Paetzold, J. C., Dima, A., Menze, B. H., Knier, B., Rueckert, D. Physiology-based simulation of the
retinal vasculature enables annotation-free segmentation of OCT angiographs. MICCAI 2022. arXiv:2207.11102.

Tetteh, G., et al. DeepVesselNet: vessel segmentation, centerline prediction, and bifurcation detection in 3-D
angiographic volumes. 2018. arXiv:1803.09340 (abstract only).

Wittmann, B., et al. Simulation-based segmentation of blood vessels in cerebral 3D OCTA images. 2024.
arXiv:2403.07116 (abstract only).

SynthMT: Synthetic data enables human-grade microtubule analysis with foundation models for segmentation. PLOS
Computational Biology, 2026. DOI 10.1371/journal.pcbi.1013901; bioRxiv 10.64898/2026.01.09.698597; code
github.com/ml-lab-htw/SynthMT.

Ait Laydi, A., Cueff, L., Crespo, M., El Mourabit, Y., Bouvrais, H. A novel attention mechanism for noise-adaptive
and robust segmentation of microtubules in microscopy images. 2025. arXiv:2507.07800.

Sid'El Moctar, S. M., Ait Laydi, A., El Mourabit, Y., Bouvrais, H. MTFlow: time-conditioned flow matching for
microtubule segmentation in noisy microscopy images. 2026. arXiv:2601.14841.

Liu, Y., Zhang, Y. A conditional generative framework for synthetic data augmentation in segmenting thin and
elongated structures in biological images. 2025. arXiv:2512.10334.

Li, M., Chen, Y., Zhang, W., Chen, Q. Image magnification network for vessel segmentation in OCTA images. 2021.
arXiv:2110.13428.

Amarnath, R. Effect of input resolution on retinal vessel segmentation performance: an empirical study across five
datasets. 2026. arXiv:2604.02977.

Carneiro-Esteves, S., Vacavant, A., Merveille, O. A plug-and-play framework for curvilinear structure segmentation
based on a learned reconnecting regularization. 2024. arXiv:2408.12943 (abstract only).

Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., Maier-Hein, K. H. nnU-Net: a self-configuring method for
deep learning-based biomedical image segmentation. Nature Methods 18:203–211, 2021. DOI 10.1038/s41592-020-01008-z;
augmentation defaults read from `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py` (master, 2026-09-04).

Mais, L., et al. FISBe: a real-world benchmark dataset for instance segmentation of long-range thin filamentous
structures. CVPR 2024. arXiv:2404.00130 (abstract only; neuron instance benchmark with the same long-range
identity problem).

Project documents used: `docs/research/2026-09-02-scale-and-gap-audit.md` (all audit numbers),
`docs/MT_SEGMENTATION_KNOWLEDGE.md` (ORION head set and decoder, l.35–73), `docs/INSTANCE_SEGMENTATION_RESEARCH.md`
(ORION overpass precedent, l.19–56).
