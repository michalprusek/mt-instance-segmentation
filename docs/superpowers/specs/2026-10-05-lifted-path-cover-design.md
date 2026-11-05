# LPC — Lifted Path Cover with a rectified-strip verifier (design, 2026-10-05)

Status: **SCREENED 2026-10-06 — NOT SUPPORTED AS SPECIFIED** (`docs/TODO.md` 2026-10-06 08:25 / 09:05 / 09:30).
S0 passed (oracle over candidates 0.838 on a perfect band, 0.831 on htw VAL; lines 0.80) but a PERMISSIVE cut makes the
candidate set worse, and on htw 0.07–0.12 of the ceiling is the existence decision over unannotated arcs. S1 passed its
declared comparator (+0.035 AUC over mean map logit on htw VAL) while a logistic on five hand features matches the CNN
and the raw image adds nothing (map-only strip = full strip). S2 was void: the instancer's junction matching is already
0.92 at clean through-crossings, so the +0.10 line could not fire; the connected-mask failures sit in arc-graph
construction, upstream of any link head. S3 not started. What survives: a +0.17 in-domain selection ceiling on the
unchanged mask, concentrated in 12–36 px links between arbitrary arc ends. Original proposal text follows unchanged.

Original status: **PROPOSAL, nothing built, nothing trained.** Requested by the user 2026-10-05 ("přijít s
nejlepším možným a ideálně novel modelem pro segmentaci MT": detect overlaps, instance better, a
linker made for filaments). Every claim marked *(inference)* is reasoning, not a measurement.

## 1. What the model has to fix — two different failures, both measured

| | failure | evidence in this repo |
|---|---|---|
| F-real | **Evidence is judged per pixel.** At cut 0.5 the map covers 85 % of GT curves; instance recall is flat at ~0.52 at every cut. The weak arm sits 8–20 px from a crossing (97 % of real crossing failures are a broken mask). 30 % of roi303 curves have SNR < 2; htw curves < 12 px are missed 88 %. | memory `recall-ceiling-is-ordering`, `junction-arity-decides-identity`; `docs/protocol.md §25` |
| F-dense | **The binary mask carries no identity.** In domain 59.5 % of failing crossings have a CONNECTED mask, both arms present, no dip; a perfect band through the current instancer scores only 0.686 on dense synthetic scenes. Raising the angle weight buys recall and loses junction identity monotonically (RELINK 1-D). | `docs/TODO.md` 2026-09-08, 2026-10-05 11:20 / 11:40 |

A bigger backbone addresses neither: in domain the model is at 93 % of the instancer ceiling.

## 2. The design

```
image ──► frozen SPARSE35 (binary map p) ──► permissive skeleton (low cut, chosen on synth)
                                                   │
                                   oriented arc ends = nodes of the LIFTED graph (x, y, theta)
                                                   │  edges = every curvature-admissible continuation
                                                   │  (kappa <= 0.25 fixed): pass-through, gap bridge, TERMINATE
                                                   ▼
            candidate arcs + links ──► RECTIFIED STRIP (raw image + p, resampled in the curve's own
                                       Frenet frame: s along, n across, W ~ 33 px)
                                                   ▼
                                 strip verifier (small CNN, synthetic-only training)
                                   · presence log-likelihood ratio per arclength  l(s)
                                   · lateral offset per arclength
                                   · per-link "same filament" log-odds
                                                   ▼
        MIN-COST PATH COVER in the lift: unit capacity per (x, y, theta) cell
          -> two filaments may share (x, y) at different theta (crossing), same-theta duplicates compete
          -> a path is emitted iff  sum_s l(s) - c(L) > 0   (per-CURVE decision, no pixel threshold)
```

- **Backbone stays frozen.** ARM 51 (orientation channels as mask outputs) cost −0.088 / −0.16 by
  damaging the mask; a separate network cannot do that. No new head is attached to the trunk, so
  the scrambled-target control (memory `five-traps`) is not needed for the trunk — it IS needed
  for the verifier (train one copy with link labels permuted).
- **Rectified strip = curve-relative coordinates by construction.** CURVEQ died because a mask read
  from translation-invariant pixel features cannot tell two identical parallels apart (AUC 0.48).
  In the strip the candidate's own filament is at n = 0 and the neighbour at n = ±3…6 px; a
  crossing filament is an oblique streak. The CondInst ablation (arXiv 2003.05664: 35.7 vs 31.4 AP
  with / without relative coordinates) is the published version of that diagnosis.
- **Identity information that the mask throws away.** In IRM contrast encodes height above the
  surface; two crossing microtubules generally differ in contrast, width and polarity. The link
  head sees the raw strip across the junction, so photometric continuity is available to it
  *(inference — whether the generator's height model makes this cue transferable is what screen S2
  measures)*.
- **Acceptance bound c(L)** is calibrated on synthetic NULL strips (background-only frames and
  paths displaced off every filament) to a fixed false-alarm rate per frame — tuned on synth, not
  on real VAL (memory `tune-on-synth-not-real-val`). It makes the minimum detectable contrast for
  a given length an explicit statement.
- **Fixed physics:** kappa_max = 0.25 rad/px is not tuned. No inference-time adaptation, no
  self-training, no real labels anywhere in training.
- **Not solved by the design:** 3-arm junctions (one filament ends under another; identity 0.33
  today). The path cover needs the TERMINATE edge with a learned cost; treat as an open risk.
  Tight bundles narrower than the strip's lateral resolution would need explaining-away
  (render-and-compare with a worm-like-chain prior) — deferred, phase 2 only if S0 shows the
  residual is there.

## 3. Prior art and what would be new

| closest prior | ID | what it does | difference here |
|---|---|---|---|
| Liu et al., "Intersection to Overpass" | DOI 10.1109/CVPRW.2019.00021 | six orientation-binned output layers, synthetic train, real MT test | that IS our ARM 51; here orientation lives in the decoder's graph, not in mask channels |
| Eckstein … Funke, MT tracking in EM | arXiv 2009.08371 | ILP on a candidate graph, evidence summed along links, curvature cost | 3D EM never shares voxels; here 2D sharing via the lift, learned likelihood, synthetic-only |
| DeFiNe | arXiv 1601.00847 | unlearned path cover that allows overlaps | learned strip likelihood + capacity per lifted cell |
| Mosinska et al. / Türetken et al. | arXiv 1905.03892 / DOI 10.1109/TPAMI.2016.2519025 | learned path classifiers on candidates from a thresholded skeleton | output is a network, not instance identity through a crossing; candidates here come from a permissive cut and the decision replaces the cut |
| Ofir et al., faint curved edges | arXiv 1505.06600, 1706.07717 | per-curve mean response vs a length-dependent bound | analytic Gaussian noise, edges, no instances |
| SE(2) sub-Riemannian geodesics (Duits, Bekkers, Mirebeau, Chen, Cohen) | arXiv 1508.02553, 1612.06137, 1612.00343, 2208.11004 | crossings disentangle in the orientation lift | hand-crafted cost, one geodesic per endpoint pair or a tree |
| mBEST / HANDLOOM (cables) | arXiv 2302.09444 / 2303.08975 | bending-energy pairing; learned tracer + crossing classifier | few objects, high SNR; mBEST is our current instancer in another field |
| CondInst / CondLaneNet / Mabon & Olivo-Marin 2026 | arXiv 2003.05664 / 2105.05003 / DOI 10.1109/ISBI61048.2026.11515629 | position-conditioned masks / multi-vector flows | never on hundreds of thin crossing filaments |
| SynthMT (direct competitor, IRM, synthetic-only) | DOI 10.64898/2026.01.09.698597 | SAM3 "thin line", SKIoU 0.74 vs human 0.80 on 66 frames | mask-level, no linker, overlap metric (not comparable with centerline-F1) |

Claimed novelty *(inference from two literature sweeps: ~40 arXiv queries, Asta snowballs from
FISBe / PatchPerPix / Liu 2019 / SynthMT; backward-reference pass NOT done)*: (i) a per-curve
existence decision by a learned, path-integrated likelihood ratio that REPLACES the pixel
threshold for instances; (ii) a path cover with unit capacity per (x, y, theta) cell on a learned
cost; (iii) the whole thing trained on synthetic data only, for in-vitro microtubules. Not
searched and likely relevant: curved-planar-reformation classifiers in vessel analysis (strip
rectification is standard there) — must be checked before the paper claims (the strip itself is
NOT claimed as new).

## 4. Screens — each can kill a component before any training

Declared in `docs/TODO.md` with final numeric lines BEFORE each is run. htw VAL is the decisive
real block (clean annotation); roi303 VAL is secondary until the blind annotation-completion pass,
because a working per-curve detector WILL raise false positives on visible unannotated filaments
there (said before reading).

| screen | question | data | cost | kill line (proposal) |
|---|---|---|---|---|
| **S0 candidate ceiling** | With GT choosing the subset, how high can the candidate set go? | `synth_relink96` (perfect band AND model map), htw VAL, roi303 VAL; cached dumps | CPU, ~2 days of code | perfect band on synth must reach >= 0.80 (instancer today 0.686); model map on htw VAL >= 0.80 (oracle mask 0.871). Below -> the graph cannot carry it, stop |
| **S1 = Screen B, repaired** (declared 2026-09-08, never read) | Does a strip pooled along a curve hypothesis beat the per-pixel map on the SAME sub-threshold pixels? | positives: GT stretches with p < 0.98; negatives: chimeras the instancer actually emitted + the neighbouring parallel's continuation | < 1 GPU-h | strip AUC must beat path-mean logit of the map by >= 0.03 on htw VAL, else the learned verifier adds nothing over a plain path-integrated score |
| **S2 link identity** | Does the link head recover identity where the mask is connected? | in-domain crossings with connected mask (identity today ~0.50; 4-arm 0.94, 3-arm 0.33) | ~1 GPU-h | >= +0.10 absolute on connected-mask failures, and the label-permuted control must NOT show it |
| **S3 full read** | Declared instrument, own thresholds replaced by c(L) from synth | htw VAL decisive, roi303 VAL secondary, 2 seeds of the verifier | days | htw >= +0.024 on both seeds; junction identity on synth not below baseline |

## 5. What it can and cannot buy

Ceilings with the current annotation and instancer: roi303 0.935, htw 0.871 (perfect mask). The
model-owned share is ≈ 0.06–0.10 on roi303 and ≈ 0.11 on htw (detection of short thin curves); on
dense synthetic scenes the gap to a perfect decoder is 0.31. So the realistic prize is
*(inference)*: most of the htw detection row, part of the roi303 low-SNR row, and a large step in
the dense-crossing regime — which is the regime the paper claims. F1 0.90 on TEST stays out of
reach without the annotation work; this design does not change that.

## 6. Not re-entering the dead list

Fixed-radius embeddings · point-cloud DIST on synth · orientation bins as mask channels (ARM 51) ·
query masks on translation-invariant features (CURVEQ) · worst-point / clDice-like loss · scale and
photometric augmentation · a global relink weight (RELINK 1-D) · threshold-only decoders (11-variant
screen).
