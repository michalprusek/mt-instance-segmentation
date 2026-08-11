# Microtubule instance segmentation — knowledge transfer

> Carried over from the BIOCEV project (2026-03 … 2026-06). This is the distilled,
> self-contained state of knowledge for developing a **synthetic dataset** and a
> **segmentation model** for microtubules (MT). Read this first.

## 1. The task

**2D instance segmentation of microtubules** in label-free / fluorescence light microscopy:
- **IRM** (interference reflection microscopy): MTs are dark/bright lines on a textured grey background, with **polarity flips** (a single MT can appear dark then bright along its length) and white interference spots.
- **TIRF**: MTs are bright lines on a black background, with laser bleaching gradients.
- Goal: separate **individual** filaments, including where they **cross** (X-junctions, shared pixels = amodal) and run **parallel & close**. Dense fields: 15–80 MTs per frame.

The hard part is NOT semantic foreground (largely solved) — it is **instance identity at crossings and between close parallels**.

## 2. Real test set (the ground truth that matters)

`data/real/alice_eval/` — **12 real Alice IRM frames** (~1344×1416 px) with **human MT instance polyline GT** (16 GT MTs/frame avg, 231 total). This is THE benchmark. Each `.h5` has `image` + `polylines/pl_XXXX` (ordered (x,y) vertex lists) + attrs `height,width,n_polylines`. Tifs alongside for tools that read images.

The 12 frames also have **pixel-co-registered TIRF channels** (IRM polyline GT transfers → free TIRF instance-GT eval). Real TIRF is otherwise nearly absent.

Larger real data (on the lab GPU server `tulen`, not copied here): 303 human-labelled colleague frames (16,076 polylines, `datasets/microtubules/cvat_export`); IRM background library v2 (428 frames, 3 microscopes); ~320 unlabelled real IRM for self-training.

## 3. Evaluation methodology (use this exact metric)

**Centerline-F1**, STRICT setting: `tol=5 px, length_coverage=0.95, precision_coverage=0.95`. A predicted instance is a true positive only if its centerline covers ≥95% of a GT polyline within 5 px AND is ≥95% precise (not over-extended). Also report the **breakdown**: covered / **clean** (Hausdorff ≤ tol) vs **chimera/overext** (covers GT but spills) / fragment / partial / missed. Inference at **1.5× upscale** separates close parallels (proven +F1).
- `clean` high + `chimera` low = good. `chimera` high = instances merge multiple MTs.
- Evaluate masks (rasterized instances) against the polyline GT.

**Reference numbers on the 12 Alice frames (strict F1):**

| method | strict F1 | notes |
|---|---|---|
| v19 PatchPerPix + **real self-train** + 1.5× upscale | **0.696** | production best; uses real pseudo-labels |
| ORION skeleton-graph + orientation arm-pairing (tuned heuristic) | **0.519** | best "split the semantic mask" heuristic |
| **TARDIS zero-shot** (pretrained on REAL TIRF) | **0.326** | clean instances, turnkey, no training on our data |
| SAM3 zero-shot ("thin line" prompt) | ~0.76 lenient / 0.76 | appearance-keyed |
| our WEAVE oracle (GT edges) | 0.16–0.27 | even perfect edges fragment |
| our end-to-end Object Condensation (synth-trained) | 0.013 | all chimeras |

## 4. What has been tried — and the DEFINITIVE finding

Many decoders were built to turn a semantic/orientation field into instances. **All synth-trained custom decoders failed on real.** Summary:

- **WEAVE** (TARDIS-style: semantic → skeleton point cloud → learned graph-edge head → cut). On synthetic data the oracle gate = **F1 1.0**, but on real the **single skeleton collapses crossings** (the real IRM foreground is a dense connected mesh: ~9300 skeleton px in ~15 tangles for 18 GT MTs); one MT loses its identity at each X. Even feeding GT edges (oracle) fragments to 0.16–0.27.
- **Tracer redesign** (skeletonize semantic once + per-point argmax orientation tangent, arc-order chaining): same crossing-collapse wall.
- **End-to-end Object Condensation** (orientation-keyed amodal: each pixel emits an embedding+β per active orientation bucket, so a crossing pixel → 2 points → 2 instances; OC loss makes the cut differentiable). Trains cleanly, **finds** the MTs (covered 213/231) but **merges adjacent/parallel MTs** → all chimeras → F1 **0.013 even in-domain on synth (0.028)**. The parallel-merge is **intrinsic to fixed-radius embedding repulsion** (research-confirmed); OC is unvalidated for thin filaments.
- **TARDIS fine-tuned on our synth**: the synth-trained FNet **over-fires ~6× on real** (point cloud 15287 vs zero-shot 2713) at every threshold → the DIST instance head produces NaN → pipeline crashes. = synth→real domain gap in the FNet.

**THE finding (consistent across the whole exploration):** real-data instance **separation on dense crossing meshes is the wall** (amodal-vs-clean-skeleton tension + parallel-merge), NOT decoder cleverness. **Real-domain training is the lever; synth-only training caps out regardless of architecture.** The only method that works on real (TARDIS zero-shot) was trained on **real** TIRF MTs; fine-tuning it on our synth **degrades** it.

→ **Implication for this project: the highest-value work is closing the synth→real gap in the SYNTHETIC DATASET** (realistic density, appearance, crossing/parallel statistics so a synth-trained foreground does not over-fire on real), not inventing another decoder. The decoder problem is effectively solved by TARDIS's DIST + ORION's orientation arm-pairing.

## 5. Reusable technical knowledge

- **TARDIS DIST works** — don't reinvent the learned instance head. It produces clean instances from a skeleton point cloud via a transformer that predicts pairwise connectivity + a greedy ≤2-neighbour graph cut. The amodal-crossing handling is its weak spot but it is still the best learned instance head available.
- **ORION-style assembler** (skeletonize semantic → orientation-binned multi-label field keeps crossings amodal → arm-pairing by collinearity at junctions) = best heuristic "split on the semantic mask" (0.519). The orientation-binned multi-label head ("intersection → overpass") is the right amodal mechanism (precedent: Liu et al. CVPRW 2019).
- **Foreground quality on real is the ceiling**, not the decoder (proven repeatedly). A semantic model that over-detects on real (as our synth-trained FNets do) breaks every downstream instance step.
- **Object Condensation** (Kieseler 2020): orientation-keyed multi-embedding is a genuinely novel idea, but parallel-merge + long-range-identity are intrinsic failure modes for thin filaments; not worth more tuning.
- **Eval gotchas:** rasterize instances to masks; the metric skeletonizes them. Sparse/scattered point sets fragment. Synth GT is dense (60–80 MT/frame) vs real (~16) — a key density mismatch driving the domain gap.

## 6. TARDIS — how to run the working baseline (0.326 on real, zero-shot)

`tardis-em` (pip). Pretrained 2D MT models auto-download to `~/.tardis_em/{fnet_attn_32/microtubules_tirf, dist_triang/2d}`.
- Base Python may lack `_sqlite3` → `pip install pysqlite3-binary` + a `sitecustomize.py` that does `sys.modules['sqlite3']=pysqlite3`.
- Zero-shot predict: `tardis_mt_tirf -dir <imgs.tif dir> -out None_csv -rt False -dv 0` → per-image `Predictions/*_instances_filter.csv` with `[ID, X, Y, Z]` (per-instance centerline points → rasterize+dilate → masks → score).
- Fine-tune FNet: `tardis_cnn_train -dir <data with train/imgs,train/masks,test/...> -cnn fnet_attn -cm 32 -cl 5 -cs 2gcl -ps 128 -cch <pretrained .pth>`. The FNet is **2D (`2gcl`), img_size 128** — passing the default `3gcl/64` CRASHES (decoder size mismatch). Masks are `<name>_mask.tif`. (Note: naive synth fine-tune over-fires on real — see §4.)

## 7. Directions for this project

1. **Synthetic dataset (primary lever).** Match real statistics so a synth-trained foreground transfers: realistic **density** (don't pack 60–80 MTs when real has ~16), IRM appearance (polarity flips, interference spots, real backgrounds), realistic crossing/parallel geometry, and thin centerline-consistent masks. Validate by: does a foreground model trained on it produce a CLEAN (non-over-firing) point cloud on the 12 Alice frames?
2. **Segmentation model.** Likely: a domain-robust semantic/orientation field (the project used frozen DINOv3 + DPT; ORION head set = semantic + K-bin multi-label orientation + distance + junction + endpoint) → a learned or TARDIS-DIST instance step. The win comes from the foreground transferring, not a new decoder.
3. **Beat the bar:** TARDIS zero-shot 0.326 (clean, real-trained) is the honest baseline to beat with a synth-only method; ORION 0.519 / v19 0.696 are the production heuristics.
