# Real IRM corpus expansion — sources, method, stats (reproducible)

> 2026-07-06: expand the label-free calibration corpus (the 320 `morphology_reference_frames/irm` frames)
> with additional real IRM microtubule images from multiple labs → a larger, MULTI-LAB corpus (directly
> addresses peer-review W1: calibration-target representativeness). Dedup, exclude eval leakage, document.
> Base corpus on tulen: `/home/prusek/BIOCEV/datasets/microtubules/morphology_reference_frames/irm/` (320).

## Sources
| # | source | provenance | content | → corpus |
|---|--------|------------|---------|----------|
| 0 | morphology_reference_frames/irm (320) | existing base | real IRM | KEEP ALL (base) |
| 1 | task_irm-2026_03_23…cvat | **Lánský lab** (ours) | 192 png `images/` | include |
| 2 | task_v7_pysoax_adela…cvat | **Lánský lab** (ours) | 3 tif `images/` | include |
| 3 | task_v7_pysoax_alice…cvat | **Lánský lab** (ours) | 12 tif `images/` = the ALICE EVAL frames | **EXCLUDE (eval leakage)** |
| 4 | Test_dataset.zip | **IRM_InVitroMT, Beber et al. 2026** | 62 tif `Prediction_source/` (+ target masks) | include SOURCE only |
| 5 | Training_dataset.zip | **IRM_InVitroMT, Beber et al. 2026** | 412 tif `Training_source/` (+ target masks) | include SOURCE only |
| 6 | HTW-KI-Werkstatt/IRM-in-vitro-microtubules (HF) | **SynthMT authors (HTW)** | 66 rows parquet: cols `id,image,mask` — 66 real IRM `image` + 66 GT `mask` | include IMAGE only (mask→W9) |
| 7 | one more (2.4GB, user's) | **Lánský lab** (ours) | downloading | include (TBD after inspect) |

## Rules (reproducibility)
1. Take only real IRM **source images** (NOT the Beber target/mask images, NOT CVAT annotation XML).
2. **EXCLUDE the 12 Alice eval frames** (source 3) — they are the held-out benchmark; adding them to the
   calibration corpus = test-set leakage. Enforced by a dedup pass against the Alice eval set.
3. Formats: keep tif and png; convert all to a common float grayscale for hashing/dedup.
4. **Dedup = PIXEL-CORRELATION** (NOT dHash — dHash over-merged low-content IRM frames, see [[real-image-corpus]]):
   resize each frame to 64×64 grayscale, z-normalize, pairwise Pearson correlation; two frames are duplicates
   if corr > THR (THR≈0.92, tuned). Greedy: iterate candidates, accept if it matches nothing already kept.
5. Dedup ORDER of exclusion for each new candidate: drop if it matches (a) any ALICE eval frame [leakage],
   else (b) any already-accepted frame [dup] (base 320 first, then accepted-new). Base 320 all kept.
6. Provenance: prefix each accepted filename with its source id (e.g. `s1_img_39.png`) so origin is traceable.
7. Output: `/home/prusek/BIOCEV/datasets/microtubules/real_corpus_v2/` = 320 base + accepted-new; update
   calibrate.py `REAL` path to point here for the next (representative, multi-lab) calibration.

## Also captured (NOT for the corpus — for a future benchmark, W9)
The CVAT tasks (task_irm, adela) carry **polyline/instance annotations**, and Beber Test/Training carry
segmentation **target masks** — these are ANNOTATED real IRM datasets → a candidate crossing-stress
INSTANCE benchmark (the "better dataset than Alice"). Kept the annotations/targets aside for W9; not deduped
into the calibration corpus.

## Stats (FINAL, 2026-07-06, `real_corpus_v2/stats.json`)
Thresholds (ASYMMETRIC, matching the established policy + leakage safety): **dup-merge corr≥0.999** on
**128×128** z-normed pixel correlation (keep DISTINCT frames, merge only pixel-identical); **Alice-leakage
exclusion corr>0.88** (LOOSE — drop anything SAME-FIELD as an Alice eval frame; different fields corr<0.5).
- Raw per source (source-images only): s1 task_irm 192 · s2 adela 3 · s4 Beber-test-source 31 ·
  s5 Beber-train-source 206 · s7 general_mt 303 · s6 HF-HTW 66 = **801 candidates** (Beber halved by the
  source-only filter; s3 alice 12 not transferred).
- **Base 320 → kept 303** (dropped **17 pre-existing same-field-as-Alice frames** — see key finding).
- New candidates: excluded **3** as same-field-as-Alice; **617** dropped as dups; **accepted 181**
  (s1 task_irm **115**, s6 HF-HTW **66**; s2/s4/s5/s7 → 0).
- **FINAL CORPUS: 303 + 181 = 484 frames**, LEAKAGE-FREE, `…/microtubules/real_corpus_v2/`.
  20 same-field-as-Alice frames excluded total (17 base + 3 new). calibrate.py `REAL` points here.

**KEY FINDINGS:**
1. **The base 320 `morphology_reference_frames` was ALREADY built from these datasets** — checked s7 frames
   are corr **1.000** (identical) to base; Beber/adela/most of s7 are identical dups. The only GENUINELY-NEW
   data is **HTW / SynthMT-authors (66)** + **task_irm (115 distinct Lánský frames not in base)**. Dedup
   verified sound (different frames corr 0.18–0.49; dups ~1.0 — NOT background-over-merge).
2. **⚠ Pre-existing LEAKAGE (rigor / W1–W2):** 17/320 base calibration frames are SAME-FIELD as Alice eval
   frames (corr>0.88) → the ESTABLISHED corpus quasi-leaked the eval set into every prior calibration. Now
   excluded (base_kept 303). Report this + re-run past results on the clean corpus.
3. Honest note (W1): the corpus is now 3 labs (Lánský + Beber + HTW), 484 frames, leakage-clean — but adding
   HTW does NOT by itself fix the corpus↔Alice waviness mismatch; that still needs a representativeness check.

## Reproduce
1. Transfer source zips to `mt_enc_exp/newdata/` (rename s1_task_irm/s2_adela/s4_beber_test/s5_beber_train/
   s7_general_mt); `huggingface_hub.snapshot_download("HTW-KI-Werkstatt/IRM-in-vitro-microtubules",
   repo_type="dataset", local_dir="newdata/hf_htw")`.
2. `dinov3_env/bin/python scripts/build_corpus.py` (in mt_enc_exp) → extracts source images (excludes
   target/mask/annotation), decodes HF parquet `image` col, dedups vs base 320 + Alice eval, writes
   `real_corpus_v2/` + `stats.json`. Script: scratchpad enc_exp/build_corpus.py.
3. Point `calibrate.py` `REAL` at `real_corpus_v2/*` (tif+png) for the next multi-lab calibration.
