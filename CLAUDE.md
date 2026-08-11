# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Goal: develop a **synthetic MT dataset** + **segmentation model** for 2D microtubule
instance segmentation (IRM/TIRF). Carried over from the BIOCEV exploration.

**Read `docs/MT_SEGMENTATION_KNOWLEDGE.md` before any MT work** — it has the task, data,
eval metric, full results of prior attempts, the definitive finding, and how to run TARDIS.

## Repo layout & state
- `synth/` holds the **synthetic MT generator** (`mt_generator.py` — stiff worm-like
  morphology + IRM dark↔bright polarity flips, composites onto real backgrounds, free GT;
  `render_samples.py` driver).
- `src/mt_bench/` — benchmark tooling: CVAT import, MT-34 builder, overlay verification,
  GT statistics (κ_max, crossings, parallels), field-of-view masking.
- `src/instance/` — the **curvature-bounded instancers** (see `docs/protocol.md §17`):
  `geometry` · `skeleton_graph` · `matching` · `instancer_a` · `lifted` · `instancer_b` ·
  `oracle` · `metrics`. `models/` is still empty (checkpoints live on tulen).

### Commands (test harness exists now)
```bash
python -m pytest -q                       # 73 tests; pythonpath=src via pytest.ini
pip install -r requirements.txt           # CPU-side deps only
python -m mt_bench.build_mt34 --cvat-dir <dir with cvat_585/ cvat_586/>
python -m mt_bench.gt_stats data/real/mt34_eval          # κ_max + crossing/parallel stats
python -c "from mt_bench.overlay import render_all; render_all('data/real/mt34_eval', \
    'data/enc_sensitivity_testset/mt34_overlays')"       # the verification gate — LOOK at these
```
**All heavier runs go on tulen** (`/home/prusek/mt_enc_exp/mt34_work`, 32 cores):
`scripts/run_oracle_eval.py` (instancer benchmark), `scripts/tune_instancer.py` (Optuna),
`scripts/semantic_compare.py`, `scripts/predict_v4b_mt34.py`, `scripts/nnunet_*`,
`scripts/tulen_tuning_chain.sh`, `scripts/final_test_run.sh`, `scripts/viz_instancers.py`.
Two environment traps: `optuna` needs the `pysqlite3` shim `synth/calibrate.py` uses, and
`rsync` over ssh breaks on the client's PQ-handshake banner — transfer with `tar` + `scp`.

### Benchmark data (`data/` and `models/` are git-ignored)
- `data/real/mt34_eval/` — **MT-34, the primary benchmark** (34 frames, 957 GT polylines:
  refreshed Alice + 22 crossing-dense frames). **32.1 crossings/frame vs Alice 2.2.**
  Split `val`/`test` in the h5 attrs — tune on VAL, score TEST once.
- `data/real/alice_eval/` (12 frames) — superseded by MT-34's task-585 half, kept for
  traceability with older numbers; `data/real/htw_eval/` (66 frames) — held-out cross-lab, SEALED.
- **Scale, corrected:** HTW GT is stored pre-multiplied by 1.5; **Alice and MT-34 GT are at
  NATIVE 1×**. The 1.5× is an *eval-pipeline convention* — `zoom(img, 1.5)` and GT×1.5 — not a
  property of the data. Getting it wrong silently collapses the metric.
- **Data-format gotcha:** polyline GT vertices in the `.h5` files are `(x=col, y=row)` — the
  transpose of NumPy `[row, col]` indexing. Mind this in any eval/rasterization code; it has
  caused two silent bugs so far and each has a regression test.

## Standing constraints & facts
- **The synthetic→real domain gap is the wall, not the decoder.** Prior synth-trained
  decoders (WEAVE, end-to-end Object Condensation) all failed on real; TARDIS (real-trained)
  works. Spend effort on the **synthetic dataset realism** (density, IRM appearance, crossing/
  parallel stats), not on inventing decoders.
- **Eval = strict centerline-F1** (tol=5, length & precision coverage 0.95) on **MT-34**
  (`data/real/mt34_eval/`), at 1.5× upscale, reported **per source** (Alice / new-22) and pooled,
  clean vs chimera. Frames with zero GT are excluded from the macro mean but keep contributing
  false positives to micro — use `instance.metrics.aggregate_benchmark`, not `aggregate_f1`.
- **Bar to beat:** TARDIS zero-shot 0.326 (honest synth-only target); ORION 0.519; v19 0.696.
- **κ_max = 0.25 rad/px is DERIVED, not tuned** — just above the 0.239 maximum over all 957
  MT-34 GT microtubules, measured at an **8 px baseline** (the same data gives 1.015 at 2 px,
  because coarse human polylines make vertex-level turns meaningless). Never let an optimiser
  fit it: the claim is that it encodes physics.
- A synth-trained foreground that **over-fires on real** breaks every downstream step — the
  single best validation of synth quality is: does a model trained on it produce a clean,
  non-over-firing prediction on the 12 Alice frames?

## Instance segmentation — priorities & main bottlenecks (user 2026-07-05)
The two hard problems that instance segmentation MUST solve — these are THE bottlenecks, prioritize them:
1. **Separating CLOSE PARALLEL microtubules** (bundles running side-by-side, few px apart) — must not
   merge them into one instance. (Fixed-radius embedding repulsion / Object-Condensation intrinsically
   fails here — parallel-merge; don't go back to it.)
2. **Resolving JUNCTIONS / crossings** (X-junctions where MTs share pixels = amodal) — each MT must
   keep its identity through the crossing, not collapse or fragment.
- **Preferred instance approach: TARDIS-style POINT-CLOUD instancing** (DIST transformer that predicts
  pairwise point connectivity on a skeleton/centerline point cloud + graph cut) — currently looks
  better than orientation-field / collinear arm-pairing for BOTH parallels and junctions. Research
  this before building. Caveat from prior work (`docs/MT_SEGMENTATION_KNOWLEDGE.md §4`): synth-trained
  TARDIS-style (WEAVE) failed on real via crossing-collapse (single skeleton) + the FNet over-firing
  ~6× → so the levers are (a) synth realism so the point cloud doesn't over-fire, (b) a centerline/
  point-cloud representation that stays amodal at crossings and resolves close parallels.
- **Instance tuning is currently DEFERRED** (generator + SEMANTIC foreground quality first, per user),
  but the generator's crossing/parallel statistics must be built to make the above solvable.

## Compute
- GPU work runs on the lab server **tulen** (`ssh prusek@tulen.utia.cas.cz` — NOT `ssh tulen`,
  which uses the wrong user). Large datasets live on tulen `/disk2/prusek`. TARDIS env at
  `/home/prusek/tardis_env`; pretrained MT models cached at `~/.tardis_em`.
- Don't poll remote jobs more than ~every 30 min.

## Living docs & TODO (keep these current)
- **`docs/TODO.md` is the living project TODO — ALWAYS consult it, keep it UPDATED, ADD new items as
  they arise, and check items off when done.** Don't let it go stale.
- Also maintain the other living docs as work progresses: `docs/PAPER_PLAN.md` (paper spine + claims/
  experiments), `docs/protocol.md` (the path taken — what worked / what didn't),
  `docs/INSTANCE_SEGMENTATION_RESEARCH.md`. Update them, don't just append.

## Repo hygiene
- Commit/push only when asked. On the default branch, branch first.
