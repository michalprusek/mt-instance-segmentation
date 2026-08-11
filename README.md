# mt-instance-segmentation

Instance segmentation of microtubules in label-free light microscopy (IRM / TIRF), trained on
a **synthetic dataset** with no real annotations, plus the **curvature-bounded instancer** that
turns a semantic foreground into individual filaments.

This repository holds code, tests and the full development protocol. **Data and model
checkpoints are not included** — the image corpora belong to the labs that recorded them.

## The problem

Two failure modes dominate microtubule instance segmentation, and general-purpose methods
fail at both:

1. **Close parallels.** Filaments running side by side a few pixels apart must not merge.
   Fixed-radius embedding repulsion fails here structurally.
2. **Crossings.** At an X-junction two microtubules share pixels. Each must keep its identity
   through the crossing rather than collapsing or fragmenting.

The physical constraint that makes both solvable: **a microtubule cannot kink.** Its
centerline curvature is bounded. Over all 957 human-annotated microtubules in our benchmark,
measured at an 8 px baseline, the maximum |dθ/ds| is **0.239 rad/px** — so κ_max = 0.25 is a
*derived* bound, not a tuned hyperparameter. It is never given to an optimiser.

> Measurement scale is essential: the same annotations give 1.015 rad/px at a 2 px baseline,
> because coarse human polylines make vertex-level turns meaningless.

## Two instancers

Both enforce κ ≤ κ_max as a hard constraint rather than a penalty.

- **A — junction matching** (`src/instance/instancer_a.py`). Contract junction clusters, fit
  tangents over a window, then solve a **min-cost perfect matching** at each junction under
  the curvature bound, with a priced "leave this arm open" option. Pairing arms globally per
  junction is what avoids PySOAX's greedy, order-dependent, unrevisitable choice.
- **B — orientation-lifted beam tracing** (`src/instance/instancer_b.py`). Trace in the lifted
  (x, y, θ) graph where consuming a path retires only the orientation slices it used, so a
  crossing stays amodal.

## Results — MT-34 benchmark, TEST split (17 frames)

Strict centerline-F1 (tol 5 px, 95 % length and precision coverage) at 1.5× upscale. Intervals
are **paired, task-stratified bootstraps over frames**, 20 000 replicates.

| method | oracle foreground | 95 % CI |
|---|---|---|
| **A** (curvature-bounded matching) | **0.920** | [0.870, 0.966] |
| B (orientation-lifted beam) | 0.813 | [0.717, 0.903] |
| PySOAX (same tuning budget) | 0.590 | [0.470, 0.724] |

| paired difference | value | 95 % CI | p |
|---|---|---|---|
| A − B | **+0.107** | [+0.047, +0.182] | < 0.001 |
| A − PySOAX | **+0.330** | [+0.219, +0.432] | < 0.001 |

A and B's *marginal* intervals overlap; only the paired comparison separates them. This is why
every head-to-head number here carries a paired interval — see `docs/protocol.md §17m`.

On the predicted (not oracle) foreground A scores 0.457 and B 0.308 — **the ranking of the
instancers does not depend on foreground quality**, even though both absolute scores roughly
halve.

PySOAX's output reaches 0.512 rad/px of curvature: physically impossible filaments.

## No human annotation enters the pipeline at any stage

The semantic model trains only on synthetic frames. The instancer has no learned weights at
all — it is a geometric algorithm with 17 hyperparameters — and those hyperparameters are
fitted on **synthetic** data, where ground truth is exact and free because the centerlines
*are* the objects the generator drew.

Fitting them on the real validation split instead is not merely unnecessary, it is worse:

| | tuned on real VAL | tuned on synthetic | paired difference | p |
|---|---|---|---|---|
| MT-34 TEST pooled | 0.416 | **0.457** | +0.041 [+0.018, +0.065] | <0.001 |
| TEST · crossing-dense half | 0.265 | **0.327** | +0.062 [+0.029, +0.095] | <0.001 |
| false positives (that half) | 404 | **326** | −78 [−120, −41] | <0.001 |
| fragmentation (that half) | 1.182 | **1.145** | −0.037 [−0.087, −0.002] | 0.028 |

The synthetic-fitted configuration weights the curvature constraint nearly twice as heavily
(`w_kappa` 8.99 → 16.11), discards short fragments far more aggressively (`min_length`
33.8 → 44.7) and fits tangents over a longer baseline — it is uniformly *more conservative
about what counts as a microtubule*. Human ground truth cannot reward that: these annotations
are human-corrected model output and are incomplete on sparse frames, so a tuner scored
against them is pushed to be permissive in order to recover filaments the annotator drew, and
permissive settings manufacture false positives everywhere else.

## The open problem is the foreground, and it is not the domain gap

With a predicted foreground the instancer drops from 0.920 to 0.416 on real data — and from
0.710 to 0.183 on *synthetic* data with exact ground truth, where there is no domain gap and
no annotation error. Measured with the same metric, the semantic model does not do better on
its own training distribution.

`src/mt_bench/fg_quality.py` addresses the reason this was hard to fix: the tolerant coverage
F1 the segmenter is tuned on **cannot rank foregrounds by their downstream instancing value**.
Over 198 same-frame model pairs:

| property | pairwise ranking accuracy |
|---|---|
| `cc_per_gt` (components per microtubule) | **0.82** |
| `gaps_per_mt` (foreground dropouts along a real filament) | **0.80** |
| `endp_per_kpx` (skeleton endpoints) | **0.79** |
| `prec2` — *the control, half of what we tune on* | **0.58** (chance) |

`prec2` correlates with downstream F1 at ρ = +0.87 across frames and still ranks models at
chance: both it and F1 track frame difficulty. Correlation was the wrong diagnostic.

**The trap this creates:** all three winners improve monotonically as a mask dilates, and raw
foreground fraction itself ranks at 0.75. Selection is therefore *constrained* minimisation —
minimise fragmentation subject to an over-firing ceiling and a collapse floor — and returns
"nothing selected" rather than a least-bad over-firing model.

## Layout

```
src/instance/    the instancers: geometry · skeleton_graph · matching · instancer_a
                 lifted · instancer_b · oracle · metrics (incl. paired bootstrap)
src/mt_bench/    benchmark tooling: CVAT import · overlays · GT statistics ·
                 field-of-view masking · fg_quality (the foreground gate)
synth/           the synthetic generator (stiff worm-like morphology, two-beam IRM
                 interference model, composited on real backgrounds, free GT)
scripts/         experiment drivers (most run on a GPU server, marked in their docstrings)
docs/protocol.md the path actually taken — including what did not work
tests/           104 tests
```

```bash
pip install -r requirements.txt
python -m pytest -q
```

## Honest status

- The instancer is frozen and measured; the semantic foreground is not final.
- The benchmark's ground truth is human-corrected model output, so it carries an agreement
  bias; a synthetic set with exact GT is scored alongside it for that reason.
- Several negative results are recorded rather than dropped: domain randomisation hurt a
  held-out lab, decoder complexity and scale augmentation coarsened thin filaments, SimGAN
  appearance refinement was a wash, and the orientation term in the matching cost is worth
  +0.010 on oracle masks and exactly 0.000 on predicted ones.
- One reproducibility failure is documented in `docs/protocol.md §17n`: a tuning script wrote
  its parameter file in place, so an earlier baseline can no longer be reproduced and its
  improvement delta is reported as development history rather than a measured result.
