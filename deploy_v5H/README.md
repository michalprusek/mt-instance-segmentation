# Microtubule instance segmentation — deployment package, v5H

Individual microtubule centerlines from a label-free light-microscopy frame (IRM / TIRF).
Self-contained: weights, model code, the network library and the instancer travel together, and
nothing is downloaded at run time.

**No human annotation enters this model at any stage.** The semantic network is trained purely
on synthetic frames; the instancer has no learned weights and its hyperparameters are fitted on
synthetic data with exact ground truth.

> **DEPLOYED** at `cvat2.utia.cas.cz:/home/cvat/cell-segmentation-hub/mt-instance-seg-v5H`,
> alongside `mt-instance-seg` (v4b), which is left in place. An earlier revision of this file
> carried a NOT-DEPLOYED banner and then a note that the banner and the session record
> disagreed; the question was settled on 2026-08-25, when tulen's working tree was wiped and
> **this deployed copy turned out to be the only surviving source of both `dino_seg_v5H.pth`
> and `params_a_model_v5H.json`** — they were recovered from here. Read *Should you deploy
> this?* below anyway: the improvement over v4b is real in point estimate and **not**
> statistically supported.

---

## Should you deploy this?

| | v4b (deployed) | **v5H (this)** |
|---|---|---|
| strict centerline-F1, real MT-34 VAL | 0.4655 | **0.4953** |
| difference | — | +0.030 [−0.024, +0.097], **p = 0.331** |

The gain is **not separable from zero** on the 17-frame validation split, and MT-34 TEST was
deliberately **not** scored, because a pre-declared rule required the validation interval to
exclude zero before spending a fifth test exposure. Run-to-run training noise was measured at
SD 0.0040, so the spread is not seed variation — the benchmark simply cannot resolve a 0.03
effect at this size.

So: this is not a measured improvement. It is a **newer, cleaner system** — trained on a
generator whose defects were found and corrected, with instancer parameters fitted to its own
foreground rather than inherited from another model. Deploy it if that is worth having;
keep v4b if you want no change without evidence. Both are defensible; the numbers above are
what the decision should rest on.

---

## Quick start

```bash
python predict.py --input frame.tif --out-dir results/ --overlay
python predict.py --input folder_of_tifs/ --out-dir results/
```

Verified 2026-08-17 on the bundled sample: **65 instances in 13.2 s** on an RTX A5000, run with
an interpreter that does **not** have `dynamic_network_architectures` installed — which is the
point of the vendoring and the same situation as the `spheroseg-ml` container.

On the CVAT host, the v4b package's recipe applies unchanged, with the directory name swapped:

```bash
cd /home/cvat/cell-segmentation-hub
docker run --rm --gpus all \
    -v /home/cvat/cell-segmentation-hub/mt-instance-seg-v5H:/pkg \
    cell-segmentation-hub-ml \
    python /pkg/predict.py --input /pkg/sample --out-dir /pkg/sample_out --overlay
```

> `docker exec spheroseg-ml ...` does not work: the running service container has no mount for
> this directory.

### Output

Unchanged from v4b — one JSON per frame, one polyline per microtubule, **already mapped back to
input resolution**; the internal 1.5× working scale never reaches the caller.

```json
{"image": "frame.tif", "shape": [1024, 1024], "n_instances": 65,
 "coordinate_order": "x=col, y=row, in original image pixels",
 "polylines": [[[241.33, 0.0], [240.39, 0.94], ...], ...]}
```

> **Coordinate order is `[x, y]` = `[col, row]`**, the transpose of NumPy indexing — the same
> convention CVAT uses. Getting it backwards has silently broken two things in this project's
> history, so it is restated in every output file.

---

## Tracking a time series

`predict.py` answers "which microtubules are in this frame". `track.py` answers "which
microtubule in frame *t* is which in frame *t+1*":

```bash
python track.py --input folder_of_frames/ --out-dir tracks/ --overlay
```

Frames are ordered by **natural sort** of their filenames (`f2` before `f10`), and that order is
the time axis. Output is `tracks.json` (every track with its polyline in each frame it appears
in) plus one `<name>.json` per frame in `predict.py`'s format with an added `track_ids` field.
With `--overlay`, colour is the **track identity**, so the same microtubule keeps its colour
across frames and a switch is visible by eye.

The association is pure geometry — no learned weights, nothing to train. Filaments in
consecutive frames are paired by a **global min-cost matching** under a displacement bound with
a priced "leave this one unmatched" option, which is what track birth and death are. Global
rather than greedy, for the same reason the instancer's junction matcher is: a greedy
nearest-neighbour link cannot revisit a wrong choice, so in a dense bundle the first filament
claims the wrong partner and every later one inherits the error. Stage drift is estimated first
and removed, from normal-flow constraints pooled over filaments at different orientations — a
median-centroid estimator reports gliding motility as drift, which is the one error a motility
assay cannot tolerate.

### What it measures, and where the wall is

There is no real cross-frame ground truth for microtubules, so this is measured on synthetic
sequences with exact correspondence — 24 sequences x 5 frames, split per sequence, 12 held out.
Two sets are reported because neither reproduces this model's training appearance exactly: the
one the tracker was originally validated on, and one from this model's own generator
calibration. `oracle` is the same instancer and tracker run on ground-truth centerlines — the
ceiling this foreground would have to reach, not a competitor.

| test split | identity switch rate | tracks per detected microtubule | objects found |
|---|---|---|---|
| **v5H**, original set | 0.014 | 2.86 | 0.88 |
| oracle, original set | 0.011 | 1.98 | 0.80 |
| **v5H**, own-generator set | 0.014 | 3.18 | 0.98 |
| oracle, own-generator set | 0.006 | 1.74 | 0.84 |

*"Objects found" is below 1.0 for the oracle too: filaments shorter than the instancer's
`min_length` never become a detection at all, whichever mask they came from.*

Read it this way. **Association is not the problem:** 3 identity switches in 207 links, barely
above what the same tracker does on perfect input. **Fragmentation is.** Each microtubule is
followed by roughly three disconnected tracks instead of one, because the per-frame foreground
breaks into pieces and a piece is a worse partner than a whole filament.

Against the predecessor, on the original set and the same instancer: **v4b 3.18 tracks per
microtubule, this model 2.52.** Those two are the *mixed* form of the metric, which counts an
undetected microtubule as zero tracks and is the only form v4b's number exists in — its
predictions were lost and cannot be re-run, so this is a cross-run comparison rather than a
paired one. It is worth knowing that this model is less fragmented in time than the one it
succeeds; the gap to the oracle is still the whole story.

> **Two things this deliberately does not do.**
> **It does not merge fragments** into one instance per frame, so do not read the track count
> as a microtubule count. That fusion step is designed and unbuilt.
> **It reports no velocities.** The estimator is sound — 0.98–1.03 of truth on ground-truth
> polylines — but a detection's extent is set by where the mask is strong rather than by which
> material the filament is made of, so it re-forms in place and under-reports gliding by ~2x,
> and the tracks that survive longest are the ones anchored to something that is not moving.
> The information is destroyed upstream of the estimator, so no velocity is emitted at all
> rather than one emitted with a caveat.

The association gates were fitted at ~5 px/frame of motion. `--max-shift` (in input pixels) and
`--min-overlap` expose them for faster acquisitions; left unset, the fitted values are used
unchanged, which is the configuration every number above was measured with.

---

## What changed from v4b

**A different network.** v4b was a frozen DINOv2 ViT-L/14 with a light high-resolution decoder.
This is **nnU-Net's ResEnc-M topology** (8 stages, features 32→512), whose seven successive /2
downsamplings give the bottleneck a receptive field spanning the whole 512 px patch. That
matters for filaments: evidence has to be integrated *along* a microtubule, and v4b's
full-resolution branch reached only 29 px.

**A corrected generator.** The training generator was found to be labelling filaments it
rendered below the noise floor — synthetic labelled microtubules sat at median SNR 0.50 against
1.45 for human-annotated real ones, so roughly a third of training positives carried no
learnable signal. Correcting the contrast is worth **+0.085 (p = 0.001)** on this architecture.
Two further generator corrections (crossing-angle statistics, a synthesised field stop) were
made and measured as **nulls** on real data; they are in the training data but claim nothing.

**Instancer parameters fitted to this foreground.** `merge_radius` 8.98 → 5.0, `prob_thr`
0.44 → 0.97 (an earlier revision of this file said 0.99; `params/params_a_model_v5H.json`,
which is what actually runs, says 0.97). A large junction-contraction radius suits a shattered mask and damages a clean
one, so the shipped v4b vector actively penalises other foregrounds. Worth +0.090 on held-out
synthetic frames and **+0.006 on real data** — quote the transferred number, not the tuning one.

**One channel instead of six — a behavioural difference invisible in the output.** v4b emitted
K = 6 orientation-keyed channels, and the instancer could use them as extra evidence when
pairing arms at a junction. This model emits one. That term (`w_ori`) measured exactly 0.000 on
predicted masks, so nothing is lost here — but **instancer B requires the orientation split and
will not work with this checkpoint.**

---

## What is in here

```
predict.py                      the entry point, one frame at a time
track.py                        the entry point for a TIME SERIES (see Tracking below)
model/net.py                    the architecture, with nnU-Net's plan copied verbatim
vendor/dynamic_network_architectures/   vendored, because the container lacks it and has no network
weights/dino_seg_v5H.pth        538 MB (no frozen backbone to carry, unlike v4b's 1.2 GB)
params/params_a_model_v5H.json  the instancer's hyperparameters, fitted to THIS foreground
instance/                       the instancer (pure numpy/scipy/skimage/networkx, no weights)
sample/training_img_114.tif     one real annotated frame, so the quick start runs immediately
```

The instancing stage is unchanged: junction clusters are contracted, tangents fitted over a
window, and each junction resolved by a **min-cost perfect matching** over its arms with a
priced "leave this arm open" option. Every join is constrained by

    κ = |dθ/ds| ≤ 0.25 rad/px

as a **hard** constraint, derived rather than tuned — just above the 0.239 rad/px maximum over
957 human-annotated microtubules at an 8 px baseline. Microtubules bend; they do not kink.

---

## Honest limits

- Not a statistically supported improvement over v4b. See the table above.
- Dense, crossing-heavy fields remain much harder than sparse ones; that gap is the open
  problem, and the same instancer on a *perfect* foreground reaches 0.92, so the loss is
  upstream of the instancing.
- The benchmark's ground truth is human-corrected model output: it carries an agreement bias
  and is demonstrably incomplete on sparse frames.
- Trained and evaluated on IRM. TIRF is supported by the architecture but not validated.
- Tuned for ~2 px-wide filaments at this project's native pixel size. Very different
  magnifications need `prob_thr` and `min_length` revisited.
- Emits one channel, so instancer B cannot consume it (see above).
- Tracking inherits all of the above and adds one: it is validated only on synthetic sequences,
  because no real cross-frame ground truth for microtubules exists. Neither synthetic set
  reproduces this model's training appearance — single-frame F1 on them is far below what the
  same weights reach on real frames — so the tracking numbers characterise the tracker on a
  degraded foreground rather than the best this model can do.

## Provenance

Full development protocol including every negative result: `docs/F1_PUSH_PLAN.md` in
<https://github.com/michalprusek/mt-instance-segmentation>. Weights `dino_seg_v5H.pth`;
instancer parameters `params_a_model_v5H.json` (fitted on stratified synthetic frames with
exact ground truth, re-ranked on a disjoint synthetic set). Packaged 2026-08-17.
