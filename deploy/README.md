# Microtubule instance segmentation — deployment package

Individual microtubule centerlines from a label-free light-microscopy frame (IRM / TIRF).
Self-contained: weights, model code and instancer travel together, and nothing is downloaded
at run time.

**No human annotation enters this model at any stage.** The semantic network is trained purely
on synthetic frames; the instancer has no learned weights at all and its hyperparameters are
fitted on synthetic data with exact ground truth.

---

## Quick start

**On this host — verified working, copy-paste as is.** It reuses the ML service's container
image but starts a *separate* throwaway container, so the running `spheroseg-ml` service is
never touched:

```bash
cd /home/cvat/cell-segmentation-hub
docker run --rm --gpus all \
    -v /home/cvat/cell-segmentation-hub/mt-instance-seg:/pkg \
    cell-segmentation-hub-ml \
    python /pkg/predict.py --input /pkg/sample --out-dir /pkg/sample_out --overlay
```

That command was run on 2026-08-11 against the bundled sample frame and produced 84 instances
in 9.8 s on the RTX A5000. Point `--input` at any directory of `.tif` files; to read from the
service's upload area, add `-v /home/cvat/cell-segmentation-hub/backend/uploads/blue:/uploads`
and use `--input /uploads/...`.

> `docker exec spheroseg-ml ...` does **not** work: the running service container has no mount
> for this directory. Either use the `docker run` form above, or add a volume mount for
> `mt-instance-seg` to `docker-compose.production.yml` if you want it inside the live service.

**Anywhere else:**

```bash
python predict.py --input frame.tif --out-dir results/ --overlay
python predict.py --input folder_of_tifs/ --out-dir results/
```

Roughly **4–10 s per 1024² frame on a GPU** (4 s on an A100-class card, 10 s on the A5000);
it runs on CPU too, considerably slower.

> Instance counts can differ by about one between machines — the same frame gives 83 on the
> development GPU and 84 here. This is ordinary floating-point nondeterminism across GPU and
> torch versions reaching a threshold decision, not a configuration problem. Weights and
> parameters are identical (checksum-verified).

### Output

One JSON per input frame:

```json
{
  "image": "frame.tif",
  "shape": [938, 1120],
  "n_instances": 19,
  "coordinate_order": "x=col, y=row, in original image pixels",
  "polylines": [[[263.33, 38.67], [263.98, 37.53], ...], ...]
}
```

One polyline per microtubule, densely sampled, **already mapped back to input resolution** —
the internal 1.5× working scale never reaches the caller.

> **Coordinate order is `[x, y]` = `[col, row]`**, the transpose of NumPy indexing. This is the
> same convention CVAT uses for polylines. Getting it backwards has silently broken two things
> in this project's history, so it is stated in every output file.

---

## Requirements

`torch`, `numpy`, `scipy`, `scikit-image`, `networkx`, `tifffile`, and `matplotlib` only for
`--overlay`. **The `spheroseg-ml` container already satisfies all of them** — verified against
the running container: python 3.10.20, torch 2.6.0+cu124 (CUDA available), numpy 1.26.4,
scipy 1.11.4, scikit-image 0.22.0, networkx 3.4.2, tifffile 2025.5.10. Nothing to install.

For a standalone environment: `pip install -r requirements.txt`.

---

## What is in here

```
predict.py                     the entry point
model/dino_seg.py              the semantic architecture (see the packaging note at its top)
model/hub/                     vendored DINOv2 repo, so torch.hub never contacts GitHub
weights/dino_seg_ori_v4b.pth   1.2 GB — full state_dict, frozen backbone included
params/params_a_model_synthtuned.json   the instancer's 17 fitted hyperparameters
instance/                      the instancer (pure numpy/scipy/skimage/networkx, no weights)
sample/training_img_114.tif    one real annotated frame, so the quick start runs immediately
```

The directory carries its own `.gitignore` containing `*`, so it stays out of the
cell-segmentation-hub repository's status without anyone editing that repository's files.

### How it works

**Stage 1 — semantic.** A frozen DINOv2 ViT-L/14 backbone with a light high-resolution decoder
predicts **K = 6 orientation-keyed foreground channels**. At a crossing the two filaments have
different local tangents and therefore land in *different* channels, so the representation
stays amodal: the crossing does not have to be resolved by the pixels alone.

**Stage 2 — instancing.** Junction clusters are contracted, tangents are fitted over a window,
and each junction is resolved by a **min-cost perfect matching** over its arms with a priced
"leave this arm open" option — a global choice per junction rather than a greedy one. Every
join is constrained by

    κ = |dθ/ds| ≤ 0.25 rad/px

as a **hard** constraint. That bound is *derived, not tuned*: it sits just above the 0.239
rad/px maximum over 957 human-annotated microtubules measured at an 8 px baseline. Microtubules
are stiff polymers — they bend, they do not kink — so a join that would require a kink is
forbidden outright rather than merely penalised.

> Measurement scale is part of the claim: the same annotations give 1.015 rad/px at a 2 px
> baseline, because coarse human polylines make vertex-level turns meaningless.

---

## Accuracy

Strict centerline-F1 (5 px tolerance, 95 % length *and* precision coverage) on **MT-34**:
34 real annotated frames, 957 ground-truth polylines, 32.1 crossings per frame on the
crossing-dense half. Intervals are paired, task-stratified bootstraps over frames.

| | mean F1 | 95 % CI |
|---|---|---|
| **this model** (predicted foreground) | **0.457** | [0.379, 0.533] |
| the same instancer on a *perfect* foreground | 0.920 | [0.870, 0.966] |
| PySOAX, tuned to the same budget | 0.590 (on the perfect foreground) | [0.470, 0.724] |

Two things worth reading off that table:

- **The instancer is not the bottleneck; the semantic foreground is.** Given a perfect
  foreground the same code reaches 0.920. The gap between 0.920 and 0.457 is all upstream.
- Against PySOAX **on identical input**, the curvature-bounded instancer scores +0.330
  [+0.219, +0.432], p < 0.001, and preserves microtubule identity through crossings where
  PySOAX scores 0.000 on all 288 non-shallow crossings. PySOAX output also reaches 0.512
  rad/px — filaments that are physically impossible.

### Honest limits

- Absolute performance on dense, crossing-heavy fields is modest (0.327 on that half). Sparse
  fields do much better (0.695).
- The benchmark's ground truth is human-corrected model output, so it carries an agreement
  bias and is demonstrably incomplete on sparse frames.
- Trained and evaluated on IRM. TIRF is supported by the architecture but not quantitatively
  validated.
- Tuned for ~2 px-wide filaments at the native pixel size of this project's data. Very
  different magnifications will need `prob_thr` and `min_length` revisited.

---

## Relationship to the microtubule model already deployed here

`backend/segmentation/models/microtubule/` runs **v7** — DINOv3-L + DPT with a 32-d embedding,
post-processed by **PySOAX** — with weights at `weights/microtubule_v7.pt`. This package is a
different network (DINOv2 + orientation channels) *and* a different post-processor.

**This package does not touch that pipeline.** It is standalone by design: the v7 wrapper has
two callers (the interactive ML queue and the Automated Essays batch assay) and both would need
re-verifying before anything is swapped.

If you do want to integrate later, the highest-value and lowest-risk step is to replace
**PySOAX only**, keeping the v7 network. `instance/` consumes a plain binary foreground mask
and returns polylines:

```python
from instance.instancer_a import instance_a
import json

params = json.load(open("params/params_a_model_synthtuned.json"))
params.pop("kappa_max", None)               # derived, never read from the params file
polylines, masks = instance_a(foreground_bool_2d, 0.25, params)
```

Optionally pass `channels=` (K orientation channels) and `prob=` (the probability map) to let
it use image evidence when bridging gaps.

**One caveat, stated plainly:** those 17 hyperparameters were fitted against *this* package's
foreground, not v7's seed map. Dropping them onto v7 unchanged is untested — re-fit them on
synthetic data with the v7 foreground before trusting the result. That re-fit is exactly what
made this package better than a real-annotation-tuned one (+0.041 pooled, p < 0.001), so it is
worth doing properly rather than skipping.

---

## Provenance

Source, full development protocol including the negative results, and the benchmark tooling:
<https://github.com/michalprusek/mt-instance-segmentation>

Weights `dino_seg_ori_v4b.pth`; instancer parameters `params_a_model_synthtuned.json`
(fitted on synthetic validation data, 100 Optuna trials). Packaged 2026-08-11.
