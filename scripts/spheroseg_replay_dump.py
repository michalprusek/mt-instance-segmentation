"""Correction replay, step 1: capture exactly what the hub's microtubule package hands to its decoder.

The hub's ``predict()`` builds the probability map, thresholds it and calls ``instance_a``. Rather
than re-implement that path for two package versions (v5H runs the network on a 1.5x-upscaled image,
SPARSE35 at native scale and resamples the probabilities), the call itself is intercepted: the mask,
the map, the channels, the parameter vector and the instances it returns are saved as they were
passed. What is stored is therefore the decoder input the annotators' proposal was made from.

Runs in an env that has the hub package's dependencies (torch, cv2); one package version per call.

    spheroseg_replay_dump.py <pkg_dir> <weights.pth> <out_dir> <tag> <img.png> [<img.png> ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

pkg, weights, out_dir, tag = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4]
imgs = [Path(p) for p in sys.argv[5:]]
out_dir.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, pkg)
import wrapper  # noqa: E402  the hub package's wrapper.py; it puts its own `instance` package on sys.path
import instance.instancer_a as hub_instancer  # noqa: E402

seen: dict = {}
_instance_a = hub_instancer.instance_a


def _spy(mask, kappa_max, params=None, **kw):
    polylines, masks = _instance_a(mask, kappa_max, params, **kw)
    seen.clear()
    seen.update(mask=np.asarray(mask, bool), kappa=float(kappa_max), params=dict(params or {}),
                channels=np.asarray(kw["channels"], np.float32), prob=np.asarray(kw["prob"], np.float32),
                polylines=[np.asarray(p, np.float64) for p in polylines])
    return polylines, masks


hub_instancer.instance_a = _spy   # predict() imports the name at call time, so this is what it gets

model = wrapper.MicrotubuleModel().load_weights(weights)
for p in imgs:
    im = Image.open(p)
    assert im.mode in ("I;16", "I;16B", "I;16L"), (p, im.mode)
    out = model.predict(np.array(im, dtype=np.uint16))
    cut = float(seen["params"].get("prob_thr", model.DEFAULT_SEED_THRESHOLD))
    assert np.array_equal(seen["mask"], seen["prob"] > cut), "captured mask is not prob > cut"
    assert len(out["centerlines_rc"]) == len(seen["polylines"])
    stem = f"{p.stem}__{tag}"
    np.savez_compressed(out_dir / f"{stem}__decoder_input.npz", prob=seen["prob"], channels=seen["channels"])
    (out_dir / f"{stem}__decoder_input.json").write_text(json.dumps({
        "image": p.name, "tag": tag, "cut": cut, "kappa": seen["kappa"], "up": float(wrapper.UP),
        "params": {k: v for k, v in seen["params"].items() if isinstance(v, (int, float))},
        "shape_eval": list(seen["prob"].shape), "shape_native": list(np.array(im).shape),
        # (x=col, y=row) in the 1.5x instancer frame, before the output simplification
        "polylines_eval": [q.tolist() for q in seen["polylines"]],
    }))
    print(f"[{tag}] {p.name}: map {seen['prob'].shape} channels {seen['channels'].shape[0]} cut {cut} "
          f"-> {len(seen['polylines'])} instances, foreground {seen['mask'].mean():.4f}", flush=True)
