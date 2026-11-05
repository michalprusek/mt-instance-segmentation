"""Regenerate the model proposal the SpheroSeg annotators started from.

Runs the hub's own MicrotubuleModel wrapper (one package version per call) on the static
segmentation-source frame of each verified project, through the same steps as
ModelLoader.predict_microtubule: 16-bit PNG -> np.uint16 -> predict(image_np), default threshold
from the package's params file, polyline i gets id ``polyline_{i+1}`` and (x=col, y=row) points.

usage: reinfer.py <pkg_dir> <weights.pth> <out_dir> <tag> <img.png> [<img.png> ...]
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

pkg, weights, out_dir, tag = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4]
imgs = [Path(p) for p in sys.argv[5:]]
out_dir.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, pkg)
import torch  # noqa: E402
import wrapper  # noqa: E402  (the hub package's wrapper.py, imported top-level like the tests do)

model = wrapper.MicrotubuleModel().load_weights(weights)
params = model.params
print(f"[{tag}] pkg={pkg} wrapper_md5={hashlib.md5(Path(pkg, 'wrapper.py').read_bytes()).hexdigest()} "
      f"torch={torch.__version__} dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'} "
      f"prob_thr={params.get('prob_thr')} min_length={params.get('min_length')} UP={wrapper.UP}", flush=True)

for p in imgs:
    im = Image.open(p)
    assert im.mode in ("I;16", "I;16B", "I;16L"), (p, im.mode)
    a = np.array(im, dtype=np.uint16)
    t0 = time.time()
    r = model.predict(a)  # seed_threshold=None, exactly as the /segment route
    dt = time.time() - t0
    polylines = []
    for idx, cl in enumerate(r["centerlines_rc"]):
        cl = np.asarray(cl, dtype=np.float64)
        polylines.append({
            "id": f"polyline_{idx + 1}",
            "points": [{"x": float(cl[i, 1]), "y": float(cl[i, 0])} for i in range(cl.shape[0])],
        })
    stem = f"{p.stem}__{tag}"
    (out_dir / f"{stem}.json").write_text(json.dumps({
        "image": p.name, "image_md5": hashlib.md5(p.read_bytes()).hexdigest(), "shape": list(a.shape),
        "tag": tag, "prob_thr": params.get("prob_thr"), "min_length": params.get("min_length"),
        "weights": str(weights), "polylines": polylines,
    }))
    np.savez_compressed(out_dir / f"{stem}__prob.npz", prob=np.asarray(r["prob"], dtype=np.float16))
    print(f"[{tag}] {p.name}: {a.shape} -> {len(polylines)} polylines in {dt:.1f}s "
          f"(prob>thr fraction {(np.asarray(r['prob']) > params.get('prob_thr')).mean():.4f})", flush=True)
