"""Polyline-on-image overlay renders -- the benchmark's verification gate.

This exists to catch two silent, metric-destroying mistakes before any number is computed:
the ``(x=col, y=row)`` transpose, and a wrong coordinate scale. Both produce plausible-looking
F1 values while the GT sits on the wrong pixels, so the overlays must be LOOKED AT.
"""
from __future__ import annotations

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mt_bench.cvat_import import read_frame_h5  # noqa: E402


def _norm01(img: np.ndarray) -> np.ndarray:
    """Display stretch based on the CENTRAL crop, not the whole frame.

    Several corpus frames have a saturated bright surround outside an octagonal field of
    view (``training_img_114``: interior 52-81, surround 255). Percentiles over the whole
    frame are then dominated by the surround and squash the microtubules to black. The
    central 60% is inside the field of view for every frame in MT-34.
    """
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape[:2]
    core = img[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
    lo, hi = np.percentile(core, [1.0, 99.0])
    return np.clip((img - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


def render_overlay(image: np.ndarray, polylines, out_png: str,
                   title: str | None = None, dpi: int = 110) -> None:
    """Draw each polyline in its own colour over the image.

    ``polylines`` are ``(N, 2)`` arrays of ``(x=col, y=row)``: column 0 goes on the
    horizontal axis, column 1 on the vertical one.
    """
    h, w = image.shape[:2]
    fig, ax = plt.subplots(figsize=(w / 200.0 + 2.0, h / 200.0 + 2.0))
    ax.imshow(_norm01(image), cmap="gray", interpolation="nearest")
    cmap = plt.get_cmap("hsv")
    for i, p in enumerate(polylines):
        p = np.asarray(p, dtype=float)
        ax.plot(p[:, 0], p[:, 1], "-", lw=1.0,
                color=cmap((i * 0.37) % 1.0), alpha=0.9)
        ax.plot(p[0, 0], p[0, 1], ".", ms=3, color="lime")
        ax.plot(p[-1, 0], p[-1, 1], ".", ms=3, color="red")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_title(title or "", fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_all(h5_dir: str, out_dir: str) -> int:
    """Render an overlay for every frame in a benchmark directory. Returns the count."""
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for path in sorted(glob.glob(os.path.join(h5_dir, "*.h5"))):
        fr = read_frame_h5(path)
        stem = os.path.splitext(os.path.basename(path))[0]
        a = fr["attrs"]
        title = (f"{stem}  |  task {a.get('source_task', '?')}  "
                 f"split={a.get('split', '?')}  n={a.get('n_polylines', '?')}  "
                 f"manual={a.get('n_manual', '?')}")
        render_overlay(fr["image"], fr["polylines"],
                       os.path.join(out_dir, f"{stem}.png"), title=title)
        n += 1
    return n
