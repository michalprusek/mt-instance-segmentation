"""CVAT 'CVAT for images 1.1' XML -> microtubule polylines, and the on-disk h5 format.

Coordinate convention: CVAT ``points="x,y;x,y"`` are ``(x=col, y=row)`` -- the transpose of
NumPy ``[row, col]`` indexing -- and we keep them in that order end to end, because
``data/real/alice_eval/*.h5`` and ``centerline_f1`` both expect ``(x, y)``.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import h5py
import numpy as np


def _parse_points(text: str) -> np.ndarray:
    return np.array([[float(v) for v in pair.split(",")]
                     for pair in text.strip().split(";")], dtype=np.float64)


def parse_cvat_xml(path: str) -> list[dict]:
    """Parse a CVAT annotations.xml into one dict per image frame.

    Returns a list of dicts with keys ``frame_id, name, width, height, polylines, sources``.
    ``polylines`` holds ``(N, 2)`` float64 arrays in ``(x=col, y=row)``; degenerate
    single-vertex polylines are dropped. ``sources`` is CVAT's per-shape ``source``
    attribute -- ``"manual"`` means a human drew or edited it, ``"file"`` means it came
    straight from the uploaded pre-annotation and was never touched.
    """
    root = ET.parse(path).getroot()
    frames = []
    for im in root.findall("image"):
        polylines, sources = [], []
        for pl in im.findall("polyline"):
            pts = _parse_points(pl.get("points"))
            if len(pts) >= 2:
                polylines.append(pts)
                sources.append(pl.get("source", "unknown"))
        frames.append({
            "frame_id": int(im.get("id")),
            "name": im.get("name"),
            "width": int(im.get("width")),
            "height": int(im.get("height")),
            "polylines": polylines,
            "sources": sources,
        })
    return frames


def write_frame_h5(out_path: str, image: np.ndarray, polylines,
                   attrs: dict | None = None) -> None:
    """Write one benchmark frame in the ``alice_eval`` layout.

    Datasets: ``image`` (H, W) float32 and ``polylines/pl_XXXX`` (N, 2) float64 in
    ``(x=col, y=row)`` at NATIVE 1x resolution. The x1.5 eval upscale is applied by the
    evaluation pipeline, not stored here.
    """
    image = np.asarray(image, dtype=np.float32)
    with h5py.File(out_path, "w") as h:
        h.create_dataset("image", data=image, compression="gzip")
        grp = h.create_group("polylines")
        for i, p in enumerate(polylines):
            grp.create_dataset(f"pl_{i:04d}", data=np.asarray(p, dtype=np.float64))
        h.attrs["height"] = int(image.shape[0])
        h.attrs["width"] = int(image.shape[1])
        h.attrs["n_polylines"] = int(len(polylines))
        for k, v in (attrs or {}).items():
            h.attrs[k] = v


def read_frame_h5(path: str) -> dict:
    """Inverse of :func:`write_frame_h5`; returns ``{image, polylines, attrs}``."""
    with h5py.File(path, "r") as h:
        image = h["image"][:]
        keys = sorted(h["polylines"].keys())
        polylines = [h["polylines"][k][:] for k in keys]
        attrs = dict(h.attrs)
    return {"image": image, "polylines": polylines, "attrs": attrs}


def assign_split(names) -> dict:
    """Alternating val/test over the sorted names (even index -> val).

    Matches the existing project convention of a deterministic alternating
    6-val/6-test split of the sorted Alice frames.
    """
    return {n: ("val" if i % 2 == 0 else "test")
            for i, n in enumerate(sorted(names))}
