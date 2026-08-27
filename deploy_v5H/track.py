#!/usr/bin/env python3
"""Cross-frame microtubule tracking, v5H: a time series of frames in, per-microtubule tracks out.

``predict.py`` answers "which microtubules are in this frame". This answers "which microtubule
in frame t is which in frame t+1", by running that same per-frame pipeline over an ordered
series and then associating the results across frames.

The association is **pure geometry, no learned weights**: filaments in consecutive frames are
paired by a global min-cost matching under a displacement bound, with a priced option to leave
a filament unmatched -- which is what track birth and death are. It is the junction matcher
from the instancer pointed at a different problem, and it is global rather than greedy for the
same reason: a greedy nearest-neighbour link commits to the locally cheapest partner and cannot
revisit it, so in a dense bundle the first filament claims the wrong one and every later
filament inherits the error.

A microscope drifts, and every filament in the field moves with it. The common-mode shift is
estimated first -- from normal-flow constraints pooled over filaments at different orientations,
because a gliding filament's centroid travels along its own contour and a median-centroid
estimator would report motility as drift. Association is then done drift-free.

What this does NOT do -- both stated because the output looks like it might
-------------------------------------------------------------------------
* **It does not merge fragments.** The per-frame foreground is shattered, so one microtubule is
  typically followed by 2-3 disconnected tracks rather than one. Fusing co-moving fragments
  into a single instance is a separate, unbuilt step; do not read the track count as a
  microtubule count.
* **It reports no velocities.** Gliding speed is not measurable from this foreground: a
  detection's extent is set by where the mask is strong rather than by which material the
  filament is made of, so it re-forms in roughly the same place and under-reports motion by
  ~2x, and the tracks that survive longest are the ones anchored to something that is not
  moving. The estimator is sound (0.98-1.03 of truth on ground-truth polylines) and the
  information is destroyed upstream of it, so no velocity is emitted rather than one emitted
  with a caveat.

Usage
-----
    python track.py --input folder_of_frames/ --out-dir tracks/
    python track.py --input folder/ --out-dir tracks/ --overlay --min-frames 2

Frames are ordered by **natural sort** of their filenames, so ``f2.tif`` precedes ``f10.tif``.
That order is the time axis and it is echoed into the output; check it if your naming is
unusual.

Outputs ``tracks.json`` (every track, with its frames and its polyline in each) plus one
``<name>.json`` per frame in ``predict.py``'s format with an added ``track_ids`` field, and
with ``--overlay`` a PNG per frame in which **colour is the track identity** -- the same
microtubule keeps its colour across frames, which is what makes a switch visible by eye.
Vertices are ``[x, y]`` = ``[col, row]`` in ORIGINAL image pixels, exactly as ``predict.py``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
from scipy.ndimage import zoom

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from predict import (DEFAULT_PARAMS, DEFAULT_WEIGHTS, UP, Segmenter,  # noqa: E402
                     display01, field_of_view, instance_polylines, load_image, norm01)


def natural_key(path: str):
    """Sort key that orders ``f2`` before ``f10``. Plain lexical sort does not."""
    name = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def save_track_overlay(path: str, img01: np.ndarray, polylines, track_ids) -> None:
    """Overlay whose colour is keyed to the TRACK, not to position in the list.

    ``predict.py``'s overlay colours by list index, which is right for a still frame and wrong
    here: the same microtubule would change colour whenever the detection order changed, hiding
    exactly the identity errors this view exists to show.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(img01.shape[1] / 110, img01.shape[0] / 110))
    ax.imshow(img01, cmap="gray", interpolation="nearest")
    cmap = plt.get_cmap("hsv")
    for p, tid in zip(polylines, track_ids):
        p = np.asarray(p, dtype=float)
        ax.plot(p[:, 0], p[:, 1], "-", lw=1.1, color=cmap((tid * 0.37) % 1.0), alpha=0.95)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                    help="directory of frames in time order (natural sort), or a glob")
    ap.add_argument("--out-dir", default="tracks")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--params", default=DEFAULT_PARAMS)
    ap.add_argument("--prob-thr", type=float, default=None,
                    help="foreground threshold; default comes from the params file")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    ap.add_argument("--overlay", action="store_true", help="also write a PNG per frame")
    ap.add_argument("--no-fov", action="store_true",
                    help="do NOT mask the field stop. Every published number for this model "
                         "was measured WITH the mask; turning it off costs ~0.023 F1.")
    ap.add_argument("--max-shift", type=float, default=None,
                    help="hard gate: no filament is associated across a larger displacement, "
                         "in INPUT image pixels (default 16.7, i.e. the fitted 25 px in the "
                         "1.5x working frame). Raise it for fast acquisitions -- the default "
                         "was fitted at ~5 px/frame of motion.")
    ap.add_argument("--min-overlap", type=float, default=None,
                    help="fraction of the shorter filament that must have a partner nearby "
                         "before two detections may be linked (default 0.35, unitless)")
    ap.add_argument("--min-frames", type=int, default=1,
                    help="drop tracks seen in fewer than this many frames. 1 keeps everything, "
                         "which is the honest default; 2 drops single-frame flotsam.")
    args = ap.parse_args()

    paths = (sorted(sum([glob.glob(os.path.join(args.input, e))
                         for e in ("*.tif", "*.tiff", "*.png", "*.jpg")], []), key=natural_key)
             if os.path.isdir(args.input) else
             sorted(glob.glob(args.input), key=natural_key))
    if len(paths) < 2:
        raise SystemExit(
            f"tracking needs at least 2 frames, found {len(paths)} at {args.input}.\n"
            f"For a single frame use predict.py -- one frame in, no tracks out, is the honest "
            f"answer rather than an error.")
    os.makedirs(args.out_dir, exist_ok=True)

    params = json.load(open(args.params))
    params.pop("kappa_max", None)      # derived, never read from a params file
    thr = args.prob_thr if args.prob_thr is not None else params.get("prob_thr", 0.35)

    # The association gates were fitted in the 1.5x working frame, so tracking is done there and
    # only the OUTPUT is mapped back. A caller thinks in input pixels, so --max-shift is taken
    # in input pixels and scaled here; leaving it unset passes the fitted defaults through
    # untouched, which is the configuration every tracking number was measured with.
    gates = {}
    if args.max_shift is not None:
        gates["max_shift"] = args.max_shift * UP
    if args.min_overlap is not None:
        gates["min_overlap"] = args.min_overlap

    seg = Segmenter(args.weights, args.device)
    print(f"device={seg.device} | {len(paths)} frame(s) | threshold={thr:.3f}", flush=True)

    per_frame, raws = [], []
    for path in paths:
        t0 = time.time()
        raw = load_image(path)
        # Association compares coordinates between frames, so frames of different sizes would
        # produce a confident answer that means nothing. An acquisition series is same-size by
        # construction; a mixed folder is a mistake and should say so rather than be tracked.
        if raws and raw.shape != raws[0].shape:
            raise SystemExit(
                f"frame sizes differ: {os.path.basename(paths[0])} is {raws[0].shape} but "
                f"{os.path.basename(path)} is {raw.shape}. Tracking compares coordinates "
                f"across frames, so a mixed-size folder cannot be one time series.")
        img01 = zoom(norm01(raw), UP, order=1)
        chans = seg.channels(img01)
        fov = None if args.no_fov else field_of_view(raw, chans.shape[1:])
        polylines = instance_polylines(chans, params, thr, fov)
        per_frame.append(polylines)
        raws.append(raw)
        print(f"  {os.path.basename(path)}: {len(polylines)} instances "
              f"({time.time() - t0:.1f}s)", flush=True)

    from instance.tracker import track_sequence
    tracks = track_sequence(per_frame, gates or None)
    tracks = [t for t in tracks if t.length >= args.min_frames]

    # Which track owns each detection, per frame -- the per-frame view of the same answer.
    # The tracker stores ``np.asarray(poly)``, which normally hands back the very same object,
    # but identity is not guaranteed for every input array, and a failed lookup would silently
    # drop a detection from the per-frame files rather than raise. Hence the equality fallback.
    owner = [{} for _ in per_frame]
    for tr in tracks:
        for k, poly in zip(tr.frames, tr.polylines):
            j = next((idx for idx, cand in enumerate(per_frame[k])
                      if cand is poly or np.array_equal(np.asarray(cand), np.asarray(poly))),
                     None)
            if j is not None:
                owner[k][j] = tr.track_id

    shape = [int(raws[0].shape[0]), int(raws[0].shape[1])]
    out_tracks = [{"track_id": int(t.track_id),
                   "frames": [int(k) for k in t.frames],
                   "polylines": [(np.asarray(p, dtype=float) / UP).round(2).tolist()
                                 for p in t.polylines]}
                  for t in tracks]
    with open(os.path.join(args.out_dir, "tracks.json"), "w") as fh:
        json.dump({"frames": [os.path.basename(p) for p in paths],
                   "shape": shape,
                   "n_frames": len(paths),
                   "n_tracks": len(out_tracks),
                   "coordinate_order": "x=col, y=row, in original image pixels",
                   "frame_order": "natural sort of the filenames; index into 'frames'",
                   "tracks": out_tracks}, fh)

    for k, (path, polylines) in enumerate(zip(paths, per_frame)):
        name = os.path.splitext(os.path.basename(path))[0]
        keep = [(j, p) for j, p in enumerate(polylines) if j in owner[k]]
        ids = [owner[k][j] for j, _ in keep]
        with open(os.path.join(args.out_dir, f"{name}.json"), "w") as fh:
            json.dump({"image": os.path.basename(path),
                       "shape": shape,
                       "frame_index": k,
                       "n_instances": len(keep),
                       "coordinate_order": "x=col, y=row, in original image pixels",
                       "track_ids": ids,
                       "polylines": [(np.asarray(p, dtype=float) / UP).round(2).tolist()
                                     for _, p in keep]}, fh)
        if args.overlay:
            hi = zoom(display01(raws[k]), UP, order=1)
            save_track_overlay(os.path.join(args.out_dir, f"{name}.png"),
                               hi, [p for _, p in keep], ids)

    seen = sum(len(t.frames) for t in tracks)
    print(f"{len(out_tracks)} tracks over {len(paths)} frames "
          f"({seen} detections linked); wrote {args.out_dir}/", flush=True)


if __name__ == "__main__":
    main()
