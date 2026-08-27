#!/usr/bin/env python3
"""Cross-frame microtubule tracking, v5H: a time series of frames in, per-microtubule tracks out.

``predict.py`` answers "which microtubules are in this frame". This answers "which microtubule
in frame t is which in frame t+1", by running that same per-frame pipeline over an ordered
series and then associating the results across frames.

The association is **pure geometry, no learned weights**: filaments in consecutive frames are
paired by a global min-cost matching under a distance bound, with a priced option to leave a
filament unmatched -- which is what track birth and death are. Structurally it is the junction
matcher from the instancer pointed at a different problem (the same device, not the same code),
and it is global rather than greedy for the same reason: a greedy nearest-neighbour link commits
to the locally cheapest partner and cannot revisit it, so in a dense bundle the first filament
claims the wrong one and every later filament inherits the error.

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
That order is the time axis, it is printed at startup and echoed into the output, and the tool
refuses inputs where it would be ambiguous -- see :func:`order_frames`.

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

#: Extensions accepted as frames, grouped by format and in preference order. ONE GROUP is used
#: per run -- see :func:`order_frames` for why mixing formats is refused rather than merged.
#: The grouping matters: ``.tif`` and ``.tiff`` are one format, and splitting them would drop
#: half a folder that happens to use both spellings.
FRAME_EXT_GROUPS = (("tif", "tiff"), ("png",), ("jpg", "jpeg"))
FRAME_EXTS = tuple(e for g in FRAME_EXT_GROUPS for e in g)


def natural_key(path: str):
    """Sort key that orders ``f2`` before ``f10``. Plain lexical sort does not."""
    name = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def _ci_glob(directory: str, ext: str) -> list:
    """Glob one extension case-insensitively.

    ``fnmatch`` -- and therefore ``glob`` -- is case-sensitive even on a case-insensitive
    filesystem, so a plain ``*.tif`` silently skips ``FRAME2.TIF``. Skipping a frame is the
    worst thing that can happen to a time series short of reordering it: the step across the
    gap is twice the real interval, the association gates reject links they should have made,
    and tracks die and are reborn -- which reports MORE fragmentation than the data has.
    """
    pat = "".join(f"[{c.lower()}{c.upper()}]" if c.isalpha() else c for c in ext)
    return glob.glob(os.path.join(directory, f"*.{pat}"))


def order_frames(spec: str) -> tuple:
    """Resolve the input into an ordered list of frame paths. Returns ``(paths, notes)``.

    The returned order IS the time axis, so every way of getting it silently wrong is refused
    here rather than tracked:

    * **Mixed extensions are not merged.** A folder holding ``f1.tif`` and ``f1.png`` -- which
      is what running ``predict.py --overlay`` into your own frame folder produces -- would
      otherwise interleave overlays with frames as separate time points, halving the apparent
      motion, and both would write the same ``f1.json``. One extension is used and the rest are
      reported as ignored.
    * **Ties are fatal, not arbitrary.** ``f2.tif`` and ``f02.tif`` produce the same natural
      key; the tie would then fall to filesystem order, so the same folder could yield a
      different time axis on a different machine.
    * **Colliding stems are fatal**, because the per-frame output file is named after the stem
      and the second would silently overwrite the first. Only reachable through a glob spanning
      directories, since one directory plus one extension gives unique names.
    """
    notes = []
    if os.path.isdir(spec):
        found = {}
        for group in FRAME_EXT_GROUPS:
            hits = [p for ext in group for p in _ci_glob(spec, ext)]
            if hits:
                found[group] = hits
        if not found:
            raise SystemExit(f"no frames found in {spec} "
                             f"(looked for {', '.join('.' + e for e in FRAME_EXTS)})")
        group = next(g for g in FRAME_EXT_GROUPS if g in found)
        paths = found[group]
        ignored = {g: len(p) for g, p in found.items() if g != group}
        if ignored:
            notes.append("using " + "/".join("." + e for e in group) + " only; ignored " +
                         ", ".join(f"{n} {'/'.join('.' + e for e in g)}"
                                   for g, n in ignored.items()) +
                         " (mixing formats would interleave them as extra time points)")
    else:
        paths = glob.glob(spec)

    stems = {}
    for p in sorted(paths):
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem in stems:
            raise SystemExit(
                f"two frames share the stem {stem!r} ({stems[stem]} and {p}); their per-frame "
                f"output files would overwrite each other.")
        stems[stem] = p

    keyed = sorted(((natural_key(p), p) for p in paths), key=lambda kp: (kp[0], kp[1]))
    for (k1, p1), (k2, p2) in zip(keyed, keyed[1:]):
        if k1 == k2:
            raise SystemExit(
                f"cannot order these frames in time: {os.path.basename(p1)} and "
                f"{os.path.basename(p2)} sort identically, so their order would depend on the "
                f"filesystem. Make the numbering consistent (pad both, or neither).")
    return [p for _, p in keyed], notes


def resolve_gates(max_shift, min_overlap, overlap_tol) -> dict:
    """Association gates from CLI values in INPUT pixels to the tracker's 1.5x working frame.

    The tracker's gates were fitted in the 1.5x frame, so association runs there and only the
    OUTPUT is mapped back. A caller thinks in input pixels, so the two distance gates are taken
    in input pixels and scaled here; ``min_overlap`` is a unitless fraction and is deliberately
    not scaled. Leaving a flag unset passes the fitted default through untouched, so the
    default configuration is exactly the one every published tracking number was measured with.
    """
    gates = {}
    if max_shift is not None:
        gates["max_shift"] = max_shift * UP
    if overlap_tol is not None:
        gates["overlap_tol"] = overlap_tol * UP
    if min_overlap is not None:
        gates["min_overlap"] = min_overlap
    return gates


def assign_owners(tracks, per_frame) -> list:
    """Which track owns each detection, per frame. Returns one dict {index: track_id} per frame.

    ``track_sequence`` covers every detection exactly once -- frame 0 starts a track per
    detection, ``match_frames`` is injective on both sides, and the births loop picks up every
    unmatched one -- so this is recovering a known correspondence, not searching for a likely
    one. It matters that it is exact: a wrong answer here does not raise, it writes a detection
    out under another track's identity and drops a second one from the per-frame file, and the
    colour-by-track overlay then renders that as fact.

    Hence the ``claimed`` set. Matching by value alone would let an earlier value-equal
    detection win over the correct later one, so two identical polylines in a frame would both
    resolve to the first index; excluding already-claimed indices removes that. A lookup that
    still fails means the invariant above is broken, which is a bug rather than a data
    condition, so it raises instead of quietly writing inconsistent files.
    """
    owner = [{} for _ in per_frame]
    claimed = [set() for _ in per_frame]
    for tr in tracks:
        for k, poly in zip(tr.frames, tr.polylines):
            j = next((idx for idx, cand in enumerate(per_frame[k])
                      if idx not in claimed[k] and
                      (cand is poly or np.array_equal(np.asarray(cand), np.asarray(poly)))),
                     None)
            if j is None:
                raise SystemExit(
                    f"internal error: a polyline of track {tr.track_id} in frame {k} matches no "
                    f"unclaimed detection in that frame. Refusing to write per-frame output "
                    f"that would disagree with tracks.json.")
            claimed[k].add(j)
            owner[k][j] = tr.track_id
    return owner


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
                    help="hard gate, in INPUT pixels, on the mean curve-to-curve distance "
                         "between two centerlines -- NOT on displacement: a filament gliding "
                         "along its own contour stays close to where it was however far the "
                         "material travelled, and is unaffected by this. It is also the "
                         "pairing radius of the drift estimator. Default 16.7 (the fitted "
                         "25 px in the 1.5x working frame).")
    ap.add_argument("--overlap-tol", type=float, default=None,
                    help="how near, in INPUT pixels, a point must be to count as overlapping. "
                         "THIS is what caps association across TRANSVERSE motion: at the "
                         "default 2.67 (4 px in the working frame) two centerlines more than "
                         "~2.7 px apart sideways stop overlapping and cannot be linked, "
                         "whatever --max-shift says. Raise it for fast transverse motion.")
    ap.add_argument("--min-overlap", type=float, default=None,
                    help="fraction of the shorter filament that must have a partner within "
                         "--overlap-tol before two detections may be linked (default 0.35, "
                         "unitless -- a fraction in [0, 1], not pixels)")
    ap.add_argument("--min-frames", type=int, default=1,
                    help="drop tracks seen in fewer than this many frames. 1 keeps everything, "
                         "which is the honest default; 2 drops single-frame flotsam. NOTE that "
                         "a dropped track's detections also leave the per-frame files, so those "
                         "then report fewer instances than predict.py would for the same frame.")
    args = ap.parse_args()

    # Gates are validated before anything expensive: a value that can never be satisfied
    # (min_overlap in pixels, say) would otherwise associate nothing at all and still exit 0.
    if args.max_shift is not None and args.max_shift <= 0:
        raise SystemExit(f"--max-shift must be positive, got {args.max_shift}")
    if args.overlap_tol is not None and args.overlap_tol <= 0:
        raise SystemExit(f"--overlap-tol must be positive, got {args.overlap_tol}")
    if args.min_overlap is not None and not 0.0 <= args.min_overlap <= 1.0:
        raise SystemExit(f"--min-overlap is a FRACTION of a filament's length and must lie in "
                         f"[0, 1], got {args.min_overlap}. Pixel tolerances are --overlap-tol "
                         f"and --max-shift.")
    if args.min_frames < 1:
        raise SystemExit(f"--min-frames must be at least 1, got {args.min_frames}")

    # Imported before the weights are loaded, not after the prediction loop: a package
    # assembled without instance/ would otherwise segment every frame and only then fail.
    from instance.tracker import track_sequence

    paths, notes = order_frames(args.input)
    if len(paths) < 2:
        raise SystemExit(
            f"tracking needs at least 2 frames, found {len(paths)} at {args.input}.\n"
            f"For a single frame use predict.py -- one frame in, no tracks out, is the honest "
            f"answer rather than an error.")
    os.makedirs(args.out_dir, exist_ok=True)

    params = json.load(open(args.params))
    params.pop("kappa_max", None)      # derived, never read from a params file
    thr = args.prob_thr if args.prob_thr is not None else params.get("prob_thr", 0.35)
    gates = resolve_gates(args.max_shift, args.min_overlap, args.overlap_tol)

    seg = Segmenter(args.weights, args.device)
    print(f"device={seg.device} | {len(paths)} frame(s) | threshold={thr:.3f}", flush=True)
    for note in notes:
        print(f"  note: {note}", flush=True)
    print("  time axis: " + " -> ".join(os.path.basename(p) for p in paths[:4])
          + (" -> ..." if len(paths) > 4 else ""), flush=True)

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

    tracks = track_sequence(per_frame, gates or None)
    kept = [t for t in tracks if t.length >= args.min_frames]
    owner = assign_owners(kept, per_frame)

    shape = [int(raws[0].shape[0]), int(raws[0].shape[1])]
    out_tracks = [{"track_id": int(t.track_id),
                   "frames": [int(k) for k in t.frames],
                   "polylines": [(np.asarray(p, dtype=float) / UP).round(2).tolist()
                                 for p in t.polylines]}
                  for t in kept]
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

    # Report the quantity that distinguishes working from not working. A count of detections
    # inside tracks does not: a track of length 1 is linked to nothing and still contributes,
    # so total association failure would print the same number as complete success.
    links = sum(t.length - 1 for t in kept)
    print(f"{len(out_tracks)} tracks over {len(paths)} frames "
          f"({links} cross-frame links); wrote {args.out_dir}/", flush=True)
    empty = [os.path.basename(p) for p, d in zip(paths, per_frame) if not d]
    if empty:
        print(f"WARNING: no detections at all in {len(empty)} frame(s): "
              f"{', '.join(empty[:5])}{' ...' if len(empty) > 5 else ''}. Every track is cut "
              f"at such a frame and restarted after it.", flush=True)
    if links == 0:
        print("WARNING: nothing was associated across frames -- every detection is its own "
              "single-frame track. Check the frame order printed above, and whether motion "
              "between frames exceeds --overlap-tol.", flush=True)


if __name__ == "__main__":
    main()
