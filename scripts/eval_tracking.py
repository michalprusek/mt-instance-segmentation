#!/usr/bin/env python3
"""Score the tracker on synthetic sequences: oracle input vs the real predicted foreground.

The tracker's zero-identity-switch result was obtained on ORACLE detections -- the ground-truth
centerlines handed straight to the matcher. That measures association in isolation and says
nothing about operation, because the real input is fragmented: the instancer produces 1.145
pieces per microtubule on predicted masks. A fragment is a worse partner than a whole filament,
and fragments are what the tracker will actually see.

This runs both and reports them side by side, so the cost of fragmentation to tracking is a
number rather than an expectation.

Metrics
-------
* **identity switches** -- how often one track changes which ground-truth object it follows;
* **track completeness** -- fraction of a GT object's frames covered by its single best track;
* **tracks per object** -- >1 means the object was followed by several disconnected tracks,
  which is fragmentation surviving into time and the size of the fusion opportunity;
* **detected fraction** -- what share of GT objects were followed by ANY track. It is reported
  because the three metrics above average over every object and score an object nobody detected
  as zero tracks and zero completeness, so a foreground that simply misses more looks LESS
  fragmented. ``tracks_per_detected_object`` and ``completeness_detected`` are the same
  quantities over the objects that were actually followed, and the pair is only interpretable
  together with this rate;
* **velocity error** -- computed and kept in the JSON, but deliberately absent from the printed
  table: protocol 18a established it is not measurable from this foreground, and a number in a
  table gets quoted.

    PYTHONPATH=src:synth ~/dinov3_env/bin/python scripts/eval_tracking.py \
        --data data/synth_seq --split test --masks oracle
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import zoom
from scipy.spatial import cKDTree

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from instance.instancer_a import instance_a  # noqa: E402
from instance.oracle import oracle_ori_channels  # noqa: E402
from instance.tracker import track_sequence, track_velocity  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402

UP = 1.5
KAPPA_MAX = 0.25


class MissingPrediction(Exception):
    """A frame has no cached prediction. A domain error, not ``SystemExit``: ``score`` is
    imported and called directly by the tests and by sweep drivers, and ``SystemExit`` derives
    from ``BaseException``, so it escapes ``except Exception`` and kills the caller."""


def load_sequences(data_dir: str, split: str):
    """Group frames by sequence, in frame order."""
    seqs = collections.defaultdict(list)
    for path in sorted(glob.glob(os.path.join(data_dir, "*.h5"))):
        fr = read_frame_h5(path)
        a = fr["attrs"]
        if str(a.get("split")) != split:
            continue
        seqs[int(a["seq_id"])].append((int(a["frame_idx"]), path, fr))
    for s in seqs:
        seqs[s].sort(key=lambda t: t[0])
    return seqs


def detections(fr, masks: str, pred_dir: str, path: str, params: dict, prob_thr: float):
    """Per-frame instance polylines, in the 1.5x frame."""
    a = fr["attrs"]
    shape = (int(a["height"]), int(a["width"]))
    if masks == "oracle":
        chans = oracle_ori_channels(fr["polylines"], shape, K=6, half_width=1.0, up=UP)
        thr = 0.5
    else:
        npz = os.path.join(pred_dir, os.path.basename(path).replace(".h5", ".npz"))
        if not os.path.exists(npz):
            # Returning None here used to drop the whole sequence from the MODEL arm while the
            # oracle arm -- which never reads pred_dir -- kept it, so the two rows printed side
            # by side as a comparison were scored on different populations, with nothing in the
            # output saying so. Which sequences go missing is not random either: an interrupted
            # prediction job loses whichever it had not reached.
            raise MissingPrediction(
                f"missing prediction for {os.path.basename(path)}: {npz}. "
                f"Refusing to score a partial model arm against a complete oracle arm.")
        chans = np.load(npz)["prob"].astype(np.float32)
        thr = effective_threshold(params, prob_thr)
    prob = chans.max(axis=0)
    pls, _ = instance_a(prob > thr, KAPPA_MAX, params, channels=chans, prob=prob)
    return pls


def effective_threshold(params: dict, prob_thr):
    """The foreground threshold the model arm will actually use.

    CLI wins over the params file, matching predict.py and track.py. The old order let the
    params file's prob_thr shadow --prob-thr entirely, so a threshold sweep would have silently
    held the threshold constant and produced a perfectly flat curve. It lives in one function
    so that the value printed at startup cannot drift away from the value used.
    """
    return prob_thr if prob_thr is not None else params.get("prob_thr", 0.35)


def missing_predictions(seqs, pred_dir: str) -> list:
    """Frames in ``seqs`` with no cached ``.npz``, checked BEFORE any scoring starts.

    Discovering this inside the scoring loop means a 300-frame run burns the whole oracle arm
    -- printing its complete-looking row -- before refusing on the model arm.
    """
    out = []
    for frames in seqs.values():
        for _, path, _ in frames:
            npz = os.path.join(pred_dir, os.path.basename(path).replace(".h5", ".npz"))
            if not os.path.exists(npz):
                out.append(os.path.basename(path))
    return out


def assign_to_gt(pred, gt, gt_ids, tol=6.0):
    """Which GT object does each predicted polyline follow? ``None`` when it follows none."""
    if not gt:
        return [None] * len(pred)
    trees = [cKDTree(np.asarray(g, dtype=float)) for g in gt]
    out = []
    for p in pred:
        p = np.asarray(p, dtype=float)
        best, best_cov = None, 0.0
        for t, gid in zip(trees, gt_ids):
            cov = float((t.query(p, k=1)[0] <= tol).mean())
            if cov > best_cov:
                best, best_cov = gid, cov
        out.append(best if best_cov >= 0.5 else None)
    return out


def score(seqs, masks, pred_dir, params, prob_thr) -> dict:
    switches = links = 0
    completeness, per_obj_tracks, vel_err, vel_ratio = [], [], [], []
    det_obj_tracks, det_completeness = [], []
    n_frames_total = n_det = 0

    for sid, frames in sorted(seqs.items()):
        gt_per_frame, ids_per_frame, det_per_frame = [], [], []
        speeds = {}
        for k, path, fr in frames:
            a = fr["attrs"]
            gt = [np.asarray(p, float) * UP for p in fr["polylines"]]
            ids = [int(x) for x in np.atleast_1d(a["inst_ids"])]
            # UNITS: the stored speed is px/frame at NATIVE scale, while every polyline here --
            # ground truth and detection alike -- lives in the 1.5x eval frame. Scaling it is
            # not cosmetic: without it the velocity error is a 1.5x unit mismatch reported as a
            # model defect, which is exactly how the first "1-2 px/frame underestimate" arose.
            for i, sp in zip(ids, np.atleast_1d(a.get("speeds", []))):
                speeds[i] = float(sp) * UP
            det = detections(fr, masks, pred_dir, path, params, prob_thr)
            gt_per_frame.append(gt)
            ids_per_frame.append(ids)
            det_per_frame.append(det)
            n_frames_total += 1
            n_det += len(det)
        if not det_per_frame:
            continue

        tracks = track_sequence(det_per_frame)
        # Which GT object does each detection follow, per frame?
        owner = [assign_to_gt(d, g, i) for d, g, i in
                 zip(det_per_frame, gt_per_frame, ids_per_frame)]

        obj_tracks = collections.defaultdict(set)
        obj_frames = collections.defaultdict(set)
        for k, ids in enumerate(ids_per_frame):
            for i in ids:
                obj_frames[i].add(k)

        # Which detection is this track's polyline? Value-equal detections would otherwise
        # both resolve to the FIRST matching index, so one GT object's id would be counted
        # twice and another never -- corrupting the identity-switch count silently. Excluding
        # indices another track already claimed removes that; track_sequence covers every
        # detection exactly once, so the claim is unique by construction.
        claimed = [set() for _ in det_per_frame]
        for tr in tracks:
            seq_ids = []
            for k, poly in zip(tr.frames, tr.polylines):
                j = next((idx for idx, p in enumerate(det_per_frame[k])
                          if idx not in claimed[k] and
                          (p is poly or np.array_equal(np.asarray(p), np.asarray(poly)))), None)
                if j is None:
                    # Unlike ``owner[k][j] is None`` -- which merely says this detection follows
                    # no GT object and is ordinary -- a failed lookup means the correspondence
                    # is broken. Swallowing it shortens ``named`` and so undercounts links AND
                    # switches, and switch_rate = switches/max(links,1) would read 0.000, a
                    # perfect score, from total failure. The deployment tool refuses this same
                    # condition; the harness that produces the published number must not be
                    # more permissive than the tool.
                    raise RuntimeError(
                        f"track {tr.track_id} has a polyline in frame {k} matching no unclaimed "
                        f"detection; the identity-switch count would be silently wrong.")
                claimed[k].add(j)
                seq_ids.append(owner[k][j])
            named = [x for x in seq_ids if x is not None]
            switches += sum(1 for x, y in zip(named, named[1:]) if x != y)
            links += max(len(named) - 1, 0)
            for x in set(named):
                obj_tracks[x].add(tr.track_id)
            if named and len(set(named)) == 1 and tr.length >= 3:
                v = track_velocity(tr)
                gt_v = speeds.get(named[0])
                if np.isfinite(v) and gt_v is not None and abs(gt_v) > 1e-6:
                    vel_err.append(abs(v) - abs(gt_v))
                    vel_ratio.append(abs(v) / abs(gt_v))

        for obj, ks in obj_frames.items():
            tids = obj_tracks.get(obj, set())
            per_obj_tracks.append(len(tids))
            if not tids:
                completeness.append(0.0)
                continue
            best = 0
            for tid in tids:
                tr = tracks[tid]
                best = max(best, sum(1 for k in tr.frames if k in ks))
            completeness.append(best / max(len(ks), 1))
            # An object nobody detected contributes ZERO tracks and ZERO completeness, so a
            # foreground that simply misses more microtubules scores as LESS fragmented. That
            # is an error in the measured model's favour, and the two populations have to be
            # separable: these lists hold only the objects that were detected at all.
            det_obj_tracks.append(len(tids))
            det_completeness.append(best / max(len(ks), 1))

    return {
        "n_frames": n_frames_total,
        "det_per_frame": n_det / max(n_frames_total, 1),
        "id_switches": switches,
        "links": links,
        "switch_rate": switches / max(links, 1),
        "completeness": float(np.mean(completeness)) if completeness else float("nan"),
        "tracks_per_object": float(np.mean(per_obj_tracks)) if per_obj_tracks else float("nan"),
        # The same quantities over DETECTED objects only, plus the detection rate that
        # separates them. tracks_per_object mixes "followed by three fragments" with
        # "never seen at all", and the second is what a missing foreground looks like --
        # scoring, wrongly, as excellent fragmentation.
        "n_objects": len(per_obj_tracks),
        "detected_frac": (float(len(det_obj_tracks) / len(per_obj_tracks))
                          if per_obj_tracks else float("nan")),
        "tracks_per_detected_object": (float(np.mean(det_obj_tracks))
                                       if det_obj_tracks else float("nan")),
        "completeness_detected": (float(np.mean(det_completeness))
                                  if det_completeness else float("nan")),
        "velocity_err_median": float(np.median(vel_err)) if vel_err else float("nan"),
        # The ratio is the honest headline: an additive error in px/frame says nothing without
        # the speed it is measured against.
        "velocity_ratio_median": float(np.median(vel_ratio)) if vel_ratio else float("nan"),
        "velocity_n": len(vel_err),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth_seq")
    ap.add_argument("--split", default="test")
    ap.add_argument("--masks", default="oracle", choices=["oracle", "model", "both"])
    ap.add_argument("--pred-dir", default="/home/prusek/mt_enc_exp/synth_seq_pred")
    ap.add_argument("--params", default="src/instance/params_a_model_synthtuned.json")
    ap.add_argument("--prob-thr", type=float, default=None,
                    help="foreground threshold for the model arm; overrides the params file. "
                         "Default: the params file's prob_thr, or 0.35 if it has none.")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    params = json.load(open(args.params))
    params.pop("kappa_max", None)
    seqs = load_sequences(args.data, args.split)
    if not seqs:
        raise SystemExit(f"no {args.split} sequences under {args.data}")
    which = ["oracle", "model"] if args.masks == "both" else [args.masks]
    if "model" in which:
        gone = missing_predictions(seqs, args.pred_dir)
        if gone:
            raise SystemExit(
                f"{len(gone)} of {sum(len(v) for v in seqs.values())} frames have no prediction "
                f"in {args.pred_dir} (e.g. {', '.join(gone[:3])}). Refusing to score a partial "
                f"model arm against a complete oracle arm.")
    thr_eff = effective_threshold(params, args.prob_thr)
    print(f"{len(seqs)} sequences, {sum(len(v) for v in seqs.values())} frames"
          + (f", model threshold {thr_eff:.3f}" if args.masks in ("model", "both") else "")
          + "\n")

    report = {}
    hdr = f"{'input':8s} {'det/frame':>10s} {'switches':>10s} {'switch rate':>12s} " \
          f"{'complete':>9s} {'trk/obj':>8s} | {'found':>6s} {'compl|det':>10s} " \
          f"{'trk/det obj':>12s}"
    print(hdr)
    for m in which:
        r = score(seqs, m, args.pred_dir, params, args.prob_thr)
        report[m] = r
        print(f"{m:8s} {r['det_per_frame']:10.1f} {r['id_switches']:4d}/{r['links']:<5d} "
              f"{r['switch_rate']:12.3f} {r['completeness']:9.3f} "
              f"{r['tracks_per_object']:8.2f} | {r['detected_frac']:6.3f} "
              f"{r['completeness_detected']:10.3f} "
              f"{r['tracks_per_detected_object']:12.2f}")
        # Velocity is deliberately absent from this table: it is not measurable from this
        # foreground (protocol 18a) and a printed number invites quotation. It stays in JSON.

    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump(report, fh, indent=2, default=float)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
