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
* **velocity error** -- against the sampled gliding speed, drift removed.

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
            return None
        chans = np.load(npz)["prob"].astype(np.float32)
        thr = params.get("prob_thr", prob_thr)
    prob = chans.max(axis=0)
    pls, _ = instance_a(prob > thr, KAPPA_MAX, params, channels=chans, prob=prob)
    return pls


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
    completeness, per_obj_tracks, vel_err = [], [], []
    n_frames_total = n_det = 0

    for sid, frames in sorted(seqs.items()):
        gt_per_frame, ids_per_frame, det_per_frame = [], [], []
        speeds = {}
        for k, path, fr in frames:
            a = fr["attrs"]
            gt = [np.asarray(p, float) * UP for p in fr["polylines"]]
            ids = [int(x) for x in np.atleast_1d(a["inst_ids"])]
            for i, sp in zip(ids, np.atleast_1d(a.get("speeds", []))):
                speeds[i] = float(sp)
            det = detections(fr, masks, pred_dir, path, params, prob_thr)
            if det is None:
                det_per_frame = []
                break
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

        for tr in tracks:
            seq_ids = []
            for k, poly in zip(tr.frames, tr.polylines):
                j = next((idx for idx, p in enumerate(det_per_frame[k])
                          if p is poly or np.array_equal(np.asarray(p), np.asarray(poly))), None)
                seq_ids.append(owner[k][j] if j is not None else None)
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

    return {
        "n_frames": n_frames_total,
        "det_per_frame": n_det / max(n_frames_total, 1),
        "id_switches": switches,
        "links": links,
        "switch_rate": switches / max(links, 1),
        "completeness": float(np.mean(completeness)) if completeness else float("nan"),
        "tracks_per_object": float(np.mean(per_obj_tracks)) if per_obj_tracks else float("nan"),
        "velocity_err_median": float(np.median(vel_err)) if vel_err else float("nan"),
        "velocity_n": len(vel_err),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth_seq")
    ap.add_argument("--split", default="test")
    ap.add_argument("--masks", default="oracle", choices=["oracle", "model", "both"])
    ap.add_argument("--pred-dir", default="/home/prusek/mt_enc_exp/synth_seq_pred")
    ap.add_argument("--params", default="src/instance/params_a_model_synthtuned.json")
    ap.add_argument("--prob-thr", type=float, default=0.35)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    params = json.load(open(args.params))
    params.pop("kappa_max", None)
    seqs = load_sequences(args.data, args.split)
    if not seqs:
        raise SystemExit(f"no {args.split} sequences under {args.data}")
    print(f"{len(seqs)} sequences, {sum(len(v) for v in seqs.values())} frames\n")

    which = ["oracle", "model"] if args.masks == "both" else [args.masks]
    report = {}
    hdr = f"{'input':8s} {'det/frame':>10s} {'switches':>10s} {'switch rate':>12s} " \
          f"{'completeness':>13s} {'tracks/obj':>11s} {'vel err':>9s}"
    print(hdr)
    for m in which:
        r = score(seqs, m, args.pred_dir, params, args.prob_thr)
        report[m] = r
        print(f"{m:8s} {r['det_per_frame']:10.1f} {r['id_switches']:4d}/{r['links']:<5d} "
              f"{r['switch_rate']:12.3f} {r['completeness']:13.3f} "
              f"{r['tracks_per_object']:11.2f} "
              f"{r['velocity_err_median']:+9.2f}")

    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump(report, fh, indent=2, default=float)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
