#!/usr/bin/env python3
"""D2/Q2 -- is the K=6 hard-bin orientation TARGET itself broken?  RUNS ON TULEN (CPU).

Everything here is a property of the TRAINING TARGET, not of the model: it reproduces
``scripts/gen_train.ori_channels`` verbatim (hard 30-degree bin from a per-vertex
``np.gradient`` tangent, ``int(round())`` point stamp, ``binary_dilation(iterations=1)``)
and measures what the network was actually asked to predict.

Four questions:
  (c) POINT-RASTERISATION GAPS -- generator vertex spacing, and connected components of the
      target (union over channels AND per channel, per microtubule).
  (a) BIN-BOUNDARY SHATTER -- fraction of GT centerline length whose tangent lies within
      +-5 deg of a 30-degree bin boundary.
  (d) CURVATURE / BIN CHANGES -- how often a single microtubule changes orientation bin, and
      the run-length distribution of contiguous same-bin stretches.
  control -- the same statistics for the ORACLE rasterisation (sub-pixel resample ds=0.4,
      segment tangents), i.e. "what the target should have looked like".

    PYTHONPATH=src:synth /home/prusek/dinov3_env/bin/python scripts/diag_target_repr.py \
        --out data/enc_sensitivity_testset/diag_target_repr.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import binary_dilation, label

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from instance.geometry import arclength, resample, segment_angles  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402

K = 6
BIN_W = np.pi / K                      # 30 deg
CONN8 = np.ones((3, 3), bool)


# ---------------------------------------------------------------------------------------
# gen_train.ori_channels, reproduced EXACTLY (one microtubule at a time so components can be
# attributed).  cl is (x=col, y=row); d[:,1]/d[:,0] = dy/dx.
# ---------------------------------------------------------------------------------------
def train_bins(cl: np.ndarray) -> np.ndarray:
    d = np.gradient(cl, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0]) % np.pi
    return (ang / BIN_W).astype(int) % K, ang


def train_channels_one(cl: np.ndarray, shape, dilate: int = 1) -> np.ndarray:
    ch = np.zeros((K,) + shape, bool)
    b, _ = train_bins(cl)
    for (x, y), bb in zip(cl, b):
        ix, iy = int(round(x)), int(round(y))
        if 0 <= iy < shape[0] and 0 <= ix < shape[1]:
            ch[bb, iy, ix] = True
    return np.stack([binary_dilation(ch[k], iterations=dilate) for k in range(K)])


def oracle_channels_one(cl: np.ndarray, shape, ds: float = 0.4) -> np.ndarray:
    """instance/oracle.oracle_ori_channels for ONE polyline at up=1, half_width=1 (3px band)."""
    ch = np.zeros((K,) + shape, bool)
    pts = resample(np.asarray(cl, float), ds=ds)
    if len(pts) < 2:
        return ch
    ang = np.rad2deg(segment_angles(pts)) % 180.0
    ang = np.concatenate([ang, ang[-1:]])
    bins = np.clip((ang // 30.0).astype(int), 0, K - 1)
    cc = np.clip(np.rint(pts[:, 0]).astype(int), 0, shape[1] - 1)
    rr = np.clip(np.rint(pts[:, 1]).astype(int), 0, shape[0] - 1)
    for b in range(K):
        s = bins == b
        if s.any():
            ch[b, rr[s], cc[s]] = True
    fp = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)   # _footprint(1.0)
    return np.stack([binary_dilation(ch[b], structure=fp) for b in range(K)])


def n_components(m: np.ndarray) -> int:
    return int(label(m, structure=CONN8)[1]) if m.any() else 0


def runs(vals: np.ndarray) -> list[int]:
    """Lengths (in vertices) of maximal constant runs."""
    if len(vals) == 0:
        return []
    edges = np.where(np.diff(vals) != 0)[0] + 1
    return list(np.diff(np.concatenate([[0], edges, [len(vals)]])))


def smooth_tangent_angle(cl: np.ndarray, win: float = 8.0) -> np.ndarray:
    """Tangent at every vertex over a +-win/2 px baseline (gt_stats' lesson: 1px is noise)."""
    s = arclength(cl)
    lo = np.searchsorted(s, s - win / 2.0, side="left")
    hi = np.searchsorted(s, s + win / 2.0, side="right") - 1
    lo = np.clip(lo, 0, len(cl) - 1)
    hi = np.clip(hi, 0, len(cl) - 1)
    same = hi <= lo
    lo[same] = np.maximum(0, hi[same] - 1)
    hi[same] = np.minimum(len(cl) - 1, lo[same] + 1)
    d = cl[hi] - cl[lo]
    return np.arctan2(d[:, 1], d[:, 0]) % np.pi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth_eval")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="data/enc_sensitivity_testset/diag_target_repr.json")
    ap.add_argument("--win", type=float, default=8.0)
    args = ap.parse_args()

    spacing, vlen = [], []
    per_mt = []          # one row per microtubule
    boundary_len = {"raw": [0.0, 0.0], "smooth": [0.0, 0.0]}   # [near-boundary, total]
    bin_hist = np.zeros(K)
    n_frames = 0

    for path in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        fr = read_frame_h5(path)
        if args.split != "all" and str(fr["attrs"].get("split")) != args.split:
            continue
        n_frames += 1
        shape = (int(fr["attrs"]["height"]), int(fr["attrs"]["width"]))
        for cl in fr["polylines"]:
            cl = np.asarray(cl, float)
            if len(cl) < 3:
                continue
            seg = np.linalg.norm(np.diff(cl, axis=0), axis=1)
            spacing.append(seg)
            vlen.append(len(cl))

            b_raw, ang_raw = train_bins(cl)
            ang_sm = smooth_tangent_angle(cl, win=args.win)
            b_sm = (ang_sm / BIN_W).astype(int) % K
            bin_hist += np.bincount(b_raw, minlength=K)

            # per-vertex arc weight (half of each adjacent segment)
            w = np.zeros(len(cl))
            w[:-1] += seg / 2.0
            w[1:] += seg / 2.0
            for tag, a in (("raw", ang_raw), ("smooth", ang_sm)):
                dist_deg = np.rad2deg(np.minimum(a % BIN_W, BIN_W - (a % BIN_W)))
                boundary_len[tag][0] += float(w[dist_deg <= 5.0].sum())
                boundary_len[tag][1] += float(w.sum())

            tr = train_channels_one(cl, shape)
            orc = oracle_channels_one(cl, shape)
            r_raw, r_sm = runs(b_raw), runs(b_sm)
            per_mt.append({
                "n_vert": len(cl),
                "length": float(arclength(cl)[-1]),
                "train_union_cc": n_components(tr.max(axis=0)),
                "train_perch_cc": int(sum(n_components(tr[k]) for k in range(K))),
                "train_active_ch": int(sum(tr[k].any() for k in range(K))),
                "oracle_union_cc": n_components(orc.max(axis=0)),
                "oracle_perch_cc": int(sum(n_components(orc[k]) for k in range(K))),
                "oracle_active_ch": int(sum(orc[k].any() for k in range(K))),
                "n_bin_changes_raw": len(r_raw) - 1,
                "n_bin_changes_smooth": len(r_sm) - 1,
                "med_run_raw": float(np.median(r_raw)),
                "med_run_smooth": float(np.median(r_sm)),
                "n_distinct_bins_raw": int(len(set(b_raw.tolist()))),
                "n_distinct_bins_smooth": int(len(set(b_sm.tolist()))),
            })

    s = np.concatenate(spacing)
    rows = per_mt

    def col(k):
        return np.array([r[k] for r in rows], float)

    res = {
        "n_frames": n_frames, "n_mt": len(rows), "split": args.split,
        "vertex_spacing": {
            "median": float(np.median(s)), "mean": float(s.mean()),
            "p1": float(np.percentile(s, 1)), "p99": float(np.percentile(s, 99)),
            "max": float(s.max()),
            "frac_gt_1.0": float((s > 1.0).mean()),
            "frac_gt_1.4": float((s > 1.4).mean()),
            "frac_gt_2.0": float((s > 2.0).mean()),
            "n_segments": int(len(s)),
        },
        "boundary_fraction_of_length": {
            "raw_gradient_tangent_pm5deg": boundary_len["raw"][0] / boundary_len["raw"][1],
            "smooth8px_tangent_pm5deg": boundary_len["smooth"][0] / boundary_len["smooth"][1],
            "total_length_px": boundary_len["raw"][1],
        },
        "bin_occupancy_frac": (bin_hist / bin_hist.sum()).tolist(),
    }
    for k in ("train_union_cc", "train_perch_cc", "train_active_ch",
              "oracle_union_cc", "oracle_perch_cc", "oracle_active_ch",
              "n_bin_changes_raw", "n_bin_changes_smooth",
              "med_run_raw", "med_run_smooth",
              "n_distinct_bins_raw", "n_distinct_bins_smooth"):
        v = col(k)
        res[k] = {"mean": float(v.mean()), "median": float(np.median(v)),
                  "p90": float(np.percentile(v, 90)), "max": float(v.max()),
                  "total": float(v.sum())}
    res["frac_mt_train_union_shattered"] = float((col("train_union_cc") > 1).mean())
    res["frac_mt_oracle_union_shattered"] = float((col("oracle_union_cc") > 1).mean())
    L = col("length")
    res["per_100px_train_perch_cc"] = float(100.0 * col("train_perch_cc").sum() / L.sum())
    res["per_100px_oracle_perch_cc"] = float(100.0 * col("oracle_perch_cc").sum() / L.sum())
    res["per_100px_bin_changes_raw"] = float(100.0 * col("n_bin_changes_raw").sum() / L.sum())
    res["per_100px_bin_changes_smooth"] = float(
        100.0 * col("n_bin_changes_smooth").sum() / L.sum())
    res["_per_mt"] = rows

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    for k, v in res.items():
        if not k.startswith("_"):
            print(k, json.dumps(v) if isinstance(v, (dict, list)) else v)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
