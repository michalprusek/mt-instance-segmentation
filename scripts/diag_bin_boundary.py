#!/usr/bin/env python3
"""D2/Q2a -- does the predicted probability DIP where the GT tangent sits on a 30-deg bin
boundary?  RUNS ON TULEN (CPU).

The inference rule is ``max over the K=6 channels > 0.35``.  If a filament's tangent lies on
a bin boundary the target itself is ambiguous, so the model can split its mass across the two
adjacent channels: the MAX falls below threshold while the SUM stays high.  That would be a
systematic, geometry-correlated hole generator -- exactly the kind of thing no connectivity
loss can repair, because the loss is being satisfied.

Measured on GT centerline samples, in the 1.5x eval frame, EXCLUDING samples near a crossing
or near an endpoint (both depress probability for unrelated reasons and would smear the curve).

    PYTHONPATH=src /home/prusek/dinov3_env/bin/python scripts/diag_bin_boundary.py \
        --data data/synth_eval --pred-dir /home/prusek/mt_enc_exp/synth_pred_v4b \
        --out data/enc_sensitivity_testset/diag_bin_boundary_synth.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from instance.geometry import arclength, resample  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.gt_stats import count_crossings  # noqa: E402

UP = 1.5
K = 6
BIN_W_DEG = 180.0 / K


def sample_tangents(pl: np.ndarray, ds: float, win: float):
    """Resample a polyline and return (points, tangent angle mod 180 deg, arclength)."""
    pts = resample(np.asarray(pl, float), ds=ds)
    if len(pts) < 2:
        return None
    s = arclength(pts)
    lo = np.clip(np.searchsorted(s, s - win / 2.0, "left"), 0, len(pts) - 1)
    hi = np.clip(np.searchsorted(s, s + win / 2.0, "right") - 1, 0, len(pts) - 1)
    bad = hi <= lo
    lo[bad] = np.maximum(0, hi[bad] - 1)
    hi[bad] = np.minimum(len(pts) - 1, lo[bad] + 1)
    d = pts[hi] - pts[lo]
    ang = np.rad2deg(np.arctan2(d[:, 1], d[:, 0])) % 180.0
    return pts, ang, s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth_eval")
    ap.add_argument("--pred-dir", default="/home/prusek/mt_enc_exp/synth_pred_v4b")
    ap.add_argument("--split", default="val")
    ap.add_argument("--ds", type=float, default=2.0, help="sample spacing, eval frame px")
    ap.add_argument("--win", type=float, default=12.0, help="tangent baseline, eval frame px")
    ap.add_argument("--excl-cross", type=float, default=15.0)
    ap.add_argument("--excl-end", type=float, default=15.0)
    ap.add_argument("--thr", type=float, default=0.35)
    ap.add_argument("--fov", action="store_true", help="apply the field-of-view mask (real data)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    D, MX, SM, T2, BINOK, ARGD = [], [], [], [], [], []
    n_frames = 0
    for path in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        fr = read_frame_h5(path)
        if args.split != "all" and str(fr["attrs"].get("split")) != args.split:
            continue
        npz = os.path.join(args.pred_dir,
                           os.path.basename(path).replace(".h5", ".npz"))
        if not os.path.exists(npz):
            print("MISSING", npz)
            continue
        prob = np.load(npz)["prob"].astype(np.float32)          # (K, H*1.5, W*1.5)
        Kc, H, W = prob.shape
        if args.fov:
            from scipy.ndimage import zoom
            from mt_bench.fov import fov_mask
            fv = zoom(fov_mask(fr["image"]).astype(np.uint8), UP, order=0).astype(bool)
            fv = np.pad(fv, ((0, max(0, H - fv.shape[0])), (0, max(0, W - fv.shape[1]))),
                        constant_values=True)[:H, :W]
            prob = prob * fv[None]
        n_frames += 1

        gt = [np.asarray(p, float) * UP for p in fr["polylines"]]
        cr = count_crossings(gt, tol=2.0 * UP, endpoint_margin=8.0 * UP)
        ctree = cKDTree(np.array([[c["x"], c["y"]] for c in cr])) if cr else None

        for pl in gt:
            out = sample_tangents(pl, args.ds, args.win)
            if out is None:
                continue
            pts, ang, s = out
            keep = (s >= args.excl_end) & (s <= s[-1] - args.excl_end)
            if ctree is not None:
                dc, _ = ctree.query(pts, k=1)
                keep &= dc > args.excl_cross
            if not keep.any():
                continue
            pts, ang = pts[keep], ang[keep]
            cc = np.clip(np.rint(pts[:, 0]).astype(int), 0, W - 1)
            rr = np.clip(np.rint(pts[:, 1]).astype(int), 0, H - 1)
            v = prob[:, rr, cc]                                   # (K, n)
            order = np.sort(v, axis=0)[::-1]
            m = ang % BIN_W_DEG
            D.append(np.minimum(m, BIN_W_DEG - m))                # deg to nearest boundary
            MX.append(order[0])
            SM.append(v.sum(axis=0))
            T2.append(order[0] + order[1])
            gtbin = np.clip((ang // BIN_W_DEG).astype(int), 0, K - 1)
            BINOK.append((np.argmax(v, axis=0) == gtbin).astype(float))
            # circular distance between predicted argmax bin and GT bin, in bins
            dbin = np.abs(np.argmax(v, axis=0) - gtbin)
            ARGD.append(np.minimum(dbin, K - dbin).astype(float))

    D = np.concatenate(D); MX = np.concatenate(MX); SM = np.concatenate(SM)
    T2 = np.concatenate(T2); BINOK = np.concatenate(BINOK); ARGD = np.concatenate(ARGD)

    edges = np.arange(0, 16.0, 1.0)
    idx = np.clip(np.digitize(D, edges) - 1, 0, len(edges) - 2)
    curve = []
    for b in range(len(edges) - 1):
        s = idx == b
        if s.sum() < 10:
            continue
        curve.append({
            "deg_lo": float(edges[b]), "deg_hi": float(edges[b + 1]), "n": int(s.sum()),
            "mean_max": float(MX[s].mean()), "median_max": float(np.median(MX[s])),
            "mean_sum": float(SM[s].mean()), "mean_top2": float(T2[s].mean()),
            "p_max_below_thr": float((MX[s] < args.thr).mean()),
            "p_sum_below_thr": float((SM[s] < args.thr).mean()),
            "argmax_bin_correct": float(BINOK[s].mean()),
            "mean_argmax_bin_dist": float(ARGD[s].mean()),
        })

    near = D <= 5.0
    far = D >= 10.0
    res = {
        "n_frames": n_frames, "n_samples": int(len(D)), "thr": args.thr,
        "ds": args.ds, "win": args.win, "pred_dir": args.pred_dir, "data": args.data,
        "frac_samples_within_5deg_of_boundary": float(near.mean()),
        "curve_deg_to_boundary": curve,
        "near_boundary_le5deg": {
            "n": int(near.sum()), "mean_max": float(MX[near].mean()),
            "p_max_below_thr": float((MX[near] < args.thr).mean()),
            "mean_sum": float(SM[near].mean()),
            "p_sum_below_thr": float((SM[near] < args.thr).mean()),
            "argmax_bin_correct": float(BINOK[near].mean()),
        },
        "far_from_boundary_ge10deg": {
            "n": int(far.sum()), "mean_max": float(MX[far].mean()),
            "p_max_below_thr": float((MX[far] < args.thr).mean()),
            "mean_sum": float(SM[far].mean()),
            "p_sum_below_thr": float((SM[far] < args.thr).mean()),
            "argmax_bin_correct": float(BINOK[far].mean()),
        },
        "overall": {
            "mean_max": float(MX.mean()), "p_max_below_thr": float((MX < args.thr).mean()),
            "mean_sum": float(SM.mean()), "p_sum_below_thr": float((SM < args.thr).mean()),
            "argmax_bin_correct": float(BINOK.mean()),
            "mean_argmax_bin_dist": float(ARGD.mean()),
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "curve_deg_to_boundary"}, indent=2))
    print("\n deg  n      max    sum   top2  P(max<thr) P(sum<thr) argmax_ok")
    for c in curve:
        print(f"{c['deg_lo']:4.0f} {c['n']:6d}  {c['mean_max']:.3f}  {c['mean_sum']:.3f}  "
              f"{c['mean_top2']:.3f}   {c['p_max_below_thr']:.4f}    "
              f"{c['p_sum_below_thr']:.4f}   {c['argmax_bin_correct']:.3f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
