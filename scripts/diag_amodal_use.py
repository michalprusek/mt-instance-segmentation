#!/usr/bin/env python3
"""D2/Q2 -- does the K=6 orientation head actually DELIVER the amodality it exists for?

The whole justification for splitting the foreground into six 30-degree orientation channels
is that a crossing writes into two different channels, so the two filaments remain separable
at the pixel they share.  That is true of the TARGET by construction.  This measures whether
it is true of the PREDICTION.

At every GT-GT crossing whose two filaments fall in DIFFERENT bins (angle >= 30 deg), the
crossing pixel is sampled and three things are recorded:

  * ``both_fire``   -- both GT bins exceed the threshold (the amodal signal is present);
  * ``one_fires``   -- only one does (the crossing has collapsed to a single orientation);
  * ``none_fires``  -- the crossing is not foreground at all.

Shallow crossings (< 30 deg) are reported separately: their two filaments share a bin, so the
representation CANNOT separate them and ``both_fire`` is undefined -- that is the by-construction
degeneracy the K=6 design accepts.

    PYTHONPATH=src /home/prusek/dinov3_env/bin/python scripts/diag_amodal_use.py \
        --data data/synth_eval --pred-dir /home/prusek/mt_enc_exp/synth_pred_v4b \
        --out data/enc_sensitivity_testset/diag_amodal_synth.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import zoom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from instance.geometry import arclength, resample  # noqa: E402
from instance.oracle import oracle_ori_channels  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.fov import fov_mask  # noqa: E402
from mt_bench.gt_stats import count_crossings  # noqa: E402

UP = 1.5
K = 6
BIN_W_DEG = 180.0 / K


def local_bin(pts: np.ndarray, s: float, win: float = 12.0) -> int:
    arc = arclength(pts)
    lo = int(np.searchsorted(arc, max(0.0, s - win / 2.0)))
    hi = int(min(len(pts) - 1, np.searchsorted(arc, s + win / 2.0)))
    if hi <= lo:
        lo, hi = max(0, hi - 1), min(len(pts) - 1, lo + 1)
    d = pts[hi] - pts[lo]
    ang = np.rad2deg(np.arctan2(d[1], d[0])) % 180.0
    return int(min(int(ang // BIN_W_DEG), K - 1)), float(ang)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth_eval")
    ap.add_argument("--pred-dir", default="/home/prusek/mt_enc_exp/synth_pred_v4b")
    ap.add_argument("--split", default="val")
    ap.add_argument("--thr", type=float, default=0.35)
    ap.add_argument("--fov", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        fr = read_frame_h5(path)
        if args.split != "all" and str(fr["attrs"].get("split")) != args.split:
            continue
        shape = (int(fr["attrs"]["height"]), int(fr["attrs"]["width"]))
        hi = (int(round(shape[0] * UP)), int(round(shape[1] * UP)))
        gt = [np.asarray(p, float) * UP for p in fr["polylines"]]
        if len(gt) < 1:
            continue
        cr = count_crossings(gt, tol=2.0 * UP, endpoint_margin=8.0 * UP)
        if not cr:
            continue
        prob = np.load(os.path.join(args.pred_dir,
                                    os.path.basename(path).replace(".h5", ".npz"))
                       )["prob"].astype(np.float32)
        if args.fov:
            fv = zoom(fov_mask(fr["image"]).astype(np.uint8), UP, order=0).astype(bool)
            fv = np.pad(fv, ((0, max(0, hi[0] - fv.shape[0])),
                             (0, max(0, hi[1] - fv.shape[1]))),
                        constant_values=True)[:hi[0], :hi[1]]
            prob = prob * fv[None]
        orc = oracle_ori_channels(fr["polylines"], shape, K=K, half_width=1.0, up=UP)
        res = [resample(p, ds=1.0) for p in gt]

        for c in cr:
            bi, _ = local_bin(res[c["i"]], c["s_i"])
            bj, _ = local_bin(res[c["j"]], c["s_j"])
            rr = int(np.clip(round(c["y"]), 0, hi[0] - 1))
            cc = int(np.clip(round(c["x"]), 0, hi[1] - 1))
            # 3x3 max around the crossing: 1 px of localisation slack, same as the mask band
            r0, r1, c0, c1 = max(0, rr - 1), rr + 2, max(0, cc - 1), cc + 2
            v = prob[:, r0:r1, c0:c1].reshape(K, -1).max(axis=1)
            o = orc[:, r0:r1, c0:c1].reshape(K, -1).max(axis=1)
            rows.append({"angle": c["angle_deg"], "same_bin": bool(bi == bj),
                         "bi": bi, "bj": bj,
                         "p_i": float(v[bi]), "p_j": float(v[bj]),
                         "p_max": float(v.max()), "p_sum": float(v.sum()),
                         "n_fire": int((v > args.thr).sum()),
                         "o_both": bool(o[bi] > 0.5 and o[bj] > 0.5)})

    def block(sel, tag):
        s = [r for r in rows if sel(r)]
        if not s:
            return {"n": 0}
        pi = np.array([r["p_i"] for r in s]); pj = np.array([r["p_j"] for r in s])
        fg = np.array([r["p_max"] for r in s]) > args.thr
        both = (pi > args.thr) & (pj > args.thr)
        one = ((pi > args.thr) ^ (pj > args.thr))
        return {"n": len(s),
                "frac_foreground_at_crossing": float(fg.mean()),
                "frac_both_gt_bins_fire": float(both.mean()),
                "frac_exactly_one_gt_bin_fires": float(one.mean()),
                "frac_neither_gt_bin_fires": float((~both & ~one).mean()),
                "mean_p_weaker_gt_bin": float(np.minimum(pi, pj).mean()),
                "mean_p_stronger_gt_bin": float(np.maximum(pi, pj).mean()),
                "mean_p_max_any_bin": float(np.mean([r["p_max"] for r in s])),
                "mean_n_bins_above_thr": float(np.mean([r["n_fire"] for r in s])),
                "oracle_frac_both_bins_set": float(np.mean([r["o_both"] for r in s])),
                "tag": tag}

    out = {
        "config": {"data": args.data, "pred_dir": args.pred_dir, "split": args.split,
                   "thr": args.thr},
        "n_crossings": len(rows),
        "frac_same_bin_by_construction": float(np.mean([r["same_bin"] for r in rows])),
        "different_bins_all": block(lambda r: not r["same_bin"], "diff-bin"),
        "different_bins_oblique_30_60": block(
            lambda r: (not r["same_bin"]) and 30 <= r["angle"] < 60, "30-60"),
        "different_bins_steep_ge60": block(
            lambda r: (not r["same_bin"]) and r["angle"] >= 60, "60-90"),
        "same_bin_shallow": block(lambda r: r["same_bin"], "same-bin"),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({**out, "_rows": rows}, fh, indent=2)
    print(json.dumps(out, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
