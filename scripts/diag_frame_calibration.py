#!/usr/bin/env python3
"""WHICH frame property predicts the model's confidence -- and its best decode threshold?

Measured 2026-09-04 on roi303 VAL: the per-frame optimal threshold of SPARSE35 spans 0.3 to
0.9995 and a per-frame oracle threshold would add +0.048 mean-F1 (frames with >= 10 GT). That is
a frame-level calibration variable. Before choosing a training fix (photometric augmentation vs
scale augmentation vs generator density) the covariate has to be named, so this script measures,
per frame, without any model in the loop:

  density   -- GT curves and GT length per megapixel
  polarity  -- sign of the image contrast under the GT (IRM: dark or bright filaments)
  contrast  -- |contrast| under the GT in units of the frame's high-pass noise (a SNR)
  width     -- FWHM of the perpendicular intensity profile at native scale (pixel-size proxy)
  stretch   -- the [1, 99] percentile span of the stored image, the fraction of pixels at the
               8-bit rails (0 / 255) and the number of distinct grey levels (quantisation)
  size      -- frame height / width

and, from the cached probability map (eval frame), the ridge-tolerant probability under the GT
centerline (p10 / p25 / p50, fraction > 0.98 / 0.9 / 0.5). The per-frame best threshold and the
F1 at the global threshold come from an existing `eval_v5.py --out-json` sweep, so nothing is
re-decoded here. Spearman correlations of (p25 under GT, best threshold, oracle-vs-global F1
gain) against every covariate are printed and written with the per-frame table.

Conventions: polylines are (x=col, y=row) native; prob is [C, H*1.5, W*1.5].

    PYTHONPATH=src:scripts python scripts/diag_frame_calibration.py \
        --data data/real/roi303_eval --pred-dir preds/SPARSE35_roi_native \
        --sweep runs/scale/thr_SPARSE35_native_wide.json --out runs/diag_roi/frame_calib.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, median_filter, zoom
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from instance.geometry import resample  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402

UP = 1.5


def _profiles(img, pts, nrm, half=8.0, step=0.5):
    offs = np.arange(-half, half + 1e-6, step)
    P = np.stack([map_coordinates(img, [(pts + d * nrm)[:, 1], (pts + d * nrm)[:, 0]],
                                  order=1, mode="nearest") for d in offs], 1)   # [N, K]
    return offs, P


def frame_width_and_contrast(img, polys, ds=6.0):
    """Median FWHM (px, native), median signed contrast, polarity share, SNR under the GT."""
    hp = img - median_filter(img, size=9)
    noise = 1.4826 * np.median(np.abs(hp - np.median(hp))) + 1e-6
    fw, con = [], []
    for q in polys:
        pts = resample(np.asarray(q, float), ds=ds)
        if len(pts) < 3:
            continue
        t = np.gradient(pts, axis=0)
        n = np.stack([-t[:, 1], t[:, 0]], 1)
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-8
        offs, P = _profiles(img, pts, n)
        outer = (np.abs(offs) >= 5.0)
        bg = np.median(P[:, outer], axis=1, keepdims=True)
        R = P - bg
        core = np.abs(offs) <= 2.5
        sgn = np.sign(R[:, core][np.arange(len(R)), np.abs(R[:, core]).argmax(1)])
        sgn[sgn == 0] = 1
        Rn = R * sgn[:, None]                      # polarity-normalised: peak positive
        # 3-tap smoothing before the peak search (noise bias of the FWHM estimator)
        Rs = (np.roll(Rn, 1, 1) + Rn + np.roll(Rn, -1, 1)) / 3.0
        pk = Rs[:, core].max(1)
        con.extend((R[:, core][np.arange(len(R)), np.abs(R[:, core]).argmax(1)]).tolist())
        for i in range(len(Rs)):
            if pk[i] / noise < 3.0:
                continue
            ic = np.flatnonzero(core)[Rs[i, core].argmax()]
            half_ = 0.5 * pk[i]
            lo = ic
            while lo > 0 and Rs[i, lo] > half_:
                lo -= 1
            hi = ic
            while hi < len(offs) - 1 and Rs[i, hi] > half_:
                hi += 1
            if lo == 0 or hi == len(offs) - 1:
                continue
            # linear interpolation of the two crossings
            xl = offs[lo] + (half_ - Rs[i, lo]) / (Rs[i, lo + 1] - Rs[i, lo] + 1e-9) * (offs[lo + 1] - offs[lo])
            xr = offs[hi - 1] + (half_ - Rs[i, hi - 1]) / (Rs[i, hi] - Rs[i, hi - 1] + 1e-9) * (offs[hi] - offs[hi - 1])
            fw.append(xr - xl)
    con = np.asarray(con)
    return {
        "fwhm_px": float(np.median(fw)) if fw else float("nan"),
        "n_fwhm": len(fw),
        "contrast_signed_median": float(np.median(con)) if len(con) else float("nan"),
        "contrast_abs_median": float(np.median(np.abs(con))) if len(con) else float("nan"),
        "snr": float(np.median(np.abs(con)) / noise) if len(con) else float("nan"),
        "bright_share": float(np.mean(con > 0)) if len(con) else float("nan"),
        "noise_hp": float(noise),
    }


def prob_under_gt(prob, polys_eval, ds=2.0):
    vals = []
    for q in polys_eval:
        pts = resample(np.asarray(q, float), ds=ds)
        if len(pts) < 3:
            continue
        t = np.gradient(pts, axis=0)
        n = np.stack([-t[:, 1], t[:, 0]], 1)
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-8
        pm = np.max(np.stack([map_coordinates(prob, [(pts + d * n)[:, 1], (pts + d * n)[:, 0]],
                                              order=1, mode="nearest")
                              for d in (-2, -1, 0, 1, 2)]), 0)
        vals.append(pm)
    v = np.concatenate(vals) if vals else np.zeros(0)
    if not len(v):
        return {}
    return {"p10": float(np.percentile(v, 10)), "p25": float(np.percentile(v, 25)),
            "p50": float(np.percentile(v, 50)), "frac_gt098": float(np.mean(v > 0.98)),
            "frac_gt09": float(np.mean(v > 0.9)), "frac_gt05": float(np.mean(v > 0.5)),
            "n_samples": int(len(v))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/roi303_eval")
    ap.add_argument("--split", default="val")
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--sweep", required=True, help="eval_v5 --out-json with per-frame rows")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.split == "test":
        raise SystemExit("refusing to touch TEST")

    sw = json.load(open(args.sweep))["results"]
    thrs = sorted(sw, key=float)
    per = {}
    for t in thrs:
        for row in sw[t]["_frames"]:
            per.setdefault(row["name"], {})[t] = row
    means = {t: np.mean([r["f1"] for r in sw[t]["_frames"] if (r["tp"] + r["fn"]) > 0]) for t in thrs}
    t_glob = max(means, key=means.get)

    rows = []
    for p in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        nm = os.path.basename(p)
        npz = os.path.join(args.pred_dir, nm.replace(".h5", ".npz"))
        if not os.path.exists(npz):
            continue
        fr = read_frame_h5(p)
        if str(fr["attrs"].get("split")) != args.split or not fr["polylines"]:
            continue
        img = np.asarray(fr["image"], float)
        H, W = img.shape
        polys = [np.asarray(q, float) for q in fr["polylines"] if len(q) >= 2]
        mpx = H * W / 1e6
        gl = float(sum(np.sum(np.hypot(*np.diff(q, axis=0).T)) for q in polys))
        lo, hi = np.percentile(img, [1, 99])
        u = np.unique(img)
        rec = {"name": nm, "H": H, "W": W, "n_gt": len(polys),
               "curves_per_mpx": len(polys) / mpx, "gt_len_per_mpx": gl / mpx,
               "p1": float(lo), "p99": float(hi), "span_1_99": float(hi - lo),
               "frac_min": float(np.mean(img <= img.min())), "frac_max": float(np.mean(img >= img.max())),
               "frac_at_0": float(np.mean(img <= 0)), "frac_at_255": float(np.mean(img >= 255)),
               "n_levels": int(len(u)), "img_min": float(img.min()), "img_max": float(img.max()),
               "median": float(np.median(img))}
        rec.update(frame_width_and_contrast(img, polys))
        prob = np.load(npz)["prob"].astype(np.float32).max(0)
        rec.update({f"pgt_{k}": v for k, v in prob_under_gt(prob, [q * UP for q in polys]).items()})
        rec["fg_frac_098"] = float(np.mean(prob > 0.98))
        rec["fg_frac_05"] = float(np.mean(prob > 0.5))
        pr = per.get(nm, {})
        if pr:
            best = max(pr, key=lambda t: pr[t]["f1"])
            rec["best_thr"] = float(best); rec["f1_best"] = pr[best]["f1"]
            rec["f1_global"] = pr[t_glob]["f1"]; rec["gain_oracle_thr"] = pr[best]["f1"] - pr[t_glob]["f1"]
            rec["f1_at_05"] = pr.get("0.5", {}).get("f1", float("nan"))
        rows.append(rec)
        print(f"{nm:16s} {H}x{W} nGT={len(polys):3d} dens={rec['curves_per_mpx']:6.1f} pol={'B' if rec['bright_share']>0.5 else 'D'}"
              f" snr={rec['snr']:5.1f} fwhm={rec['fwhm_px']:.2f} span={rec['span_1_99']:.0f} lev={rec['n_levels']:3d}"
              f" rail={100*(rec['frac_min']+rec['frac_max']):.1f}% | pgt p25={rec.get('pgt_p25', float('nan')):.2f} p50={rec.get('pgt_p50', float('nan')):.2f}"
              f" >0.98={rec.get('pgt_frac_gt098', float('nan')):.2f} | thr*={rec.get('best_thr', float('nan')):.3f} f1@g={rec.get('f1_global', float('nan')):.2f} f1*={rec.get('f1_best', float('nan')):.2f}",
              flush=True)

    # ---- correlations (frames with >= 10 GT for the threshold targets)
    cov = ["curves_per_mpx", "gt_len_per_mpx", "bright_share", "snr", "contrast_abs_median",
           "fwhm_px", "span_1_99", "frac_min", "frac_max", "n_levels", "H", "noise_hp", "median"]
    tgt = ["pgt_p25", "pgt_p50", "pgt_frac_gt098", "best_thr", "gain_oracle_thr", "f1_global"]
    big = [r for r in rows if r["n_gt"] >= 10]
    corr = {}
    print("\nSpearman rho (frames with >=10 GT, n=%d):  rows=target, cols=covariate" % len(big))
    print("%-16s" % "" + "".join("%13s" % c[:12] for c in cov))
    for t in tgt:
        line = "%-16s" % t
        corr[t] = {}
        for c in cov:
            x = np.array([r.get(c, np.nan) for r in big], float)
            y = np.array([r.get(t, np.nan) for r in big], float)
            ok = np.isfinite(x) & np.isfinite(y)
            rho, pv = spearmanr(x[ok], y[ok]) if ok.sum() > 5 else (np.nan, np.nan)
            corr[t][c] = (float(rho), float(pv))
            line += "%9.2f%s" % (rho, ("***" if pv < 0.001 else "** " if pv < 0.01 else "*  " if pv < 0.05 else "   "))
        print(line)
    # polarity split
    for t in ("pgt_p25", "best_thr", "f1_global", "gain_oracle_thr"):
        b = [r[t] for r in big if r["bright_share"] > 0.5 and np.isfinite(r.get(t, np.nan))]
        d = [r[t] for r in big if r["bright_share"] <= 0.5 and np.isfinite(r.get(t, np.nan))]
        print(f"{t}: bright frames n={len(b)} median={np.median(b) if b else float('nan'):.3f} | dark frames n={len(d)} median={np.median(d) if d else float('nan'):.3f}")
    with open(args.out, "w") as fh:
        json.dump({"global_thr": t_glob, "rows": rows, "spearman_ge10": corr}, fh, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
