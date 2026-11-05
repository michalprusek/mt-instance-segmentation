#!/usr/bin/env python3
"""Does the IMAGE lose contrast on a filament as it approaches a crossing -- and does the render?

Pre-screen (model-free first, model second) for the lift-over hypothesis, 2026-09-16:

  In interference reflection microscopy the contrast of a microtubule is set by its HEIGHT above
  the coverslip (two-beam interference, -cos(2k h); dark at contact, vanishing at the ~56 nm
  zero-crossing, bright beyond). At a crossing one filament must climb OVER the other: a stiff
  rod crossing a ~25 nm obstacle is lifted over a RAMP of hundreds of nm on either side, so its
  contrast fades (and can flip sign) over a stretch of several to ~20 px before and after the
  crossing point. The generator composites filaments at their own heights with no lift-over ramp,
  so it never renders that fade. The crossing failure the campaign measured on real data is an
  EROSION of the probability ridge on one arm 8-20 px from the junction (82.8 % of failures have
  the ridge below threshold at the crossing on >= 1 filament, 62 % have an arm missing; opposite-
  polarity crossings resolve at 0.082), which is what a lifted arm would look like to a network
  that has never seen one.

This script asks the image, not the model: along every GT filament, the perpendicular-max
|I - background| (the same estimator as diag_along_filament.py, so annotation drift cannot fake
a fade), expressed as a RATIO to that filament's own median contrast far from every crossing,
binned by arc-length distance to the nearest crossing on that filament. Reported per block:

  * median ratio per distance bin, and the fraction of samples below 0.5 (a real fade);
  * the WEAKER of the two filaments at each crossing, per bin (the lifted one is unknown, so the
    minimum over the pair is the honest statistic);
  * the fraction of samples whose SIGNED contrast has the opposite sign to the filament's own
    far-from-crossing sign (a polarity flip = the height crossed the zero-crossing);
  * a control: the same statistics around pseudo-crossings placed at random points >= 40 px
    from any real crossing, so the bins can be read against a null;
  * optionally, with --pred-dir, the model's probability along the filament in the same bins,
    so the image fade and the network's erosion are read side by side.

If real filaments fade toward crossings and synthetic ones do not, the generator lacks a
physical mechanism (not a prior to be fitted: the ramp follows from mechanics and is to be
RANDOMISED wide, per the domain-randomisation rule) and a generator arm is declared. If both
fade, or neither, the lift-over hypothesis dies here for CPU-minutes.

    PYTHONPATH=src:scripts python scripts/diag_crossing_contrast.py \
        --data data/real/roi303_eval --split val --pred-dir preds/SPARSE35_roi_ep40_native \
        --out crossing_contrast_roi303.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from instance.geometry import arclength, resample  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.gt_stats import count_crossings  # noqa: E402

UP = 1.5                      # eval frame scale of the cached probability maps
BINS = ((0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 40))   # native px from the crossing
FAR = 40.0                    # native px: "far from every crossing" for the per-filament reference
WIN = 40.0                    # native px: half-window around each crossing that is analysed


def source_of(name: str) -> str:
    if name.startswith("htw_"):
        return "htw"
    if name.startswith("alice_"):
        return "alice"
    if name.startswith("training_img_"):
        return "586"
    if name.startswith("roi_img_"):
        return "roi303"
    return "synth"


def normals(pts: np.ndarray) -> np.ndarray:
    d = np.gradient(pts, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], axis=1)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)


def signed_profile(img: np.ndarray, pts: np.ndarray, perp: float, bg_lo: float, bg_hi: float):
    """Signed contrast at the perpendicular |max| within +-perp px, background = median of a
    perpendicular annulus bg_lo..bg_hi px. Same construction as diag_along_filament.profile,
    but the SIGN is kept so a polarity flip can be counted."""
    H, W = img.shape
    nrm = normals(pts)
    offs_sig = np.arange(-perp, perp + 1e-9, 0.5)
    offs_bg = np.concatenate([np.arange(-bg_hi, -bg_lo + 1e-9, 1.0),
                              np.arange(bg_lo, bg_hi + 1e-9, 1.0)])

    def sample(off):
        p = pts[:, None, :] + nrm[:, None, :] * off[None, :, None]
        cc = np.clip(np.rint(p[..., 0]).astype(int), 0, W - 1)
        rr = np.clip(np.rint(p[..., 1]).astype(int), 0, H - 1)
        return img[rr, cc]

    bg = np.median(sample(offs_bg), axis=1)
    sig = sample(offs_sig) - bg[:, None]
    k = np.abs(sig).argmax(axis=1)
    return sig[np.arange(len(pts)), k]


def prob_along(prob: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Model probability at the GT samples in the eval frame: max within +-2 eval px along the
    normal, so a ridge one pixel off the annotation still counts as present."""
    H, W = prob.shape
    q = pts * UP
    nrm = normals(q)
    offs = np.arange(-2.0, 2.0 + 1e-9, 1.0)
    p = q[:, None, :] + nrm[:, None, :] * offs[None, :, None]
    cc = np.clip(np.rint(p[..., 0]).astype(int), 0, W - 1)
    rr = np.clip(np.rint(p[..., 1]).astype(int), 0, H - 1)
    return prob[rr, cc].max(axis=1)


class Acc:
    """Per-block accumulator: per-bin lists of ratio / flip / prob, for real and pseudo crossings,
    for 'any filament' and 'weaker filament of the pair'."""

    def __init__(self):
        self.d = {}

    def add(self, key, values):
        self.d.setdefault(key, []).extend(np.asarray(values, float).tolist())

    def summary(self):
        out = {}
        for key, vals in self.d.items():
            a = np.asarray(vals)
            if a.size == 0:
                continue
            out[key] = {"n": int(a.size), "p25": float(np.percentile(a, 25)),
                        "p50": float(np.median(a)), "p75": float(np.percentile(a, 75)),
                        "mean": float(a.mean())}
            if key.startswith(("ratio", "wratio", "null_ratio")):
                out[key]["frac_below_0.5"] = float((a < 0.5).mean())
            if key.startswith(("prob", "wprob", "null_prob")):
                out[key]["frac_below_0.98"] = float((a < 0.98).mean())
                out[key]["frac_below_0.5"] = float((a < 0.5).mean())
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--pred-dir", default="")
    ap.add_argument("--perp", type=float, default=2.0)
    ap.add_argument("--bg-lo", type=float, default=6.0)
    ap.add_argument("--bg-hi", type=float, default=12.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    accs: dict[str, Acc] = {}
    counts: dict[str, dict] = {}
    n_done = 0
    for path in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        name = os.path.basename(path)
        fr = read_frame_h5(path)
        if args.split != "all" and str(fr["attrs"].get("split", "")) != args.split:
            continue
        img = np.asarray(fr["image"], float)
        polys = [np.asarray(p, float) for p in fr["polylines"] if len(p) >= 2]
        if len(polys) < 2:
            continue
        src = source_of(name)
        acc = accs.setdefault(src, Acc())
        cnt = counts.setdefault(src, {"frames": 0, "crossings": 0, "curves": 0, "pseudo": 0})
        cnt["frames"] += 1

        prob = None
        if args.pred_dir:
            npz = os.path.join(args.pred_dir, name.replace(".h5", ".npz"))
            if os.path.exists(npz):
                prob = np.load(npz)["prob"].astype(np.float32)
                prob = prob.max(axis=0) if prob.ndim == 3 else prob

        # the SAME resampling count_crossings uses (ds = 1), so s_i indexes these samples
        res = [resample(p, ds=1.0) for p in polys]
        crossings = count_crossings(polys, tol=2.0, ds=1.0, endpoint_margin=8.0)
        cnt["crossings"] += len(crossings)
        cnt["curves"] += len(res)

        # per curve: signed contrast profile, model prob, and the arc positions of its crossings
        prof = {}
        pro = {}
        cross_s: dict[int, list[float]] = {}
        for c in crossings:
            cross_s.setdefault(c["i"], []).append(c["s_i"])
            cross_s.setdefault(c["j"], []).append(c["s_j"])
        for k, pts in enumerate(res):
            if len(pts) < 8:
                continue
            prof[k] = signed_profile(img, pts, args.perp, args.bg_lo, args.bg_hi)
            if prob is not None:
                pro[k] = prob_along(prob, pts)

        def far_reference(k):
            """Median |contrast| and dominant sign over samples >= FAR px from every crossing."""
            pts = res[k]
            s = arclength(pts)
            dist = np.full(len(pts), np.inf)
            for sc in cross_s.get(k, []):
                dist = np.minimum(dist, np.abs(s - sc))
            far = dist >= FAR
            if far.sum() < 10:
                return None
            c = prof[k][far]
            return float(np.median(np.abs(c))), float(np.sign(np.median(c))), s, dist

        # real crossings
        for c in crossings:
            pair_bins: dict[tuple, list] = {}
            pair_pbins: dict[tuple, list] = {}
            for k, sc in ((c["i"], c["s_i"]), (c["j"], c["s_j"])):
                if k not in prof:
                    continue
                ref = far_reference(k)
                if ref is None:
                    continue
                med, sgn, s, _ = ref
                if med <= 0:
                    continue
                dd = np.abs(s - sc)
                win = dd <= WIN
                ratio = np.abs(prof[k][win]) / med
                flip = (np.sign(prof[k][win]) != sgn).astype(float)
                for lo, hi in BINS:
                    m = (dd[win] >= lo) & (dd[win] < hi)
                    if not m.any():
                        continue
                    key = f"{lo}-{hi}"
                    acc.add(f"ratio/{key}", ratio[m])
                    acc.add(f"flip/{key}", flip[m])
                    pair_bins.setdefault((lo, hi), []).append(float(np.median(ratio[m])))
                    if k in pro:
                        pv = pro[k][win][m]
                        acc.add(f"prob/{key}", pv)
                        pair_pbins.setdefault((lo, hi), []).append(float(np.median(pv)))
            for (lo, hi), vals in pair_bins.items():
                if len(vals) == 2:
                    acc.add(f"wratio/{lo}-{hi}", [min(vals)])
            for (lo, hi), vals in pair_pbins.items():
                if len(vals) == 2:
                    acc.add(f"wprob/{lo}-{hi}", [min(vals)])

        # pseudo crossings: random points on curves, >= FAR px from every real crossing and from
        # both ends, one per real crossing so the null has the same weight
        n_pseudo = max(len(crossings), 1)
        eligible = [k for k in prof if len(res[k]) > 2 * (FAR + WIN)]
        for _ in range(n_pseudo):
            if not eligible:
                break
            k = int(rng.choice(eligible))
            ref = far_reference(k)
            if ref is None:
                continue
            med, sgn, s, dist = ref
            if med <= 0:
                continue
            ok = np.where((dist >= FAR + WIN) & (s >= WIN + 8) & (s <= s[-1] - WIN - 8))[0]
            if ok.size == 0:
                continue
            sc = float(s[int(rng.choice(ok))])
            cnt["pseudo"] += 1
            dd = np.abs(s - sc)
            win = dd <= WIN
            ratio = np.abs(prof[k][win]) / med
            flip = (np.sign(prof[k][win]) != sgn).astype(float)
            for lo, hi in BINS:
                m = (dd[win] >= lo) & (dd[win] < hi)
                if not m.any():
                    continue
                acc.add(f"null_ratio/{lo}-{hi}", ratio[m])
                acc.add(f"null_flip/{lo}-{hi}", flip[m])
                if k in pro:
                    acc.add(f"null_prob/{lo}-{hi}", pro[k][win][m])
        n_done += 1
        if args.limit and n_done >= args.limit:
            break

    out = {"data": args.data, "split": args.split, "pred_dir": args.pred_dir, "bins": BINS,
           "far_px": FAR, "counts": counts, "blocks": {s: a.summary() for s, a in accs.items()}}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)

    for src, summ in out["blocks"].items():
        c = counts[src]
        print(f"\n=== {src}: frames {c['frames']}  curves {c['curves']}  crossings {c['crossings']}  "
              f"pseudo {c['pseudo']}  (ratio = |contrast| / own far median; flip = sign differs from own far sign)")
        print(f"  {'dist px':<9}{'ratio p50':>10}{'<0.5':>7}{'weaker p50':>12}{'flip':>7}"
              f"{'| null p50':>11}{'<0.5':>7}{'flip':>7}"
              + (f"{'| prob p50':>11}{'<.98':>7}{'weaker':>8}{'| null p':>9}" if any(k.startswith('prob') for k in summ) else ""))
        for lo, hi in BINS:
            key = f"{lo}-{hi}"
            r = summ.get(f"ratio/{key}"); w = summ.get(f"wratio/{key}"); f = summ.get(f"flip/{key}")
            nr = summ.get(f"null_ratio/{key}"); nf = summ.get(f"null_flip/{key}")
            line = (f"  {key:<9}{r['p50'] if r else float('nan'):>10.3f}{r['frac_below_0.5'] if r else float('nan'):>7.2f}"
                    f"{w['p50'] if w else float('nan'):>12.3f}{f['mean'] if f else float('nan'):>7.2f}"
                    f"{nr['p50'] if nr else float('nan'):>11.3f}{nr['frac_below_0.5'] if nr else float('nan'):>7.2f}"
                    f"{nf['mean'] if nf else float('nan'):>7.2f}")
            p = summ.get(f"prob/{key}"); wp = summ.get(f"wprob/{key}"); np_ = summ.get(f"null_prob/{key}")
            if p:
                line += (f"{p['p50']:>11.3f}{p['frac_below_0.98']:>7.2f}{wp['p50'] if wp else float('nan'):>8.3f}"
                         f"{np_['p50'] if np_ else float('nan'):>9.3f}")
            print(line)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
