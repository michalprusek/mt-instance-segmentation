#!/usr/bin/env python3
"""Is the evidence for a WEAK filament stretch recoverable by integrating ALONG the filament, and
does that integration rank it above the network's own false foreground?

Pre-screen for the along-filament lane (advisor 2026-09-16). The survey's "20 px along the tangent
sees SNR ~4.5x" was arithmetic; this measures it on the frames the model fails on, against the
negatives that actually matter.

Why the negatives are the model's own blobs: a first version used laterally displaced (5-8 px)
near-filament negatives and the network separated weak positives from them with AUC 0.994 -- the
weak stretches are not indistinguishable from their neighbourhood; they lose to the GLOBAL threshold
that the model's high-confidence OFF-annotation responses force (39 % of roi303 FP, median 37 px).
So the ordering that matters is: weak on-filament sample  vs  a sample on an off-annotation
prediction the model emits at p > thr. At the operating point the network ranks every blob above
every weak positive by construction (recall of the weak set = 0). The question is whether an
along-the-structure statistic of the RAW image inverts that ordering.

Positives: samples on GT filaments >= 40 px from any crossing and >= 20 px from either end, with the
model's p < --weak-thr (max within +-2 eval px). Statistics along the GT tangent.
Negatives: the instancer's own predictions at --thr whose centerline lies < 50 % within 5 eval px of
any GT (the off_other class of diag_fp_anatomy), sampled along their OWN tangent, with the polyline
extended straight by 20 px at both ends so a short blob's window reaches into what surrounds it.
Reference negatives: the same positives displaced laterally by 5-8 px (near-filament background).

Statistics per sample, raw image, per-sample perpendicular-annulus background:
  pix     perpendicular-max |contrast| at the sample
  coh_L   |mean over [-L/2, L/2] along the tangent of the SIGNED perpendicular-max contrast|
  inc_L   mean over the same window of |contrast|
  for L = 10, 20, 40 native px.

Readout: AUC weak-vs-blob per statistic, and the recall of weak positives at the threshold that
rejects 50 / 80 / 90 % of blob samples. Kill line: at 80 % blob rejection, coh/inc at L = 20 or 40
recover < 20 % of the weak positives on roi303 -> the along-filament information does not beat the
blobs either; the lane is dead for CPU-minutes.

    PYTHONPATH=src:scripts python scripts/diag_line_integral.py --data data/real/roi303_eval --split val \\
        --pred-dir preds/SPARSE35_roi_ep40_native --thr 0.98 --shard 0/6 --out li_roi303_0.json
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
from instance.instancer_a import instance_a  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.gt_stats import count_crossings  # noqa: E402

UP, KAPPA_MAX = 1.5, 0.25
LS = (10, 20, 40)
FAR_CROSS, FAR_END = 40.0, 20.0
DISP = (5.0, 8.0)
EXT = 20.0          # native px of straight extension at each end of a predicted polyline
PERP = [2.0]        # native px, perpendicular window of the contrast read (CLI --perp)
NETWIN = [2.0]      # eval px, perpendicular window of the network read (CLI --net-window)


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


def normals(pts):
    d = np.gradient(pts, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], axis=1)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)


def signed_contrast(img, pts, nrm, perp=2.0, bg_lo=6.0, bg_hi=12.0):
    perp = PERP[0]
    H, W = img.shape
    offs_sig = np.arange(-perp, perp + 1e-9, 0.5)
    offs_bg = np.concatenate([np.arange(-bg_hi, -bg_lo + 1e-9, 1.0), np.arange(bg_lo, bg_hi + 1e-9, 1.0)])

    def sample(off):
        p = pts[:, None, :] + nrm[:, None, :] * off[None, :, None]
        cc = np.clip(np.rint(p[..., 0]).astype(int), 0, W - 1)
        rr = np.clip(np.rint(p[..., 1]).astype(int), 0, H - 1)
        return img[rr, cc]

    bg = np.median(sample(offs_bg), axis=1)
    sig = sample(offs_sig) - bg[:, None]
    k = np.abs(sig).argmax(axis=1)
    return sig[np.arange(len(pts)), k]


def prob_at(prob, pts_native):
    H, W = prob.shape
    q = pts_native * UP
    nrm = normals(q)
    offs = np.arange(-NETWIN[0], NETWIN[0] + 1e-9, 1.0)
    p = q[:, None, :] + nrm[:, None, :] * offs[None, :, None]
    cc = np.clip(np.rint(p[..., 0]).astype(int), 0, W - 1)
    rr = np.clip(np.rint(p[..., 1]).astype(int), 0, H - 1)
    return prob[rr, cc].max(axis=1)


def windowed(c, L):
    h = max(1, L // 2)
    n = len(c)
    cs = np.concatenate([[0.0], np.cumsum(c)])
    ca = np.concatenate([[0.0], np.cumsum(np.abs(c))])
    lo = np.clip(np.arange(n) - h, 0, n)
    hi = np.clip(np.arange(n) + h + 1, 0, n)
    cnt = np.maximum(hi - lo, 1)
    return np.abs(cs[hi] - cs[lo]) / cnt, (ca[hi] - ca[lo]) / cnt


def feats_along(img, pts, keep=None):
    """All statistics at the points of a 1-px-resampled polyline (optionally only rows `keep`)."""
    nrm = normals(pts)
    c = signed_contrast(img, pts, nrm)
    f = {"pix": np.abs(c)}
    for L in LS:
        coh, inc = windowed(c, L)
        f[f"coh_{L}"] = coh; f[f"inc_{L}"] = inc
    if keep is not None:
        f = {k: v[keep] for k, v in f.items()}
    return f


def extend(pts, ext):
    """Straight extension of a polyline by `ext` px at both ends (unit-step samples)."""
    if len(pts) < 2:
        return pts, np.zeros(len(pts), bool)
    t0 = pts[0] - pts[1]; t0 /= np.linalg.norm(t0) + 1e-9
    t1 = pts[-1] - pts[-2]; t1 /= np.linalg.norm(t1) + 1e-9
    n = int(ext)
    head = pts[0] + t0[None, :] * np.arange(n, 0, -1)[:, None]
    tail = pts[-1] + t1[None, :] * np.arange(1, n + 1)[:, None]
    out = np.concatenate([head, pts, tail], 0)
    orig = np.zeros(len(out), bool); orig[n:n + len(pts)] = True
    return out, orig


def auc(pos, neg):
    from scipy.stats import rankdata
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def recall_at_reject(pos, neg, reject):
    """Recall of positives at the threshold that rejects a fraction `reject` of negatives."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    t = np.quantile(neg, reject)
    return float((pos > t).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--thr", type=float, required=True, help="the operating threshold (VAL-chosen)")
    ap.add_argument("--params", default="src/instance/params_a_derived.json")
    ap.add_argument("--weak-thr", type=float, default=None, help="default = --thr")
    ap.add_argument("--shard", default="0/1", help="i/n: process frames i, i+n, i+2n, ...")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perp", type=float, default=2.0, help="native px; 3.3 = the metric's 5 eval px")
    ap.add_argument("--net-window", type=float, default=2.0, help="eval px; 5 = the metric's tolerance")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    PERP[0] = args.perp; NETWIN[0] = args.net_window
    weak_thr = args.thr if args.weak_thr is None else args.weak_thr
    si, sn = (int(x) for x in args.shard.split("/"))
    rng = np.random.default_rng(args.seed + si)
    params = json.load(open(args.params))

    raw: dict[str, dict[str, list]] = {}
    files = sorted(glob.glob(os.path.join(args.data, "*.h5")))[si::sn]
    n_done = 0
    for path in files:
        name = os.path.basename(path)
        fr = read_frame_h5(path)
        if args.split != "all" and str(fr["attrs"].get("split", "")) != args.split:
            continue
        npz = os.path.join(args.pred_dir, name.replace(".h5", ".npz"))
        if not os.path.exists(npz):
            continue
        ch = np.load(npz)["prob"].astype(np.float32)
        prob = ch.max(axis=0) if ch.ndim == 3 else ch
        img = np.asarray(fr["image"], float)
        polys = [np.asarray(p, float) for p in fr["polylines"] if len(p) >= 2]
        if not polys:
            continue
        src = source_of(name)
        S = raw.setdefault(src, {})
        res = [resample(p, ds=1.0) for p in polys]
        gt_tree = cKDTree(np.concatenate([r for r in res if len(r)], 0))
        crossings = count_crossings(polys, tol=2.0, ds=1.0, endpoint_margin=8.0)
        cross_s: dict[int, list] = {}
        for c in crossings:
            cross_s.setdefault(c["i"], []).append(c["s_i"]); cross_s.setdefault(c["j"], []).append(c["s_j"])

        # ---- positives (weak / strong) and the displaced reference negatives
        for k, pts in enumerate(res):
            if len(pts) < 2 * FAR_END + 10:
                continue
            s = arclength(pts)
            dist = np.full(len(pts), np.inf)
            for sc in cross_s.get(k, []):
                dist = np.minimum(dist, np.abs(s - sc))
            ok = (dist >= FAR_CROSS) & (s >= FAR_END) & (s <= s[-1] - FAR_END)
            if ok.sum() < 5:
                continue
            p_pos = prob_at(prob, pts)
            fpos = feats_along(img, pts)
            weak = ok & (p_pos < weak_thr); strong = ok & ~(p_pos < weak_thr)
            d = rng.uniform(*DISP) * rng.choice([-1.0, 1.0])
            q = pts + normals(pts) * d
            negm = ok & (gt_tree.query(q)[0] > 3.0)
            fneg = feats_along(img, q)
            for key in fpos:
                S.setdefault(f"{key}/weak", []).extend(fpos[key][weak].tolist())
                S.setdefault(f"{key}/strong", []).extend(fpos[key][strong].tolist())
                S.setdefault(f"{key}/disp", []).extend(fneg[key][negm].tolist())
            S.setdefault("net/weak", []).extend(p_pos[weak].tolist())
            S.setdefault("net/strong", []).extend(p_pos[strong].tolist())
            S.setdefault("net/disp", []).extend(prob_at(prob, q)[negm].tolist())

        # ---- operating-point negatives: the model's own off-annotation predictions at --thr
        pr_polys, _ = instance_a(prob > args.thr, KAPPA_MAX, params=params, channels=ch if ch.ndim == 3 else ch[None], prob=prob)
        n_blob = 0
        for pl in pr_polys:
            pl = np.asarray(pl, float)
            if len(pl) < 2:
                continue
            pts_nat = resample(pl / UP, ds=1.0)                # back to native px
            if len(pts_nat) < 4:
                continue
            on_gt = float((gt_tree.query(pts_nat)[0] <= 5.0 / UP).mean())
            if on_gt >= 0.5:
                continue                                       # not an off-annotation prediction
            n_blob += 1
            ext_pts, orig = extend(pts_nat, EXT)
            fb = feats_along(img, ext_pts, keep=orig)
            for key in fb:
                S.setdefault(f"{key}/blob", []).extend(fb[key].tolist())
            S.setdefault("net/blob", []).extend(prob_at(prob, pts_nat).tolist())
            S.setdefault("blob_len", []).append(float(len(pts_nat)))
        n_done += 1
        if args.limit and n_done >= args.limit:
            break

    out = {"data": args.data, "split": args.split, "pred_dir": args.pred_dir, "thr": args.thr,
           "weak_thr": weak_thr, "shard": args.shard, "perp": args.perp, "net_window": args.net_window, "raw": raw}
    with open(args.out, "w") as fh:
        json.dump(out, fh)
    report(raw, weak_thr)


def report(raw, weak_thr):
    for src, S in raw.items():
        nw, ns = len(S.get("net/weak", [])), len(S.get("net/strong", []))
        nb, nd = len(S.get("net/blob", [])), len(S.get("net/disp", []))
        bl = np.asarray(S.get("blob_len", []), float)
        print(f"\n=== {src}: plain-stretch on-filament samples {nw + ns} (weak p<{weak_thr}: {nw}, "
              f"{100 * nw / max(nw + ns, 1):.1f} %); blob samples {nb} on {bl.size} off-annotation predictions "
              f"(len p50 {np.median(bl) if bl.size else float('nan'):.0f} px); displaced reference negatives {nd}")
        print(f"  {'statistic':<8}{'AUC w/blob':>11}{'AUC s/blob':>11}{'R@rej50':>9}{'R@rej80':>9}{'R@rej90':>9}"
              f"{'| AUC w/disp':>13}{'R@disp1%':>10}")
        for key in ["net", "pix"] + [f"{t}_{L}" for L in LS for t in ("coh", "inc")]:
            pw, ps = S.get(f"{key}/weak", []), S.get(f"{key}/strong", [])
            nbv, ndv = S.get(f"{key}/blob", []), S.get(f"{key}/disp", [])
            print(f"  {key:<8}{auc(pw, nbv):>11.3f}{auc(ps, nbv):>11.3f}{recall_at_reject(pw, nbv, 0.5):>9.3f}"
                  f"{recall_at_reject(pw, nbv, 0.8):>9.3f}{recall_at_reject(pw, nbv, 0.9):>9.3f}"
                  f"{auc(pw, ndv):>13.3f}{recall_at_reject(pw, ndv, 0.99):>10.3f}")


def merge_main() -> None:
    """python diag_line_integral.py --merge out_0.json out_1.json ... : pool shards and report."""
    files = sys.argv[2:]
    raw: dict = {}
    weak_thr = None
    for f in files:
        d = json.load(open(f)); weak_thr = d["weak_thr"]
        for src, S in d["raw"].items():
            T = raw.setdefault(src, {})
            for k, v in S.items():
                T.setdefault(k, []).extend(v)
    report(raw, weak_thr)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--merge":
        merge_main()
    else:
        main()
