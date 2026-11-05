#!/usr/bin/env python3
"""Are the WEAK ARMS the faint filaments the generator never renders?  RUNS ON TULEN (CPU).

The 2026-09-04 crossing diagnostic says 97.6 % of lost crossings have a disconnected mask and 62 %
an arm the model never found, and the frame-level covariates say confidence tracks SNR and the
bright share. This measures the same thing at the FILAMENT level, on both sides of the domain gap,
with one common instrument:

  per polyline (real GT or synthetic instance), on the native, [1, 99]-stretched image:
    * signed contrast  = mean(profile at |d| <= 1 px) - median(profile at 4 <= |d| <= 6 px), sampled
      every 2 px of arc length along the normal of the (sigma = 1 smoothed) image;
    * noise            = 1.4826 * MAD of the frame's high-pass residual (img - gaussian(img, 2));
    * snr              = |median signed contrast| / noise, polarity = sign of the median, consistency
      = fraction of samples that share the median's sign;
  real side only, from the cached probability map (1.5x eval frame, channel 0):
    * p25 / p50 of the probability along the polyline, fraction >= thr, and "covered" = that
      fraction >= 0.95 (the strict-F1 coverage rule applied to the mask).

Outputs one JSON with the real rows, the synthetic rows and a summary: the real SNR distribution by
polarity vs the synthetic one, the coverage curve against SNR (where the model loses filaments),
and the share of real filament LENGTH below the synthetic 5th percentile -- the number that says
whether the contrast prior is the lever.

    PYTHONPATH=.:src:scripts:synth:third_party python scripts/diag_filament_contrast.py \
        --pred-dir /disk2/prusek/mt_work/preds/SPARSE35_roi_native --thr 0.98 \
        --calib calib_V6_c070_narrow.json --n-synth 80 --out runs/diag_roi/contrast.json
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "src", "scripts", "synth", "third_party"):
    sys.path.insert(0, os.path.join(ROOT, d))

from mt_bench.cvat_import import read_frame_h5  # noqa: E402

UP = 1.5
TAPS = np.arange(-6, 7, 1.0)
SYNTH_META: list = []      # one record per synthetic frame: the photometric draws behind it


def stretch(img):
    lo, hi = np.percentile(img, [1, 99])
    return np.clip((np.asarray(img, np.float64) - lo) / (hi - lo + 1e-6), 0, 1)


def resample(poly, step=2.0):
    """(x, y) vertices -> points every `step` px of arc length, with unit tangents."""
    P = np.asarray(poly, float)
    if len(P) < 2:
        return None, None
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    L = np.concatenate([[0], np.cumsum(seg)])
    if L[-1] < step:
        return None, None
    s = np.arange(0, L[-1], step)
    x = np.interp(s, L, P[:, 0]); y = np.interp(s, L, P[:, 1])
    Q = np.stack([x, y], 1)
    t = np.gradient(Q, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    return Q, t


def profile_stats(img_s, Q, t):
    """Signed contrast per sample from the normal profile of the smoothed image."""
    n = np.stack([-t[:, 1], t[:, 0]], 1)                       # unit normal (x, y)
    xs = Q[:, 0][:, None] + TAPS[None, :] * n[:, 0][:, None]
    ys = Q[:, 1][:, None] + TAPS[None, :] * n[:, 1][:, None]
    prof = map_coordinates(img_s, [ys.ravel(), xs.ravel()], order=1, mode="nearest").reshape(xs.shape)
    core = prof[:, np.abs(TAPS) <= 1].mean(1)
    bg = np.median(prof[:, np.abs(TAPS) >= 4], axis=1)
    return core - bg


def frame_noise(img):
    hp = img - gaussian_filter(img, 2.0)
    return 1.4826 * np.median(np.abs(hp - np.median(hp))) + 1e-9


MARGIN = 7.0        # samples closer than this to the frame edge are dropped (the taps reach 6 px)
ISO_R = 8.0         # a sample is "isolated" when no OTHER polyline sample lies within this radius


def polyline_rows(img01, polys, prob=None, thr=0.98, tag=None):
    from scipy.spatial import cKDTree
    img_s = gaussian_filter(img01, 1.0)
    noise = frame_noise(img01)
    H, W = img01.shape
    rs = [resample(poly) for poly in polys]
    allpts = [(k, Q) for k, (Q, t) in enumerate(rs) if Q is not None]
    if allpts:
        P = np.concatenate([Q for _, Q in allpts]); owner = np.concatenate([np.full(len(Q), k) for k, Q in allpts])
        tree = cKDTree(P)
    rows = []
    for i, (Q, t) in enumerate(rs):
        if Q is None:
            continue
        inside = (Q[:, 0] >= MARGIN) & (Q[:, 0] < W - MARGIN) & (Q[:, 1] >= MARGIN) & (Q[:, 1] < H - MARGIN)
        if inside.sum() < 3:
            continue
        Q, t = Q[inside], t[inside]
        nb = tree.query_ball_point(Q, ISO_R)
        iso = np.array([all(owner[j] == i for j in lst) for lst in nb])
        c = profile_stats(img_s, Q, t)
        med = float(np.median(c))
        pol = 1 if med >= 0 else -1
        a = np.abs(c) / noise
        r = {"frame": tag, "idx": i, "length_px": float(2.0 * len(Q)), "n_samples": int(len(Q)),
             "contrast_signed_median": med, "contrast_abs_p25": float(np.percentile(np.abs(c), 25)),
             # snr = |median SIGNED contrast| / noise -- a filament that flips polarity along its
             # length scores ~0 here even when every stretch is visible; snr_abs = median |c| / noise
             # is the visibility statistic, and the per-sample fractions below separate the two.
             "snr": abs(med) / noise, "snr_abs": float(np.median(a)),
             "snr_p25": float(np.percentile(np.abs(c), 25)) / noise,
             "polarity": pol, "polarity_consistency": float(np.mean(np.sign(c) == pol)),
             "frac_weak_1": float(np.mean(a < 1.0)), "frac_weak_08": float(np.mean(a < 0.8)),
             "frac_flip_visible": float(np.mean((np.sign(c) != pol) & (a >= 1.0))),
             "frac_iso": float(iso.mean()), "n_iso": int(iso.sum()),
             "iso_frac_weak_1": float(np.mean(a[iso] < 1.0)) if iso.any() else None,
             "iso_snr_abs": float(np.median(a[iso])) if iso.any() else None,
             "noniso_frac_weak_1": float(np.mean(a[~iso] < 1.0)) if (~iso).any() else None,
             "noise": float(noise)}
        if prob is not None:
            pv = map_coordinates(prob, [Q[:, 1] * UP, Q[:, 0] * UP], order=1, mode="nearest")
            r.update({"p25": float(np.percentile(pv, 25)), "p50": float(np.median(pv)),
                      "frac_ge_thr": float(np.mean(pv >= thr)),
                      "covered": bool(np.mean(pv >= thr) >= 0.95),
                      # sample-level pairs for the pooled correlation: (|contrast|/noise, p)
                      "_pairs": np.stack([np.abs(c) / noise, pv], 1).round(4).tolist()})
        rows.append(r)
    return rows


def real_side(args):
    excl = set(args.exclude.split(",")) if args.exclude else set()
    rows = []
    for p in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        stem = os.path.basename(p)[:-3]
        if stem in excl:
            continue
        fr = read_frame_h5(p)
        if str(fr["attrs"].get("split")) != args.split:
            continue
        npz = os.path.join(args.pred_dir, stem + ".npz")
        if not os.path.exists(npz):
            continue
        allch = np.load(npz)["prob"].astype(np.float32)
        prob = allch[min(args.channel, allch.shape[0] - 1)]
        img01 = stretch(fr["image"])
        rows += polyline_rows(img01, fr["polylines"], prob, args.thr, tag=stem)
    return rows


def synth_side(args):
    from gen_train import build_cfg
    from mt_generator import generate_frame
    from PIL import Image
    cal = json.load(open(args.calib))
    cfg = build_cfg(cal["best_params"], mask_hw=1.0)
    if cal.get("gen_overrides"):
        cfg = dataclasses.replace(cfg, **cal["gen_overrides"])
    bgs = []
    for d in ("/home/prusek/mt_enc_exp/irm_backgrounds_v2",
              "/home/prusek/BIOCEV/datasets/microtubules/IRM_backgrounds_v2",
              os.path.join(ROOT, "data/real/irm_backgrounds_v2")):
        bgs = sorted(glob.glob(os.path.join(d, "*.tif")))
        if bgs:
            break
    if not bgs:
        raise SystemExit("no IRM backgrounds found")
    rng = np.random.default_rng(args.seed)
    rows, g = [], 640
    synth_meta = SYNTH_META
    for k in range(args.n_synth):
        b = np.asarray(Image.open(bgs[int(rng.integers(len(bgs)))]), np.float32)
        if b.ndim == 3:
            b = b.mean(2)
        if b.shape[0] < g or b.shape[1] < g:
            b = np.asarray(Image.fromarray(b).resize((max(g, b.shape[1]), max(g, b.shape[0]))), np.float32)
        y0, x0 = int(rng.integers(0, b.shape[0] - g + 1)), int(rng.integers(0, b.shape[1] - g + 1))
        bg = b[y0:y0 + g, x0:x0 + g]
        img, inst, meta = generate_frame(bg, rng, cfg)
        img01 = stretch(img)
        # the BACKGROUND crop's own statistics: if a frame is invisible because a dirty or graded
        # background dominates the [1, 99] stretch, these -- not the photometric draws -- explain it
        b1, b50, b99 = np.percentile(bg, [1, 50, 99])
        bhp = bg - gaussian_filter(bg, 2.0)
        meta = dict(meta, bg_range_rel=float((b99 - b1) / max(b50, 1e-6)), bg_median=float(b50),
                    bg_hp_mad_rel=float(1.4826 * np.median(np.abs(bhp - np.median(bhp))) / max(b50, 1e-6)),
                    img_p1=float(np.percentile(img, 1)), img_p99=float(np.percentile(img, 99)),
                    field_rel=float((np.percentile(img, 99) - np.percentile(img, 1)) / max(b99 - b1, 1e-6)))
        fr = polyline_rows(img01, [z["centerline"] for z in inst], None, tag=f"synth_{k}")
        keep = {k_: float(v) for k_, v in meta.items() if isinstance(v, (int, float, bool, np.floating))}
        for r in fr:
            r["render"] = inst[r["idx"]].get("render")
        synth_meta.append({"frame": f"synth_{k}", **keep, "n_inst": len(inst)})
        rows += fr
    return rows


def q(v, ps=(5, 10, 25, 50, 75, 90)):
    v = np.asarray(v, float)
    return {str(p): float(np.percentile(v, p)) for p in ps} if len(v) else {}


def summarise(real, synth, thr):
    S = {}
    for pol, name in ((1, "bright"), (-1, "dark")):
        rr = [r for r in real if r["polarity"] == pol]
        ss = [r for r in synth if r["polarity"] == pol]
        S[name] = {"n_real": len(rr), "n_synth": len(ss),
                   "real_snr": q([r["snr"] for r in rr]), "synth_snr": q([r["snr"] for r in ss]),
                   "real_len_share": float(sum(r["length_px"] for r in rr) / max(1, sum(r["length_px"] for r in real))),
                   "synth_len_share": float(sum(r["length_px"] for r in ss) / max(1, sum(r["length_px"] for r in synth)))}
    # coverage vs snr, pooled and per polarity; and the real length share below the synth p5
    edges = [0, 1, 1.5, 2, 3, 4, 6, 8, 12, 1e9]
    def curve(rows):
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = [r for r in rows if lo <= r["snr"] < hi]
            if sel:
                out.append({"snr_lo": lo, "snr_hi": hi, "n": len(sel),
                            "len_share": float(sum(r["length_px"] for r in sel) / sum(r["length_px"] for r in rows)),
                            "covered": float(np.mean([r["covered"] for r in sel])),
                            "p25_median": float(np.median([r["p25"] for r in sel]))})
        return out
    S["coverage_vs_snr_all"] = curve(real)
    S["coverage_vs_snr_dark"] = curve([r for r in real if r["polarity"] == -1])
    S["coverage_vs_snr_bright"] = curve([r for r in real if r["polarity"] == 1])
    synth_snr = np.array([r["snr"] for r in synth])
    if len(synth_snr):
        p5, p10 = np.percentile(synth_snr, [5, 10])
        tot = sum(r["length_px"] for r in real)
        S["synth_snr_p5"] = float(p5); S["synth_snr_p10"] = float(p10)
        S["real_len_share_below_synth_p5"] = float(sum(r["length_px"] for r in real if r["snr"] < p5) / tot)
        S["real_len_share_below_synth_p10"] = float(sum(r["length_px"] for r in real if r["snr"] < p10) / tot)
        S["real_uncovered_len_share_below_synth_p5"] = float(
            sum(r["length_px"] for r in real if r["snr"] < p5 and not r["covered"])
            / max(1, sum(r["length_px"] for r in real if not r["covered"])))
    # sample-level: does p track LOCAL contrast within filaments?  pooled Spearman
    pairs = np.array([pr for r in real for pr in r.get("_pairs", [])], float)
    if len(pairs) > 100:
        from scipy.stats import spearmanr
        rho, pv = spearmanr(pairs[:, 0], pairs[:, 1])
        S["sample_spearman_localsnr_vs_p"] = [float(rho), float(pv)]
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = pairs[(pairs[:, 0] >= lo) & (pairs[:, 0] < hi)]
            if len(sel):
                S.setdefault("sample_frac_p_lt_thr_by_localsnr", []).append(
                    {"snr_lo": lo, "snr_hi": hi, "n": int(len(sel)), "frac_p_lt_thr": float(np.mean(sel[:, 1] < thr))})
    def lw(rows, key):
        w = np.array([r["n_samples"] for r in rows], float)
        v = np.array([r[key] for r in rows], float)
        return float((v * w).sum() / max(w.sum(), 1))
    bands = [(0, 50), (50, 150), (150, 1e9)]
    S["sample_level"] = {}
    for side, rows in (("real", real), ("synth", synth)):
        if not rows:
            continue
        iso_rows = [r for r in rows if r["n_iso"] > 0]
        wiso = np.array([r["n_iso"] for r in iso_rows], float)
        non_rows = [r for r in rows if r["n_samples"] - r["n_iso"] > 0]
        wnon = np.array([r["n_samples"] - r["n_iso"] for r in non_rows], float)
        S["sample_level"][side] = {
            "share_isolated": float(sum(r["n_iso"] for r in rows) / max(1, sum(r["n_samples"] for r in rows))),
            "iso_weak_lt1": float((np.array([r["iso_frac_weak_1"] for r in iso_rows]) * wiso).sum() / max(wiso.sum(), 1)),
            "noniso_weak_lt1": float((np.array([r["noniso_frac_weak_1"] for r in non_rows]) * wnon).sum() / max(wnon.sum(), 1)),
            "iso_snr_abs_median": float(np.median([r["iso_snr_abs"] for r in iso_rows])) if iso_rows else None,
            "weak_lt1": lw(rows, "frac_weak_1"), "weak_lt08": lw(rows, "frac_weak_08"),
            "flip_visible": lw(rows, "frac_flip_visible"),
            "snr_abs_lenweighted_q": q([r["snr_abs"] for r in rows]),
            "by_length": [{"lo": lo, "hi": hi, "n": len(sel), "weak_lt1": lw(sel, "frac_weak_1"),
                           "flip_visible": lw(sel, "frac_flip_visible"),
                           "snr_abs_median": float(np.median([r["snr_abs"] for r in sel]))}
                          for lo, hi in bands for sel in [[r for r in rows if lo <= r["length_px"] < hi]] if sel]}
    if synth and SYNTH_META:
        from scipy.stats import spearmanr
        import collections
        byf = collections.defaultdict(list)
        for r in synth:
            byf[r["frame"]].append(r)
        fw = {}
        for f, v in byf.items():
            w = np.array([r["n_iso"] for r in v], float)
            if w.sum() > 20:
                fw[f] = float((w * np.array([r["iso_frac_weak_1"] or 0 for r in v])).sum() / w.sum())
        metas = [m for m in SYNTH_META if m["frame"] in fw]
        y = np.array([fw[m["frame"]] for m in metas])
        S["invisible_mode_attribution"] = {}
        for key in ("contrast", "ina", "h_base_nm", "width_sigma", "psf_sigma", "poisson_gain", "read_noise", "n_inst",
                    "bg_range_rel", "bg_median", "bg_hp_mad_rel", "field_rel"):
            x = np.array([m.get(key, np.nan) for m in metas], float)
            if np.isfinite(x).sum() > 10:
                rho, pv = spearmanr(x, y)
                S["invisible_mode_attribution"][key] = {"spearman": float(rho), "p": float(pv),
                    "median_invisible(weak>0.4)": float(np.median(x[y > 0.4])) if (y > 0.4).any() else None,
                    "median_visible(weak<0.1)": float(np.median(x[y < 0.1])) if (y < 0.1).any() else None}
        # per-instance: rendered strength vs measured weakness
        rr = [r for r in synth if r.get("render")]
        if rr:
            a = np.array([r["render"]["amp"] * r["render"]["mod_abs_mean"] for r in rr]); wk = np.array([r["frac_weak_1"] for r in rr])
            S["instance_strength_vs_weak"] = {"spearman": float(spearmanr(a, wk)[0]),
                "amp_x_mod_q_weak>0.5": q(a[wk > 0.5]) if (wk > 0.5).any() else None,
                "amp_x_mod_q_weak<0.1": q(a[wk < 0.1]) if (wk < 0.1).any() else None}
    S["real_polarity_consistency"] = q([r["polarity_consistency"] for r in real])
    S["synth_polarity_consistency"] = q([r["polarity_consistency"] for r in synth])
    S["covered_overall"] = float(np.mean([r["covered"] for r in real])) if real else None
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/roi303_eval")
    ap.add_argument("--split", default="val")
    ap.add_argument("--exclude", default="")
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--channel", type=int, default=0)
    ap.add_argument("--thr", type=float, default=0.98)
    ap.add_argument("--calib", default="calib_V6_c070_narrow.json")
    ap.add_argument("--n-synth", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.split == "test":
        raise SystemExit("refusing to touch TEST")
    real = real_side(args)
    print(f"real: {len(real)} polylines on {len({r['frame'] for r in real})} frames", flush=True)
    synth = synth_side(args) if args.n_synth > 0 else []
    print(f"synth: {len(synth)} instances on {args.n_synth} frames", flush=True)
    S = summarise(real, synth, args.thr)
    for r in real:
        r.pop("_pairs", None)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"args": vars(args), "summary": S, "real": real, "synth": synth, "synth_meta": SYNTH_META},
              open(args.out, "w"))
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
