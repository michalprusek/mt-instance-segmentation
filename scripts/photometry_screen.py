"""TRAINING-FREE photometric label-validity screen (generator <-> real benchmark).

Two statistics, both measured with ONE estimator on both sides, in the geometry the
network actually sees (real images and GT are zoomed x1.5, exactly as `eval_v5.py`
does before `predict`; synthetic frames are measured at the native generator scale
they are trained at):

  A) LABEL VALIDITY ALONG THE FILAMENT.  For every isolated filament, sample the
     perpendicular profile every 2 px and take the local signed contrast
     c(s) = core(s) - far(s).  Report
       * `flip`  = fraction of the filament's length whose sign of c disagrees with
                   the filament's dominant sign  (a rendered polarity reversal), and
       * `weak`  = fraction of its length with |c|/noise < 1 (locally undetectable,
                   yet labelled foreground).
     The metric requires 95 % length coverage, so `weak` length is length the network
     is trained to hallucinate and cannot deliver at test time.

  B) CLOSE-PAIR RESOLVABILITY.  For every pair of centerlines whose closest approach
     is 2-12 px, test the image for a saddle between them (dip below the two cores,
     in units of local noise).  This is the OVERSHOOT guard for (A): raising contrast
     until the synthetic filaments are crisper than real ones would make synthetic
     close pairs separable in a way real ones are not.

A candidate generator config is ACCEPTED only if BOTH statistics land inside the band
spanned by the three real blocks.  No ground truth is used on the synthetic side (the
generator supplies exact centerlines); real GT is used only to place the measurement
window, never to fit a parameter -- the parameter target is the real statistic itself.

Usage:
    python3 scripts/photometry_screen.py --calib <calib.json> [--override k=v ...]
    python3 scripts/photometry_screen.py --real-only
"""
from __future__ import annotations
import argparse, dataclasses, glob, json, os, sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "synth"))

UP = 1.5                                  # the scale eval_v5.py feeds the network
OFF = np.arange(-8, 8.01, 0.5)            # perpendicular sampling offsets, px
ISO_R = 15.0                              # a filament is "isolated" if no other is within this
LEN_BAND = (40.0, 90.0)                   # matched length band, px, as-seen


def _resample(p, ds=2.0):
    d = np.r_[0.0, np.cumsum(np.hypot(*(np.diff(p, axis=0).T)))]
    if d[-1] < 1e-6:
        return None, 0.0
    n = max(int(d[-1] / ds) + 1, 3)
    t = np.linspace(0, d[-1], n)
    return np.c_[np.interp(t, d, p[:, 0]), np.interp(t, d, p[:, 1])], float(d[-1])


def _isolated(qs, k, rmin=ISO_R):
    q = qs[k]
    for j, o in enumerate(qs):
        if j == k:
            continue
        if np.hypot(q[:, None, 0] - o[None, :, 0], q[:, None, 1] - o[None, :, 1]).min() < rmin:
            return False
    return True


def _norm01(img):
    im = np.asarray(img, float)
    lo, hi = np.percentile(im, [1, 99])
    return np.clip((im - lo) / max(hi - lo, 1e-9), 0, 1)


def along_stats(img, polys, len_band=LEN_BAND):
    """(A) per-filament sign-flip and sub-CNR-1 length fractions."""
    from scipy.ndimage import map_coordinates, uniform_filter1d
    im = _norm01(img)
    subs = [_resample(p, 2.0) for p in polys]
    keep = [(q, L) for q, L in subs if q is not None and len(q) >= 6]
    qs = [q for q, _ in keep]
    out = []
    for k, (q, L) in enumerate(keep):
        if not (len_band[0] <= L <= len_band[1]) or not _isolated(qs, k):
            continue
        t = np.gradient(q, axis=0)
        t /= (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
        nr = np.c_[-t[:, 1], t[:, 0]]
        pr = np.array([map_coordinates(im, [(q + o * nr)[:, 1], (q + o * nr)[:, 0]],
                                       order=1, mode="nearest") for o in OFF])
        core = pr[np.abs(OFF) <= 1.0].mean(axis=0)
        far = pr[np.abs(OFF) >= 6.0].mean(axis=0)
        noise = float(pr[np.abs(OFF) >= 6.0].std(axis=0).mean())
        c = uniform_filter1d(core - far, size=3)
        dom = np.sign(np.median(c)) or 1.0
        out.append((float(np.mean(np.sign(c) != dom)),
                    float(np.mean(np.abs(c) / max(noise, 1e-9) < 1.0))))
    return out


def pair_stats(img, polys):
    """(B) saddle depth between labelled close pairs, in units of local noise."""
    from scipy.ndimage import map_coordinates
    im = _norm01(img)
    qs = [q for q, _ in (_resample(p, 1.0) for p in polys) if q is not None]
    out = []
    for i in range(len(qs)):
        for j in range(i + 1, len(qs)):
            a, b = qs[i], qs[j]
            D = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
            k = np.unravel_index(np.argmin(D), D.shape)
            d = float(D[k])
            if not (2.0 <= d <= 12.0):
                continue
            pa, pb = a[k[0]], b[k[1]]
            t = np.linspace(-0.6, 1.6, 45)
            pts = pa[None, :] + t[:, None] * (pb - pa)[None, :]
            v = map_coordinates(im, [pts[:, 1], pts[:, 0]], order=1, mode="nearest")
            n = (pb - pa) / (np.linalg.norm(pb - pa) + 1e-9)
            perp = np.array([-n[1], n[0]])
            far = np.array([float(map_coordinates(im, [[(pa + o * perp)[1]], [(pa + o * perp)[0]]],
                                                  order=1, mode="nearest")[0])
                            for o in np.arange(-12, 12.1, 1.0)])
            nz = float(np.std(far)) + 1e-9
            ca, cb, mid = v[np.argmin(np.abs(t))], v[np.argmin(np.abs(t - 1.0))], v[np.argmin(np.abs(t - 0.5))]
            sgn = np.sign(0.5 * (ca + cb) - 0.5 * (v[0] + v[-1])) or 1.0
            out.append((d, float(sgn * (0.5 * (ca + cb) - mid) / nz)))
    return out


def report(tag, along, pairs):
    a = np.array(along) if along else np.zeros((0, 2))
    p = np.array(pairs) if pairs else np.zeros((0, 2))
    line = f"{tag:32s} n_iso={len(a):5d} "
    if len(a):
        line += (f"| flipped-length {a[:, 0].mean() * 100:5.1f}% "
                 f"| >10%-flipped filaments {np.mean(a[:, 0] > .10) * 100:5.1f}% "
                 f"| sub-CNR1 length {a[:, 1].mean() * 100:5.1f}% ")
    if len(p):
        for lo, hi in ((2, 4), (4, 6), (6, 9), (9, 12)):
            m = (p[:, 0] >= lo) & (p[:, 0] < hi)
            if m.sum() >= 3:
                line += f"| res{lo}-{hi} {np.mean(p[m, 1] >= 1.0) * 100:4.1f}% "
    print(line, flush=True)
    return dict(tag=tag, n=len(a),
                flip=float(a[:, 0].mean()) if len(a) else None,
                weak=float(a[:, 1].mean()) if len(a) else None)


def real_blocks(bench):
    import h5py
    from scipy.ndimage import zoom
    A, P = defaultdict(list), defaultdict(list)
    for f in sorted(glob.glob(os.path.join(bench, "*.h5"))):
        with h5py.File(f, "r") as h:
            src = str(h.attrs["source_task"])
            imz = zoom(_norm01(h["image"][:]), UP, order=1)
            pl = [h["polylines"][k][:] * UP for k in sorted(h["polylines"].keys())]
            A[src] += along_stats(imz, pl)
            P[src] += pair_stats(imz, pl)
    return A, P


def synth_frames(cfg, bg_dir, n, size=640, seed0=7000):
    from PIL import Image
    from mt_generator import generate_frame
    bgs = sorted(glob.glob(os.path.join(bg_dir, "*.tif")))
    if not bgs:
        raise SystemExit(f"no backgrounds in {bg_dir}")
    for i in range(n):
        rng = np.random.default_rng(seed0 + i)
        b = np.asarray(Image.open(bgs[int(rng.integers(len(bgs)))]).convert("F"), np.float32)
        while b.ndim > 2:
            b = b[..., 0]
        H, W = b.shape
        y0, x0 = int(rng.integers(0, H - size + 1)), int(rng.integers(0, W - size + 1))
        img, inst, _ = generate_frame(b[y0:y0 + size, x0:x0 + size], rng, cfg)
        pol = []
        for ins in inst:
            c = ins["centerline"]
            ok = (c[:, 0] >= 0) & (c[:, 0] < size) & (c[:, 1] >= 0) & (c[:, 1] < size)
            best, cur = [], []
            for t, f in enumerate(ok):
                if f:
                    cur.append(t)
                else:
                    best, cur = (cur if len(cur) > len(best) else best), []
            best = cur if len(cur) > len(best) else best
            if len(best) >= 2:
                pol.append(c[best])
        yield img, pol


def build_cfg_from_calib(calib_path, overrides):
    from mt_generator import GenConfig
    cal = json.load(open(calib_path))
    p = cal["best_params"]
    cfg = dataclasses.replace(
        GenConfig(),
        n_mt_range=(max(2, p["density_center"] - 6), p["density_center"] + 6),
        cluster_frac=p.get("cluster_frac", 0.0),
        contrast_range=(0.5 * p["contrast_center"], 1.5 * p["contrast_center"]),
        width_mean=p.get("width_mean", 1.3), width_std=0.35,
        curve_base=p.get("curve_base", 0.05),
        ina_range=(max(0.5, p.get("ina", 0.9) - 0.12), p.get("ina", 0.9) + 0.12),
        height_base_nm_range=(max(5.0, p.get("height_base", 40.0) - 20.0), p.get("height_base", 40.0) + 20.0),
        length_log_mean=p["length_log_mean"], length_log_sigma=p.get("length_log_sigma", 0.45),
        waviness_frame_prob=p["waviness_frame_prob"],
        waviness_amp_range=(0.6 * p["waviness_amp_center"], 1.4 * p["waviness_amp_center"]),
        detach_frame_prob=p["detach_frame_prob"], detach_prob=p["detach_prob"],
        orient_kappa_range=(0.0, p.get("orient_kappa", 0.0)),
        kink_prob=p.get("kink_prob", 0.12), short_prob=p.get("short_prob", 0.15),
        psf_sigma_range=(0.7 * p["psf_center"], 1.3 * p["psf_center"]),
        mask_half_width=1.0)
    if cal.get("gen_overrides"):
        cfg = dataclasses.replace(cfg, **cal["gen_overrides"])
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/real/mt34_htw_eval")
    ap.add_argument("--bg-dir", default="data/real/irm_backgrounds_v2")
    ap.add_argument("--calib", default=None)
    ap.add_argument("--override", action="append", default=[],
                    help="GenConfig field, e.g. height_base_nm_range=5,35 or contrast_range=0.04,0.12")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--real-only", action="store_true")
    a = ap.parse_args()

    print("REAL, as the network sees it (image and GT x1.5); isolated filaments 40-90 px:")
    A, P = real_blocks(a.bench)
    band = []
    for k in ("585", "586", "htw"):
        band.append(report(f"  real {k}", A[k], P[k]))
    lo_f = min(b["flip"] for b in band if b["flip"] is not None)
    hi_f = max(b["flip"] for b in band if b["flip"] is not None)
    lo_w = min(b["weak"] for b in band if b["weak"] is not None)
    hi_w = max(b["weak"] for b in band if b["weak"] is not None)
    print(f"\nACCEPTANCE BAND (spanned by the three real blocks): "
          f"flipped-length {lo_f*100:.1f}-{hi_f*100:.1f}%   sub-CNR1 length {lo_w*100:.1f}-{hi_w*100:.1f}%")
    if a.real_only or not a.calib:
        return

    ov = {}
    for s in a.override:
        k, v = s.split("=", 1)
        parts = [float(x) for x in v.split(",")]
        ov[k] = tuple(parts) if len(parts) > 1 else parts[0]
    cfg = build_cfg_from_calib(a.calib, ov)
    al, pr = [], []
    for img, pol in synth_frames(cfg, a.bg_dir, a.n):
        al += along_stats(img, pol)
        pr += pair_stats(img, pol)
    print()
    r = report(f"  synth {os.path.basename(a.calib)}{' +' + ';'.join(a.override) if a.override else ''}", al, pr)
    ok = (r["flip"] is not None and lo_f <= r["flip"] <= hi_f and lo_w <= r["weak"] <= hi_w)
    print(f"\nVERDICT: {'ACCEPT — inside the real band on both statistics' if ok else 'REJECT — outside the real band'}")


if __name__ == "__main__":
    main()
