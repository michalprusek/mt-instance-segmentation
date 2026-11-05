#!/usr/bin/env python3
"""Train/test SCALE and PARALLEL-GAP audit of the foreground network (measurement only).

WHAT. Two things nobody had measured about the shipped foreground model (SPARSE35,
``scripts/train_v5.py``):

A. The apparent WIDTH of a filament (FWHM of the perpendicular intensity profile), measured the
   same way on (1) synthetic training frames rendered at NATIVE resolution exactly as
   ``train_v5.OnlineDS`` renders them, (2) real roi303 VAL frames at native resolution and
   (3) the same real frames after the eval pipeline's ``zoom(img, 1.5, order=1)`` -- which is
   what the network actually sees at inference. The training target is a 3 px line
   (``ori_channels_soft`` with half_width 1). From the cached predictions the script also
   measures how wide the PREDICTED probability ridge is (band width at p > 0.5 / 0.9 / 0.99).
   Question: is the network asked to reproduce a 3 px target on filaments wider than the ones
   it was trained on, and by how much?

B. The probability profile in the GAP between two close PARALLEL microtubules (the project's
   measured worst failure: bundle recovery 0.23 vs 0.72 with an oracle mask). For every pair of
   GT polylines running parallel within 2-9 px (native) for >= 40 px, the probability is
   sampled along the line joining the two centerlines every 4 px of arc length, and compared
   with an ISOLATED filament's profile at the same distances ("virtual gap") and with the raw
   IMAGE intensity dip along the same line. Question: does the probability dip between close
   parallels at all (then decoding could separate them) or is it flat (then the network fuses
   them and only training can change it)? The image-dip column says whether the information
   was there to begin with.

C. Transpose sanity gate: the median probability at GT centerline points must be > 0.8 and the
   probability 15 px away must be low; the script REFUSES to write any prediction-derived
   output otherwise. (This project has hit the (x=col, y=row) transpose trap three times.)

WHY the measurement details are what they are:
- Profiles are sampled bilinearly, +-8 px at native (+-12 px at 1.5x -- the SAME physical
  window, otherwise the wider 1.5x tails contaminate the background estimate), local
  background = median of the outer 3 px (4.5 px at 1.5x) on each side, polarity normalised by
  the sign of the strongest excursion within +-2.5 px (IRM filaments flip dark<->bright).
- Human polylines are coarse, so the tangent is a chord over +-4 px of arc length and the peak
  is searched within +-2.5 px (native) of the GT point; the offset is reported as a free
  GT-quality number.
- Acceptance (peak SNR >= 3 against the outer-region MAD) is decided at NATIVE resolution and
  the same sample points are re-used at 1.5x, so the two real sets are paired.
- Only ISOLATED samples enter Part A: no other polyline within 12 px (native) of the sample
  point, >= 8 px from either end of the polyline, inside the eroded field of view.
- Stretches in Part B require the nearest polyline to the segment midpoint to be one of the
  pair (third-filament guard) and are counted once per unordered pair.

Runs locally for the synthetic + real-width parts; the prediction parts need the cached
SPARSE35 maps on tulen (``/disk2/prusek/mt_work/preds/SPARSE35_roi``):

    PYTHONPATH=src:synth:scripts python scripts/diag_scale_and_gap.py --part synth real
    nice -n 10 PYTHONPATH=.:src:scripts:synth:third_party python scripts/diag_scale_and_gap.py \
        --part real pred gap --pred-dir /disk2/prusek/mt_work/preds/SPARSE35_roi

Outputs JSON under ``--out`` (default data/enc_sensitivity_testset/scale_gap/).
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import binary_erosion, map_coordinates, zoom
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("src", "synth", "scripts", "third_party"):
    sys.path.insert(0, os.path.join(ROOT, _sub))

from instance.geometry import resample  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.fov import fov_mask  # noqa: E402

UP = 1.5                                   # the eval-pipeline upscale (eval_v5.UP)
TARGET_WIDTH_PX = 3.0                      # train_v5.ori_channels_soft, half_width=1 -> 3 px line
PROFILE_DS = 0.25                          # sampling step along a profile (px of the frame sampled)
NATIVE_WIN = dict(half_extent=8.0, bg_width=3.0, peak_search=2.5)
EVAL_WIN = {k: v * UP for k, v in NATIVE_WIN.items()}
STEP_A = 8.0                               # native px between profile samples (Part A)
STEP_B = 4.0                               # native px between samples along a stretch (Part B)
ISO_DIST = 12.0                            # native px: "no other polyline within"
ISO_DIST_DISPLACED = 20.0                  # native px: isolation for the 15 px displaced control
END_MARGIN = 8.0                           # native px kept clear of polyline ends
BORDER_MARGIN = 16.0                       # native px kept clear of the frame border
FOV_ERODE = 16                             # native px erosion of the field-of-view mask
MIN_STRETCH_PX = 40.0                      # native px
MAX_ANGLE_DEG = 15.0
GAP_RANGE_NATIVE = (2.0, 9.0)
GAP_BINS_EVAL = [(3.0, 4.5), (4.5, 6.0), (6.0, 7.5), (7.5, 9.0), (9.0, 13.5)]
THRS = (0.99, 0.9, 0.5)
SNR_MIN = 3.0


# ----------------------------------------------------------------------------- small helpers
def pct(x, q=(25, 50, 75)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"p{int(v)}": None for v in q} | {"n": 0}
    out = {f"p{int(v)}": float(np.percentile(x, v)) for v in q}
    out["n"] = int(x.size)
    return out


def eval_frame_shape(shape, up=UP):
    return tuple(int(round(s * up)) for s in shape)


def fold_angle_diff(a, b):
    """Unsigned difference between two UNDIRECTED tangent angles, in [0, pi/2]."""
    d = np.mod(a - b + np.pi / 2, np.pi) - np.pi / 2
    return np.abs(d)


def prep_polyline(pl, ds=1.0, chord=4):
    """Resample to ~1 px and attach a chord tangent (+-``chord`` samples). Points are (x, y)."""
    pts = resample(np.asarray(pl, float), ds=ds)
    if len(pts) < 2:
        return None
    n = len(pts)
    i0 = np.clip(np.arange(n) - chord, 0, n - 1)
    i1 = np.clip(np.arange(n) + chord, 0, n - 1)
    t = pts[i1] - pts[i0]
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    return pts, t


def sample_line(img, p, direction, offs):
    """Bilinear samples of ``img`` at ``p + offs * direction``; ``p``/``direction`` are (x, y)."""
    xs = p[0] + offs * direction[0]
    ys = p[1] + offs * direction[1]
    return map_coordinates(img, [ys, xs], order=1, mode="nearest")


def normal_of(t):
    return np.array([-t[1], t[0]])


def fwhm_estimate(d, prof, half_extent, bg_width, peak_search, snr_min=SNR_MIN):
    """Background-subtracted, polarity-normalised FWHM of one perpendicular profile.

    Returns dict(peak, snr, d0, fwhm, ok). ``ok`` is False when the peak SNR is below
    ``snr_min`` or the profile never drops to half maximum inside the window.
    """
    outer = np.abs(d) >= half_extent - bg_width
    bg = float(np.median(prof[outer]))
    sig = 1.4826 * float(np.median(np.abs(prof[outer] - bg))) + 1e-12
    q = prof - bg
    # A 3-tap (0.75 px) boxcar before the peak search: the max of a NOISY profile sits on a noise
    # spike, which raises the half-maximum level and biases FWHM low. 0.75 px broadens a 3.5 px
    # FWHM feature by ~2 %, identically for every set. The SNR is judged on the RAW profile.
    q = np.convolve(q, np.ones(3) / 3.0, mode="same")
    q[0], q[-1] = q[1], q[-2]
    idx = np.flatnonzero(np.abs(d) <= peak_search)
    i0 = idx[np.argmax(np.abs(q[idx]))]
    sgn = 1.0 if q[i0] >= 0 else -1.0
    q = q * sgn
    peak = float(q[i0])
    snr = float(np.max(np.abs(prof[idx] - bg))) / sig
    res = dict(peak=peak, snr=snr, d0=float(d[i0]), fwhm=float("nan"), ok=False,
               sign=sgn, q=q)
    if peak <= 0 or snr < snr_min:
        return res
    half = 0.5 * peak
    i = i0
    while i > 0 and q[i] >= half:
        i -= 1
    if q[i] >= half:
        return res
    left = d[i] + (half - q[i]) / (q[i + 1] - q[i]) * (d[i + 1] - d[i])
    j = i0
    while j < len(q) - 1 and q[j] >= half:
        j += 1
    if q[j] >= half:
        return res
    right = d[j - 1] + (q[j - 1] - half) / (q[j - 1] - q[j]) * (d[j] - d[j - 1])
    res.update(fwhm=float(right - left), ok=True)
    return res


def band_width(d, p, level, peak_search):
    """Width of the contiguous band with ``p > level`` around the peak nearest the centre.

    Returns (width, peak). width is NaN when the peak is below ``level`` (a miss) and +inf when
    the band reaches the window edge on either side.
    """
    idx = np.flatnonzero(np.abs(d) <= peak_search)
    i0 = idx[np.argmax(p[idx])]
    peak = float(p[i0])
    if peak <= level:
        return float("nan"), peak
    i = i0
    while i > 0 and p[i] > level:
        i -= 1
    if p[i] > level:
        return float("inf"), peak
    left = d[i] + (level - p[i]) / (p[i + 1] - p[i]) * (d[i + 1] - d[i])
    j = i0
    while j < len(p) - 1 and p[j] > level:
        j += 1
    if p[j] > level:
        return float("inf"), peak
    right = d[j - 1] + (p[j - 1] - level) / (p[j - 1] - p[j]) * (d[j] - d[j - 1])
    return float(right - left), peak


def mean_profile_fwhm(d, profiles, win):
    """FWHM of the mean of peak-aligned, peak-normalised profiles."""
    if not profiles:
        return None, None
    m = np.nanmean(np.stack(profiles), 0)
    m = np.nan_to_num(m, nan=0.0)
    r = fwhm_estimate(d, m, win["half_extent"], win["bg_width"], win["peak_search"], snr_min=0.0)
    return (r["fwhm"] if r["ok"] else None), m.tolist()


def aligned_profile(d, q, d0, align=True):
    """Background-subtracted, polarity-normalised profile scaled to peak 1; ``align`` shifts the
    found peak to d=0 (biased NARROW under noise: the noise spike always lands at 0), otherwise
    the profile stays centred on the GT point (exact for synthetic GT; broadened by the human
    GT offset, 0.5 px median, on real)."""
    q_al = np.interp(d + d0, d, q, left=np.nan, right=np.nan) if align else q.copy()
    pk = np.nanmax(q_al[np.abs(d) <= (1.0 if align else 2.5)])
    return q_al / (pk + 1e-12)


# ----------------------------------------------------------------------------- frames
class Frame:
    """One frame with prepared polylines, isolation tree and validity mask (all NATIVE px)."""

    def __init__(self, name, image, polylines, keep=None):
        self.name, self.image = name, np.asarray(image, float)
        self.H, self.W = self.image.shape
        self.polys = [q for q in (prep_polyline(pl) for pl in polylines) if q is not None]
        pts = [p for p, _ in self.polys]
        self.ids = np.concatenate([np.full(len(p), k) for k, p in enumerate(pts)]) if pts else np.zeros(0, int)
        self.allpts = np.concatenate(pts, 0) if pts else np.zeros((0, 2))
        self.tree = cKDTree(self.allpts) if len(self.allpts) else None
        if keep is None:
            keep = np.ones((self.H, self.W), bool)
        self.keep = keep

    def valid(self, p):
        """Inside the eroded field of view and away from the frame border."""
        x, y = p
        if not (BORDER_MARGIN <= x <= self.W - 1 - BORDER_MARGIN and
                BORDER_MARGIN <= y <= self.H - 1 - BORDER_MARGIN):
            return False
        return bool(self.keep[int(round(y)), int(round(x))])

    def isolated(self, p, k, r):
        """True when no point of any polyline other than ``k`` lies within ``r`` px of ``p``."""
        if self.tree is None:
            return True
        ii = self.tree.query_ball_point(p, r)
        return not ii or bool(np.all(self.ids[ii] == k))

    def samples(self, step):
        """Sample points every ``step`` px of arc, END_MARGIN clear of the ends.

        Yields (k, index_in_polyline, point(x,y), unit tangent)."""
        for k, (pts, tan) in enumerate(self.polys):
            n = len(pts)
            lo, hi = int(np.ceil(END_MARGIN)), n - 1 - int(np.ceil(END_MARGIN))
            if hi <= lo:
                continue
            for i in range(lo, hi + 1, int(step)):
                yield k, i, pts[i], tan[i]


def eroded_fov(image):
    keep = fov_mask(image)
    if keep.all():
        return keep
    return binary_erosion(keep, structure=np.ones((3, 3), bool), iterations=FOV_ERODE,
                          border_value=1)


def load_real_frames(roi_dir, split="val"):
    out = []
    for p in sorted(glob.glob(os.path.join(roi_dir, "*.h5"))):
        fr = read_frame_h5(p)
        if str(fr["attrs"].get("split")) == split and fr["polylines"]:
            out.append((os.path.basename(p), fr))
    return out


def pick_evenly(items, n):
    if n <= 0 or n >= len(items):
        return list(items)
    idx = np.unique(np.linspace(0, len(items) - 1, n).round().astype(int))
    return [items[i] for i in idx]


def load_prob(pred_dir, name, native_shape):
    a = np.load(os.path.join(pred_dir, name.replace(".h5", ".npz")))["prob"]
    if a.ndim != 3 or a.shape[0] != 1:
        raise SystemExit(f"{name}: expected a 1-channel (binary head) map, got {a.shape}")
    prob = a[0].astype(np.float32)
    if prob.shape != eval_frame_shape(native_shape):
        raise SystemExit(f"{name}: prob {prob.shape} != eval frame {eval_frame_shape(native_shape)}")
    return prob


# ----------------------------------------------------------------------------- Part A: width
def width_on_frame(frame, image, scale, win, step, accepted=None):
    """Perpendicular FWHM at isolated samples of ``frame`` measured on ``image`` (= frame.image
    scaled by ``scale``). If ``accepted`` (a set of (k, i)) is given, measure exactly those
    samples and skip the SNR gate (paired re-measurement)."""
    d = np.arange(-win["half_extent"], win["half_extent"] + 1e-9, PROFILE_DS)
    rows, profiles, profiles_un = [], [], []
    n_cand = n_iso = 0
    for k, i, p, t in frame.samples(step):
        key = (k, i)
        if accepted is not None and key not in accepted:
            continue
        if accepted is None:
            n_cand += 1
            if not frame.valid(p) or not frame.isolated(p, k, ISO_DIST):
                continue
            n_iso += 1
        prof = sample_line(image, p * scale, normal_of(t), d)
        r = fwhm_estimate(d, prof, win["half_extent"], win["bg_width"], win["peak_search"],
                          snr_min=0.0 if accepted is not None else SNR_MIN)
        rows.append(dict(k=k, i=i, fwhm=r["fwhm"], d0=r["d0"], snr=r["snr"], ok=r["ok"],
                         peak=r["peak"], sign=r["sign"]))
        if r["ok"] and np.isfinite(r["fwhm"]):
            profiles.append(aligned_profile(d, r["q"], r["d0"]))
            profiles_un.append(aligned_profile(d, r["q"], r["d0"], align=False))
    return rows, (profiles, profiles_un), d, dict(n_candidates=n_cand, n_isolated=n_iso)


def summarise_width(rows, profiles, d, win, per_frame=None):
    profiles, profiles_un = profiles
    ok = [r for r in rows if r["ok"] and np.isfinite(r["fwhm"])]
    fw = np.array([r["fwhm"] for r in ok])
    mean_fwhm, mean_prof = mean_profile_fwhm(d, profiles, win)
    mean_fwhm_un, mean_prof_un = mean_profile_fwhm(d, profiles_un, win)
    out = dict(n_samples=len(rows), n_accepted=len(ok),
               accept_frac=(len(ok) / len(rows)) if rows else None,
               fwhm=pct(fw) | {"p5": float(np.percentile(fw, 5)) if fw.size else None,
                              "p95": float(np.percentile(fw, 95)) if fw.size else None,
                              "mean": float(fw.mean()) if fw.size else None},
               fwhm_of_mean_profile=mean_fwhm,
               fwhm_of_mean_profile_unaligned=mean_fwhm_un,
               abs_peak_offset_px=pct(np.abs([r["d0"] for r in ok])),
               snr_accepted=pct([r["snr"] for r in ok]),
               fwhm_by_snr={f">={lo}": pct([r["fwhm"] for r in ok if r["snr"] >= lo]) for lo in (3, 6, 10, 20)},
               mean_profile=dict(d=d.tolist(), q=mean_prof, q_unaligned=mean_prof_un))
    if per_frame:
        med = [np.median([r["fwhm"] for r in rs if r["ok"] and np.isfinite(r["fwhm"])] or [np.nan])
               for rs in per_frame.values()]
        out["per_frame_median_fwhm"] = {k: (None if not np.isfinite(v) else float(v))
                                        for k, v in zip(per_frame.keys(), med)}
        out["per_frame_median_fwhm_pct"] = pct(med)
    return out


def analytic_synth_fwhm(cfg, n=200000, seed=0):
    """Expected FWHM from the config alone: Gaussian core sigma ws (frame-level) convolved with
    the PSF sigma -> FWHM = 2.3548 * sqrt(ws^2 + psf^2). A measured synth median far from this
    means the profile pipeline is wrong, not the generator."""
    rng = np.random.default_rng(seed)
    ws = np.clip(rng.normal(cfg.width_mean, cfg.width_std, n), *cfg.width_clip)
    psf = rng.uniform(*cfg.psf_sigma_range, n)
    return pct(2.3548 * np.sqrt(ws ** 2 + psf ** 2))


def _build_cfg_replica(p, mask_hw=None):
    """Byte-for-byte replica of tulen's ``scripts/gen_train.py::build_cfg`` (md5 aa815065...),
    which ``train_v5.py`` imports and which is not in this repo."""
    from mt_generator import GenConfig
    extra = {"mask_half_width": mask_hw} if mask_hw and mask_hw > 0 else {}
    return dataclasses.replace(
        GenConfig(),
        n_mt_range=(max(2, p["density_center"] - 6), p["density_center"] + 6),
        cluster_frac=p.get("cluster_frac", 0.0),
        contrast_range=(0.5 * p["contrast_center"], 1.5 * p["contrast_center"]),
        width_mean=p.get("width_mean", 1.3), width_std=0.35,
        curve_base=p.get("curve_base", p.get("curve_center", 0.05)),
        ina_range=(max(0.5, p.get("ina", 0.9) - 0.12), p.get("ina", 0.9) + 0.12),
        height_base_nm_range=(max(5.0, p.get("height_base", 40.0) - 20.0),
                              p.get("height_base", 40.0) + 20.0),
        length_log_mean=p["length_log_mean"], length_log_sigma=p.get("length_log_sigma", 0.45),
        waviness_frame_prob=p["waviness_frame_prob"],
        waviness_amp_range=(0.6 * p["waviness_amp_center"], 1.4 * p["waviness_amp_center"]),
        detach_frame_prob=p["detach_frame_prob"], detach_prob=p["detach_prob"],
        orient_kappa_range=(0.0, p.get("orient_kappa", 0.0)),
        kink_prob=p.get("kink_prob", 0.12), short_prob=p.get("short_prob", 0.15),
        psf_sigma_range=(0.7 * p["psf_center"], 1.3 * p["psf_center"]),
        **extra,
    )


def build_training_cfg(calib_path):
    """The config exactly as train_v5.main builds it: gen_train.build_cfg(best_params,
    mask_hw=1.0) followed by dataclasses.replace(**gen_overrides)."""
    cal = json.load(open(calib_path))
    try:
        from gen_train import build_cfg
        cfg, src = build_cfg(cal["best_params"], mask_hw=1.0), "gen_train.build_cfg"
    except ImportError:
        cfg, src = _build_cfg_replica(cal["best_params"], mask_hw=1.0), "inline replica"
    if cal.get("gen_overrides"):
        cfg = dataclasses.replace(cfg, **cal["gen_overrides"])
    return cfg, src


def find_bg_dir(explicit):
    cands = [explicit] if explicit else []
    cands += [os.path.join(ROOT, "data", "real", "irm_backgrounds_v2"),
              "/home/prusek/mt_enc_exp/irm_backgrounds_v2",              # train_v5's first choice
              "/home/prusek/BIOCEV/datasets/microtubules/IRM_backgrounds_v2"]  # its fallback
    for d in cands:
        if d and glob.glob(os.path.join(d, "*.tif")):
            return d
    raise SystemExit("no IRM backgrounds found")


def render_synth_frame(bgs, cfg, rng, g=640):
    """Bit-for-bit the background handling of train_v5.OnlineDS.__getitem__ (IRM branch)."""
    from PIL import Image
    from mt_generator import generate_frame
    bg = None
    for _ in range(8):
        b = np.asarray(Image.open(bgs[int(rng.integers(len(bgs)))]).convert("F"), np.float32)
        while b.ndim > 2:
            b = b[..., 0]
        if b.shape[0] >= g and b.shape[1] >= g:
            bg = b
            break
    if bg is None:
        bg = np.asarray(Image.fromarray(b).resize((max(g, b.shape[1]), max(g, b.shape[0]))),
                        np.float32)
    H, W = bg.shape
    y0, x0 = int(rng.integers(0, H - g + 1)), int(rng.integers(0, W - g + 1))
    bg = bg[y0:y0 + g, x0:x0 + g]
    img, inst, meta = generate_frame(bg, rng, cfg)
    return img, inst, meta


def part_synth(args):
    cfg, src = build_training_cfg(args.calib)
    bg_dir = find_bg_dir(args.bg_dir)
    bgs = sorted(glob.glob(os.path.join(bg_dir, "*.tif")))
    print(f"[synth] cfg from {src}  bgs={len(bgs)} in {bg_dir}  n_frames={args.n_synth}", flush=True)
    rows_all, profiles, per_frame = [], ([], []), {}
    d = None
    t0 = time.time()
    for i in range(args.n_synth):
        rng = np.random.default_rng(args.seed + i)
        img, inst, meta = render_synth_frame(bgs, cfg, rng)
        fr = Frame(f"synth_{i:02d}", img, [z["centerline"] for z in inst])
        rows, prof, d, info = width_on_frame(fr, fr.image, 1.0, NATIVE_WIN, STEP_A)
        rows_all += rows
        profiles[0].extend(prof[0])
        profiles[1].extend(prof[1])
        per_frame[fr.name] = rows
        print(f"  {fr.name}: {len(inst)} MTs, {info['n_candidates']} samples, {info['n_isolated']} isolated, "
              f"{sum(r['ok'] for r in rows)} accepted  regime={meta['regime']}", flush=True)
    summ = summarise_width(rows_all, profiles, d, NATIVE_WIN, per_frame)
    summ["analytic_expected_fwhm"] = analytic_synth_fwhm(cfg)
    summ["config_source"] = src
    summ["bg_dir"] = bg_dir
    summ["cfg_width"] = dict(width_mean=cfg.width_mean, width_std=cfg.width_std,
                             width_clip=list(cfg.width_clip), psf_sigma_range=list(cfg.psf_sigma_range))
    summ["target_width_px"] = TARGET_WIDTH_PX
    summ["scale"] = "native (as trained)"
    summ["n_frames"] = args.n_synth
    print(f"[synth] FWHM native px: {summ['fwhm']}  mean-profile FWHM {summ['fwhm_of_mean_profile']} "
          f"(unaligned {summ['fwhm_of_mean_profile_unaligned']})  "
          f"analytic {summ['analytic_expected_fwhm']}  ({time.time()-t0:.0f}s)", flush=True)
    return summ


def real_frames_for_width(args):
    frames = load_real_frames(args.roi)
    return pick_evenly(frames, args.n_real)


def build_frame(name, fr):
    return Frame(name, fr["image"], fr["polylines"], keep=eroded_fov(fr["image"]))


def part_real(args, frames=None):
    frames = frames or real_frames_for_width(args)
    print(f"[real] {len(frames)} roi303 VAL frames", flush=True)
    nat_rows, nat_prof, ev_rows, ev_prof = [], ([], []), [], ([], [])
    nat_pf, ev_pf, ratios = {}, {}, []
    d_nat = d_ev = None
    accepted_by_frame = {}
    t0 = time.time()
    for name, fr in frames:
        f = build_frame(name, fr)
        rows, prof, d_nat, info = width_on_frame(f, f.image, 1.0, NATIVE_WIN, STEP_A)
        acc = {(r["k"], r["i"]) for r in rows if r["ok"] and np.isfinite(r["fwhm"])}
        accepted_by_frame[name] = acc
        imz = zoom(f.image, UP, order=1)
        rows_e, prof_e, d_ev, _ = width_on_frame(f, imz, UP, EVAL_WIN, STEP_A, accepted=acc)
        # paired ratio 1.5x / native on identical sample points
        by_key = {(r["k"], r["i"]): r["fwhm"] for r in rows_e if r["ok"] and np.isfinite(r["fwhm"])}
        for r in rows:
            key = (r["k"], r["i"])
            if key in acc and key in by_key:
                ratios.append(by_key[key] / r["fwhm"])
        nat_rows += rows
        ev_rows += rows_e
        for dst, src_ in ((nat_prof, prof), (ev_prof, prof_e)):
            dst[0].extend(src_[0])
            dst[1].extend(src_[1])
        nat_pf[name], ev_pf[name] = rows, rows_e
        print(f"  {name} {f.H}x{f.W}: {len(f.polys)} curves, {info['n_candidates']} samples, "
              f"{info['n_isolated']} isolated, {len(acc)} accepted, fov {f.keep.mean():.2f}", flush=True)
    nat = summarise_width(nat_rows, nat_prof, d_nat, NATIVE_WIN, nat_pf)
    ev = summarise_width(ev_rows, ev_prof, d_ev, EVAL_WIN, ev_pf)
    nat["scale"], ev["scale"] = "native 1x", "eval 1.5x (zoom order=1), same samples"
    out = dict(n_frames=len(frames), frames=[n for n, _ in frames], native=nat, eval15=ev,
               paired_ratio_eval_over_native=pct(ratios) | {"mean": float(np.mean(ratios)) if ratios else None},
               target_width_px=TARGET_WIDTH_PX)
    print(f"[real] native FWHM {nat['fwhm']}\n[real] 1.5x   FWHM {ev['fwhm']}\n"
          f"[real] paired ratio {out['paired_ratio_eval_over_native']}  ({time.time()-t0:.0f}s)", flush=True)
    return out, accepted_by_frame


# ----------------------------------------------------------------------------- Part C: sanity
def sanity_check(frames, pred_dir):
    """Transpose gate. Median prob at exact GT points (eval frame, [row=y*1.5, col=x*1.5]) must be
    > 0.8; the same points under the TRANSPOSED convention and 15 px (eval) off the filament must
    be low. Returns the numbers; the caller refuses to continue on failure."""
    exact, ridge, transposed, displaced = [], [], [], []
    per_frame = {}
    for name, fr in frames:
        f = build_frame(name, fr)
        prob = load_prob(pred_dir, name, f.image.shape)
        Hs, Ws = prob.shape
        ex, tr, di, ri = [], [], [], []
        for k, (pts, tan) in enumerate(f.polys):
            for i in range(0, len(pts), 2):
                p = pts[i]
                if not f.valid(p):
                    continue
                x, y = p * UP
                ex.append(prob[int(round(y)), int(round(x))])
                offs = np.arange(-2 * UP, 2 * UP + 1e-9, PROFILE_DS)
                ri.append(sample_line(prob, p * UP, normal_of(tan[i]), offs).max())
                xi, yi = int(round(x)), int(round(y))
                if 0 <= xi < Hs and 0 <= yi < Ws:
                    tr.append(prob[xi, yi])              # deliberately TRANSPOSED control
                if f.isolated(p, k, ISO_DIST_DISPLACED):
                    for s in (-15.0, 15.0):
                        di.append(sample_line(prob, p * UP, normal_of(tan[i]), np.array([s]))[0])
        per_frame[name] = dict(exact=float(np.median(ex)) if ex else None,
                               displaced=float(np.median(di)) if di else None)
        exact += ex
        ridge += ri
        transposed += tr
        displaced += di
    res = dict(n_points=len(exact),
               median_prob_at_gt=float(np.median(exact)),
               mean_prob_at_gt=float(np.mean(exact)),
               median_max_prob_within_2px=float(np.median(ridge)),
               median_prob_transposed=float(np.median(transposed)) if transposed else None,
               mean_prob_transposed=float(np.mean(transposed)) if transposed else None,
               n_displaced=len(displaced),
               median_prob_displaced_15px=float(np.median(displaced)) if displaced else None,
               mean_prob_displaced_15px=float(np.mean(displaced)) if displaced else None,
               per_frame=per_frame)
    res["pass"] = bool(res["median_prob_at_gt"] > 0.8 and
                       (res["median_prob_displaced_15px"] is not None and res["median_prob_displaced_15px"] < 0.2) and
                       (res["median_prob_transposed"] is None or
                        res["median_prob_transposed"] < 0.5 * res["median_prob_at_gt"]))
    print(f"[sanity] n={res['n_points']}  median p@GT={res['median_prob_at_gt']:.3f} "
          f"(mean {res['mean_prob_at_gt']:.3f})  median max-p within 2px={res['median_max_prob_within_2px']:.3f}  "
          f"transposed median={res['median_prob_transposed']}  displaced-15px median="
          f"{res['median_prob_displaced_15px']} (mean {res['mean_prob_displaced_15px']}, n={res['n_displaced']})  "
          f"PASS={res['pass']}", flush=True)
    return res


# ----------------------------------------------------------------------------- Part A: pred ridge
def part_pred(args, frames, accepted_by_frame):
    """Width of the predicted probability ridge across the SAME isolated, accepted samples."""
    levels = (0.5, 0.9, 0.99)
    d = np.arange(-EVAL_WIN["half_extent"], EVAL_WIN["half_extent"] + 1e-9, PROFILE_DS)
    widths = {lv: [] for lv in levels}
    unbounded = {lv: 0 for lv in levels}
    miss = {lv: 0 for lv in levels}
    peaks, offsets, curves = [], [], []
    n = 0
    for name, fr in frames:
        f = build_frame(name, fr)
        prob = load_prob(args.pred_dir, name, f.image.shape)
        acc = accepted_by_frame[name]
        for k, i, p, t in f.samples(STEP_A):
            if (k, i) not in acc:
                continue
            prof = sample_line(prob, p * UP, normal_of(t), d)
            n += 1
            idx = np.flatnonzero(np.abs(d) <= EVAL_WIN["peak_search"])
            i0 = idx[np.argmax(prof[idx])]
            peaks.append(float(prof[i0]))
            offsets.append(abs(float(d[i0])))
            # profile as a function of |d| from the ridge peak, both sides averaged
            al = np.interp(d + d[i0], d, prof, left=np.nan, right=np.nan)
            half = len(d) // 2
            curves.append(np.nanmean(np.stack([al[half:], al[half::-1]]), 0))
            for lv in levels:
                w, _ = band_width(d, prof, lv, EVAL_WIN["peak_search"])
                if np.isnan(w):
                    miss[lv] += 1
                elif np.isinf(w):
                    unbounded[lv] += 1
                else:
                    widths[lv].append(w)
    curve = np.nanmedian(np.stack(curves), 0) if curves else None
    curve_mean = np.nanmean(np.stack(curves), 0) if curves else None
    out = dict(n_samples=n, peak_prob=pct(peaks) | {"frac_ge_0.99": float(np.mean(np.array(peaks) >= 0.99)) if peaks else None,
                                                    "frac_ge_0.5": float(np.mean(np.array(peaks) >= 0.5)) if peaks else None},
               abs_peak_offset_px_eval=pct(offsets),
               band_width_px_eval={str(lv): pct(widths[lv]) | {"miss_frac": miss[lv] / n if n else None,
                                                             "unbounded": unbounded[lv]} for lv in levels},
               prob_vs_distance_eval_px=dict(d=d[len(d) // 2:].tolist(),
                                             median=None if curve is None else curve.tolist(),
                                             mean=None if curve_mean is None else curve_mean.tolist()),
               window="eval frame, +-12 px, peak searched within +-3.75 px of the GT point")
    for lv in levels:
        print(f"[pred] band p>{lv}: {out['band_width_px_eval'][str(lv)]}", flush=True)
    print(f"[pred] peak prob {out['peak_prob']}", flush=True)
    return out


# ----------------------------------------------------------------------------- Part B: gaps
def line_stats(prob, imz, a, b):
    """Probability and image-intensity statistics along the eval-frame segment a -> b (x, y)."""
    v = b - a
    L = float(np.linalg.norm(v))
    u = v / max(L, 1e-9)
    ts = np.arange(0.0, L, PROFILE_DS)
    ts = np.append(ts, L)
    p = sample_line(prob, a, u, ts)
    p_mid = float(np.interp(L / 2, ts, p))
    ext = np.arange(-2.0, 2.0 + 1e-9, PROFILE_DS)
    pa_max = float(sample_line(prob, a, u, ext).max())
    pb_max = float(sample_line(prob, b, u, ext).max())
    ti = np.arange(-8.0, L + 8.0 + 1e-9, PROFILE_DS)
    I = sample_line(imz, a, u, ti)
    outer = (ti <= -4.0) | (ti >= L + 4.0)
    bg = float(np.median(I[outer]))
    J = np.abs(I - bg)
    A_a = float(J[(ti >= -1.5) & (ti <= 1.5)].max())
    A_b = float(J[(ti >= L - 1.5) & (ti <= L + 1.5)].max())
    ref = min(A_a, A_b)
    interior = (ti > 1.5) & (ti < L - 1.5)
    j_min = float(J[interior].min()) if interior.any() else float(np.interp(L / 2, ti, J))
    j_mid = float(np.interp(L / 2, ti, J))
    return dict(L=L, p_a=float(p[0]), p_b=float(p[-1]), p_mid=p_mid, p_min=float(p.min()),
                pa_max=pa_max, pb_max=pb_max,
                dip_min=float(np.clip(1 - j_min / ref, -1, 1)) if ref > 0 else float("nan"),
                dip_mid=float(np.clip(1 - j_mid / ref, -1, 1)) if ref > 0 else float("nan"),
                same_sign=bool(np.sign(np.interp(0.0, ti, I) - bg) == np.sign(np.interp(L, ti, I) - bg)))


def runs_of(mask, min_len):
    """Start/end index pairs of consecutive True runs of length >= min_len."""
    out, start = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    return out


def bin_of(gap_eval):
    for lo, hi in GAP_BINS_EVAL:
        if lo <= gap_eval < hi or (hi == GAP_BINS_EVAL[-1][1] and gap_eval == hi):
            return f"{lo}-{hi}"
    return None


def part_gap(args, frames):
    min_samples = int(MIN_STRETCH_PX / STEP_B) + 1
    max_ang = np.deg2rad(MAX_ANGLE_DEG)
    stretches, iso_runs = [], []
    near_hist = np.zeros(12, int)          # nearest-other distance of near-parallel samples, native px bins
    n_pairs_checked = n_guard_dropped = 0
    iso_d = np.arange(-21.5, 21.5 + 1e-9, PROFILE_DS)
    half = len(iso_d) // 2
    curve_d = iso_d[half:]
    iso_curves_p, iso_curves_j = [], []
    t0 = time.time()
    for fi, (name, fr) in enumerate(frames):
        f = build_frame(name, fr)
        prob = load_prob(args.pred_dir, name, f.image.shape)
        imz = zoom(f.image, UP, order=1)
        polys = f.polys
        trees = [cKDTree(p) for p, _ in polys]
        bbox = [(p.min(0), p.max(0)) for p, _ in polys]
        angs = [np.arctan2(t[:, 1], t[:, 0]) for _, t in polys]
        # sample indices per polyline (END_MARGIN clear)
        sidx = []
        for pts, _ in polys:
            n = len(pts)
            lo, hi = int(np.ceil(END_MARGIN)), n - 1 - int(np.ceil(END_MARGIN))
            sidx.append(np.arange(lo, hi + 1, int(STEP_B)) if hi > lo else np.zeros(0, int))
        # ---- isolated runs (virtual gaps)
        for k, (pts, tan) in enumerate(polys):
            si = sidx[k]
            if si.size == 0:
                continue
            iso = np.array([f.valid(pts[i]) and f.isolated(pts[i], k, ISO_DIST) for i in si])
            for s, e in runs_of(iso, min_samples):
                recs = []
                for i in si[s:e]:
                    p, t = pts[i] * UP, normal_of(tan[i])
                    pp = sample_line(prob, p, t, iso_d)
                    I = sample_line(imz, p, t, iso_d)
                    bg = float(np.median(I[np.abs(iso_d) >= 13.5]))
                    J = np.abs(I - bg)
                    A0 = float(J[np.abs(iso_d) <= 1.5].max())
                    # centre the profile on the prob ridge (within +-3.75 px) like the pairs use GT points:
                    # here the GT point itself is the "centerline" -- keep exact for comparability with p_a
                    rec = dict(p0=float(pp[half]))
                    for side in (1, -1):
                        dd = iso_d[half::side] if side == 1 else -iso_d[half::-1]
                        pv = pp[half::side] if side == 1 else pp[half::-1]
                        jv = J[half::side] if side == 1 else J[half::-1]
                        for lo, hi in GAP_BINS_EVAL:
                            dc = 0.5 * (lo + hi)
                            key = f"{lo}-{hi}"
                            inner = (dd > 0) & (dd <= dc)
                            rec.setdefault(key, []).append(dict(
                                p_mid=float(np.interp(dc / 2, dd, pv)),
                                p_min=float(pv[inner].min()),
                                p_far=float(np.interp(dc, dd, pv)),
                                dip_min=float(np.clip(1 - jv[(dd > 1.5) & (dd <= dc)].min() / A0, -1, 1)) if A0 > 0 else float("nan"),
                                dip_mid=float(np.clip(1 - np.interp(dc / 2, dd, jv) / A0, -1, 1)) if A0 > 0 else float("nan")))
                    iso_curves_p.append(0.5 * (pp[half:] + pp[half::-1]))
                    iso_curves_j.append(0.5 * (J[half:] + J[half::-1]) / (A0 + 1e-12))
                    recs.append(rec)
                run = dict(frame=name, k=int(k), n=len(recs), p0=float(np.median([r["p0"] for r in recs])), bins={})
                for lo, hi in GAP_BINS_EVAL:
                    key = f"{lo}-{hi}"
                    vals = [v for r in recs for v in r[key]]
                    run["bins"][key] = {q: float(np.nanmedian([v[q] for v in vals]))
                                       for q in ("p_mid", "p_min", "p_far", "dip_min", "dip_mid")}
                    run["bins"][key].update({f"frac_min_lt_{thr}": float(np.mean([v["p_min"] < thr for v in vals]))
                                             for thr in THRS})
                iso_runs.append(run)
        # ---- parallel pairs
        for k in range(len(polys)):
            si = sidx[k]
            if si.size == 0:
                continue
            pts_k, tan_k = polys[k]
            for m in range(k + 1, len(polys)):
                if (bbox[k][0] > bbox[m][1] + GAP_RANGE_NATIVE[1]).any() or \
                   (bbox[m][0] > bbox[k][1] + GAP_RANGE_NATIVE[1]).any():
                    continue
                n_pairs_checked += 1
                dist, j = trees[m].query(pts_k[si])
                ang = fold_angle_diff(angs[k][si], angs[m][j])
                near_par = ang < max_ang
                for dd_ in dist[near_par]:
                    if dd_ < len(near_hist):
                        near_hist[int(dd_)] += 1
                ok = near_par & (dist >= GAP_RANGE_NATIVE[0]) & (dist <= GAP_RANGE_NATIVE[1])
                for q in np.flatnonzero(ok):
                    a, b = pts_k[si[q]], polys[m][0][j[q]]
                    if not (f.valid(a) and f.valid(b)):
                        ok[q] = False
                        continue
                    mid = 0.5 * (a + b)
                    ids = set(f.ids[f.tree.query_ball_point(mid, 0.5 * dist[q] * 0.999)])
                    if ids - {k, m}:
                        ok[q] = False
                        n_guard_dropped += 1
                for s, e in runs_of(ok, min_samples):
                    recs = []
                    for q in range(s, e):
                        a, b = pts_k[si[q]] * UP, polys[m][0][j[q]] * UP
                        r = line_stats(prob, imz, a, b)
                        r["gap_eval"] = float(dist[q] * UP)
                        recs.append(r)
                    st = dict(frame=name, a=int(k), b=int(m), n=len(recs),
                              length_native=float((len(recs) - 1) * STEP_B),
                              gap_eval=float(np.median([r["gap_eval"] for r in recs])),
                              gap_eval_min=float(np.min([r["gap_eval"] for r in recs])),
                              gap_eval_max=float(np.max([r["gap_eval"] for r in recs])),
                              p_center=float(np.median([v for r in recs for v in (r["p_a"], r["p_b"])])),
                              p_center_max=float(np.median([v for r in recs for v in (r["pa_max"], r["pb_max"])])),
                              p_mid=float(np.median([r["p_mid"] for r in recs])),
                              p_min=float(np.median([r["p_min"] for r in recs])),
                              dip_min=float(np.nanmedian([r["dip_min"] for r in recs])),
                              dip_mid=float(np.nanmedian([r["dip_mid"] for r in recs])),
                              same_sign_frac=float(np.mean([r["same_sign"] for r in recs])))
                    for thr in THRS:
                        st[f"frac_min_lt_{thr}"] = float(np.mean([r["p_min"] < thr for r in recs]))
                    stretches.append(st)
        if (fi + 1) % 10 == 0 or fi == len(frames) - 1:
            print(f"  [{fi+1}/{len(frames)}] {name}: stretches so far {len(stretches)}, "
                  f"isolated runs {len(iso_runs)}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- aggregate per bin
    def agg_pairs(sts):
        out = {}
        for lo, hi in GAP_BINS_EVAL:
            key = f"{lo}-{hi}"
            b = [s for s in sts if bin_of(s["gap_eval"]) == key]
            row = dict(n_stretches=len(b), n_samples=int(sum(s["n"] for s in b)),
                       total_length_native_px=float(sum(s["length_native"] for s in b)),
                       n_frames=len({s["frame"] for s in b}))
            if b:
                for q in ("p_center", "p_center_max", "p_mid", "p_min", "dip_min", "dip_mid"):
                    row[f"median_{q}"] = float(np.nanmedian([s[q] for s in b]))
                    row[f"p25_{q}"] = float(np.nanpercentile([s[q] for s in b], 25))
                    row[f"p75_{q}"] = float(np.nanpercentile([s[q] for s in b], 75))
                for thr in THRS:
                    row[f"frac_stretches_min_lt_{thr}"] = float(np.mean([s["p_min"] < thr for s in b]))
                    row[f"mean_frac_length_min_lt_{thr}"] = float(np.mean([s[f"frac_min_lt_{thr}"] for s in b]))
                dip = np.array([s["dip_min"] for s in b])
                pm = np.array([s["p_min"] for s in b])
                row["frac_image_dips_gt_0.5"] = float(np.nanmean(dip > 0.5))
                row["frac_image_dips_and_prob_flat_0.99"] = float(np.nanmean((dip > 0.5) & (pm >= 0.99)))
                row["frac_image_dips_and_prob_dips_0.99"] = float(np.nanmean((dip > 0.5) & (pm < 0.99)))
                row["frac_prob_dips_0.99_given_image_dips"] = float(np.mean(pm[dip > 0.5] < 0.99)) if (dip > 0.5).any() else None
                row["frac_same_sign"] = float(np.mean([s["same_sign_frac"] >= 0.5 for s in b]))
            out[key] = row
        return out

    def agg_iso(runs):
        out = {}
        for lo, hi in GAP_BINS_EVAL:
            key = f"{lo}-{hi}"
            row = dict(n_runs=len(runs), n_samples=int(sum(r["n"] for r in runs)),
                       median_p_center=float(np.median([r["p0"] for r in runs])) if runs else None)
            if runs:
                for q in ("p_mid", "p_min", "p_far", "dip_min", "dip_mid"):
                    row[f"median_{q}"] = float(np.nanmedian([r["bins"][key][q] for r in runs]))
                for thr in THRS:
                    row[f"frac_runs_min_lt_{thr}"] = float(np.mean([r["bins"][key]["p_min"] < thr for r in runs]))
                    row[f"mean_frac_length_min_lt_{thr}"] = float(np.mean([r["bins"][key][f"frac_min_lt_{thr}"] for r in runs]))
            out[key] = row
        return out

    curve_p = np.stack(iso_curves_p) if iso_curves_p else None
    curve_j = np.stack(iso_curves_j) if iso_curves_j else None
    res = dict(n_frames=len(frames), n_pairs_checked=n_pairs_checked, n_guard_dropped=n_guard_dropped,
               n_stretches=len(stretches), n_isolated_runs=len(iso_runs),
               near_parallel_nearest_other_hist_native_px={f"{i}-{i+1}": int(v) for i, v in enumerate(near_hist)},
               pairs_by_gap_bin_eval=agg_pairs(stretches),
               isolated_virtual_gap_by_bin_eval=agg_iso(iso_runs),
               isolated_profile_vs_distance_eval_px=dict(
                   d=curve_d.tolist(),
                   prob_median=None if curve_p is None else np.median(curve_p, 0).tolist(),
                   prob_mean=None if curve_p is None else curve_p.mean(0).tolist(),
                   prob_p25=None if curve_p is None else np.percentile(curve_p, 25, 0).tolist(),
                   prob_p75=None if curve_p is None else np.percentile(curve_p, 75, 0).tolist(),
                   image_abs_contrast_norm_median=None if curve_j is None else np.median(curve_j, 0).tolist()),
               stretches=stretches,
               method=dict(step_native=STEP_B, min_stretch_native=MIN_STRETCH_PX, max_angle_deg=MAX_ANGLE_DEG,
                           gap_range_native=list(GAP_RANGE_NATIVE), iso_dist_native=ISO_DIST,
                           bins_eval=[list(b) for b in GAP_BINS_EVAL]))
    print("[gap] pairs by gap bin (eval px):", flush=True)
    for key, row in res["pairs_by_gap_bin_eval"].items():
        if row["n_stretches"]:
            print(f"  {key:>9}: n={row['n_stretches']:4d}  p_center={row['median_p_center']:.3f}  "
                  f"p_mid={row['median_p_mid']:.3f}  p_min={row['median_p_min']:.3f}  "
                  f"<0.99: {row['frac_stretches_min_lt_0.99']:.2f}  <0.9: {row['frac_stretches_min_lt_0.9']:.2f}  "
                  f"<0.5: {row['frac_stretches_min_lt_0.5']:.2f}  img-dip {row['median_dip_min']:.2f}", flush=True)
        else:
            print(f"  {key:>9}: n=0", flush=True)
    print("[gap] isolated virtual gaps:", flush=True)
    for key, row in res["isolated_virtual_gap_by_bin_eval"].items():
        if row["n_runs"]:
            print(f"  {key:>9}: n={row['n_runs']:4d}  p_center={row['median_p_center']:.3f}  "
                  f"p_mid={row['median_p_mid']:.3f}  p_min={row['median_p_min']:.3f}  "
                  f"<0.99: {row['frac_runs_min_lt_0.99']:.2f}  <0.9: {row['frac_runs_min_lt_0.9']:.2f}  "
                  f"<0.5: {row['frac_runs_min_lt_0.5']:.2f}  img-dip {row['median_dip_min']:.2f}", flush=True)
    return res


# ----------------------------------------------------------------------------- main
def dump(obj, path):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, float) and not np.isfinite(o):
            return None
        raise TypeError(str(type(o)))
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=conv)
    print(f"wrote {path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", nargs="+", default=["synth", "real"],
                    choices=["synth", "real", "pred", "gap"],
                    help="synth/real run anywhere; pred/gap need --pred-dir (tulen)")
    ap.add_argument("--roi", default=os.path.join(ROOT, "data", "real", "roi303_eval"))
    ap.add_argument("--pred-dir", default="/disk2/prusek/mt_work/preds/SPARSE35_roi")
    ap.add_argument("--calib", default=os.path.join(ROOT, "calib_V6_c070_narrow.json"))
    ap.add_argument("--bg-dir", default=None, help="IRM backgrounds; default = train_v5's search order")
    ap.add_argument("--n-synth", type=int, default=20)
    ap.add_argument("--n-real", type=int, default=30, help="roi303 VAL frames for the width parts")
    ap.add_argument("--gap-frames", type=int, default=0, help="0 = all roi303 VAL frames")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "enc_sensitivity_testset", "scale_gap"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    parts = set(args.part)

    if "synth" in parts:
        dump(part_synth(args), os.path.join(args.out, "synth_width.json"))

    width_frames = accepted = None
    if parts & {"real", "pred"}:
        width_frames = real_frames_for_width(args)
        real, accepted = part_real(args, width_frames)
        if "real" in parts:
            dump(real, os.path.join(args.out, "real_width.json"))

    if parts & {"pred", "gap"}:
        gap_frames = load_real_frames(args.roi)
        if args.gap_frames:
            gap_frames = pick_evenly(gap_frames, args.gap_frames)
        san_frames = gap_frames if "gap" in parts else width_frames
        sanity = sanity_check(san_frames, args.pred_dir)
        dump(sanity, os.path.join(args.out, "sanity.json"))
        if not sanity["pass"]:
            print("SANITY GATE FAILED -- refusing to report prediction-derived numbers. "
                  "Median prob at exact GT points, at the ridge within 2 px, and under the transposed "
                  "convention are printed above; if the ridge value passes and the transposed one is "
                  "near zero, this is a GT-offset finding, not a transpose bug -- rule on it before "
                  "relaxing anything.", flush=True)
            sys.exit(2)
        if "pred" in parts:
            dump(part_pred(args, width_frames, accepted), os.path.join(args.out, "pred_ridge.json"))
        if "gap" in parts:
            dump(part_gap(args, gap_frames), os.path.join(args.out, "gap.json"))


if __name__ == "__main__":
    main()
