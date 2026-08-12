"""Synthetic microtubule generator — MORPHOLOGY (modality-agnostic) + per-modality RENDER (IRM / TIRF).

Factorization (the paper's cross-modality argument): `sample_scene()` produces the microtubule GEOMETRY
(centerlines + instance masks) — the microtubules themselves, identical across imaging modalities — and a
separate RENDERER paints appearance: `render_irm()` (dark filaments, multiplicative, interference halo,
bright detachment) or `render_tirf()` (bright fluorescent filaments on a dark background, additive, no
halo, detachment DIMS). Calibrate the morphology once (on IRM, where we have data + the co-registered
TIRF is the SAME physical MTs), then reuse those morphology params for TIRF and recalibrate only the
appearance. `generate_frame(bg, rng, cfg, modality="irm"|"tirf")` ties them together (default irm =
backward compatible).

Conventions: centerline points are (x=col, y=row) float arrays, matching the project's .h5 GT.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d


# ----------------------------------------------------------------------------- config
@dataclass
class GenConfig:
    # =========================== MORPHOLOGY (modality-agnostic — the microtubules) ===========================
    # --- density / placement
    n_mt_range: Tuple[int, int] = (8, 22)            # microtubules per frame
    cluster_frac: float = 0.0                        # 0 = uniform; ->1 = few tight clusters (Thomas process)
    orient_kappa_range: Tuple[float, float] = (0.0, 3.0)  # NEW frame-level orientation ANISOTROPY (von Mises
    #                                                       concentration): 0 = isotropic, high = nematic/aligned
    # --- parallel bundles (irregular so they aren't perfectly regular)
    bundle_prob: float = 0.20
    bundle_size_range: Tuple[int, int] = (2, 3)
    bundle_gap_range: Tuple[float, float] = (3.0, 9.0)
    bundle_gap_std: float = 0.45
    bundle_angle_div: float = 0.05
    bundle_stagger: float = 0.35
    # --- length (lognormal) + NEW bimodality: a population of SHORT dynamic seeds
    length_log_mean: float = 5.9                     # exp(5.9) ~ 365 px
    length_log_sigma: float = 0.45
    length_range: Tuple[float, float] = (80.0, 950.0)
    short_prob: float = 0.15                         # NEW fraction of MTs that are short seeds
    short_length_log_mean: float = 4.7               # NEW exp(4.7) ~ 110 px
    step_px: float = 1.0
    # --- bending: LENGTH-COUPLED stiffness (Pampaloni PNAS 2006: persistence length RISES with contour
    #     length => curvature = lateral-dev/length FALLS with length; broad per-MT log-normal scatter =>
    #     short MTs wavier, long ones near-straight, from ONE physical rule). Replaces a global curvature.
    curve_base: float = 0.05                          # mean curvature (lateral dev/length) at the ref length
    curve_len_ref: float = 300.0                      # px reference contour length
    curve_len_exp: float = 0.7                        # how strongly curvature falls with length (stiffness ~ L^exp)
    curve_log_std: float = 0.55                       # per-MT log-normal scatter (the broad Lp variance)
    curve_frac_range: Tuple[float, float] = (0.004, 0.16)  # clamp bounds for the resulting per-MT curvature
    arc_bias_range: Tuple[float, float] = (0.0, 0.5)
    smooth_sigma: float = 2.0
    # --- waviness = FRAME-LEVEL condition (all MTs of a wavy frame share the regime)
    waviness_frame_prob: float = 0.35
    waviness_amp_range: Tuple[float, float] = (1.5, 9.0)
    waviness_wavelength_range: Tuple[float, float] = (120.0, 350.0)
    waviness_amp_jitter: float = 0.4
    # --- MULTI-REGIME morphology (research: span the FULL in-vitro variability across assay types) ---
    #   Each frame samples a REGIME. STATIC/stabilized = the calibrated stiff, near-straight WLC (dominant, matches
    #   surface-immobilized / taxol Alice-like data). GLIDING = motor-propelled, ~10x WIGGLIER via an effective PATH
    #   persistence length (0.1-0.5mm; Sci Rep 2022 s41598-022-06941-x / Hess lab) + occasional tight arcs/rings
    #   (Liu-Tüzel-Ross 2011). DYNAMIC = dynamic-instability with an EXPONENTIAL (broad, mixed grow/shrink) length.
    regime_gliding_prob: float = 0.38                # fraction of frames in the WIGGLY gliding regime
    regime_dynamic_prob: float = 0.18                # fraction in the dynamic-instability regime (exp length)
    lp_gliding_px_range: Tuple[float, float] = (1000.0, 5000.0)   # gliding effective path Lp (px @ ~100nm/px)
    lp_log_std: float = 0.4                          # per-MT Lp scatter (log-normal)
    ring_prob: float = 0.15                          # per-MT chance (gliding) of a tight arc / loop / ring / spool
    ring_radius_px_range: Tuple[float, float] = (6.0, 26.0)       # 0.6-2.6 um ring radius (avg ~1um)
    dynamic_length_mean_px: float = 320.0            # exponential steady-state length mean (dynamic regime)
    # --- sharp bends / HAIRPINS (smooth up-to-180deg reversals): frame-gated + per-MT
    hairpin_frame_prob: float = 0.30
    hairpin_prob: float = 0.15
    n_bend_range: Tuple[int, int] = (1, 2)
    bend_angle_range: Tuple[float, float] = (1.2, 3.05)
    bend_width_range: Tuple[float, float] = (22.0, 60.0)
    # --- NEW KINKS / lattice defects: localized SHARP small-angle bends (still smooth, narrow window),
    #     distinct from the large smooth hairpins. Per-MT.
    kink_prob: float = 0.12
    n_kink_range: Tuple[int, int] = (1, 3)
    kink_angle_range: Tuple[float, float] = (0.15, 0.6)   # rad (~9deg .. ~34deg)
    kink_width_range: Tuple[float, float] = (3.0, 10.0)   # px arc length (narrow => sharp but finite)

    # =========================== IRM APPEARANCE ===========================
    width_mean: float = 1.3                          # core half-width MEAN (px) — FRAME-level (PSF/resolution-set)
    width_std: float = 0.35                          # FRAME-to-frame width spread (cross-microscope resolution)
    width_clip: Tuple[float, float] = (0.7, 2.4)
    contrast_range: Tuple[float, float] = (0.025, 0.09)   # interference amplitude A (|contrast| at a fringe)
    contrast_rel_std: float = 0.3                         # per-MT amplitude variation (reflectivity spread)
    contrast_floor: float = 0.22                          # min |interference contrast| so a LABELED MT is never
    #                                                       fully invisible at the height zero-crossing (else the
    #                                                       mask supervises signal-free pixels → over-firing; PR fix)
    # ---- IRM TWO-BEAM INTERFERENCE (Simmert 2018; Mahamdeh & Howard 2018/19): I = B + A·cos(2k·h)·E_INA(h),
    #      k = 2π·n_water/λ. MT height h [nm] above the coverslip sets contrast SIGN: DARK at contact (π shift),
    #      BRIGHT once elevated past the ~λ/(8n)≈56 nm zero-crossing; INA envelope decays higher fringes. ----
    wavelength_nm: float = 600.0                          # illumination wavelength (nm)
    n_water: float = 1.33                                 # water index sets height→phase (NOT tubulin)
    ina_range: Tuple[float, float] = (0.7, 1.1)           # FRAME-level illumination NA (strongest contrast knob)
    height_base_nm_range: Tuple[float, float] = (20.0, 90.0)    # FRAME-level base height regime (dark ⇄ brighter MTs)
    height_along_std_nm: float = 15.0                     # low-freq height variation along the MT (thermal/attach)
    height_detach_nm_range: Tuple[float, float] = (60.0, 160.0) # DETACHED segment height → crosses into BRIGHT
    # NEW appearance diversity: tapered tips + along-length intensity heterogeneity
    tip_taper_prob: float = 0.4                      # NEW chance an MT has tapered (dimmer) ends
    tip_taper_frac_range: Tuple[float, float] = (0.03, 0.15)  # NEW taper length as frac of MT length
    along_intensity_std: float = 0.15                # NEW per-MT low-freq intensity variation along length
    # dirt / debris specks + out-of-focus scatterers: coverslip debris reflects BRIGHT in IRM (→ positive field
    # skew) and is HEAVY-TAILED (rare very-strong specks → high kurtosis). Amplitude ~ log-normal, bright-biased.
    # These field-level stats have NO literature (research) → FIT empirically to the real skew(+1.63)/kurt(~10.4).
    spot_rate: float = 48.0                          # mean # specks / frame (Poisson) — FIT to real skew/kurt
    spot_rate_range: Optional[Tuple[float, float]] = None  # DOMAIN RANDOMIZATION: if set, the per-frame dirt RATE is
    #                                                       itself sampled uniform from this band (clean⇄dirty fields
    #                                                       are experiment-variable → randomize, don't fit to one corpus)
    spot_size_range: Tuple[float, float] = (1.0, 5.0)
    spot_log_mean: float = -1.40                     # log-normal amplitude location (fit; median exp(-1.4)=0.25)
    spot_log_std: float = 0.85                       # log-normal amplitude scale (heavy tail → kurtosis, fit)
    spot_bright_frac: float = 0.88                   # fraction of specks BRIGHT (+) → positive skew (fit)
    # detachment event (SHARED across modalities): a locally detached segment leaves the surface. IRM: its height
    # rises into height_detach_nm_range → BRIGHT via interference; TIRF: it leaves the evanescent field → DIMS.
    detach_frame_prob: float = 0.5
    detach_prob: float = 0.40
    n_detach_range: Tuple[int, int] = (1, 2)
    detach_len_range: Tuple[float, float] = (10.0, 55.0)
    detach_transition_sigma: float = 8.0
    # (whole-frame inversion REMOVED — MTs are always DARK; real-data polarity ambiguity is a TRAINING-time
    #  augmentation, not an unphysical whole-frame photometric negative baked into the generator.)
    # IRM optics / sensor
    psf_sigma_range: Tuple[float, float] = (0.8, 1.6)
    poisson_gain_range: Tuple[float, float] = (90.0, 380.0)
    read_noise_range: Tuple[float, float] = (0.004, 0.018)
    # (texture_amp REMOVED — composite is on REAL empty-field backgrounds that already carry the coverslip /
    #  interference texture; adding synthetic texture on top would double-count.)

    # =========================== TIRF APPEARANCE (bright fluorescent line on DARK background, additive) =======
    tirf_signal_range: Tuple[float, float] = (0.4, 1.0)   # additive bright signal amplitude (rel. to range)
    tirf_bg_boost: float = 1.0                       # scale of the (dark) background contribution
    tirf_detach_dim: float = 0.35                    # detached segment DIMS to this fraction (TIRF: less excitation)
    tirf_psf_sigma_range: Tuple[float, float] = (1.0, 2.0)
    tirf_poisson_gain_range: Tuple[float, float] = (20.0, 120.0)   # fewer photons => stronger shot noise
    tirf_read_noise_range: Tuple[float, float] = (0.008, 0.03)

    # =========================== render geometry (both modalities) ===========================
    render_half_width: float = 6.0                   # perpendicular stamping radius (px)
    mask_half_width: float = 1.6                     # half-width of the instance GT mask (px)


# ----------------------------------------------------------------------------- morphology
def _sample_length(rng, cfg: GenConfig, regime="static") -> float:
    if regime == "dynamic":                                              # dynamic instability -> EXPONENTIAL length
        L = float(rng.exponential(cfg.dynamic_length_mean_px))           # (broad, mixed growing+shrinking; Dogterom-Leibler)
    else:
        lm = cfg.short_length_log_mean if rng.random() < cfg.short_prob else cfg.length_log_mean  # bimodal seeds+long
        L = float(np.exp(rng.normal(lm, cfg.length_log_sigma)))
    return float(np.clip(L, *cfg.length_range))


def _apply_bends(cl: np.ndarray, rng, n_range, angle_range, width_range, smooth_sigma, alternate=True):
    """Inject localized SMOOTH bends by re-integrating the tangent angle with smootherstep (C2) ramps —
    finite, continuous curvature (no corner). Used for both large hairpins (alternate=True => S-curves,
    not loops) and small sharp kinks (alternate=False => random-sign defects)."""
    n = len(cl)
    if n < 12:
        return cl
    d = np.diff(cl, axis=0)
    seg = np.hypot(d[:, 0], d[:, 1]) + 1e-8
    ang = np.arctan2(d[:, 1], d[:, 0])
    s = np.concatenate([[0.0], np.cumsum(seg)])[:-1]
    L = float(s[-1] + seg[-1])
    extra = np.zeros_like(ang)
    sign = 1.0 if rng.random() < 0.5 else -1.0
    for _ in range(int(rng.integers(*n_range, endpoint=True))):
        dtheta = rng.uniform(*angle_range) * sign
        sign = -sign if alternate else (1.0 if rng.random() < 0.5 else -1.0)
        w = min(rng.uniform(*width_range), 0.8 * L)
        s0 = rng.uniform(0.15, 0.85) * L
        t = np.clip((s - (s0 - w / 2)) / w, 0.0, 1.0)
        extra = extra + dtheta * (t ** 3 * (10 - 15 * t + 6 * t ** 2))    # smootherstep
    newang = ang + extra
    steps = np.stack([np.cos(newang), np.sin(newang)], 1) * seg[:, None]
    out = np.concatenate([cl[:1], cl[:1] + np.cumsum(steps, axis=0)], axis=0)
    out[:, 0] = gaussian_filter1d(out[:, 0], smooth_sigma)
    out[:, 1] = gaussian_filter1d(out[:, 1], smooth_sigma)
    return out


def sample_centerline(rng, cfg: GenConfig, shape, regime="static", length=None, start=None, angle0=None):
    """Microtubule centerline (x,y), REGIME-dependent:
      'gliding' = motor-propelled WIGGLY path: a 2D worm-like-chain tangent-angle random walk with an effective PATH
                  persistence length (var(dθ)=ds/Lp, Lp~0.1-0.5mm) + an OPTIONAL tight constant curvature (arc/ring).
      'static'/'dynamic' = the calibrated STIFF near-straight filament (length-coupled Pampaloni curvature + gentle arc)."""
    H, W = shape
    L = length if length is not None else _sample_length(rng, cfg, regime)
    n = max(int(L / cfg.step_px), 8)
    theta0 = rng.uniform(0, 2 * np.pi) if angle0 is None else angle0
    if start is None:
        start = np.array([rng.uniform(0, W), rng.uniform(0, H)])
    if regime == "gliding":
        ds = cfg.step_px
        Lp = max(rng.uniform(*cfg.lp_gliding_px_range) * float(np.exp(rng.normal(0.0, cfg.lp_log_std))), 300.0)
        kap = np.zeros(n)                                                 # per-step constant curvature (ring/arc)
        if rng.random() < cfg.ring_prob:                                 # tight arc / loop / ring (motor mode)
            R = rng.uniform(*cfg.ring_radius_px_range); sign = 1.0 if rng.random() < 0.5 else -1.0
            loops = rng.uniform(0.4, 1.5)                                 # 0.4-1.5 loops (arc/loop/ring, NOT a dense spool)
            seg = min(n, max(4, int(loops * 2 * np.pi * R / ds)))         # apply curvature over a SEGMENT only
            s0 = int(rng.integers(0, max(1, n - seg))); kap[s0:s0 + seg] = sign / R
        theta = theta0 + np.cumsum(rng.normal(0.0, np.sqrt(2.0 * ds / Lp), n) + kap * ds)  # 2D WLC: Var(Δθ)=2ds/Lp
        x = np.cumsum(np.cos(theta) * ds); y = np.cumsum(np.sin(theta) * ds)
        x = gaussian_filter1d(x, cfg.smooth_sigma); y = gaussian_filter1d(y, cfg.smooth_sigma)
        return np.stack([x - x[0] + start[0], y - y[0] + start[1]], axis=1)
    s = np.arange(n) * cfg.step_px
    # per-MT LENGTH-COUPLED stiffness (Pampaloni 2006): curvature falls with length, broad log-normal scatter
    stiff = (L / cfg.curve_len_ref) ** cfg.curve_len_exp
    cfrac = cfg.curve_base / max(stiff, 0.15) * float(np.exp(rng.normal(0.0, cfg.curve_log_std)))
    cfrac = float(np.clip(cfrac, *cfg.curve_frac_range))
    walk = np.cumsum(rng.normal(0, 1, n)); walk = gaussian_filter1d(walk, max(n * 0.18, 5.0))
    walk = walk - np.linspace(walk[0], walk[-1], n); walk = walk / (np.abs(walk).max() + 1e-8)
    transverse = cfrac * L * walk
    arc = rng.uniform(*cfg.arc_bias_range) * (0.04 * L) * (1 - (2 * s / (L + 1e-8) - 1) ** 2)
    transverse = transverse + (arc if rng.random() < 0.5 else -arc)
    ct, st = np.cos(theta0), np.sin(theta0)
    x = gaussian_filter1d(s * ct - transverse * st, cfg.smooth_sigma)
    y = gaussian_filter1d(s * st + transverse * ct, cfg.smooth_sigma)
    return np.stack([x - x[0] + start[0], y - y[0] + start[1]], axis=1)


def sample_instance(rng, cfg: GenConfig, shape, regime="static", start=None, angle0=None,
                    hairpin_active=False) -> List[np.ndarray]:
    """One object: a single MT (with optional hairpins + kinks) or a parallel bundle sharing the base."""
    base = sample_centerline(rng, cfg, shape, regime=regime, start=start, angle0=angle0)
    if hairpin_active and rng.random() < cfg.hairpin_prob:
        base = _apply_bends(base, rng, cfg.n_bend_range, cfg.bend_angle_range, cfg.bend_width_range,
                            cfg.smooth_sigma, alternate=True)
    if rng.random() < cfg.kink_prob:                                     # NEW localized sharp kinks / defects
        base = _apply_bends(base, rng, cfg.n_kink_range, cfg.kink_angle_range, cfg.kink_width_range,
                            0.6, alternate=False)
    if rng.random() >= cfg.bundle_prob:
        return [base]
    k = int(rng.integers(*cfg.bundle_size_range, endpoint=True))
    base_gap = rng.uniform(*cfg.bundle_gap_range)
    offsets, pos = [], -(k - 1) / 2.0 * base_gap
    for _ in range(k):
        offsets.append(pos); pos += base_gap * max(0.2, 1 + rng.normal(0, cfg.bundle_gap_std))
    c = base.mean(0); out = []
    for off in offsets:
        a = rng.normal(0, cfg.bundle_angle_div)
        ca, sa = np.cos(a), np.sin(a)
        rot = (base - c) @ np.array([[ca, sa], [-sa, ca]]) + c
        tang = np.gradient(rot, axis=0)
        normal = np.stack([-tang[:, 1], tang[:, 0]], 1) / (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-8)
        jit = gaussian_filter1d(rng.normal(0, 0.4, len(rot)), 6.0)
        m = rot + (off + jit)[:, None] * normal
        st = int(cfg.bundle_stagger * len(m) * rng.random())
        if st > 1:
            m = m[st:] if rng.random() < 0.5 else m[:len(m) - st]
        out.append(m)
    return out


def rasterize_mask(cl: np.ndarray, shape, half_width: float) -> np.ndarray:
    """Modality-agnostic instance GT mask: pixels within `half_width` of the centerline."""
    H, W = shape
    mask = np.zeros((H, W), dtype=bool)
    tang = np.gradient(cl, axis=0)
    normal = np.stack([-tang[:, 1], tang[:, 0]], 1) / (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-8)
    for d in np.arange(-half_width, half_width + 1e-6, 0.5):
        ix = np.round(cl[:, 0] + d * normal[:, 0]).astype(int)
        iy = np.round(cl[:, 1] + d * normal[:, 1]).astype(int)
        ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        mask[iy[ok], ix[ok]] = True
    return mask


def sample_scene(shape, rng, cfg: GenConfig):
    """MODALITY-AGNOSTIC morphology sampler: returns (instances, morph_cond). instances = list of
    {centerline (x,y), mask}. morph_cond carries frame-level MORPHOLOGY conditions (waviness, hairpin,
    orientation regime) — the appearance conditions are sampled later by the renderer."""
    H, W = shape
    # FRAME REGIME: gliding (wiggly) / dynamic (exp length) / static (calibrated stiff, near-straight) — spans the
    # full in-vitro variability across assay types.
    r = rng.random()
    regime = ("gliding" if r < cfg.regime_gliding_prob
              else "dynamic" if r < cfg.regime_gliding_prob + cfg.regime_dynamic_prob else "static")
    # nematic alignment is DENSITY/regime-gated (research: crossings isotropic at low density; local nematic streams
    # appear in the gliding regime) — non-gliding frames render more isotropic.
    okappa = rng.uniform(*cfg.orient_kappa_range) * (1.0 if regime == "gliding" else 0.35)
    cond = {
        "regime": regime,
        "wavy": regime == "gliding",
        "hairpin_active": rng.random() < cfg.hairpin_frame_prob,
        "orient_mean": rng.uniform(0, 2 * np.pi),
        "orient_kappa": okappa,          # regime-gated nematic alignment strength
    }
    n_obj = int(rng.integers(*cfg.n_mt_range, endpoint=True))
    cf = cfg.cluster_frac
    if cf > 0:
        n_clust = max(1, int(round(n_obj * (1 - cf))))
        centers = np.stack([rng.uniform(0, W, n_clust), rng.uniform(0, H, n_clust)], 1)
        csig = min(W, H) * (0.04 + 0.22 * (1 - cf))
    instances = []
    for _ in range(n_obj):
        start = centers[rng.integers(n_clust)] + rng.normal(0, csig, 2) if cf > 0 else None
        # NEW orientation from a von Mises (kappa=0 => isotropic; high => aligned). Undirected => *0.5 spread.
        angle0 = float(rng.vonmises(cond["orient_mean"], cond["orient_kappa"]))
        for cl in sample_instance(rng, cfg, (H, W), regime=cond["regime"], start=start, angle0=angle0,
                                  hairpin_active=cond["hairpin_active"]):
            mask = rasterize_mask(cl, (H, W), cfg.mask_half_width)
            if mask.any():
                instances.append({"centerline": cl, "mask": mask})
    return instances, cond


# ----------------------------------------------------------------------------- appearance helpers
def cross_profile(d, width_sigma):
    """Cross-section PERPENDICULAR to the filament: a single Gaussian core (sub-diffraction filament → PSF-limited
    dip/ridge). Contrast SIGN varies only ALONG the filament (via height), not across it — no modeled lateral
    interference fringe (the sign flips are along-length, not perpendicular side-lobes)."""
    return np.exp(-0.5 * (d / width_sigma) ** 2)


def _tip_taper(n, rng, cfg: GenConfig):
    """Per-MT tip-taper envelope in [0,1] (dimmer ends) — applied on top of the along-length modulation."""
    env = np.ones(n)
    if n > 20 and rng.random() < cfg.tip_taper_prob:
        tl = int(np.clip(rng.uniform(*cfg.tip_taper_frac_range) * n, 3, n // 2))
        ramp = np.linspace(0.25, 1.0, tl)
        env[:tl] = ramp; env[-tl:] = ramp[::-1]
    return env


def _along_intensity(n, rng, cfg: GenConfig):
    """Per-MT low-frequency intensity heterogeneity along the filament (~1 +/- std)."""
    if cfg.along_intensity_std <= 0 or n < 8:
        return np.ones(n)
    z = gaussian_filter1d(rng.normal(0, 1, n), max(n * 0.15, 4.0))
    z = z / (z.std() + 1e-8)
    return np.clip(1.0 + cfg.along_intensity_std * z, 0.3, 1.7)


def mt_height_field(rng, n, cfg: GenConfig, h_base, detach_active=True):
    """Per-MT HEIGHT above the coverslip h(s) [nm]: the FRAME-level base height regime `h_base` (assay/ATP
    condition — low ⇒ dark MTs, elevated ⇒ brighter) + small per-MT offset + low-frequency thermal/attachment
    variation + occasional DETACHED (elevated) segments. Drives the IRM interference contrast."""
    h = np.full(n, max(0.0, h_base + rng.normal(0.0, 0.5 * cfg.height_along_std_nm)))
    if cfg.height_along_std_nm > 0 and n >= 8:
        z = gaussian_filter1d(rng.normal(0, 1, n), max(n * 0.15, 4.0)); z = z / (z.std() + 1e-8)
        h = h + cfg.height_along_std_nm * z
    if detach_active and rng.random() < cfg.detach_prob and n > 20:
        for _ in range(rng.integers(*cfg.n_detach_range, endpoint=True)):
            seg = int(rng.uniform(*cfg.detach_len_range) / cfg.step_px)
            if seg < n:
                s0 = rng.integers(0, n - seg); h[s0:s0 + seg] = rng.uniform(*cfg.height_detach_nm_range)
    return np.clip(gaussian_filter1d(h, cfg.detach_transition_sigma / cfg.step_px), 0.0, None)


def height_to_contrast(h_nm, cfg: GenConfig, ina):
    """Two-beam interference contrast (signed, ~[-1,1]) from MT height: -cos(2k·h) makes contact (h≈0) DARK
    (π reflection phase shift) and elevated segments (>~56 nm) BRIGHT. Finite-aperture INA envelope decays
    higher fringes (Simmert 2018: α = arcsin(INA / n_water)). k = 2π·n_water/λ [rad/nm]."""
    k = 2.0 * np.pi * cfg.n_water / cfg.wavelength_nm
    alpha = np.arcsin(min(ina / cfg.n_water, 0.999))
    env = np.exp(-(2.0 * k * h_nm * np.sin(alpha / 2.0) ** 2) ** 2)
    return -np.cos(2.0 * k * h_nm) * env


def along_length_tirf(rng, n, cfg: GenConfig, detach_active=True):
    """TIRF modulation: BRIGHT baseline (+1); a DETACHED segment leaves the evanescent field => DIMS."""
    m = np.ones(n)
    if detach_active and rng.random() < cfg.detach_prob and n > 20:
        for _ in range(rng.integers(*cfg.n_detach_range, endpoint=True)):
            seg = int(rng.uniform(*cfg.detach_len_range) / cfg.step_px)
            if seg < n:
                s0 = rng.integers(0, n - seg); m[s0:s0 + seg] = cfg.tirf_detach_dim
    return gaussian_filter1d(m, cfg.detach_transition_sigma / cfg.step_px)


def _paint(field, cl, amp, modulation, width_sigma, cfg: GenConfig):
    """Accumulate one filament's signed contrast onto `field` (mask handled separately in sample_scene)."""
    H, W = field.shape
    tang = np.gradient(cl, axis=0)
    normal = np.stack([-tang[:, 1], tang[:, 0]], 1) / (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-8)
    offs = np.arange(-cfg.render_half_width, cfg.render_half_width + 1e-6, 0.5)
    prof = cross_profile(offs, width_sigma)
    for d, pv in zip(offs, prof):
        ix = np.round(cl[:, 0] + d * normal[:, 0]).astype(int)
        iy = np.round(cl[:, 1] + d * normal[:, 1]).astype(int)
        ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        vals = amp * pv * modulation
        np.add.at(field, (iy[ok], ix[ok]), vals[ok])


def add_spots(field, rng, cfg: GenConfig):
    """Dirt/debris specks + out-of-focus scatterers: heavy-tailed (log-normal) amplitudes, bright-biased — the
    physical source of the real IRM field's POSITIVE skew + HEAVY tails (fit empirically, no literature)."""
    H, W = field.shape
    rate = rng.uniform(*cfg.spot_rate_range) if cfg.spot_rate_range else cfg.spot_rate
    for _ in range(rng.poisson(rate)):
        cy, cx = int(rng.integers(0, H)), int(rng.integers(0, W))
        sig = rng.uniform(*cfg.spot_size_range)
        mag = float(np.exp(rng.normal(cfg.spot_log_mean, cfg.spot_log_std)))       # heavy-tailed magnitude
        amp = mag * (1.0 if rng.random() < cfg.spot_bright_frac else -1.0)         # bright-biased sign
        r = int(sig * 3)
        y0, y1 = max(0, cy-r), min(H, cy+r+1); x0, x1 = max(0, cx-r), min(W, cx+r+1)
        yy, xx = np.ogrid[y0-cy:y1-cy, x0-cx:x1-cx]
        field[y0:y1, x0:x1] += amp * np.exp(-(yy**2 + xx**2) / (2 * sig**2))


def _shot_read_noise(z, rng, gain_range, read_range):
    gain = rng.uniform(*gain_range)
    z = rng.poisson(np.clip(z, 0, None) * gain) / gain
    return z + rng.normal(0, rng.uniform(*read_range), z.shape)


def add_irm_noise(img, rng, cfg: GenConfig):
    """Poisson shot + gaussian read noise, in [0,1] space. NO synthetic mid-frequency texture: the composite is
    on REAL empty-field backgrounds that already carry the coverslip / interference texture — adding synthetic
    texture on top would double-count it."""
    lo, hi = np.percentile(img, [0.5, 99.5]); span = max(hi - lo, 1e-6)
    z = _shot_read_noise(np.clip((img - lo) / span, 0, 1), rng, cfg.poisson_gain_range, cfg.read_noise_range)
    return lo + np.clip(z, 0, 1) * span


# ----------------------------------------------------------------------------- renderers
def render_irm(instances, cond, background, rng, cfg: GenConfig, noise_rng=None):
    """IRM appearance via TWO-BEAM INTERFERENCE: each MT's per-segment HEIGHT h(s) [nm] sets a signed contrast
    -cos(2k·h)·E_INA(h) — DARK at contact (π shift), BRIGHT when elevated. Multiplicative composite bg*(1+field),
    then IRM sensor noise. (No whole-frame inversion: polarity ambiguity is a training augmentation.)"""
    H, W = background.shape
    field = np.zeros((H, W))
    contrast = rng.uniform(*cfg.contrast_range)
    ina = rng.uniform(*cfg.ina_range)                # FRAME-level illumination NA (strongest contrast knob)
    h_base = rng.uniform(*cfg.height_base_nm_range)  # FRAME-level base height regime (assay/ATP: dark⇄bright MTs)
    # FRAME-level render width: apparent MT width is set by the microscope PSF / pixel-size (constant across a
    # frame), NOT a per-MT property — single MTs are all ~25 nm << diffraction limit, so every MT shares one width.
    ws = float(np.clip(rng.normal(cfg.width_mean, cfg.width_std), *cfg.width_clip))
    detach_active = rng.random() < cfg.detach_frame_prob
    for ins in instances:
        cl = ins["centerline"]; n = len(cl)
        amp = contrast * float(np.clip(rng.normal(1.0, cfg.contrast_rel_std), 0.1, 3.0))
        h = mt_height_field(rng, n, cfg, h_base, detach_active)             # height [nm] above the coverslip
        c = height_to_contrast(h, cfg, ina)                                 # signed two-beam interference contrast
        c = np.sign(c) * np.maximum(np.abs(c), cfg.contrast_floor)          # floor: labeled MT never fully invisible
        mod = c * _tip_taper(n, rng, cfg)
        _paint(field, cl, amp, mod, ws, cfg)
    add_spots(field, rng, cfg)
    field = gaussian_filter(field, rng.uniform(*cfg.psf_sigma_range))
    img = background.astype(np.float64) * (1.0 + field)
    img = add_irm_noise(img, noise_rng if noise_rng is not None else rng, cfg)
    return img, {"inverted": False, "detach_active": detach_active, "ina": float(ina)}


def render_tirf(instances, cond, background, rng, cfg: GenConfig, noise_rng=None):
    """TIRF appearance: BRIGHT fluorescent filaments on a DARK background, ADDITIVE (bg + signal),
    NO interference halo, detachment DIMS (less evanescent excitation), fluorescence shot noise."""
    H, W = background.shape
    field = np.zeros((H, W))
    signal = rng.uniform(*cfg.tirf_signal_range)
    ws = float(np.clip(rng.normal(cfg.width_mean, cfg.width_std), *cfg.width_clip))   # FRAME-level (see render_irm)
    detach_active = rng.random() < cfg.detach_frame_prob
    for ins in instances:
        cl = ins["centerline"]; n = len(cl)
        amp = signal * float(np.clip(rng.normal(1.0, cfg.contrast_rel_std), 0.1, 3.0))
        mod = along_length_tirf(rng, n, cfg, detach_active) * _tip_taper(n, rng, cfg) * _along_intensity(n, rng, cfg)
        _paint(field, cl, amp, mod, ws, cfg)
    field = np.clip(field, 0, None)                                  # fluorescence is emissive (>=0)
    field = gaussian_filter(field, rng.uniform(*cfg.tirf_psf_sigma_range))
    bg = background.astype(np.float64)
    bg = (bg - bg.min()) / (bg.max() - bg.min() + 1e-6)              # normalize bg to [0,1] (dark field)
    img = cfg.tirf_bg_boost * bg + field                            # ADDITIVE bright signal on dark bg
    lo, hi = np.percentile(img, [0.5, 99.5]); span = max(hi - lo, 1e-6)
    z = _shot_read_noise(np.clip((img - lo) / span, 0, 1),
                         noise_rng if noise_rng is not None else rng,
                         cfg.tirf_poisson_gain_range, cfg.tirf_read_noise_range)
    img = lo + np.clip(z, 0, 1) * span
    return img, {"inverted": False, "detach_active": detach_active}


def generate_frame(background, rng, cfg: GenConfig, modality: str = "irm"):
    """Sample MORPHOLOGY once, then RENDER it for the chosen modality. Returns (image, instances, meta).
    Backward compatible: modality defaults to 'irm'."""
    instances, cond = sample_scene(background.shape, rng, cfg)
    render = render_tirf if modality == "tirf" else render_irm
    img, rmeta = render(instances, cond, background, rng, cfg)
    for ins in instances:
        ins["polarity_base"] = 1.0 if modality == "tirf" else -1.0
    meta = {"modality": modality, "regime": cond["regime"], "wavy": cond["wavy"], "hairpin_active": cond["hairpin_active"],
            "orient_kappa": cond["orient_kappa"], "n_instances": len(instances), **rmeta}
    return img, instances, meta
