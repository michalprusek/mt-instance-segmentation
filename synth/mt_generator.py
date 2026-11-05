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
import dataclasses
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np
from scipy.ndimage import label as ndimage_label, gaussian_filter, gaussian_filter1d


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
    # --- FRAME-LEVEL length scale. Characteristic filament length is a property of the
    #     PREPARATION, not of the individual microtubule: a given experiment yields either short
    #     filaments throughout or long ones, never a per-filament coin flip. Measured 2026-08-27:
    #     median ground-truth length is 29 px in the HTW corpus against 173-358 px in MT-34, and
    #     whole frames are consistently one or the other. Each frame therefore draws an additive
    #     shift in log-length applied to BOTH log-normal arms and to the dynamic regime's mean.
    #     Default (0.0, 0.0) reproduces the previous behaviour exactly, so unflagged runs are
    #     bit-identical.
    frame_length_shift_range: Tuple[float, float] = (0.0, 0.0)
    #     Filament COUNT has to move with the frame's length scale, or short-filament frames come
    #     out nearly empty and the network learns that short microtubules are rare rather than
    #     small. At coupling 1.0 the expected total contour length per frame is held constant
    #     (n scaled by exp(-length_shift)), which keeps the foreground fraction in the band the
    #     appearance calibration was fitted at. 0.0 disables the coupling and is the default, so
    #     unflagged runs stay bit-identical.
    frame_length_count_coupling: float = 0.0
    frame_n_mt_cap: int = 120                        # ceiling on the coupled count, for tractability
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
    # THIN REGIME (arm 63, 2026-11-02): with this probability a frame draws its core half-width from width_thin_range instead,
    # reaching FWHM ~1.4-3.6 px with the shipped PSF. The upper range is untouched: this widens the prior downward, it does not
    # refit it. Measured: htw 80 % of filaments <= 3 px FWHM vs 4 % in the V16 render. 0 = OFF, no draw.
    width_thin_prob: float = 0.0
    width_thin_range: Tuple[float, float] = (0.35, 1.2)
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
    # --- INTERFERENCE ZERO-CROSSING GUARD.  -cos(2k·h) vanishes and CHANGES SIGN at
    #     h = λ/(8·n_water) = 56.4 nm.  The calibrated base-height range (46.9, 86.9) straddles it, and
    #     `height_along_std_nm` then walks a single filament back and forth ACROSS it, so one labelled
    #     microtubule is rendered part dark and part bright.  `contrast_floor` (applied as
    #     sign(c)·max(|c|,floor)) preserves that sign, so instead of fading the crossed segment it is
    #     amplified back to full strength in the WRONG polarity: the guard against invisible labels
    #     manufactures visible wrong-signed ones.  Measured 2026-08-28 against the real benchmark with
    #     one estimator on both sides (scripts/photometry_screen.py):
    #         flipped label length   synth 12.0 %  vs real 2.2-3.4 %
    #         sub-CNR-1 label length synth 22.8 %  vs real 3.2-4.6 %
    #     `height_crossing_guard_nm` keeps the FRAME's base height that far away from the crossing (so a
    #     frame is decidedly dark or decidedly bright, which is what the real blocks are: 585/htw ~100 %
    #     dark, 586 ~97 % bright), and `height_clamp_to_regime` keeps the along-filament variation on that
    #     same side -- except inside a genuine detachment event, which really does lift the filament and
    #     is observed on 4.7-11.1 % of real filaments.  Both default to OFF/0.0 = previous behaviour.
    height_crossing_guard_nm: float = 0.0
    height_clamp_to_regime: bool = False
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
    # RIDGE DISTRACTORS (arm 41). Measured 2026-09-09: real off-filament background carries ~40 %
    # MORE elongated ridge structure than a render (share 0.0062 vs 0.0044, q75 0.0193 vs 0.0069)
    # while the render is 2.3x NOISIER in amplitude — the network learns to reject loud white noise
    # and never has to reject quiet filament-SHAPED structure, which is what real frames put in the
    # 0.5-0.97 probability band where the whole recall gap lives (docs/TODO.md read 38b / 39).
    # These are physically NOT microtubules — debris streaks, scratches, out-of-focus blur — so
    # their length and curvature priors are deliberately OUTSIDE the MT priors, and they are never
    # labelled. They must not be used to suppress unannotated real filaments: that would be
    # fitting the annotation convention, which this project has ruled out.
    streak_rate: float = 0.0                         # mean # distractors / frame (Poisson); 0 = OFF
    streak_rate_range: Optional[Tuple[float, float]] = None   # domain randomisation of the rate
    streak_len_range: Tuple[float, float] = (15.0, 70.0)      # px; SHORTER than the MT prior
    streak_width_range: Tuple[float, float] = (0.8, 2.5)      # px sigma; can be fatter than an MT
    streak_amp_range: Tuple[float, float] = (0.05, 0.45)      # relative to the MT contrast scale
    streak_curv_range: Tuple[float, float] = (0.0, 0.02)      # rad/px; straight OR (below) kinked
    streak_kink_frac: float = 0.35                   # share given a sharp kink no MT may have
    # INTERFERENCE RING DISTRACTORS (arm 61, 2026-11-01). Newton's rings of particles / vesicles resting on or near the
    # coverslip: the gap h(r) = h0 + r^2 * npp / (2 R) [nm] goes through the SAME two-beam contrast as the filaments
    # (height_to_contrast), so fringe spacing and the number of visible fringes follow from the optics; partial arcs come
    # from a soft angular window. Measured 2026-10-31: 20 % of the htw off-annotation predictions are arcs of such
    # patterns (9 of 33 frames) and the generator had none. Physically not microtubules, never labelled. The ring INA
    # prior is WIDER than the filament one so that 1-12 fringes occur (domain randomisation, not a fit). 0 = OFF.
    fringe_rate: float = 0.0                         # mean # ring patterns / frame (Poisson); 0 = OFF
    fringe_rate_range: Optional[Tuple[float, float]] = None   # domain randomisation of the rate
    fringe_radius_px_range: Tuple[float, float] = (20.0, 400.0)   # sphere radius R, log-uniform (2-40 um at ~100 nm/px)
    fringe_h0_nm_range: Tuple[float, float] = (0.0, 250.0)        # gap at the centre: contact (dark) .. lifted
    fringe_extent_px_range: Tuple[float, float] = (12.0, 90.0)    # footprint radius (soft edge)
    fringe_amp_range: Tuple[float, float] = (0.5, 3.0)            # relative to the frame's MT contrast scale
    fringe_arc_prob: float = 0.6                     # share rendered as a partial arc
    fringe_arc_deg_range: Tuple[float, float] = (60.0, 240.0)
    fringe_ina_range: Tuple[float, float] = (0.3, 1.1)
    fringe_nm_per_px: float = 100.0
    # COMPACT DEBRIS / PARTICLES (arm 62, 2026-11-01). Measured 2026-10-31: 25 % of the htw off-annotation predictions are
    # particles / debris, and the generator's only debris were round Gaussian specks (add_spots). Two kinds, both COMPACT
    # by construction (principal-axis ratio <= debris_max_axis_ratio, rejection-sampled) so that no elongated negative is
    # taught (arm 41): AGGREGATES of small particles, each a contact disc through height_to_contrast (dark core / bright
    # halo of a real particle), and IRREGULAR BLOBS (a thresholded smooth random field) with an optional opposite-signed
    # halo. Physically not microtubules, never labelled. 0 = OFF.
    debris_rate: float = 0.0                          # mean # debris objects / frame (Poisson); 0 = OFF
    debris_rate_range: Optional[Tuple[float, float]] = None
    debris_agg_frac: float = 0.5                      # share of aggregates (rest: irregular blobs)
    debris_agg_n_range: Tuple[int, int] = (2, 12)     # particles per aggregate
    debris_agg_sigma_range: Tuple[float, float] = (2.0, 12.0)    # px, isotropic scatter of an aggregate
    debris_particle_R_range: Tuple[float, float] = (2.0, 12.0)   # px, sphere radius of one particle (Newton profile)
    debris_particle_ext_range: Tuple[float, float] = (1.5, 5.0)  # px, footprint of one particle
    debris_particle_h0_nm_range: Tuple[float, float] = (0.0, 60.0)
    debris_blob_size_range: Tuple[float, float] = (6.0, 30.0)    # px, blob window diameter
    debris_blob_halo_prob: float = 0.5
    debris_blob_halo_px_range: Tuple[float, float] = (1.0, 3.0)
    debris_amp_range: Tuple[float, float] = (0.5, 3.0)           # relative to the frame's MT contrast scale
    debris_max_axis_ratio: float = 2.5
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
    # ---- NOISE MODEL, second generation (2026-09-04; all defaults reproduce the old behaviour).
    #   noise_ref: "span" normalises the composite by its [0.5, 99.5] span before adding shot + read
    #   noise, so a dirty / graded background crop gets noise proportional to ITS range while the
    #   filament modulation stays proportional to the local intensity -- the measured cause of the
    #   27 % "invisible" synthetic frames (bg relative range 0.88 vs 0.085, Spearman +0.79).
    #   "median" normalises by noise_span_rel x the frame median (the photon-count scale), which is
    #   what a camera does. noise_corr_frac_range: per-frame fraction of the noise VARIANCE that is
    #   spatially correlated (white noise filtered with a Gaussian of noise_corr_sigma_range), at
    #   constant total variance so the SNR calibration is untouched -- the real 16-bit frames' off-
    #   filament residual has lag-1 autocorrelation q10/50/90 = -0.01 / +0.19 / +0.62 while every
    #   synthetic frame was white (-0.04), and injecting that texture at inference cut the trained
    #   model's label recall at SNR 1-2 from 0.93 to 0.56-0.75 (the real 0.73-0.81) while white noise
    #   of the same amplitude did nothing: docs/TODO.md 0.0-CAMPAIGN 2026-09-04 23:20.
    #   noise_impulse_frac_range: fraction of pixels hit by an impulse of +-U(noise_impulse_amp_range)
    #   x the white sigma (hot pixels / dust: the real kurtosis tail, median 4, p90 122 on 16-bit).
    noise_ref: str = "span"
    noise_span_rel_range: Tuple[float, float] = (0.3, 0.3)   # per-frame, log-uniform: the photon-count scale
    noise_corr_frac_range: Tuple[float, float] = (0.0, 0.0)
    noise_corr_sigma_range: Tuple[float, float] = (0.8, 1.5)
    noise_impulse_frac_range: Tuple[float, float] = (0.0, 0.0)
    noise_impulse_amp_range: Tuple[float, float] = (5.0, 20.0)
    # ---- NOISE-REGIME MIXTURE (2026-10-05). The correlated-noise regime (calib_V6_noise.json) REPLACING the
    #   shipped one buys htw +0.023 / +0.028 and costs roi303 -0.018 on both seeds (docs/TODO.md 2026-10-05
    #   15:30). With noise_alt_prob > 0 each frame draws its regime: with that probability the noise_* fields
    #   are replaced by `noise_alt` (a dict of GenConfig noise fields), otherwise the frame keeps the fields
    #   above. 0.0 draws nothing, so every earlier run stays bit-identical.
    noise_alt_prob: float = 0.0
    noise_alt: Optional[dict] = None
    # ---- FLAT-FIELDED BACKGROUND (2026-10-05). Labs export frames after flat-field / background correction;
    #   the generator composites onto RAW empty-field crops only, so a background with low-pass std below
    #   ~0.08 of the stretched range is never rendered (real htw: 0.007-0.05). With this probability the
    #   crop is divided by its own Gaussian low-pass (sigma log-uniform in the range, px: a few px leaves
    #   only pixel noise, tens of px leave the coverslip texture) and rescaled to its mean before the
    #   filaments are composited. 0.0 draws nothing.
    bg_flatten_prob: float = 0.0
    bg_flatten_sigma_range: Tuple[float, float] = (3.0, 60.0)
    # (texture_amp REMOVED — composite is on REAL empty-field backgrounds that already carry the coverslip /
    #  interference texture; adding synthetic texture on top would double-count.)

    # =========================== TIRF APPEARANCE (bright fluorescent line on DARK background, additive) =======
    tirf_signal_range: Tuple[float, float] = (0.4, 1.0)   # additive bright signal amplitude (rel. to range)
    tirf_bg_boost: float = 1.0                       # scale of the (dark) background contribution
    tirf_detach_dim: float = 0.35                    # detached segment DIMS to this fraction (TIRF: less excitation)
    tirf_psf_sigma_range: Tuple[float, float] = (1.0, 2.0)
    tirf_poisson_gain_range: Tuple[float, float] = (20.0, 120.0)   # fewer photons => stronger shot noise
    tirf_read_noise_range: Tuple[float, float] = (0.008, 0.03)
    # --- sparse fluorophore labelling -------------------------------------------------------
    # A TIRF microtubule is NOT a continuous line. Fluorophores sit discretely along the lattice,
    # so at a low labelling ratio the filament reads as a chain of puncta separated by dark gaps
    # (plainly visible at native resolution in the real 500x-diluted frames; measured by
    # scripts/tirf_puncta_stats.py). IRM images MASS and is continuous; TIRF images LABELS and is
    # not -- a difference in the SHAPE of the signal, not just its polarity. The GT mask is built
    # from the centerline in sample_scene and stays continuous, so a model trained on this learns
    # to bridge the gaps rather than to end an instance at one.
    tirf_label_sparse: bool = False                  # OFF by default: the renderer stays bit-identical
    tirf_label_density_log10: Tuple[float, float] = (-1.0, 0.7)   # log10 fluorophores/px: 0.1 .. 5
    tirf_label_amp_cv: float = 0.6                   # per-fluorophore brightness spread (dye, bleaching)

    # =========================== render geometry (both modalities) ===========================
    render_half_width: float = 6.0                   # perpendicular stamping radius (px)
    paint_aa: bool = False                           # OFF = the shipped renderer, bit-identical. ON = orientation-isotropic
                                                     # stamp (exact perpendicular distance instead of rounded 0.5-px offsets;
                                                     # research round 2026-09-16: the rounded stamp renders axis-aligned
                                                     # filaments 20-40 % brighter than oblique ones, invariant under rot90/flip)
    paint_aa_gain: float = 1.0574                    # multiplies the isotropic stamp so its ORIENTATION-AVERAGED peak equals
                                                     # the shipped renderer's: 1.953 at PSF 0.705, sigma 1.78 (scripts/paint_aa_effect_test.py 2026-09-16);
                                                     # without it the fix is a contrast arm in disguise
    mask_half_width: float = 1.6                     # half-width of the instance GT mask (px)


# ----------------------------------------------------------------------------- morphology
def _sample_length(rng, cfg: GenConfig, regime="static", length_shift: float = 0.0) -> float:
    """``length_shift`` is the frame's characteristic log-length offset (see
    ``GenConfig.frame_length_shift_range``); 0.0 leaves the per-filament priors untouched."""
    if regime == "dynamic":                                              # dynamic instability -> EXPONENTIAL length
        L = float(rng.exponential(cfg.dynamic_length_mean_px * np.exp(length_shift)))  # (broad, mixed growing+shrinking; Dogterom-Leibler)
    else:
        lm = cfg.short_length_log_mean if rng.random() < cfg.short_prob else cfg.length_log_mean  # bimodal seeds+long
        L = float(np.exp(rng.normal(lm + length_shift, cfg.length_log_sigma)))
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


def sample_centerline(rng, cfg: GenConfig, shape, regime="static", length=None, start=None, angle0=None,
                      length_shift: float = 0.0):
    """Microtubule centerline (x,y), REGIME-dependent:
      'gliding' = motor-propelled WIGGLY path: a 2D worm-like-chain tangent-angle random walk with an effective PATH
                  persistence length (var(dθ)=ds/Lp, Lp~0.1-0.5mm) + an OPTIONAL tight constant curvature (arc/ring).
      'static'/'dynamic' = the calibrated STIFF near-straight filament (length-coupled Pampaloni curvature + gentle arc)."""
    H, W = shape
    L = length if length is not None else _sample_length(rng, cfg, regime, length_shift)
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
                    hairpin_active=False, length_shift: float = 0.0) -> List[np.ndarray]:
    """One object: a single MT (with optional hairpins + kinks) or a parallel bundle sharing the base."""
    base = sample_centerline(rng, cfg, shape, regime=regime, start=start, angle0=angle0,
                             length_shift=length_shift)
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
    # one characteristic length per FRAME, not per filament (see frame_length_shift_range)
    length_shift = float(rng.uniform(*cfg.frame_length_shift_range))
    cond = {
        "regime": regime,
        "length_shift": length_shift,
        "wavy": regime == "gliding",
        "hairpin_active": rng.random() < cfg.hairpin_frame_prob,
        "orient_mean": rng.uniform(0, 2 * np.pi),
        "orient_kappa": okappa,          # regime-gated nematic alignment strength
    }
    n_obj = int(rng.integers(*cfg.n_mt_range, endpoint=True))
    if cfg.frame_length_count_coupling and length_shift:
        # shorter filaments -> proportionally more of them, so total polymer per frame is stable
        n_obj = int(round(n_obj * float(np.exp(-length_shift * cfg.frame_length_count_coupling))))
        n_obj = int(np.clip(n_obj, 1, cfg.frame_n_mt_cap))
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
                                  hairpin_active=cond["hairpin_active"],
                                  length_shift=cond["length_shift"]):
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


def interference_zero_nm(cfg: GenConfig) -> float:
    """Height at which the two-beam contrast -cos(2k·h) vanishes and flips sign: λ/(8·n_water)."""
    return cfg.wavelength_nm / (8.0 * cfg.n_water)


def sample_base_height(rng, cfg: GenConfig) -> float:
    """FRAME base height, kept clear of the interference zero-crossing when a guard is set.

    Rejection-sample inside ``height_base_nm_range``; if the guard swallows the whole range, fall back
    to the nearer edge rather than looping forever.
    """
    lo, hi = cfg.height_base_nm_range
    if cfg.height_crossing_guard_nm <= 0:
        return float(rng.uniform(lo, hi))
    z = interference_zero_nm(cfg)
    g = cfg.height_crossing_guard_nm
    dark = (lo, min(hi, z - g))
    bright = (max(lo, z + g), hi)
    spans = [(a, b) for a, b in (dark, bright) if b > a]
    if not spans:
        return float(lo if abs(lo - z) > abs(hi - z) else hi)
    w = np.array([b - a for a, b in spans], float)
    a, b = spans[int(rng.choice(len(spans), p=w / w.sum()))]
    return float(rng.uniform(a, b))


def mt_height_field(rng, n, cfg: GenConfig, h_base, detach_active=True):
    """Per-MT HEIGHT above the coverslip h(s) [nm]: the FRAME-level base height regime `h_base` (assay/ATP
    condition — low ⇒ dark MTs, elevated ⇒ brighter) + small per-MT offset + low-frequency thermal/attachment
    variation + occasional DETACHED (elevated) segments. Drives the IRM interference contrast."""
    h = np.full(n, max(0.0, h_base + rng.normal(0.0, 0.5 * cfg.height_along_std_nm)))
    if cfg.height_along_std_nm > 0 and n >= 8:
        z = gaussian_filter1d(rng.normal(0, 1, n), max(n * 0.15, 4.0)); z = z / (z.std() + 1e-8)
        h = h + cfg.height_along_std_nm * z
    if cfg.height_clamp_to_regime and cfg.height_crossing_guard_nm > 0:
        # keep the thermal/attachment walk on the frame's own side of the crossing, so a filament is
        # rendered with ONE polarity; genuine detachment below is applied after this and may cross.
        z = interference_zero_nm(cfg)
        g = 0.5 * cfg.height_crossing_guard_nm
        h = np.minimum(h, z - g) if h_base < z else np.maximum(h, z + g)
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


def along_label_density(rng, n, lam, cfg: GenConfig):
    """Discrete fluorophores along the lattice, as a multiplicative modulation with MEAN 1.

    Positions are a Poisson process of linear density ``lam`` (fluorophores per pixel); each carries
    a gamma-distributed brightness with the configured coefficient of variation. Two design points:

    * Normalising by the mean keeps ``lam`` a GRANULARITY knob and nothing else. Without it, a lower
      labelling density would also make the frame dimmer, and a sweep over labelling density would
      silently be a sweep over exposure -- the two effects could not be told apart afterwards.
    * At high ``lam`` the sum concentrates on its mean, the modulation flattens to 1, and the
      filament goes continuous. That is the well-labelled regime AND the old behaviour, so one prior
      spans both instead of the generator having two disjoint TIRF modes.

    No smoothing is applied here on purpose: a fluorophore is a point source, and the PSF that turns
    it into a spot is the ``gaussian_filter`` render_tirf already applies to the whole field.
    """
    mu = max(lam * cfg.step_px, 1e-9)
    k = rng.poisson(mu, n).astype(np.float64)
    if cfg.tirf_label_amp_cv > 0:
        shape = 1.0 / cfg.tirf_label_amp_cv ** 2          # gamma shape of ONE fluorophore
        hit = k > 0
        w = np.zeros(n)
        if hit.any():
            # a sum of k iid gamma(shape, 1/shape) is gamma(k*shape, 1/shape); mean k, so the
            # normalisation below is untouched by the jitter
            w[hit] = rng.gamma(k[hit] * shape, 1.0 / shape)
        k = w
    return k / mu


def _paint(field, cl, amp, modulation, width_sigma, cfg: GenConfig):
    """Accumulate one filament's signed contrast onto `field` (mask handled separately in sample_scene)."""
    if cfg.paint_aa:
        return _paint_exact(field, cl, amp, modulation, width_sigma, cfg)
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


def _paint_exact(field, cl, amp, modulation, width_sigma, cfg: GenConfig):
    """Orientation-isotropic stamp: every pixel within render_half_width of the centerline receives
    2 * cross_profile(exact perpendicular distance) * amp * modulation(nearest sample) * paint_aa_gain.

    The factor 2 keeps the MASS per unit length of the shipped stamp (its 0.5-px offset sum is twice
    the profile's integral), so with gain 1.0 the two renderers agree on total contrast and differ
    only in how it is spread across pixels: the shipped one depends on the filament's angle to the
    pixel grid (np.round on 0.5-px offsets and 1-px samples; peak max/min 1.40 raw, 1.26 at PSF
    0.705), this one does not (residual max/min ~1.02 from pixel-centre sampling of the Gaussian).
    Candidate pixels are the same set the shipped stamp touches; the value is exact.
    """
    from scipy.spatial import cKDTree
    H, W = field.shape
    n = len(cl)
    if n < 2:
        return
    # dense centerline (0.25 px) for the distance query; modulation carried by nearest original sample
    seg = np.linalg.norm(np.diff(cl, axis=0), axis=1)
    t = np.concatenate([[0.0], np.cumsum(seg)])
    if t[-1] <= 0:
        return
    td = np.arange(0.0, t[-1] + 1e-9, 0.25)
    dense = np.stack([np.interp(td, t, cl[:, 0]), np.interp(td, t, cl[:, 1])], 1)
    src = np.clip(np.searchsorted(t, td, side="right") - 1, 0, n - 1)
    tree = cKDTree(dense)
    tang = np.gradient(cl, axis=0)
    normal = np.stack([-tang[:, 1], tang[:, 0]], 1) / (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-8)
    offs = np.arange(-cfg.render_half_width, cfg.render_half_width + 1e-6, 0.5)
    ix = np.round(cl[:, None, 0] + offs[None, :] * normal[:, None, 0]).astype(int).ravel()
    iy = np.round(cl[:, None, 1] + offs[None, :] * normal[:, None, 1]).astype(int).ravel()
    ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    lin = np.unique(iy[ok] * W + ix[ok])
    py, px = lin // W, lin % W
    dist, j = tree.query(np.stack([px, py], 1).astype(float))
    inside = dist <= cfg.render_half_width
    vals = 2.0 * cfg.paint_aa_gain * amp * cross_profile(dist[inside], width_sigma) * modulation[src[j[inside]]]
    np.add.at(field, (py[inside], px[inside]), vals)


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


def add_streaks(field, rng, cfg: GenConfig, meta=None):
    """Elongated NON-microtubule structure: debris streaks, scratches, out-of-focus blur.

    The generator already has point-like dirt (`add_spots`). What it lacked, measured, is
    elongated structure: a render carries 40 % less ridge share off-filament than a real frame
    and a third of its upper quartile. A network that never had to reject a quiet filament-shaped
    thing scores exactly that in the band where the recall gap lives.

    Deliberately outside the microtubule priors so the arm cannot be accused of teaching the model
    to suppress real filaments: shorter, either perfectly straight or sharply kinked (a kink no
    curvature-bounded microtubule may have), often fatter, and always unlabelled.
    """
    rate = rng.uniform(*cfg.streak_rate_range) if cfg.streak_rate_range else cfg.streak_rate
    if rate <= 0:
        return 0
    H, W = field.shape
    n = int(rng.poisson(rate))
    for _ in range(n):
        L = float(rng.uniform(*cfg.streak_len_range))
        sig = float(rng.uniform(*cfg.streak_width_range))
        amp = float(rng.uniform(*cfg.streak_amp_range)) * (1.0 if rng.random() < 0.5 else -1.0)
        y0, x0 = rng.uniform(0, H), rng.uniform(0, W)
        th = rng.uniform(0, 2 * np.pi)
        kap = float(rng.uniform(*cfg.streak_curv_range))
        n_pts = max(int(L), 3)
        pts, y, x, t = [], y0, x0, th
        kink_at = int(n_pts * rng.uniform(0.3, 0.7)) if rng.random() < cfg.streak_kink_frac else -1
        for i in range(n_pts):
            pts.append((y, x))
            t += kap * (1.0 if rng.random() < 0.5 else -1.0)
            if i == kink_at:
                t += rng.uniform(0.6, 1.4) * (1.0 if rng.random() < 0.5 else -1.0)
            y += np.sin(t); x += np.cos(t)
        pts = np.array(pts)
        keep = (pts[:, 0] >= 0) & (pts[:, 0] < H) & (pts[:, 1] >= 0) & (pts[:, 1] < W)
        pts = pts[keep]
        if len(pts) < 3:
            continue
        r = int(np.ceil(3 * sig))
        for (py, px) in pts:
            iy, ix = int(py), int(px)
            ya, yb = max(0, iy - r), min(H, iy + r + 1)
            xa, xb = max(0, ix - r), min(W, ix + r + 1)
            if yb <= ya or xb <= xa:
                continue
            yy, xx = np.ogrid[ya - py:yb - py, xa - px:xb - px]
            field[ya:yb, xa:xb] += amp * np.exp(-(yy ** 2 + xx ** 2) / (2 * sig ** 2)) / max(sig, 0.5)
    if meta is not None:
        meta["streaks"] = n
    return n


def add_fringe_rings(field, rng, cfg: GenConfig, contrast: float, meta=None):
    """Interference ring patterns (arm 61): Newton's rings h(r) = h0 + r^2 npp / (2 R) through height_to_contrast,
    soft footprint exp(-(r / ext)^4), optional soft angular window (a partial arc). Centres may lie up to half a
    footprint outside the frame, so a crop also sees arcs cut by its edge. Unlabelled. Returns the number drawn."""
    rate = rng.uniform(*cfg.fringe_rate_range) if cfg.fringe_rate_range else cfg.fringe_rate
    if rate <= 0:
        return 0
    H, W = field.shape
    n = int(rng.poisson(rate))
    for _ in range(n):
        R = float(np.exp(rng.uniform(*np.log(cfg.fringe_radius_px_range))))
        h0 = float(rng.uniform(*cfg.fringe_h0_nm_range))
        ext = float(rng.uniform(*cfg.fringe_extent_px_range))
        amp = contrast * float(rng.uniform(*cfg.fringe_amp_range))
        ina = float(rng.uniform(*cfg.fringe_ina_range))
        cy, cx = rng.uniform(-0.5 * ext, H + 0.5 * ext), rng.uniform(-0.5 * ext, W + 0.5 * ext)
        arc = rng.random() < cfg.fringe_arc_prob
        th0 = float(rng.uniform(0.0, 2.0 * np.pi)); half = 0.5 * np.deg2rad(float(rng.uniform(*cfg.fringe_arc_deg_range)))
        r0 = int(np.ceil(1.3 * ext))
        y0, y1 = max(0, int(cy) - r0), min(H, int(cy) + r0 + 1)
        x0, x1 = max(0, int(cx) - r0), min(W, int(cx) + r0 + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dy, dx = yy - cy, xx - cx
        r = np.hypot(dy, dx)
        h = h0 + r ** 2 * cfg.fringe_nm_per_px / (2.0 * R)
        c = height_to_contrast(h, cfg, ina) * np.exp(-(r / ext) ** 4)
        if arc:
            d = np.abs(np.angle(np.exp(1j * (np.arctan2(dy, dx) - th0))))
            c = c * np.clip((half - d) / 0.3 + 0.5, 0.0, 1.0)
        field[y0:y1, x0:x1] += amp * c
    if meta is not None:
        meta["fringe_rings"] = n
    return n


def _axis_ratio(yx):
    """sqrt of the eigenvalue ratio of the 2-D point cloud's covariance (1 = isotropic)."""
    if len(yx) < 3:
        return 1.0
    ev = np.linalg.eigvalsh(np.cov(np.asarray(yx, float).T) + 1e-6 * np.eye(2))
    return float(np.sqrt(ev[1] / max(ev[0], 1e-9)))


def _footprint_ratio(patch):
    """Axis ratio of the pixels above 20 % of the patch's peak magnitude."""
    a = np.abs(patch)
    return _axis_ratio(np.argwhere(a > 0.2 * a.max())) if a.max() > 0 else 1.0


def _debris_patch(rng, cfg: GenConfig):
    """One compact debris object as a signed local patch (unit amplitude); None if no compact draw in 10 tries."""
    for _ in range(10):
        if rng.random() < cfg.debris_agg_frac:                                  # aggregate of small particles
            k = int(rng.integers(cfg.debris_agg_n_range[0], cfg.debris_agg_n_range[1] + 1))
            off = rng.normal(0.0, float(rng.uniform(*cfg.debris_agg_sigma_range)), (k, 2))
            if _axis_ratio(off) > cfg.debris_max_axis_ratio:
                continue
            half = int(np.ceil(np.abs(off).max() + 3 * cfg.debris_particle_ext_range[1])) + 1
            yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(float); patch = np.zeros(yy.shape)
            for oy, ox in off:
                R = float(rng.uniform(*cfg.debris_particle_R_range)); ext = float(rng.uniform(*cfg.debris_particle_ext_range))
                ina = float(rng.uniform(*cfg.fringe_ina_range)); h0 = float(rng.uniform(*cfg.debris_particle_h0_nm_range))
                r = np.hypot(yy - oy, xx - ox)
                patch += height_to_contrast(h0 + r ** 2 * cfg.fringe_nm_per_px / (2.0 * R), cfg, ina) * np.exp(-(r / ext) ** 4)
            if _footprint_ratio(patch) > cfg.debris_max_axis_ratio:   # judged on the RENDERED footprint (a 2-particle
                continue                                              # dumbbell has a degenerate offset covariance)
            return patch
        size = float(rng.uniform(*cfg.debris_blob_size_range)); half = int(np.ceil(size / 2)) + 4  # irregular blob
        noise = gaussian_filter(rng.normal(0.0, 1.0, (2 * half + 1, 2 * half + 1)), max(size / 6.0, 1.0))
        yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
        inside = np.hypot(yy, xx) <= size / 2.0
        q = float(rng.uniform(0.4, 0.8))
        shape = inside & (noise >= np.quantile(noise[inside], q))
        lab, nl = ndimage_label(shape)
        if nl == 0:
            continue
        big = np.argmax(np.bincount(lab.ravel())[1:]) + 1; shape = lab == big
        if shape.sum() < 4 or _axis_ratio(np.argwhere(shape)) > cfg.debris_max_axis_ratio:
            continue
        body = gaussian_filter(shape.astype(float), 0.8)
        if rng.random() < cfg.debris_blob_halo_prob:
            w = float(rng.uniform(*cfg.debris_blob_halo_px_range))
            ring = gaussian_filter(shape.astype(float), w) - body
            body = body - 2.0 * np.clip(ring, 0.0, None)
        if _footprint_ratio(body) > cfg.debris_max_axis_ratio:
            continue
        return body * (1.0 if rng.random() < 0.5 else -1.0)
    return None


def add_debris(field, rng, cfg: GenConfig, contrast: float, meta=None):
    """Compact debris / particles (arm 62): aggregates of Newton-profile particles and irregular blobs, each <= the
    axis-ratio bound, amplitude debris_amp_range x the frame's MT contrast, sign random. Unlabelled. Returns the count."""
    rate = rng.uniform(*cfg.debris_rate_range) if cfg.debris_rate_range else cfg.debris_rate
    if rate <= 0:
        return 0
    H, W = field.shape
    n = int(rng.poisson(rate)); drawn = 0
    for _ in range(n):
        patch = _debris_patch(rng, cfg)
        if patch is None:
            continue
        amp = contrast * float(rng.uniform(*cfg.debris_amp_range))
        h2, w2 = patch.shape[0] // 2, patch.shape[1] // 2
        cy, cx = int(rng.integers(0, H)), int(rng.integers(0, W))
        y0, y1, x0, x1 = cy - h2, cy - h2 + patch.shape[0], cx - w2, cx - w2 + patch.shape[1]
        py0, px0 = max(0, -y0), max(0, -x0)
        y0c, y1c, x0c, x1c = max(0, y0), min(H, y1), max(0, x0), min(W, x1)
        if y1c <= y0c or x1c <= x0c:
            continue
        field[y0c:y1c, x0c:x1c] += amp * patch[py0:py0 + (y1c - y0c), px0:px0 + (x1c - x0c)]
        drawn += 1
    if meta is not None:
        meta["debris"] = drawn
    return drawn


def _shot_read_noise(z, rng, gain_range, read_range, meta=None):
    """Same random-draw sequence as always; `meta` (a dict) receives the drawn gain / read sigma so a
    diagnostic can attribute a frame's SNR to its photometric draw without changing the stream."""
    gain = rng.uniform(*gain_range)
    z = rng.poisson(np.clip(z, 0, None) * gain) / gain
    read = rng.uniform(*read_range)
    if meta is not None:
        meta["poisson_gain"] = float(gain); meta["read_noise"] = float(read)
    return z + rng.normal(0, read, z.shape)


def add_irm_noise(img, rng, cfg: GenConfig, meta=None):
    """Poisson shot + gaussian read noise, in [0,1] space, optionally with a spatially correlated share
    and sparse impulses (see the GenConfig noise_* fields). With the defaults this is the original
    function, draw for draw. NO synthetic mid-frequency texture beyond that: the composite is on REAL
    empty-field backgrounds that already carry the coverslip / interference texture."""
    if cfg.noise_alt_prob > 0 and cfg.noise_alt:
        alt = bool(rng.random() < cfg.noise_alt_prob)
        if alt:
            cfg = dataclasses.replace(cfg, noise_alt_prob=0.0,
                                      **{k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg.noise_alt.items()})
        if meta is not None:
            meta["noise_alt"] = alt
    if cfg.noise_ref == "median":
        # ABSOLUTE intensity: z = img / (rel * median), Poisson on that, nothing subtracted and nothing
        # clipped, so a dark filament's 3-10 % dip is never flattened (an earlier draft subtracted
        # med - span/2 and clipped at zero, which put a floor inside the dip range for rel < 0.2 --
        # caught in review 2026-09-04 23:55 before any training). Shot noise then scales with the
        # square root of the LOCAL intensity, as a camera's does; `rel` is the photon-count scale.
        a, b = cfg.noise_span_rel_range
        rel = float(np.exp(rng.uniform(np.log(a), np.log(b)))) if b > a else float(a)
        med = float(np.median(img)); span = max(rel * med, 1e-6); lo = 0.0
        if meta is not None:
            meta["noise_span_rel"] = rel
        z0 = np.clip(img, 0, None) / span
    else:
        lo, hi = np.percentile(img, [0.5, 99.5]); span = max(hi - lo, 1e-6)
        z0 = np.clip((img - lo) / span, 0, 1)
    z = _shot_read_noise(z0, rng, cfg.poisson_gain_range, cfg.read_noise_range, meta=meta)
    corr_hi = cfg.noise_corr_frac_range[1]
    imp_hi = cfg.noise_impulse_frac_range[1]
    if corr_hi > 0 or imp_hi > 0:
        w = z - z0                                   # the white part the camera model produced
        s_w = float(w.std()) + 1e-9
        if corr_hi > 0:
            f = float(rng.uniform(*cfg.noise_corr_frac_range))
            sig = float(rng.uniform(*cfg.noise_corr_sigma_range))
            c = gaussian_filter(rng.normal(0.0, 1.0, z.shape), sig)
            c *= s_w / (c.std() + 1e-9)
            z = z0 + np.sqrt(1.0 - f) * w + np.sqrt(f) * c   # same total variance, a share of it correlated
            if meta is not None:
                meta["noise_corr_frac"] = f; meta["noise_corr_sigma"] = sig
        if imp_hi > 0:
            pf = float(rng.uniform(*cfg.noise_impulse_frac_range))
            hit = rng.random(z.shape) < pf
            amp = rng.uniform(*cfg.noise_impulse_amp_range, size=int(hit.sum())) * s_w
            z[hit] += amp * rng.choice([-1.0, 1.0], size=int(hit.sum()))
            if meta is not None:
                meta["noise_impulse_frac"] = pf
    if cfg.noise_ref == "median":
        return np.clip(z * span, 0.0, None)          # intensities stay non-negative; no upper clip
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
    h_base = sample_base_height(rng, cfg)            # FRAME-level base height regime (assay/ATP: dark⇄bright MTs)
    # FRAME-level render width: apparent MT width is set by the microscope PSF / pixel-size (constant across a
    # frame), NOT a per-MT property — single MTs are all ~25 nm << diffraction limit, so every MT shares one width.
    ws = float(np.clip(rng.normal(cfg.width_mean, cfg.width_std), *cfg.width_clip))
    if cfg.width_thin_prob > 0 and rng.random() < cfg.width_thin_prob:   # arm 63
        ws = float(rng.uniform(*cfg.width_thin_range))
    detach_active = rng.random() < cfg.detach_frame_prob
    for ins in instances:
        cl = ins["centerline"]; n = len(cl)
        amp = contrast * float(np.clip(rng.normal(1.0, cfg.contrast_rel_std), 0.1, 3.0))
        h = mt_height_field(rng, n, cfg, h_base, detach_active)             # height [nm] above the coverslip
        c = height_to_contrast(h, cfg, ina)                                 # signed two-beam interference contrast
        c = np.sign(c) * np.maximum(np.abs(c), cfg.contrast_floor)          # floor: labeled MT never fully invisible
        mod = c * _tip_taper(n, rng, cfg)
        # provenance of the rendered strength, for diagnostics (no effect on the image or the draws)
        ins["render"] = {"amp": float(amp), "h_mean_nm": float(h.mean()), "c_abs_mean": float(np.abs(c).mean()),
                         "mod_abs_mean": float(np.abs(mod).mean())}
        _paint(field, cl, amp, mod, ws, cfg)
    add_spots(field, rng, cfg)
    add_streaks(field, rng, cfg)          # arm 41; streak_rate defaults to 0.0, so this is a no-op
    add_fringe_rings(field, rng, cfg, contrast)  # arm 61; fringe_rate defaults to 0.0, so this is a no-op
    add_debris(field, rng, cfg, contrast)  # arm 62; debris_rate defaults to 0.0, so this is a no-op
    psf = rng.uniform(*cfg.psf_sigma_range)
    field = gaussian_filter(field, psf)
    img = background.astype(np.float64) * (1.0 + field)
    rmeta = {"inverted": False, "detach_active": detach_active, "ina": float(ina), "contrast": float(contrast),
             "h_base_nm": float(h_base), "width_sigma": float(ws), "psf_sigma": float(psf)}
    img = add_irm_noise(img, noise_rng if noise_rng is not None else rng, cfg, meta=rmeta)
    return img, rmeta


def render_tirf(instances, cond, background, rng, cfg: GenConfig, noise_rng=None):
    """TIRF appearance: BRIGHT fluorescent filaments on a DARK background, ADDITIVE (bg + signal),
    NO interference halo, detachment DIMS (less evanescent excitation), fluorescence shot noise."""
    H, W = background.shape
    field = np.zeros((H, W))
    signal = rng.uniform(*cfg.tirf_signal_range)
    ws = float(np.clip(rng.normal(cfg.width_mean, cfg.width_std), *cfg.width_clip))   # FRAME-level (see render_irm)
    detach_active = rng.random() < cfg.detach_frame_prob
    # Labelling density is FRAME-level: the labelled-tubulin ratio is a property of the preparation,
    # so every microtubule in one field shares it. Per-filament brightness spread already comes from
    # contrast_rel_std and _along_intensity.
    lam = 10.0 ** rng.uniform(*cfg.tirf_label_density_log10) if cfg.tirf_label_sparse else None
    for ins in instances:
        cl = ins["centerline"]; n = len(cl)
        amp = signal * float(np.clip(rng.normal(1.0, cfg.contrast_rel_std), 0.1, 3.0))
        mod = along_length_tirf(rng, n, cfg, detach_active) * _tip_taper(n, rng, cfg) * _along_intensity(n, rng, cfg)
        if lam is not None:
            lab = along_label_density(rng, n, lam, cfg)
            # A microtubule that incorporated NO labelled dimer emits nothing and is simply not
            # there in TIRF -- nobody would annotate it either. Marking it lets the caller drop it
            # from the target rather than ask the model to find a filament the image does not
            # contain, which is the one way a synthetic frame can actively teach over-firing.
            # MEASURED SCOPE, so this does not read as more than it is: at the default prior it
            # fires essentially never (0 of 410 filaments over 25 frames), because a 300 px
            # microtubule at 0.1 fluorophores/px draws Poisson(30) and is never empty. It is a
            # floor for extreme density settings, not the cure for dim filaments -- those are dim
            # because of SNR, and 20 % of them were already invisible under the old continuous
            # renderer. The rule is parameter-free on purpose: exactly zero, nothing fitted.
            if not np.any(lab > 0):
                ins["tirf_unlabelled"] = True
                continue
            mod = mod * lab
        _paint(field, cl, amp, mod, ws, cfg)
    field = np.clip(field, 0, None)                                  # fluorescence is emissive (>=0)
    field = gaussian_filter(field, rng.uniform(*cfg.tirf_psf_sigma_range))
    bg = background.astype(np.float64)
    bg = (bg - bg.min()) / (bg.max() - bg.min() + 1e-6)              # normalize bg to [0,1] (dark field)
    img = cfg.tirf_bg_boost * bg + field                            # ADDITIVE bright signal on dark bg
    lo, hi = np.percentile(img, [0.5, 99.5]); span = max(hi - lo, 1e-6)
    tmeta = {"inverted": False, "detach_active": detach_active}
    z = _shot_read_noise(np.clip((img - lo) / span, 0, 1),
                         noise_rng if noise_rng is not None else rng,
                         cfg.tirf_poisson_gain_range, cfg.tirf_read_noise_range, meta=tmeta)
    img = lo + np.clip(z, 0, 1) * span
    return img, tmeta


def flatten_background(background, sigma):
    """Flat-field correction of an empty-field crop: divide by its Gaussian low-pass, keep the mean level.
    Removes illumination gradients and interference fringes wider than ~sigma, keeps the fine coverslip
    texture and the pixel noise."""
    bg = background.astype(np.float64)
    lp = gaussian_filter(bg, sigma, mode="reflect")
    return bg / np.maximum(lp, 1e-6 * max(float(bg.mean()), 1e-12)) * float(bg.mean())


def generate_frame(background, rng, cfg: GenConfig, modality: str = "irm"):
    """Sample MORPHOLOGY once, then RENDER it for the chosen modality. Returns (image, instances, meta).
    Backward compatible: modality defaults to 'irm'."""
    instances, cond = sample_scene(background.shape, rng, cfg)
    render = render_tirf if modality == "tirf" else render_irm
    flat_sigma = None
    if cfg.bg_flatten_prob > 0 and rng.random() < cfg.bg_flatten_prob:
        flat_sigma = float(np.exp(rng.uniform(*np.log(cfg.bg_flatten_sigma_range))))   # log-uniform
        background = flatten_background(background, flat_sigma)
    img, rmeta = render(instances, cond, background, rng, cfg)
    if flat_sigma is not None:
        rmeta["bg_flatten_sigma"] = flat_sigma
    for ins in instances:
        ins["polarity_base"] = 1.0 if modality == "tirf" else -1.0
    meta = {"modality": modality, "regime": cond["regime"], "wavy": cond["wavy"], "hairpin_active": cond["hairpin_active"],
            "orient_kappa": cond["orient_kappa"], "n_instances": len(instances), **rmeta}
    return img, instances, meta
