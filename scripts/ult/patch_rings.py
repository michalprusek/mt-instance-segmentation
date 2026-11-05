#!/usr/bin/env python3
"""ARM 61 (docs/TODO.md 2026-11-01 01:45): add interference-ring distractors to synth/mt_generator.py. Anchored on exact
strings, refuses to apply twice or to a file whose anchors moved. Default OFF: with fringe_rate 0 and no range, no random
number is drawn, so every existing recipe renders bit-identically.
    python patch_rings.py path/to/mt_generator.py"""
import sys
p = sys.argv[1]; s = open(p).read()
if "def add_fringe_rings" in s: sys.exit("already patched")
A1 = "    streak_kink_frac: float = 0.35                   # share given a sharp kink no MT may have\n"
A2 = "def _shot_read_noise(z, rng, gain_range, read_range, meta=None):\n"
A3 = "    add_streaks(field, rng, cfg)          # arm 41; streak_rate defaults to 0.0, so this is a no-op\n"
for a in (A1, A2, A3):
    if s.count(a) != 1: sys.exit(f"anchor not unique / missing: {a!r}")
CFG = A1 + '''    # INTERFERENCE RING DISTRACTORS (arm 61, 2026-11-01). Newton's rings of particles / vesicles resting on or near the
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
'''
FN = '''def add_fringe_rings(field, rng, cfg: GenConfig, contrast: float, meta=None):
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


''' + A2
CALL = A3 + "    add_fringe_rings(field, rng, cfg, contrast)  # arm 61; fringe_rate defaults to 0.0, so this is a no-op\n"
s = s.replace(A1, CFG, 1).replace(A2, FN, 1).replace(A3, CALL, 1)
open(p, "w").write(s); print("patched", p)
