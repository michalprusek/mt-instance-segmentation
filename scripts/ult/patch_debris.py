#!/usr/bin/env python3
"""ARM 62 (docs/TODO.md 2026-11-01 07:55): add compact debris / particle distractors to synth/mt_generator.py (on top of
the arm-61 patch). Anchored on exact strings, refuses to apply twice. Default OFF: with debris_rate 0 and no range, no
random number is drawn, so every existing recipe (V14, V15) renders bit-identically.
    python patch_debris.py path/to/mt_generator.py"""
import sys
p = sys.argv[1]; s = open(p).read()
if "def add_debris" in s: sys.exit("already patched")
if "def add_fringe_rings" not in s: sys.exit("needs the arm-61 patch first")
A1 = "    fringe_nm_per_px: float = 100.0\n"
A2 = "def _shot_read_noise(z, rng, gain_range, read_range, meta=None):\n"
A3 = "    add_fringe_rings(field, rng, cfg, contrast)  # arm 61; fringe_rate defaults to 0.0, so this is a no-op\n"
for a in (A1, A2, A3):
    if s.count(a) != 1: sys.exit(f"anchor not unique / missing: {a!r}")
CFG = A1 + '''    # COMPACT DEBRIS / PARTICLES (arm 62, 2026-11-01). Measured 2026-10-31: 25 % of the htw off-annotation predictions are
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
'''
FN = '''def _axis_ratio(yx):
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


''' + A2
CALL = A3 + "    add_debris(field, rng, cfg, contrast)  # arm 62; debris_rate defaults to 0.0, so this is a no-op\n"
s = s.replace(A1, CFG, 1).replace(A2, FN, 1).replace(A3, CALL, 1)
if "ndimage_label" not in s.split("def _axis_ratio")[0]:
    s = s.replace("from scipy.ndimage import ", "from scipy.ndimage import label as ndimage_label, ", 1)
open(p, "w").write(s); print("patched", p)
