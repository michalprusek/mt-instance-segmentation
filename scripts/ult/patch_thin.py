#!/usr/bin/env python3
"""ARM 63 (docs/TODO.md 2026-11-02): a frame-level THIN regime for the IRM render. Measured: htw filament FWHM median 2.61 px, 80 % <= 3
px; roi303 4.43 px; V16 synth 4.62 px, 4 % <= 3 px. Default OFF: with width_thin_prob 0 no random number is drawn, so every existing
recipe renders bit-identically. usage: patch_thin.py path/to/mt_generator.py"""
import sys
p = sys.argv[1]; s = open(p).read()
if "width_thin_prob" in s: sys.exit("already patched")
A1 = "    width_clip: Tuple[float, float] = (0.7, 2.4)\n"
A2 = "    ws = float(np.clip(rng.normal(cfg.width_mean, cfg.width_std), *cfg.width_clip))\n    detach_active = rng.random() < cfg.detach_frame_prob\n    for ins in instances:\n        cl = ins[\"centerline\"]; n = len(cl)\n        amp = contrast * float(np.clip(rng.normal(1.0, cfg.contrast_rel_std), 0.1, 3.0))\n        h = mt_height_field"
for a in (A1, A2):
    if s.count(a) != 1: sys.exit(f"anchor not unique / missing: {a!r}")
C = A1 + """    # THIN REGIME (arm 63, 2026-11-02): with this probability a frame draws its core half-width from width_thin_range instead,
    # reaching FWHM ~1.4-3.6 px with the shipped PSF. The upper range is untouched: this widens the prior downward, it does not
    # refit it. Measured: htw 80 % of filaments <= 3 px FWHM vs 4 % in the V16 render. 0 = OFF, no draw.
    width_thin_prob: float = 0.0
    width_thin_range: Tuple[float, float] = (0.35, 1.2)
"""
L = "    ws = float(np.clip(rng.normal(cfg.width_mean, cfg.width_std), *cfg.width_clip))\n"
NEW = L + "    if cfg.width_thin_prob > 0 and rng.random() < cfg.width_thin_prob:   # arm 63\n        ws = float(rng.uniform(*cfg.width_thin_range))\n"
s = s.replace(A1, C, 1).replace(A2, A2.replace(L, NEW, 1), 1)
open(p, "w").write(s); print("patched", p)
