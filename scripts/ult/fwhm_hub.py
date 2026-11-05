#!/usr/bin/env python3
"""Report only (docs/TODO.md 2026-11-02 14:10): which benchmark block do the hub's verified MT frames resemble in filament width? Same estimator as fwhm_cov.py, lifted verbatim."""
import glob, os, re, sys
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter1d
sys.path[:0] = ["src", "scripts"]
from instance.geometry import resample
from mt_bench.cvat_import import read_frame_h5
src = open(sys.argv[1]).read(); body = src[src.index("OFF = np.arange"):src.index("def real(")]; exec(body)
rows = []
for p in sorted(glob.glob("data/real/spheroseg_verified/h5/*.h5")):
    fr = read_frame_h5(p); img = np.asarray(fr["image"], np.float64); v = [r[0] for r in (fwhm(img, q) for q in fr["polylines"] if len(q) >= 2) if r]
    rows += v; print(os.path.basename(p)[:44], img.shape, img.dtype, "curves", len(fr["polylines"]), "measured", len(v), "FWHM median %.2f <=3px %.0f%%" % (np.median(v), 100 * np.mean(np.array(v) <= 3)) if v else "none", dict(fr["attrs"]).get("modality", ""))
print("ALL n", len(rows), "median %.2f p10 %.2f p90 %.2f <=3px %.0f%%" % (np.median(rows), *np.percentile(rows, [10, 90]), 100 * np.mean(np.array(rows) <= 3)))
print("reference: htw median 2.61 (80 % <=3), roi303 v2 4.43 (5 %), V16 synth 4.62 (4 %), V17 3.91 (27 %)")
