#!/usr/bin/env python3
"""Report only: FWHM of a labelled set with the estimator of fwhm_cov.py, lifted verbatim. usage: fwhm_set.py DATA mt34|all"""
import glob, os, sys, collections
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter1d
sys.path[:0] = [".", "src", "scripts"]
from instance.geometry import resample
from mt_bench.cvat_import import read_frame_h5
from step1_read import source_of
src = open("fwhm_cov.py").read(); exec(src[src.index("OFF = np.arange"):src.index("def real(")])
S = collections.defaultdict(list)
for p in sorted(glob.glob(sys.argv[1] + "/*.h5")):
    fr = read_frame_h5(p); n = os.path.basename(p)
    if str(fr["attrs"].get("split", "")) != "val" or (sys.argv[2] == "mt34" and source_of(n) == "htw"): continue
    img = np.asarray(fr["image"], np.float64); S[source_of(n)] += [r[0] for r in (fwhm(img, q) for q in fr["polylines"] if len(q) >= 2) if r]
S["ALL"] = [x for v in list(S.values()) for x in v]
for k, v in S.items(): print(k, "n", len(v), "median %.2f p10 %.2f p90 %.2f <=3px %.0f%%" % (np.median(v), *np.percentile(v, [10, 90]), 100 * np.mean(np.array(v) <= 3)))
print("reference: htw 2.61 (80 %), roi303 v2 4.43 (5 %), hub verified 3.30 (20 %), V16 synth 4.62 (4 %), V17 3.91 (27 %)")
