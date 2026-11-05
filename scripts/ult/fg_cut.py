#!/usr/bin/env python3
"""Report only: mean foreground fraction per cut over all VAL frames, DEB vs THIN, roi303 v2 and htw (docs/TODO.md 2026-11-02 14:30)."""
import glob, os, sys, numpy as np
from multiprocessing import Pool
sys.path.insert(0, "scripts")
from step1_read import source_of
from mt_bench.cvat_import import read_frame_h5
W = "/data2/prusek/mt_work/preds/"; C = (0.90, 0.96, 0.98, 0.99, 0.995, 0.999, 0.9995)
def one(f):
    a = np.load(f)["prob"][0]; return [float((a > c).mean()) for c in C]
htw = [os.path.basename(p).replace(".h5", ".npz") for p in glob.glob("data/real/mt34_htw_eval/*.h5") if source_of(os.path.basename(p)) == "htw"]
with Pool(16) as pl:
    for b, pat, keep in (("roi303", "GAUSS_{}_roiv2_ep40_native", None), ("htw", "{}_hb_ep40_native", set(htw))):
        print("==", b, " cuts", C); T = {}
        for g, ms in (("DEB", [f"DEB_S{k}" for k in (3, 4, 5, 6)]), ("THIN", [f"THIN_S{k}" for k in (3, 4, 5, 6)])):
            fs = [f for m in ms for f in sorted(glob.glob(W + pat.format(m) + "/*.npz")) if keep is None or os.path.basename(f) in keep]; T[g] = np.mean(pl.map(one, fs, chunksize=4), 0)
            print(f"   {g:5s} n {len(fs)} " + " ".join(f"{100 * v:.3f}%" for v in T[g]))
        print("   THIN / DEB " + " ".join(f"{v:.3f}" for v in T["THIN"] / T["DEB"]))
