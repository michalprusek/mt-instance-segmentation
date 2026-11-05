#!/usr/bin/env python3
"""How does the L+H decoder's time scale with the number of arcs? Real VAL frames and synthetic over-firing maps. usage: lh_scale.py HUB_PKG DUMP_DIR NAMES.json"""
import json, os, sys, time
import numpy as np
from multiprocessing import Pool
from scipy.ndimage import gaussian_filter
PKG = sys.argv[1]; sys.path[:0] = [PKG, os.path.join(PKG, "vendor")]
import instance.instancer_lh as L
PAR = {k: v for k, v in json.load(open(os.path.join(PKG, "params_v16.json"))).items() if not k.startswith("_")}
def measure(prob):
    t0 = time.perf_counter(); mask = L.hysteresis_mask(prob, 0.995, PAR["hyst_t_lo"], PAR["hyst_radius"]); arcs, arms = L._arc_graph(mask, PAR); t1 = time.perf_counter()
    logit = L.to_logit(prob); partner = L._match_ends(arms, logit, 0.25, PAR) if arcs else {}; t2 = time.perf_counter()
    return len(arcs), t1 - t0, t2 - t1
def real(a):
    d, n = a; prob = np.load(os.path.join(d, n.replace(".h5", ".npz")))["prob"].astype(np.float32)[0]; return (n,) + measure(prob)
def synth(a):
    size, sigma, q, seed = a; rng = np.random.default_rng(seed); f = gaussian_filter(rng.normal(size=(size, size)), sigma); t = np.quantile(f, q)
    prob = np.where(f > t, 0.9995, 0.3).astype(np.float32); return (f"synth {size} s{sigma} q{q}",) + measure(prob)
if __name__ == "__main__":
    d, names = sys.argv[2], json.load(open(sys.argv[3]))
    with Pool(16) as p: R = p.map(real, [(d, n) for n in names], chunksize=1)
    R.sort(key=lambda r: r[1]); print("real frames, arcs: median", R[len(R) // 2][1], "p95", R[int(len(R) * .95)][1], "max", R[-1][1], "| densest:", [(r[0][:14], r[1], round(r[2], 1), round(r[3], 1)) for r in R[-4:]], flush=True)
    grid = [(768, 1.5, 0.9, 0), (1024, 1.5, 0.9, 0), (1280, 1.5, 0.9, 0), (1536, 1.5, 0.9, 0), (1792, 1.5, 0.9, 0), (2048, 1.5, 0.9, 0)]
    with Pool(6) as p:
        for r in p.imap(synth, grid): print(r[0], "arcs", r[1], "graph s", round(r[2], 1), "match s", round(r[3], 1), flush=True)
