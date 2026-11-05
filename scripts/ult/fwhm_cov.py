#!/usr/bin/env python3
"""ARM 63 pre-screen (docs/TODO.md 2026-11-02 07:15): filament FWHM, one estimator on all sets. For each labelled curve >= 20 native px:
perpendicular profiles +-10 px at 1 px steps along it on the native image, each re-centred on the extremum within +-3 px of the smoothed
profile (so human polyline offset does not inflate real and exact labels do not deflate synth), median profile, baseline = median of
|d| >= 7, FWHM by linear interpolation at half height. Real: htw VAL, roi303 v2 VAL. Synth: V16 frames rendered as the trainer does
(640 px crop, IRM). usage: fwhm_cov.py N_SYNTH"""
import dataclasses, glob, json, os, sys
import numpy as np
from multiprocessing import Pool
from PIL import Image
from scipy.ndimage import map_coordinates, gaussian_filter1d
sys.path[:0] = [".", "src", "scripts", "synth", "third_party"]
from instance.geometry import resample
from mt_bench.cvat_import import read_frame_h5
from step1_read import source_of
import mt_generator as N
from gen_train import build_cfg
OFF = np.arange(-10.0, 10.01, 1.0); SUB = np.arange(-7.0, 7.01, 0.25)
def fwhm(img, q):
    p = resample(np.asarray(q, float), ds=1.0)
    if len(p) < 20: return None
    t = np.gradient(p, axis=0); t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9; nx, ny = -t[:, 1], t[:, 0]
    xs = p[:, 0, None] + nx[:, None] * OFF[None]; ys = p[:, 1, None] + ny[:, None] * OFF[None]
    if xs.min() < 0 or ys.min() < 0 or xs.max() > img.shape[1] - 1 or ys.max() > img.shape[0] - 1:
        k = (xs.min(1) >= 0) & (ys.min(1) >= 0) & (xs.max(1) <= img.shape[1] - 1) & (ys.max(1) <= img.shape[0] - 1)
        if k.sum() < 20: return None
        xs, ys = xs[k], ys[k]
    pr = map_coordinates(img, [ys, xs], order=1); pr = gaussian_filter1d(pr, 3.0, axis=0)              # along the curve: beats the noise
    base = np.median(pr[:, np.abs(OFF) >= 7], axis=1, keepdims=True); pr = pr - base
    ctr = np.abs(OFF) <= 3; sgn = np.sign(np.median(pr[:, np.abs(OFF) <= 1]))
    if sgn == 0: return None
    s = sgn * pr; ali = np.zeros((len(s), len(SUB)))
    for i, r in enumerate(s):
        c = OFF[ctr][np.argmax(r[ctr])]; ali[i] = np.interp(SUB + c, OFF, r, left=0, right=0)
    m = np.median(ali, 0); h = m.max()
    if h <= 0: return None
    i0 = int(np.argmax(m)); a = i0
    while a > 0 and m[a] > h / 2: a -= 1
    b = i0
    while b < len(m) - 1 and m[b] > h / 2: b += 1
    if m[a] > h / 2 or m[b] > h / 2: return None
    la = SUB[a] + (h / 2 - m[a]) / (m[a + 1] - m[a] + 1e-12) * 0.25; lb = SUB[b - 1] + (m[b - 1] - h / 2) / (m[b - 1] - m[b] + 1e-12) * 0.25
    return float(lb - la), float(h)
def real(path):
    fr = read_frame_h5(path); name = os.path.basename(path)
    if str(fr["attrs"].get("split", "")) != "val": return []
    if "htw_eval" in path and source_of(name) != "htw": return []
    img = np.asarray(fr["image"], np.float64); out = []
    for q in fr["polylines"]:
        if len(q) >= 2:
            v = fwhm(img, q)
            if v: out.append(v[0])
    return out
_c = json.load(open("calib_V16_debris.json")); CFG = dataclasses.replace(build_cfg(_c["best_params"], mask_hw=1.0), **_c["gen_overrides"])
BG = sorted(glob.glob("/data2/prusek/mt_work/IRM_backgrounds_v2/*.tif"))
def syn(seed):
    rng = np.random.default_rng(630000 + seed); b = np.asarray(Image.open(BG[int(rng.integers(len(BG)))]).convert("F"), np.float32)
    y, x = int(rng.integers(0, b.shape[0] - 640)), int(rng.integers(0, b.shape[1] - 640)); img, inst, meta = N.generate_frame(b[y:y + 640, x:x + 640], rng, CFG)
    out = []
    for z in inst[:40]:
        v = fwhm(np.asarray(img, np.float64), z["centerline"])
        if v: out.append(v[0])
    ws, ps = meta["width_sigma"], meta["psf_sigma"]
    return out, float(np.sqrt(ws ** 2 + ps ** 2))
if __name__ == "__main__":
    n = int(sys.argv[1]); R = {}
    with Pool(40) as pl:
        R["htw"] = [x for o in pl.map(real, sorted(glob.glob("data/real/mt34_htw_eval/*.h5"))) for x in o]
        R["roi303"] = [x for o in pl.map(real, sorted(glob.glob("data/real/roi303_v2/*.h5"))) for x in o]
        sy = pl.map(syn, range(n))
    R["synth V16"] = [x for o, _ in sy for x in o]; se = np.array([s for _, s in sy])
    print("set        n     p10   p25   p50   p75   p90   | share FWHM<=2.5  <=3.0  <=3.5  <=4.0")
    for k, v in R.items():
        v = np.array(v); print(f"{k:10s} {len(v):5d} " + " ".join(f"{q:5.2f}" for q in np.percentile(v, [10, 25, 50, 75, 90])) + "  |      " + "  ".join(f"{(v <= t).mean():.3f}" for t in (2.5, 3.0, 3.5, 4.0)))
    print("synth frame sigma_eff: p5 %.2f p50 %.2f p95 %.2f -> nominal FWHM %.2f %.2f %.2f; share of frames nominal FWHM <= 3: %.3f" % (*np.percentile(se, [5, 50, 95]), *(2.355 * np.percentile(se, [5, 50, 95])), (2.355 * se <= 3).mean()))
    json.dump({k: list(map(float, v)) for k, v in R.items()}, open("/data2/prusek/mt_work/runs/ult/fwhm_cov.json", "w"))
