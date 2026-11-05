#!/usr/bin/env python3
"""Diagnostic (docs/TODO.md 2026-11-04 17:05): correlation of background high-pass residuals between consecutive frames, real windows against static synthetic clips.
usage: noise_corr.py GEOM_DIR FRAMES_DIR LIST_JSON V16_JSONS(comma) CLIPS_DIR"""
import glob, json, os, sys
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
GEOM, ROOT, LST, POL, CLIPS = sys.argv[1:6]; sys.path.insert(0, GEOM)
from mt_geometry_cost import resample
from griddrift import best_shift
def stretch(a):
    lo, hi = np.percentile(a, [1, 99]); return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)
def resid(a): a = stretch(a.astype(np.float64)); return a - ndi.gaussian_filter(a, 2.0)
def bgmask(shape, curves_rc):
    m = np.ones(shape, bool)
    for c in curves_rc:
        q = np.rint(resample(np.asarray(c, float), 1.0)).astype(int); ok = (q[:, 0] >= 0) & (q[:, 0] < shape[0]) & (q[:, 1] >= 0) & (q[:, 1] < shape[1]); m[q[ok, 0], q[ok, 1]] = False
    m = ndi.distance_transform_edt(m) > 8; m[:16] = m[-16:] = False; m[:, :16] = m[:, -16:] = False; return m
def move(a, d):
    dy, dx = int(np.rint(d[0])), int(np.rint(d[1])); H, W = a.shape
    return a[np.clip(np.arange(H) - dy, 0, H - 1)][:, np.clip(np.arange(W) - dx, 0, W - 1)]
def pair(a, b, ca, cb):
    """a, b images; ca, cb curves (row, col). b is moved back onto a."""
    pa = np.vstack([resample(np.asarray(c, float), 1.0) for c in ca if len(c) > 1]); pb = np.vstack([resample(np.asarray(c, float), 1.0) for c in cb if len(c) > 1])
    d = best_shift(pa, pb); ra, rb = resid(a), move(resid(b), -d); m = bgmask(a.shape, ca) & move(bgmask(b.shape, cb), -d)
    x, y = ra[m], rb[m]; return float(np.corrcoef(x, y)[0, 1]), float(x.std())
P = {}
for p in POL.split(","): P.update(json.load(open(p)))
real = []
for r in json.load(open(LST)):
    F = r["frames"]; v = r["vid"]; cs = []
    for i in range(0, len(F) - 1, 3):
        if F[i]["md5"] == F[i + 1]["md5"]: continue
        a, b = (np.asarray(Image.open(os.path.join(ROOT, F[j]["path"]))).astype(np.float64) for j in (i, i + 1))
        cs.append(pair(a, b, P[v]["frames"][F[i]["frame"]]["polys"], P[v]["frames"][F[i + 1]["frame"]]["polys"]))
    c = np.array(cs); real.append(c.mean(0)); print("real", v[:8], f"corr {c[:, 0].mean():.3f} resid sd {c[:, 1].mean():.4f}", flush=True)
syn = []
for path in sorted(glob.glob(os.path.join(CLIPS, "static_*.npy"))):
    imgs = np.load(path); gt = json.load(open(path[:-4] + ".json"))["gt"]; cs = []
    for i in range(0, len(imgs) - 1, 2):
        cs.append(pair(imgs[i], imgs[i + 1], [np.asarray(p)[:, ::-1] for p in gt[i].values()], [np.asarray(p)[:, ::-1] for p in gt[i + 1].values()]))
    c = np.array(cs); syn.append(c.mean(0)); print("synth", os.path.basename(path)[:-4], f"corr {c[:, 0].mean():.3f} resid sd {c[:, 1].mean():.4f}", flush=True)
R, Sy = np.array(real), np.array(syn)
print(f"REAL  12 windows: residual correlation t,t+1 median {np.median(R[:, 0]):.3f} (range {R[:, 0].min():.3f}-{R[:, 0].max():.3f}), resid sd median {np.median(R[:, 1]):.4f}")
print(f"SYNTH 12 static : residual correlation t,t+1 median {np.median(Sy[:, 0]):.3f} (range {Sy[:, 0].min():.3f}-{Sy[:, 0].max():.3f}), resid sd median {np.median(Sy[:, 1]):.4f}")
print("NOISE_CORR_DONE")
