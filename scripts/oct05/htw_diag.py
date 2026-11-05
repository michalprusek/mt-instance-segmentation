#!/usr/bin/env python3
"""D2 (b)(c)(f) declared 2026-10-05: per-GT-curve read. Length, measured FWHM, amplitude SNR, centerline saturation,
metric-matched (declared instrument), map-covered, and matched on the ORACLE band. VAL only.

    python htw_diag.py DATA PRED_DIR THR OUT.json [ORACLE_PRED_DIR] [NAME_PREFIX] [NPROC]
"""
import glob, json, os, sys
import numpy as np
from multiprocessing import Pool
from scipy.ndimage import map_coordinates, maximum_filter, gaussian_filter
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from instance.instancer_a import instance_a
from instance.metrics import centerline_f1
from instance.geometry import resample
from mt_bench.cvat_import import read_frame_h5
UP, KAPPA = 1.5, 0.25
P = {k: v for k, v in json.load(open("src/instance/params_a_derived.json")).items() if not k.startswith("_")}
data, pred, thr, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
orac = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != "-" else None
pref = sys.argv[6] if len(sys.argv) > 6 else ""
NP = int(sys.argv[7]) if len(sys.argv) > 7 else 12
OFF = np.arange(-9, 9.01, 0.5)


def profile(img, pts):
    t = np.gradient(pts, axis=0); t /= (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9); n = np.stack([-t[:, 1], t[:, 0]], 1)
    xs = pts[:, None, 0] + OFF[None] * n[:, None, 0]; ys = pts[:, None, 1] + OFF[None] * n[:, None, 1]
    v = map_coordinates(img, [ys, xs], order=1, mode="nearest")
    return np.median(v, axis=0)


def fwhm(pr):
    base = np.median(np.r_[pr[:6], pr[-6:]]); c = pr - base; mid = len(c) // 2
    k = mid - 2 + int(np.argmax(np.abs(c[mid - 2:mid + 3]))); amp = c[k]
    if abs(amp) < 1e-9: return float("nan"), 0.0
    h = c / amp; lo = k
    while lo > 0 and h[lo] >= 0.5: lo -= 1
    hi = k
    while hi < len(h) - 1 and h[hi] >= 0.5: hi += 1
    l = lo + (0.5 - h[lo]) / (h[lo + 1] - h[lo] + 1e-12) if h[lo] < 0.5 else lo
    r = hi - (0.5 - h[hi]) / (h[hi - 1] - h[hi] + 1e-12) if h[hi] < 0.5 else hi
    return float((r - l) * 0.5), float(amp)


def matched_set(prob, gt, t):
    mask = prob > t
    _, masks = instance_a(mask, KAPPA, params=P, channels=prob[None], prob=prob)
    r = centerline_f1(masks, gt, tol=5.0, length_coverage=0.8, precision_coverage=0.8, end_tol=0.0)
    pr = r.get("pairs", [])
    return r, pr


def one(path):
    name = os.path.basename(path); fr = read_frame_h5(path)
    img = np.asarray(fr["image"], float); lo, hi = np.percentile(img, [1, 99]); z = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)
    hp = z - gaussian_filter(z, 3.0); noise = float(1.4826 * np.median(np.abs(hp - np.median(hp)))) + 1e-9
    prob = np.load(os.path.join(pred, name.replace(".h5", ".npz")))["prob"].astype(np.float32)[0]
    pm = maximum_filter(prob, size=11)
    polys = [np.asarray(q, float) for q in fr["polylines"]]; gt = [q * UP for q in polys]
    r, pairs = matched_set(prob, gt, thr)
    ro, po = (None, None)
    if orac and os.path.exists(os.path.join(orac, name.replace(".h5", ".npz"))):
        op = np.load(os.path.join(orac, name.replace(".h5", ".npz")))["prob"].astype(np.float32); op = op[0] if op.ndim == 3 else op
        ro, po = matched_set(op, gt, 0.5)
    rows = []
    for k, q in enumerate(polys):
        if len(q) < 2: continue
        pts = resample(q, ds=1.0)
        if len(pts) < 3: pts = np.linspace(q[0], q[-1], 3)
        L = float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())
        w, amp = fwhm(profile(z, pts)); e = pts * UP
        xi = np.clip(np.round(e[:, 0]).astype(int), 0, prob.shape[1] - 1); yi = np.clip(np.round(e[:, 1]).astype(int), 0, prob.shape[0] - 1)
        xn = np.clip(np.round(pts[:, 0]).astype(int), 0, z.shape[1] - 1); yn = np.clip(np.round(pts[:, 1]).astype(int), 0, z.shape[0] - 1)
        rows.append(dict(frame=name, k=k, len=L, fwhm=w, amp=amp, snr=abs(amp) / noise, sat=float(((z[yn, xn] <= 0) | (z[yn, xn] >= 1)).mean()),
                         cov=float((pm[yi, xi] >= thr).mean()), cov50=float((pm[yi, xi] >= 0.5).mean())))
    return name, rows, dict(tp=r["tp"], fp=r["fp"], fn=r["fn"]), [(p["gt"], bool(p["tp"])) for p in pairs], ([(p["gt"], bool(p["tp"])) for p in po] if po is not None else None), \
        (dict(tp=ro["tp"], fp=ro["fp"], fn=ro["fn"]) if ro else None)


if __name__ == "__main__":
    paths = [p for p in sorted(glob.glob(os.path.join(data, pref + "*.h5")))
             if str(read_frame_h5(p)["attrs"].get("split")) == "val" and os.path.exists(os.path.join(pred, os.path.basename(p).replace(".h5", ".npz")))]
    with Pool(NP) as pool:
        res = pool.map(one, paths, chunksize=1)
    json.dump(res, open(out, "w")); print(len(res), "frames", sum(len(r[1]) for r in res), "curves; pairs sample:", res[0][3][:5])
