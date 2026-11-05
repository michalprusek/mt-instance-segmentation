#!/usr/bin/env python3
"""S0, a temporal window with no training (docs/TODO.md 2026-11-03 21:10): V16 on the mean image and on the mean map of t-1, t, t+1, plus where one curve becomes two.
usage: fuse.py PKG WEIGHTS FRAMES_DIR LIST_JSON V16_JSONS(comma) GEOM_DIR OUT_JSON k/K [noop]"""
import json, os, sys
from collections import defaultdict
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
PKG, W, ROOT, LST, POL, GEOM, OUT = sys.argv[1:8]; k, K = (int(x) for x in sys.argv[8].split("/"))
sys.path[:0] = [os.path.dirname(PKG), PKG, os.path.join(PKG, "vendor"), GEOM]
from wrapper import MicrotubuleModel, UP, KAPPA_MAX
from instance.instancer_a import instance_a
from instance.instancer_lh import TooManyArcs, instance_lh
from mt_geometry_cost import resample
from griddrift import best_shift
NOOP = len(sys.argv) > 9 and sys.argv[9] == "noop"
TOL = 5 / 1.5; MINCOV = 20; CUT = 0.995
m = MicrotubuleModel().load_weights(W)
print("params", m._params_path.name, m.params.get("fusion"), m.params.get("decoder"), m.params.get("prob_thr"), flush=True)
assert m.params.get("decoder") == "lh" and m.params.get("prob_thr") == CUT
P = {}
for p in POL.split(","): P.update(json.load(open(p)))
def disk(r):
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return yy ** 2 + xx ** 2 <= r ** 2
FOOT = {"r3": disk(3), "r5": disk(5)}   # eval px: 2 native px tolerates registration, 3.33 native is the metric's tolerance and fills gaps up to 6.7 native px
def read(p):
    a = np.asarray(Image.open(p)); assert a.ndim == 2, (p, a.shape)
    return a.astype(np.float32)
def pts(cs):
    a = [resample(c, 1.0) for c in cs]
    return np.vstack(a) if a else np.zeros((0, 2))
def move(a, d):
    """Whole-pixel shift with replicated edges; a is (H, W) or (C, H, W). A point at p in the source lands at p + d."""
    dy, dx = int(np.rint(d[0])), int(np.rint(d[1]))
    if dy == 0 and dx == 0: return a
    H, W = a.shape[-2:]; y = np.clip(np.arange(H) - dy, 0, H - 1); x = np.clip(np.arange(W) - dx, 0, W - 1)
    return a[..., y, :][..., :, x]
def decode(chans):
    """The wrapper's L+H call on a given chans_eval, float16 rounding included."""
    prob = chans.max(axis=0)
    try: pl = instance_lh(prob.astype(np.float16).astype(np.float32), CUT, KAPPA_MAX, m.params)
    except TooManyArcs: pl, _ = instance_a(prob > CUT, KAPPA_MAX, m.params, channels=chans, prob=prob, return_masks=False)
    return [np.asarray(c, dtype=np.float64)[:, ::-1] / UP for c in pl]
def wrap_output(cs):
    from wrapper import _simplify_polyline
    eps = float(m.params.get("polyline_eps_px", 0.0) or 0.0)
    return [(_simplify_polyline(c, eps) if eps > 0 else c).tolist() for c in cs]
def attribute(A, B, shift, peak):
    """One curve of A continuing as two curves of B: bundle, map or decoder. A, B resampled at 1 px; peak is B's frame at the eval scale."""
    out = dict(bundle=0, map_r3=0, decoder_r3=0, map_r5=0, decoder_r5=0)
    if not A or not B: return out
    own = np.concatenate([np.full(len(b), i) for i, b in enumerate(B)]); T = cKDTree(np.vstack(B))
    for a in A:
        q = a + shift; near = T.query_ball_point(q, TOL); cov = defaultdict(int)
        for hits in near:
            for c in {own[h] for h in hits}: cov[c] += 1
        two = [c for c, v in cov.items() if v >= MINCOV]
        if len(two) < 2: continue
        two = set(two); both = sum(len({own[h] for h in hits} & two) >= 2 for hits in near)
        if both > 0.5 * sorted(cov[c] for c in two)[-2]: out["bundle"] += 1; continue
        d, idx = T.query(q); lab = np.array([own[i] if (x <= TOL and own[i] in two) else -1 for i, x in zip(idx, d)])
        on = np.flatnonzero(lab >= 0)
        for i, j in zip(on, on[1:]):
            if lab[i] == lab[j]: continue
            s = np.rint(q[i:j + 1] * UP).astype(int); H, W = peak["r3"].shape; s[:, 0] = s[:, 0].clip(0, H - 1); s[:, 1] = s[:, 1].clip(0, W - 1)
            for r in ("r3", "r5"): out[("map_" if peak[r][s[:, 0], s[:, 1]].min() < CUT else "decoder_") + r] += 1
    return out
R = {}
for r in json.load(open(LST))[k::K]:
    vid = r["vid"]; F = r["frames"]; p = [os.path.join(ROOT, f["path"]) for f in F]; T = len(F)
    C = [[np.asarray(c, dtype=float) for c in P[vid]["frames"][f["frame"]]["polys"] if len(c) > 1] for f in F]; Pt = [pts(c) for c in C]
    if NOOP:
        t = T // 2; a = read(p[t]); base = P[vid]["frames"][F[t]["frame"]]["polys"]
        same = lambda x: len(x) == len(base) and all(np.array_equal(np.asarray(u), np.asarray(v)) for u, v in zip(x, base))
        mean = np.mean([a, move(a, np.zeros(2)), move(a, np.zeros(2))], axis=0).astype(a.dtype)
        b = [c.tolist() for c in m.predict(mean)["centerlines_rc"]]
        ch = m.infer_maps(a)["chans_eval"].astype(np.float32); c = wrap_output(decode(np.mean([ch, move(ch, np.zeros(2)), move(ch, np.zeros(2))], axis=0)))
        print(vid[:8], "noop meanimg", same(b), "meanmap", same(c), len(base), flush=True); assert same(b) and same(c)
        continue
    imgs = [read(x) for x in p]; ch = []; attr = defaultdict(int)
    for i in range(T):
        ch.append(m.infer_maps(imgs[i])["chans_eval"].astype(np.float16))
    peak = [{r: ndi.maximum_filter(c.max(axis=0).astype(np.float32), footprint=f).astype(np.float16) for r, f in FOOT.items()} for c in ch]
    res = [dict(meanimg=None, meanmap=None) for _ in range(T)]
    fwd = [best_shift(Pt[i], Pt[i + 1]) for i in range(T - 1)]   # a point of i lands at + fwd[i] in i + 1
    for i in range(T - 1):
        if F[i]["md5"] == F[i + 1]["md5"]: continue
        A, B = [resample(c, 1.0) for c in C[i]], [resample(c, 1.0) for c in C[i + 1]]
        for key, v in attribute(A, B, fwd[i], peak[i + 1]).items(): attr[key] += v
        for key, v in attribute(B, A, -fwd[i], peak[i]).items(): attr[key] += v
    del peak
    for i in range(1, T - 1):
        dm, dp = fwd[i - 1], -fwd[i]      # bring t-1 and t+1 onto t
        img = np.mean([imgs[i], move(imgs[i - 1], dm), move(imgs[i + 1], dp)], axis=0, dtype=np.float32)
        o = m.predict(img); res[i]["meanimg"] = [np.asarray(c).tolist() for c in o["centerlines_rc"]]; dec = o["decoder"]
        fused = np.mean([ch[i].astype(np.float32), move(ch[i - 1], dm * UP).astype(np.float32), move(ch[i + 1], dp * UP).astype(np.float32)], axis=0)
        res[i]["meanmap"] = wrap_output(decode(fused))
        print(vid[:8], F[i]["frame"], "n", len(C[i]), len(res[i]["meanimg"]), len(res[i]["meanmap"]), dec, flush=True)
    R[vid] = dict(attr=dict(attr), shift=[list(map(float, x)) for x in fwd], frames={F[i]["frame"]: res[i] for i in range(1, T - 1)})
    json.dump(R, open(OUT, "w"))
print("FUSE_DONE", k, flush=True)
