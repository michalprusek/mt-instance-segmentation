#!/usr/bin/env python3
"""S0, the two arms added 21:20 (docs/TODO.md 2026-11-03 21:10): median of the three maps at the deployed cut, and their mean at the cut keeping frame t's own foreground share.
usage: fuse2.py PKG WEIGHTS FRAMES_DIR LIST_JSON V16_JSONS(comma) GEOM_DIR OUT_JSON k/K [noop]"""
import json, os, sys
import numpy as np
from PIL import Image
PKG, W, ROOT, LST, POL, GEOM, OUT = sys.argv[1:8]; k, K = (int(x) for x in sys.argv[8].split("/"))
sys.path[:0] = [os.path.dirname(PKG), PKG, os.path.join(PKG, "vendor"), GEOM]
from wrapper import MicrotubuleModel, UP, KAPPA_MAX, _simplify_polyline
from instance.instancer_a import instance_a
from instance.instancer_lh import TooManyArcs, instance_lh
from mt_geometry_cost import resample
from griddrift import best_shift
NOOP = len(sys.argv) > 9 and sys.argv[9] == "noop"
CUT = 0.995
m = MicrotubuleModel().load_weights(W)
assert m.params.get("decoder") == "lh" and m.params.get("prob_thr") == CUT
P = {}
for p in POL.split(","): P.update(json.load(open(p)))
def read(p):
    a = np.asarray(Image.open(p)); assert a.ndim == 2, (p, a.shape)
    return a.astype(np.float32)
def pts(cs):
    a = [resample(c, 1.0) for c in cs]
    return np.vstack(a) if a else np.zeros((0, 2))
def move(a, d):
    dy, dx = int(np.rint(d[0])), int(np.rint(d[1]))
    if dy == 0 and dx == 0: return a
    H, W_ = a.shape[-2:]; y = np.clip(np.arange(H) - dy, 0, H - 1); x = np.clip(np.arange(W_) - dx, 0, W_ - 1)
    return a[..., y, :][..., :, x]
def decode(chans, cut):
    prob = chans.max(axis=0)
    try: pl = instance_lh(prob.astype(np.float16).astype(np.float32), cut, KAPPA_MAX, m.params)
    except TooManyArcs: pl, _ = instance_a(prob > cut, KAPPA_MAX, m.params, channels=chans, prob=prob, return_masks=False)
    eps = float(m.params.get("polyline_eps_px", 0.0) or 0.0)
    return [(_simplify_polyline(c, eps) if eps > 0 else c).tolist() for c in (np.asarray(x, dtype=np.float64)[:, ::-1] / UP for x in pl)]
def fused_cut(mean, own):
    """The cut on the mean map that keeps the foreground share frame t has at CUT, rounded as the decoder rounds."""
    share = float((own.astype(np.float16).astype(np.float32) > CUT).mean())
    rounded = mean.max(axis=0).astype(np.float16).astype(np.float32)
    return float(np.quantile(rounded, 1.0 - share)), share, float((rounded > CUT).mean())
R = {}
for r in json.load(open(LST))[k::K]:
    vid = r["vid"]; F = r["frames"]; p = [os.path.join(ROOT, f["path"]) for f in F]; T = len(F)
    C = [[np.asarray(c, dtype=float) for c in P[vid]["frames"][f["frame"]]["polys"] if len(c) > 1] for f in F]; Pt = [pts(c) for c in C]
    if NOOP:
        t = T // 2; base = P[vid]["frames"][F[t]["frame"]]["polys"]; ch = m.infer_maps(read(p[t]))["chans_eval"].astype(np.float32)
        same = lambda x: len(x) == len(base) and all(np.array_equal(np.asarray(u), np.asarray(v)) for u, v in zip(x, base))
        q = fused_cut(np.mean([ch, ch, ch], axis=0), ch.max(axis=0))[0]
        a = decode(np.median([ch, ch, ch], axis=0), CUT); b = decode(np.mean([ch, ch, ch], axis=0), q)
        print(vid[:8], "noop medianmap", same(a), "meanmap_q", same(b), f"cut {q:.5f}", len(base), len(b), flush=True); assert same(a)
        continue
    ch = [m.infer_maps(read(x))["chans_eval"].astype(np.float16) for x in p]
    fwd = [best_shift(Pt[i], Pt[i + 1]) for i in range(T - 1)]; res = {}
    for i in range(1, T - 1):
        st = np.stack([ch[i].astype(np.float32), move(ch[i - 1], fwd[i - 1] * UP).astype(np.float32), move(ch[i + 1], -fwd[i] * UP).astype(np.float32)])
        mean = st.mean(axis=0); q, share, raw = fused_cut(mean, ch[i].astype(np.float32).max(axis=0))
        res[F[i]["frame"]] = dict(medianmap=decode(np.median(st, axis=0), CUT), meanmap_q=decode(mean, q), cut_q=q, share=share, share_mean_at_cut=raw)
        x = res[F[i]["frame"]]; print(vid[:8], F[i]["frame"], "n", len(C[i]), len(x["medianmap"]), len(x["meanmap_q"]), f"q {q:.5f} share {share:.4f} mean@cut {raw:.4f}", flush=True)
    R[vid] = dict(frames=res)
    json.dump(R, open(OUT, "w"))
print("FUSE2_DONE", k, flush=True)
