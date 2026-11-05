#!/usr/bin/env python3
"""S1 (docs/TODO.md 2026-11-04 08:10): V16 maps through L+H with the temporal vote, the lam ladder, on the real windows or the synthetic clips. One pass: votes come from the per-frame curves of t-1 and t+1 only.
usage: s1_links.py PKG WEIGHTS GEOM_DIR OUT_JSON k/K real FRAMES_DIR LIST_JSON V16_JSONS(comma)
       s1_links.py PKG WEIGHTS GEOM_DIR OUT_JSON k/K synth CLIPS_DIR"""
import glob, json, os, sys
import numpy as np
from PIL import Image
PKG, W, GEOM, OUT = sys.argv[1:5]; k, K = (int(x) for x in sys.argv[5].split("/")); MODE = sys.argv[6]
sys.path[:0] = [os.path.dirname(PKG), PKG, os.path.join(PKG, "vendor"), GEOM, os.path.dirname(os.path.abspath(__file__))]
from wrapper import MicrotubuleModel, UP, KAPPA_MAX, _simplify_polyline
from instance.instancer_lh import TooManyArcs
from mt_geometry_cost import resample
from griddrift import best_shift
from lh_temporal import Neighbour, decode_ladder
CUT = 0.995; LAMS = (0.0, 0.5, 1.0, 2.0, 4.0)
m = MicrotubuleModel().load_weights(W)
assert m.params.get("decoder") == "lh" and m.params.get("prob_thr") == CUT
EPS = float(m.params.get("polyline_eps_px", 0.0) or 0.0)
def prob_of(img):
    return m.infer_maps(img)["prob_eval"].astype(np.float16).astype(np.float32)
def to_rc(cs):
    return [np.asarray(c, dtype=np.float64)[:, ::-1] / UP for c in cs]
def final(cs):
    return [(_simplify_polyline(c, EPS) if EPS > 0 else c).tolist() for c in to_rc(cs)]
def pts(cs):
    a = [resample(c, 1.0) for c in cs]
    return np.vstack(a) if a else np.zeros((0, 2))
def neighbour(cs, d):
    return Neighbour([((resample(c, 1.0) + d) * UP)[:, ::-1] for c in cs])
def ladder(prob, cm, cp, dm, dp):
    try: return decode_ladder(prob, CUT, KAPPA_MAX, m.params, [neighbour(cm, dm), neighbour(cp, dp)], LAMS), False
    except TooManyArcs: return None, True
R = {}; fallback = 0
if MODE == "real":
    ROOT, LST, POL = sys.argv[7:10]; P = {}
    for p in POL.split(","): P.update(json.load(open(p)))
    for r in json.load(open(LST))[k::K]:
        vid = r["vid"]; F = r["frames"]; T = len(F); bad = 0; res = {}
        raw = [P[vid]["frames"][f["frame"]]["polys"] for f in F]; C = [[np.asarray(c, dtype=float) for c in x if len(c) > 1] for x in raw]; Pt = [pts(c) for c in C]
        fwd = [best_shift(Pt[i], Pt[i + 1]) for i in range(T - 1)]
        for i in range(1, T - 1):
            a = np.asarray(Image.open(os.path.join(ROOT, F[i]["path"]))); assert a.ndim == 2
            out, fb = ladder(prob_of(a.astype(np.float32)), C[i - 1], C[i + 1], fwd[i - 1], -fwd[i])
            if fb: fallback += 1; res[F[i]["frame"]] = {f"lam{l:g}": raw[i] for l in LAMS}; continue
            res[F[i]["frame"]] = {f"lam{l:g}": final(out[l]) for l in LAMS}
            same = res[F[i]["frame"]]["lam0"]; base = raw[i]
            ok = len(same) == len(base) and all(np.array_equal(np.asarray(u), np.asarray(v)) for u, v in zip(same, base)); bad += not ok
            print(vid[:8], F[i]["frame"], "n", " ".join(str(len(res[F[i]["frame"]][f"lam{l:g}"])) for l in LAMS), "noop", ok, flush=True)
        assert bad == 0, (vid, bad)                                         # lam 0 is V16's curves exactly
        R[vid] = res; json.dump(dict(frames=R, fallback=fallback), open(OUT, "w"))
else:
    for path in sorted(glob.glob(os.path.join(sys.argv[7], "*.npy")))[k::K]:
        name = os.path.basename(path)[:-4]; imgs = np.load(path); T = len(imgs); probs = [prob_of(x) for x in imgs]; base = []
        for p in probs:
            try: base.append(decode_ladder(p, CUT, KAPPA_MAX, m.params, [], (0.0,))[0.0])
            except TooManyArcs: base.append([]); fallback += 1
        C = [[c for c in to_rc(b) if len(c) > 1] for b in base]; Pt = [pts(c) for c in C]; fwd = [best_shift(Pt[i], Pt[i + 1]) for i in range(T - 1)]
        res = {"per_frame": [[np.round(c, 2).tolist() for c in b] for b in base], "lam": {}}
        for i in range(1, T - 1):
            out, fb = ladder(probs[i], C[i - 1], C[i + 1], fwd[i - 1], -fwd[i])
            if fb: fallback += 1; continue
            res["lam"][str(i)] = {f"lam{l:g}": [np.round(c, 2).tolist() for c in out[l]] for l in LAMS if l > 0}
            same = out[0.0]; ok = len(same) == len(base[i]) and all(np.array_equal(u, v) for u, v in zip(same, base[i])); assert ok, (name, i)   # the neighbour path with lam 0 is the plain one
            print(name, i, "n", len(base[i]), " ".join(str(len(out[l])) for l in LAMS if l > 0), flush=True)
        R[name] = dict(res, shift=[list(map(float, d)) for d in fwd]); json.dump(dict(clips=R, fallback=fallback), open(OUT, "w"))
print("S1_DONE", k, "fallback", fallback, flush=True)
