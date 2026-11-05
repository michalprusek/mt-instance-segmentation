#!/usr/bin/env python3
"""R2 and R3 (docs/TODO.md 2026-11-04 08:10): a model on the real windows or the synthetic clips, single frame and, for TMP, with raw neighbours; edge frames go as (t, t, t). S1 with lam 1, unchanged, on the temporal curves.
usage: tmp_infer.py PKG WEIGHTS GEOM_DIR OUT_JSON CUT temporal|single real FRAMES_DIR LIST_JSON
       tmp_infer.py PKG WEIGHTS GEOM_DIR OUT_JSON CUT temporal|single synth CLIPS_DIR"""
import glob, json, os, sys
import numpy as np
import torch
from PIL import Image
PKG, W, GEOM, OUT = sys.argv[1:5]; CUT = float(sys.argv[5]); TEMPORAL = sys.argv[6] == "temporal"; SET = sys.argv[7]
sys.path[:0] = [os.path.dirname(PKG), PKG, os.path.join(PKG, "vendor"), GEOM, os.path.dirname(os.path.abspath(__file__))]
from wrapper import MicrotubuleModel, UP, KAPPA_MAX, _normalize, _simplify_polyline, eval_shape, resample_to_eval
from instance.instancer_a import instance_a
from instance.instancer_lh import TooManyArcs
from net import IMA_M, IMA_S, TILE
from mt_geometry_cost import resample
from griddrift import best_shift
from lh_temporal import Neighbour, decode_ladder
m = MicrotubuleModel().load_weights(W); assert m.params.get("fusion") == "gauss" and m.params.get("decoder") == "lh"
EPS = float(m.params.get("polyline_eps_px", 0.0) or 0.0)
def channels3(st):
    """The wrapper's gauss path with a three-channel tile. st: (3, H, W), each frame already stretched."""
    mean = torch.tensor(IMA_M).view(3, 1, 1); std = torch.tensor(IMA_S).view(3, 1, 1); _, height, width = st.shape; stride = TILE // 2
    def starts(extent):
        s = list(range(0, max(1, extent - TILE + 1), stride)) or [0]
        if s[-1] != max(0, extent - TILE): s.append(max(0, extent - TILE))
        return s
    acc = cnt = None
    with torch.no_grad():
        for y in starts(height):
            for x in starts(width):
                tile = st[:, y:y + TILE, x:x + TILE]; th, tw = tile.shape[1:]
                pad_h, pad_w = (-th) % 128, (-tw) % 128
                if pad_h or pad_w: tile = np.pad(tile, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect" if pad_h < th and pad_w < tw else "edge")
                t = ((torch.from_numpy(tile.astype(np.float32)) - mean) / std)[None].to(m._device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(m._device).startswith("cuda")): out = m._model(t)
                while isinstance(out, (tuple, list)): out = out[0]
                out = np.clip(out.float()[0].cpu().numpy(), -30.0, 30.0)
                wy, wx = np.arange(th) - (th - 1) / 2, np.arange(tw) - (tw - 1) / 2
                weight = np.exp(-(wy[:, None] ** 2 + wx[None] ** 2) / (2 * (TILE / 8) ** 2)); weight = np.maximum(weight, weight.max() * 1e-3)
                if acc is None: acc, cnt = np.zeros((out.shape[0], height, width)), np.zeros((height, width))
                acc[:, y:y + th, x:x + tw] += out[:, :th, :tw] * weight; cnt[y:y + th, x:x + tw] += weight
    return 1 / (1 + np.exp(-acc / cnt[None]))
def prob3(a, b, c):
    st = np.stack([_normalize(np.asarray(z, np.float64)) for z in (a, b, c)])
    return resample_to_eval(channels3(st), eval_shape(st.shape[1:])).max(axis=0).astype(np.float16).astype(np.float32)
def decode(prob):
    try: return decode_ladder(prob, CUT, KAPPA_MAX, m.params, [], (0.0,))[0.0], False
    except TooManyArcs:
        pl, _ = instance_a(prob > CUT, KAPPA_MAX, m.params, channels=prob[None], prob=prob, return_masks=False)
        return pl, True
def to_rc(cs): return [np.asarray(c, dtype=np.float64)[:, ::-1] / UP for c in cs]
def final(cs): return [(_simplify_polyline(c, EPS) if EPS > 0 else c).tolist() for c in to_rc(cs)]
def pts(cs):
    a = [resample(c, 1.0) for c in cs if len(c) > 1]
    return np.vstack(a) if a else np.zeros((0, 2))
def neighbour(cs, d): return Neighbour([((resample(c, 1.0) + d) * UP)[:, ::-1] for c in cs if len(c) > 1])
def run(frames, label, proof):
    """-> per frame the curves (eval, x y), S1's curves with lam 1, the foreground share at the cut, single and temporal."""
    T = len(frames); idx = list(range(T)); out = []; fb = 0
    single = [prob3(a, a, a) for a in frames]
    if proof:
        ref = m.infer_maps(frames[T // 2])["prob_eval"].astype(np.float16).astype(np.float32)
        assert np.array_equal(ref, single[T // 2]), "(t, t, t) through the three-channel tile is not the wrapper's map"; print(label, "proof ok", flush=True)
    probs = [(prob3(frames[i - 1], frames[i], frames[i + 1]) if TEMPORAL and 0 < i < T - 1 else single[i]) for i in idx]
    base = []
    for p in probs:
        c, f = decode(p); base.append(c); fb += f
    C = [to_rc(b) for b in base]; Pt = [pts(c) for c in C]; fwd = [best_shift(Pt[i], Pt[i + 1]) for i in range(T - 1)]
    for i in idx:
        s1 = None
        if 0 < i < T - 1:
            try: s1 = decode_ladder(probs[i], CUT, KAPPA_MAX, m.params, [neighbour(C[i - 1], fwd[i - 1]), neighbour(C[i + 1], -fwd[i])], (1.0,))[1.0]
            except TooManyArcs: s1 = base[i]
        out.append(dict(curves=base[i], s1=s1, share_single=float((single[i] > CUT).mean()), share=float((probs[i] > CUT).mean())))
        print(label, i, "n", len(base[i]), "s1", None if s1 is None else len(s1), f"fg {out[-1]['share_single']:.4f} {out[-1]['share']:.4f}", flush=True)
    return out, fb
R = {}; fallback = 0
if SET == "real":
    ROOT, LST = sys.argv[8:10]
    for j, r in enumerate(json.load(open(LST))):
        F = r["frames"]; frames = []
        for f in F:
            a = np.asarray(Image.open(os.path.join(ROOT, f["path"]))); assert a.ndim == 2; frames.append(a.astype(np.float32))
        res, fb = run(frames, r["vid"][:8], True); fallback += fb
        R[r["vid"]] = {f["frame"]: dict(polys=final(x["curves"]), s1=None if x["s1"] is None else final(x["s1"]), share_single=x["share_single"], share=x["share"]) for f, x in zip(F, res)}
        json.dump(dict(frames=R, fallback=fallback, cut=CUT, temporal=TEMPORAL), open(OUT, "w"))
else:
    for path in sorted(glob.glob(os.path.join(sys.argv[8], "*.npy"))):
        name = os.path.basename(path)[:-4]; res, fb = run(list(np.load(path)), name, False); fallback += fb
        R[name] = [dict(curves=[np.round(c, 2).tolist() for c in x["curves"]], s1=None if x["s1"] is None else [np.round(c, 2).tolist() for c in x["s1"]], share_single=x["share_single"], share=x["share"]) for x in res]
        json.dump(dict(clips=R, fallback=fallback, cut=CUT, temporal=TEMPORAL), open(OUT, "w"))
print("TMP_INFER_DONE", fallback, flush=True)
