#!/usr/bin/env python3
"""Fixture of the hub's microtubule temporal vote (hub backend/segmentation/tests/fixtures/mt_vote), docs/TODO.md 2026-11-04 22:10 G2.

One synthetic clip triplet from S1's held-out clips (no user data: the hub repository is public), and what the RESEARCH path
emits for its middle frame: s1_links.py's real-video functions on the pristine hub package (main 055fdb86) with lh_temporal.py,
griddrift.py and the geom copy of mt_geometry_cost.py. The hub's new code is the thing tested, so none of it is imported here.

Frames are written as 16-bit PNG and READ BACK before anything is computed (the quantisation the fixture carries is the one the
reference saw); one scale per triplet when a clip exceeds 16 bits. The stored curves are `predict`'s centerlines with >= 2 points,
the shape the hub keeps. Frame choice, declared before any candidate is scored: the middle frame (t = 5) of the first clip in name
order whose vote changes at least 3 curves AND whose result moves when the shifts are negated AND when the neighbours are
transposed, so that each of those mutations is visible to the test.
usage: make_mt_fixture_vote.py PKG WEIGHTS GEOM_DIR CLIPS_DIR OUT_DIR"""
import glob, hashlib, json, os, sys
import numpy as np
from PIL import Image
PKG, W, GEOM, CLIPS, OUT = sys.argv[1:6]
sys.path[:0] = [os.path.dirname(PKG), PKG, os.path.join(PKG, "vendor"), GEOM, os.path.dirname(os.path.abspath(__file__))]
from wrapper import MicrotubuleModel, UP, KAPPA_MAX, _simplify_polyline
from mt_geometry_cost import resample
from griddrift import best_shift
from lh_temporal import Neighbour, decode_ladder
os.makedirs(OUT, exist_ok=True)
m = MicrotubuleModel().load_weights(W); EPS = float(m.params["polyline_eps_px"]); CUT = float(m.params["prob_thr"])
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def png16(a, scale, path):
    Image.fromarray(np.clip(np.rint(a * scale), 0, 65535).astype(np.uint16)).save(path)
    return np.asarray(Image.open(path))
def stored(img): return [np.asarray(c, float) for c in m.predict(img)["centerlines_rc"] if len(c) >= 2]
def pts(cs):
    a = [resample(c, 1.0) for c in cs]
    return np.vstack(a) if a else np.zeros((0, 2))
def final(cs): return [(_simplify_polyline(c, EPS) if EPS > 0 else c) for c in (np.asarray(x, dtype=np.float64)[:, ::-1] / UP for x in cs)]
def same(a, b): return len(a) == len(b) and all(x.shape == y.shape and np.allclose(x, y, rtol=0, atol=1e-9) for x, y in zip(a, b))
def nb(cs, d, flip=True):
    return Neighbour([(((resample(c, 1.0) + d) * UP)[:, ::-1] if flip else (resample(c, 1.0) + d) * UP) for c in cs])
def wire(cs): return [[[float(p[1]), float(p[0])] for p in c] for c in cs]
T = 5
for path in sorted(glob.glob(f"{CLIPS}/*.npy")):
    name = os.path.basename(path)[:-4]; imgs = np.load(path)[T - 1:T + 2]; scale = min(1.0, 65535.0 / float(imgs.max()))
    tri = [png16(imgs[j], scale, f"{OUT}/_{j}.png") for j in range(3)]
    C = [stored(x) for x in tri]; P = [pts(c) for c in C]
    d_prev, d_next = best_shift(P[0], P[1]), -best_shift(P[1], P[2])
    prob = m.infer_maps(tri[1])["prob_eval"].astype(np.float16).astype(np.float32)
    out = decode_ladder(prob, CUT, KAPPA_MAX, m.params, [nb(C[0], d_prev), nb(C[2], d_next)], (0.0, 1.0))
    lam0, lam1 = final(out[0.0]), final(out[1.0])
    assert same(lam0, C[1]), name                                         # lam 0 is the stored per-frame output
    neg = final(decode_ladder(prob, CUT, KAPPA_MAX, m.params, [nb(C[0], -d_prev), nb(C[2], -d_next)], (1.0,))[1.0])
    trn = final(decode_ladder(prob, CUT, KAPPA_MAX, m.params, [nb(C[0], d_prev, False), nb(C[2], d_next, False)], (1.0,))[1.0])
    moved = sum(not any(x.shape == y.shape and np.allclose(x, y, atol=1e-9) for y in lam0) for x in lam1)
    print(name, "n", len(lam0), len(lam1), "moved", moved, "shift", np.round(d_prev, 1), np.round(d_next, 1), "neg differs", not same(neg, lam1), "transposed differs", not same(trn, lam1), flush=True)
    if moved >= 3 and not same(neg, lam1) and not same(trn, lam1): break
else:
    raise SystemExit("no clip qualifies")
os.replace(f"{OUT}/_1.png", f"{OUT}/frame_t.png")
for j in (0, 2): os.remove(f"{OUT}/_{j}.png")
np.savez_compressed(f"{OUT}/reference_prob_eval_scale.npz", prob=prob.astype(np.float16))
json.dump(dict(previous=wire(C[0]), current=wire(C[1]), following=wire(C[2]),
               voted_rc=[c.tolist() for c in lam1], per_frame_rc=[c.tolist() for c in lam0],
               shift_previous=list(map(float, d_prev)), shift_next=list(map(float, d_next))), open(f"{OUT}/reference_vote.json", "w"))
json.dump(dict(clip=name, clip_seed_rule="build_clips.py: 20261104 + 1000 * regime index + i", frames=[T - 1, T, T + 1], png_scale=scale,
               ckpt=W, ckpt_sha256=sha(W), thr=CUT, polyline_eps_px=EPS, vote_lam=1.0, moved_curves=moved,
               n_per_frame=len(lam0), n_voted=len(lam1), negated_shift_differs=True, transposed_differs=True,
               sources={os.path.basename(f): sha(f) for f in [sys.modules[k].__file__ for k in ("lh_temporal", "griddrift", "mt_geometry_cost", "wrapper", "instance.instancer_lh")] + [os.path.join(PKG, "params_v16.json")]},
               frame_png_sha256=sha(f"{OUT}/frame_t.png")), open(f"{OUT}/manifest.json", "w"), indent=1)
print("FIXTURE_DONE", name, flush=True)
