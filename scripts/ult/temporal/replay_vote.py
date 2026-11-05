#!/usr/bin/env python3
"""G1 (docs/TODO.md 2026-11-04 22:10): the hub's temporal vote, run as the route runs it, must reproduce S1's lam 1 on every interior real frame.
What the backend sends is emulated: the stored per-frame curves (>= 2 points) as [[[x, y], ...]] JSON; the image as `_mt_image_array` reads it.
usage: replay_vote.py HUB_SEGMENTATION_DIR WEIGHTS FRAMES_DIR LIST_JSON V16_JSONS(comma) S1_JSONS(comma) OUT_JSON k/K"""
import json, os, sys, time
import numpy as np
from PIL import Image
SEG, W, ROOT, LST, POL, S1, OUT = sys.argv[1:8]; k, K = (int(x) for x in sys.argv[8].split("/"))
PKG = os.path.join(SEG, "models", "microtubule")
sys.path[:0] = [SEG, PKG, os.path.join(PKG, "vendor")]
from api.mt_temporal_vote import parse_curves, same_curves, verdict
from api.mt_geometry_cost import align_neighbours
from wrapper import MicrotubuleModel
def load(spec, key=None):
    R = {}
    for p in spec.split(","):
        d = json.load(open(p)); R.update(d[key] if key else d)
    return R
P, E = load(POL), load(S1, "frames")
m = MicrotubuleModel().load_weights(W)
def image_array(path):   # ml/model_loader.py `_mt_image_array`, the branch each PNG mode takes
    im = Image.open(path)
    if im.mode in ("I;16", "I;16B", "I;16L"): return np.array(im, dtype=np.uint16)
    if im.mode == "I": return np.array(im, dtype=np.int32)
    if im.mode == "F": return np.array(im, dtype=np.float32)
    return np.array(im.convert("L"), dtype=np.uint8)
def wire(cs): return json.dumps([[[p[1], p[0]] for p in c] for c in cs if len(c) >= 2])
R = {}; rows = []
for r in json.load(open(LST))[k::K]:
    vid = r["vid"]; F = r["frames"]; raw = [P[vid]["frames"][f["frame"]]["polys"] for f in F]
    for i in range(1, len(F) - 1):
        fr = F[i]["frame"]; t0 = time.perf_counter()
        prev, cur, nxt = (parse_curves(wire(raw[j]), n) for j, n in ((i - 1, "previous"), (i, "current"), (i + 1, "following")))
        nb, shifts = align_neighbours(prev, cur, nxt); t1 = time.perf_counter()
        out = m.predict_voted(image_array(os.path.join(ROOT, F[i]["path"])), nb); t2 = time.perf_counter()
        applied, reason, changed = verdict(out["per_frame"], out["voted"], cur)
        exp = [np.asarray(c, dtype=float) for c in E[vid][fr]["lam1"]]; exp0 = [np.asarray(c, dtype=float) for c in E[vid][fr]["lam0"]]
        got = out["voted"] or []
        bit = len(got) == len(exp) and all(np.array_equal(a, b) for a, b in zip(got, exp))
        row = dict(vid=vid, frame=fr, applied=applied, reason=reason, changed=changed, n0=len(out["per_frame"] or []), n1=len(got), n_exp=len(exp),
                   match_1e9=same_curves(got, exp, 1e-9), bitwise=bit, lam0_match=same_curves(out["per_frame"] or [], exp0, 1e-9),
                   exp_changed=not same_curves(exp, exp0, 1e-9), shift=[list(map(float, d)) for d in shifts], t_align=t1 - t0, t_vote=t2 - t1, shape=list(Image.open(os.path.join(ROOT, F[i]["path"])).size))
        rows.append(row)
        print(vid[:8], fr, "applied", applied, reason, "changed", changed, "n", row["n0"], row["n1"], row["n_exp"], "match", row["match_1e9"], "bit", bit, f"{row['t_align']:.2f}+{row['t_vote']:.2f}s", flush=True)
    json.dump(rows, open(OUT, "w"))
print("REPLAY_DONE", k, len(rows), "applied", sum(r["applied"] for r in rows), "match", sum(r["match_1e9"] for r in rows), "bitwise", sum(r["bitwise"] for r in rows), flush=True)
