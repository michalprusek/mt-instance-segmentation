#!/usr/bin/env python3
"""D3 mechanism read (declared 2026-10-05 08:54 in docs/TODO.md): weak-point share on annotated filaments vs distance to the
nearest INTERIOR tile edge of the declared fusion (tile 512, stride 387, native px), for the declared dump and the Gaussian
one. ADDENDUM (written before this script ran, 2026-10-05 22:00): the D3 gain was precision, so the same bins also get the
off-annotation foreground share (pixels >= thr farther than 5 eval px from any GT polyline). CPU only, VAL only.

    python tile_edge_read.py DATA SPLIT PRED_A PRED_B THR OUT.json
"""
import glob, json, os, sys
import numpy as np
from scipy.ndimage import distance_transform_edt
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from mt_bench.cvat_import import read_frame_h5  # noqa: E402

data, split, pa, pb, thr, out = sys.argv[1:7]; thr = float(thr)
UP, TILE, S0 = 1.5, 512, int(round(512 * 392 / 518))
BINS = [0, 8, 16, 32, 64, 1e9]


def starts(n):
    s = list(range(0, max(1, n - TILE + 1), S0)) or [0]
    if s[-1] != max(0, n - TILE): s.append(max(0, n - TILE))
    return s


def edge_dist(n_native, n_eval):
    """Distance (eval px) of every eval row/col to the nearest tile edge that is not the frame border."""
    e = set()
    for s in starts(n_native):
        for v in (s, s + TILE):
            if 0 < v < n_native: e.add(v * UP)
    c = np.arange(n_eval, dtype=float)
    return np.full(n_eval, 1e9) if not e else np.min(np.abs(c[:, None] - np.array(sorted(e))[None]), axis=1)


def sample(poly, step=1.0):
    p = np.asarray(poly, float) * UP; d = np.hypot(*np.diff(p, axis=0).T); L = d.sum()
    if L < 1: return p[:1]
    t = np.concatenate([[0], np.cumsum(d)]); s = np.arange(0, L, step)
    return np.stack([np.interp(s, t, p[:, 0]), np.interp(s, t, p[:, 1])], 1)   # (x=col, y=row)


nb = len(BINS) - 1
acc = {k: np.zeros(nb) for k in ("pts", "weakA", "weakB", "area", "offA", "offB", "fgA", "fgB")}
nfr = 0; multi = 0
for p in sorted(glob.glob(os.path.join(data, "*.h5"))):
    fr = read_frame_h5(p)
    if str(fr["attrs"].get("split")) != split or not fr["polylines"]: continue
    nm = os.path.basename(p).replace(".h5", ".npz")
    fa, fb = os.path.join(pa, nm), os.path.join(pb, nm)
    if not (os.path.exists(fa) and os.path.exists(fb)): continue
    A = np.load(fa)["prob"].astype(np.float32); B = np.load(fb)["prob"].astype(np.float32)
    A = A[0] if A.ndim == 3 else A; B = B[0] if B.ndim == 3 else B
    H, W = A.shape; h, w = fr["image"].shape[:2]
    dy, dx = edge_dist(h, H), edge_dist(w, W)
    D = np.minimum(dy[:, None], dx[None])
    if D.min() > 1e8: continue                       # single-tile frame: fusion cannot act
    multi += 1
    gt = np.ones((H, W), bool)
    for q in fr["polylines"]:
        s = sample(q); r = np.clip(np.round(s[:, 1]).astype(int), 0, H - 1); c = np.clip(np.round(s[:, 0]).astype(int), 0, W - 1)
        gt[r, c] = False
        b = np.digitize(D[r, c], BINS) - 1
        np.add.at(acc["pts"], b, 1); np.add.at(acc["weakA"], b, A[r, c] < thr); np.add.at(acc["weakB"], b, B[r, c] < thr)
    off = distance_transform_edt(gt) > 5
    bi = np.digitize(D, BINS) - 1
    for k in range(nb):
        m = bi == k
        acc["area"][k] += m.sum(); acc["fgA"][k] += (A[m] >= thr).sum(); acc["fgB"][k] += (B[m] >= thr).sum()
        acc["offA"][k] += ((A >= thr) & off & m).sum(); acc["offB"][k] += ((B >= thr) & off & m).sum()
    nfr += 1
res = {"frames": nfr, "multi_tile": multi, "thr": thr, "bins": BINS[:-1], **{k: v.tolist() for k, v in acc.items()}}
json.dump(res, open(out, "w"), indent=1)
print("frames", nfr, "thr", thr)
print("bin(eval px)   pts   weakA  weakB  | area%  offA/area  offB/area  rel")
for k in range(nb):
    a = acc
    print(f"{BINS[k]:>4}+ {int(a['pts'][k]):>9} {a['weakA'][k]/max(a['pts'][k],1):.4f} {a['weakB'][k]/max(a['pts'][k],1):.4f} | "
          f"{100*a['area'][k]/a['area'].sum():5.1f} {a['offA'][k]/a['area'][k]:.5f} {a['offB'][k]/a['area'][k]:.5f} "
          f"{a['offB'][k]/max(a['offA'][k],1):.3f}")
