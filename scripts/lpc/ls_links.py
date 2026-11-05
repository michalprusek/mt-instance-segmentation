#!/usr/bin/env python3
"""L-strip, step 1 (declared docs/TODO.md 2026-10-06 14:10): labelled candidate links of SYNTHETIC frames, exactly the
candidate set instancer L scores (every curvature-admissible pair of ends of different arcs within 60 px, NO lower gap
bound), with their map-logit strips and hand features. Label rule = S1's (scripts/lpc/s1_links.py).

    ls_links.py DATA PRED_DIR OUT.npz SPLIT [NPROC]
"""
import glob, json, os, sys
import numpy as np
from multiprocessing import Pool
from scipy.spatial import cKDTree
sys.path[:0] = [".", "src", "scripts", "scripts/lpc"]
from lpc_lib import UP, TOL, link_edges, owners, to_logit
from instance.geometry import resample
from mt_bench.cvat_import import read_frame_h5
data, pred, out, split = sys.argv[1:5]; NP = int(sys.argv[5]) if len(sys.argv) > 5 else 12
CUT, BR = 0.98, 60.0
HAND = json.load(open("src/instance/params_l_hand.json"))


def one(path):
    name = os.path.basename(path); fr = read_frame_h5(path)
    prob = np.load(os.path.join(pred, name.replace(".h5", ".npz")))["prob"].astype(np.float32)[0]
    gt = [np.asarray(q, float) * UP for q in fr["polylines"] if len(q) >= 2]
    arcs, edges, (F, S, N) = link_edges(prob, CUT, HAND, BR, strips=True)
    if not edges: return name, S, N, np.zeros(0, np.int8), F, 0
    owner, proj = owners(arcs, gt); trees = [cKDTree(resample(q, ds=1.0)) for q in gt]; order = {}
    for j in set(owner.values()):
        for n, k in enumerate(sorted((k for k, o in owner.items() if o == j), key=lambda k: (proj[k][0] + proj[k][1]) / 2)): order[k] = n
    Y = np.zeros(len(edges), np.int8); n_amb = 0                      # -1 = ambiguous, dropped
    for e, ((ka, wa), (kb, wb), gap, _) in enumerate(edges):
        oa, ob = owner.get(ka), owner.get(kb)
        if oa is None or ob is None or oa != ob: continue             # synthetic: an unowned arc is a true negative
        pa = arcs[ka][-1 if wa == "end" else 0]; pb = arcs[kb][-1 if wb == "end" else 0]; nb = max(2, int(round(gap)) + 1)
        d, _ = trees[oa].query(pa[None] + (pb - pa)[None] * np.linspace(0, 1, nb)[:, None], k=1)
        fa = proj[ka][1] if wa == "end" else proj[ka][0]; fb = proj[kb][1] if wb == "end" else proj[kb][0]
        ia = (proj[ka][0] + proj[ka][1]) / 2; ib = (proj[kb][0] + proj[kb][1]) / 2
        facing = (ia < ib and ia <= fa <= fb <= ib + 1e-9) or (ib < ia and ib <= fb <= fa <= ia + 1e-9)
        if abs(order[ka] - order[kb]) == 1 and facing and float((d <= TOL).mean()) >= 0.8: Y[e] = 1
        else: Y[e] = -1; n_amb += 1
    m = Y >= 0
    return name, S[m], N[m], Y[m], F[m], n_amb


if __name__ == "__main__":
    paths = [p for p in sorted(glob.glob(os.path.join(data, "*.h5")))
             if os.path.exists(os.path.join(pred, os.path.basename(p).replace(".h5", ".npz")))
             and (split == "all" or str(read_frame_h5(p)["attrs"].get("split")) == split)]
    with Pool(NP) as pool: R = pool.map(one, paths, chunksize=1)
    y = np.concatenate([r[3] for r in R])
    np.savez(out, logit=np.concatenate([r[1] for r in R]), lens=np.concatenate([r[2] for r in R]), y=y, feats=np.concatenate([r[4] for r in R]),
             frame=np.concatenate([np.full(len(r[3]), i, np.int32) for i, r in enumerate(R)]), names=np.array([r[0] for r in R]))
    g = np.concatenate([r[4] for r in R])[:, 2]
    print(len(R), "frames", len(y), "links pos", int(y.sum()), "| gap < 4:", int((g < 4).sum()), "pos", int(y[g < 4].sum()), "| ambiguous dropped", sum(r[5] for r in R))
