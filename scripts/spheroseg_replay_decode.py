"""Correction replay, step 2: decoder A and decoder L on the SAME captured map, scored against the
human annotation with the declared pair rule (docs/TODO.md 2026-10-13 21:20, declared before this ran).

Run from the research tree (cwd = the tree whose ``scripts/lpc`` measured L), e.g. on tulen:

    cd /disk2/prusek/mt_work/mt34_work
    python <this file> <dump_dir> <h5_dir> <out.json> [nproc]

Decoders, all on the captured map at the seed cut unless named otherwise:
  hubA   the hub package's own instances, as captured -- what the annotators were shown (the control)
  A      this tree's instance_a with the captured parameter vector (how far the two A's have drifted)
  A15    v5H frames only: this tree's A with min_length 15 instead of 44.7
  L      link_edges + chain, params_l_hand.json, bridge 60, t 0 (the declared instancer)
  L995   SPARSE35 frames only: L at cut 0.995 -- a different mask, secondary
"""
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path[:0] = [".", "src", "scripts", "scripts/lpc"]
from lpc_lib import KAPPA, UP, chain, link_edges  # noqa: E402
from instance.geometry import resample  # noqa: E402
from instance.instancer_a import instance_a  # noqa: E402
from instance.metrics import centerline_f1, rasterise_polyline  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402

dump_dir, h5_dir, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
NPROC = int(sys.argv[4]) if len(sys.argv) > 4 else 9
MODEL = json.load(open("src/instance/params_l_hand.json"))
BRIDGE, T = 60.0, 0.0


def score(polys, gt, shape):
    masks = [rasterise_polyline(resample(np.asarray(p, float), ds=0.5), shape) for p in polys]
    r = centerline_f1(masks, gt, tol=5.0, length_coverage=0.8, precision_coverage=0.8, end_tol=0.0)
    hit = [False] * len(gt)
    for pr in r["pairs"]:
        if pr["tp"]:
            hit[pr["gt"]] = True
    return {"tp": r["tp"], "fp": r["fp"], "fn": r["fn"], "n_pred": len(polys), "hit": hit,
            "polylines_native": [(np.asarray(p, float) / UP).round(3).tolist() for p in polys]}


def one(meta_path):
    meta = json.load(open(meta_path))
    z = np.load(meta_path.replace(".json", ".npz"))
    prob, chans, cut = z["prob"].astype(np.float32), z["channels"].astype(np.float32), float(meta["cut"])
    assert abs(float(meta["up"]) - UP) < 1e-9 and abs(float(meta["kappa"]) - KAPPA) < 1e-9
    pid = os.path.basename(meta_path)[:8]
    h5 = [p for p in glob.glob(os.path.join(h5_dir, "*.h5")) if read_frame_h5(p)["attrs"]["project_id"].startswith(pid)]
    assert len(h5) == 1, (pid, h5)
    fr = read_frame_h5(h5[0])
    gt = [np.asarray(q, float) * UP for q in fr["polylines"]]
    shape = prob.shape
    params = {k: v for k, v in meta["params"].items() if k not in ("prob_thr", "polyline_eps_px")}

    dec = {"hubA": score([np.asarray(p, float) for p in meta["polylines_eval"]], gt, shape)}
    pA, _ = instance_a(prob > cut, KAPPA, params=params, channels=chans, prob=prob)
    dec["A"] = score(pA, gt, shape)
    if abs(params.get("min_length", 15.0) - 15.0) > 1e-6:
        pA15, _ = instance_a(prob > cut, KAPPA, params={**params, "min_length": 15.0}, channels=chans, prob=prob)
        dec["A15"] = score(pA15, gt, shape)
    arcs, edges = link_edges(prob, cut, MODEL, BRIDGE)
    dec["L"] = score(chain(arcs, edges, BRIDGE, T) if arcs else [], gt, shape)
    if meta["tag"] == "sparse35":
        arcs2, edges2 = link_edges(prob, 0.995, MODEL, BRIDGE)
        dec["L995"] = score(chain(arcs2, edges2, BRIDGE, T) if arcs2 else [], gt, shape)
    return {"project": pid, "tag": meta["tag"], "cut": cut, "shape_eval": list(shape), "n_gt": len(gt),
            "h5": os.path.basename(h5[0]), "decoders": dec}


if __name__ == "__main__":
    metas = sorted(glob.glob(os.path.join(dump_dir, "*__decoder_input.json")))
    with Pool(min(NPROC, len(metas))) as pool:
        rows = pool.map(one, metas, chunksize=1)
    json.dump({"bridge": BRIDGE, "t": T, "model": "src/instance/params_l_hand.json", "rows": rows}, open(out_path, "w"))
    for r in rows:
        print(r["project"], r["tag"], "cut", r["cut"], "gt", r["n_gt"],
              {k: (v["tp"], v["fp"], v["fn"]) for k, v in r["decoders"].items()}, flush=True)
