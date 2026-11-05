#!/usr/bin/env python3
"""Read 36b-2 on the frame the fix acts on: the 3-arm junctions, not all 454 crossings.

Read 35 measured that a 4-arm junction resolves 0.944 and a 3-arm one 0.333, and that a 3-arm
junction is a crossing split into two clusters joined by a shared arc the graph can traverse
once. The pass-through is defined ONLY there. Scoring it on all crossings buries a 33-crossing
effect inside 454, of which 79 are broken masks the fix can never touch -- a +0.02 there is a
9-crossing swing, indistinguishable from noise.

So this reads, per `ov_frac`:
  * identity on the 3-arm `node` crossings (the frame the fix acts on; base 0.333)
  * identity on the 4-arm ones (must NOT fall: they already work at 0.944)
  * identity on every crossing of the same frames (the honest wider number)
  * the FRACTION OF JUNCTION-TO-JUNCTION ARCS DUPLICATED -- the void check. If it is zero the
    criterion did not fire and the read is void, not negative; if it climbs while identity does
    not, the duplicated arcs were single filaments and the mechanism is wrong.

    PYTHONPATH=src:scripts python scripts/read_36b_arity.py --data data/synth_eval_s35 \
        --pred-dir <dump> --diag <topo2_synth/crossing_diag.json> \
        --params src/instance/params_a.json --thr 0.99 --fracs 0.02,0.05,0.10,0.20
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from instance.geometry import arclength, resample  # noqa: E402
from instance.instancer_a import (_arc_overlap_frac, default_params,  # noqa: E402
                                  instance_a)
from instance.metrics import _covering_instances, _mask_trees, centerline_f1  # noqa: E402
from instance.oracle import oracle_instance_masks  # noqa: E402
from instance.skeleton_graph import build_arc_graph  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.gt_stats import count_crossings  # noqa: E402

UP, KAPPA_MAX = 1.5, 0.25


def crossing_identity(masks, crossings, res, lens):
    """Per-crossing identity, keyed by (x, y). Identical sampling to `diag_crossing_map.py`."""
    trees = _mask_trees(masks)
    out = {}
    for c in crossings:
        good, usable = True, True
        for ik, sk in (("i", "s_i"), ("j", "s_j")):
            k, s0 = c[ik], c[sk]
            pts, L = res[k], lens[k]
            if s0 - 15 < 0 or s0 + 15 > L:
                usable = False
                break
            pb = pts[min(int(round(s0 - 15)), len(pts) - 1)]
            pa = pts[min(int(round(s0 + 15)), len(pts) - 1)]
            good &= bool(_covering_instances(trees, pb, 5.0)
                         & _covering_instances(trees, pa, 5.0))
        if usable:
            out[(round(float(c["x"]), 3), round(float(c["y"]), 3))] = good
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--diag", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--thr", type=float, required=True)
    ap.add_argument("--fracs", default="0.02,0.05,0.10,0.20")
    ap.add_argument("--ov-thr", type=float, default=0.5)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fracs = [float(x) for x in args.fracs.split(",")]
    params = {**default_params(), **json.load(open(args.params))}
    params.pop("kappa_max", None); params.pop("prob_thr", None)

    diag = json.load(open(args.diag))["rows"]
    arity = {(r["frame"], round(r["x"], 3), round(r["y"], 3)): r.get("n_arms_junction")
             for r in diag if r.get("topo") == "node"}
    frames = {r["frame"] for r in diag}

    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    dup = collections.defaultdict(list)
    f1s = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        stem = os.path.basename(p)[:-3]
        npz = os.path.join(args.pred_dir, stem + ".npz")
        if stem not in frames or not os.path.exists(npz):
            continue
        fr = read_frame_h5(p)
        if str(fr["attrs"].get("split")) != args.split:
            continue
        prob = np.load(npz)["prob"].astype(np.float32)[0]
        gt = [np.asarray(q, float) * UP for q in fr["polylines"]]
        if len(gt) < 2:
            continue
        ovacc = np.zeros(prob.shape, np.int16)
        for gm in oracle_instance_masks(gt, prob.shape, half_width=1.0, up=1.0):
            ovacc += gm.astype(np.int16)
        ov = (ovacc >= 2).astype(np.float32)
        mask = prob > args.thr
        res = [resample(q, ds=1.0) for q in gt]
        lens = [float(arclength(q)[-1]) if len(q) > 1 else 0.0 for q in res]
        cr = count_crossings(gt, tol=2.0 * UP, endpoint_margin=8.0 * UP)

        # the void check: how many junction-to-junction arcs each criterion would duplicate
        g = build_arc_graph(mask, merge_radius=params["merge_radius"],
                            min_arc_len=int(params["min_arc_len"]),
                            bridge_max_len=params["bridge_max_len"])
        jj = [resample(a, ds=params["ds"]) for a, (j0, j1) in zip(g.arcs, g.arc_ends)
              if j0 is not None and j1 is not None and j0 != j1 and len(a) >= 2]
        fr_vals = [_arc_overlap_frac(a, ov, args.ov_thr) for a in jj]

        for tag, kw in [("base", {})] + [(f"{f:g}", {"overlap": ov}) for f in fracs]:
            pr = dict(params)
            if kw:
                pr["ov_frac"] = float(tag); pr["ov_thr"] = args.ov_thr
            polys, _ = instance_a(mask, KAPPA_MAX, pr, channels=prob[None], prob=prob, **kw)
            pm = oracle_instance_masks(polys, prob.shape, half_width=1.0, up=1.0)
            ident = crossing_identity(pm, cr, res, lens)
            for (x, y), ok in ident.items():
                acc[tag][arity.get((stem, x, y), "other")].append(ok)
            f1s[tag].append(centerline_f1(pm, gt, tol=5.0, length_coverage=0.95,
                                          precision_coverage=0.95)["f1"])
            if kw:
                dup[tag].append((sum(v >= float(tag) for v in fr_vals), len(jj)))
        print(f"{stem}: {len(cr)} crossings, {len(jj)} jj-arcs", flush=True)

    summ = {}
    for tag in ["base"] + [f"{f:g}" for f in fracs]:
        row = {}
        for key in (3, 4, 2, 5, "other"):
            v = acc[tag].get(key, [])
            if v:
                row[f"arms_{key}"] = [round(float(np.mean(v)), 4), len(v)]
        allv = [b for lst in acc[tag].values() for b in lst]
        row["all_on_these_frames"] = [round(float(np.mean(allv)), 4), len(allv)]
        row["macro_f1_95"] = round(float(np.mean(f1s[tag])), 4)
        if tag in dup:
            d = np.array(dup[tag])
            row["arcs_duplicated"] = int(d[:, 0].sum())
            row["jj_arcs_total"] = int(d[:, 1].sum())
            row["dup_share"] = round(float(d[:, 0].sum() / max(d[:, 1].sum(), 1)), 4)
        summ[tag] = row
    json.dump(summ, open(args.out, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
