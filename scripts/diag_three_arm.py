#!/usr/bin/env python3
"""What IS a 3-arm junction at a crossing the instancer loses?

`diag_crossing_map.py` measures that within the `node` class a 4-arm junction keeps identity
0.944 (n = 107) and a 3-arm junction only 0.333 (n = 33): the pairing is not the problem, the
junction's SHAPE is.  Two mechanisms produce a 3-arm junction where four arms should be, and
they need different fixes:

  bridge  -- the two branch points of a shallow X sit further apart than `bridge_max_len`, so
             `_absorb_crossing_bridges` leaves them separate and the arc BETWEEN them is a
             junction-to-junction bridge.  Each junction then has A-outer, B-outer and the
             bridge; `match_junction` gives the bridge one partner per end, so one filament is
             always left open.  Fix: duplicate the bridge where multiplicity says 2.
  pruned  -- the fourth arm was dropped by `min_arc_len` or swallowed by `merge_radius`.
             Fix: protect the arm where multiplicity is high.

Reads the rows written by `diag_crossing_map.py`, rebuilds the arc graph on the frames that
hold the crossings of interest, and reports, per crossing, the arcs attached to the nearest
junction: their length and whether the far end is another junction.

    PYTHONPATH=src:scripts python scripts/diag_three_arm.py --data <dir> --pred-dir <dir> \
        --diag <crossing_diag.json> --params src/instance/params_a.json --thr 0.99
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from instance.geometry import total_length  # noqa: E402
from instance.skeleton_graph import build_arc_graph  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--diag", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--thr", type=float, required=True)
    ap.add_argument("--channel", type=int, default=0)
    ap.add_argument("--split", default="val")
    ap.add_argument("--arms", type=int, default=3, help="junction arity to characterise")
    ap.add_argument("--klass", default="node")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    params = json.load(open(args.params))
    rows = json.load(open(args.diag))["rows"]
    want = [r for r in rows if r.get("topo") == args.klass
            and r.get("n_arms_junction") == args.arms]
    by_frame = collections.defaultdict(list)
    for r in want:
        by_frame[r["frame"]].append(r)
    print(f"{len(want)} crossings in class {args.klass} with {args.arms} arms, "
          f"on {len(by_frame)} frames ({sum(not r['identity'] for r in want)} failing)")

    out, tally = [], collections.Counter()
    for p in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        stem = os.path.basename(p)[:-3]
        if stem not in by_frame:
            continue
        npz = os.path.join(args.pred_dir, stem + ".npz")
        if not os.path.exists(npz):
            continue
        fr = read_frame_h5(p)
        if str(fr["attrs"].get("split")) != args.split:
            continue
        prob = np.load(npz)["prob"].astype(np.float32)[args.channel]
        g = build_arc_graph(prob > args.thr, merge_radius=params["merge_radius"],
                            min_arc_len=int(params["min_arc_len"]),
                            bridge_max_len=params["bridge_max_len"])
        if not g.junctions:
            continue
        jt = cKDTree(np.asarray(g.junctions))
        att = collections.defaultdict(list)      # junction id -> [(len, far end is junction)]
        for arc, (j0, j1) in zip(g.arcs, g.arc_ends):
            if len(arc) < 2:
                continue
            L = float(total_length(arc))
            for jj, other in ((j0, j1), (j1, j0)):
                if jj is not None:
                    att[jj].append((L, other is not None))
        for r in by_frame[stem]:
            jid = int(jt.query(np.array([[r["x"], r["y"]]], float))[1][0])
            a = att.get(jid, [])
            bridges = [L for L, isj in a if isj]
            rec = {"frame": stem, "x": r["x"], "y": r["y"], "identity": r["identity"],
                   "angle": r["angle"], "n_arms": len(a), "n_bridges": len(bridges),
                   "bridge_lens": [round(L, 1) for L in bridges],
                   "arc_lens": sorted(round(L, 1) for L, _ in a)}
            kind = "bridge" if bridges else "pruned"
            rec["kind"] = kind
            tally[(kind, bool(r["identity"]))] += 1
            out.append(rec)

    print()
    for kind in ("bridge", "pruned"):
        ok, bad = tally[(kind, True)], tally[(kind, False)]
        n = ok + bad
        if n:
            print(f"{kind:7s} n={n:4d}  identity={ok / n:.3f}")
    bl = [L for r in out for L in r["bridge_lens"]]
    if bl:
        print(f"bridge arc length q10/50/90 = {np.round(np.percentile(bl, [10, 50, 90]), 1)}"
              f"   (bridge_max_len = {params['bridge_max_len']:.1f})")
    fa = [r for r in out if not r["identity"]]
    if fa:
        print(f"failing: bridge {sum(r['kind'] == 'bridge' for r in fa)} / "
              f"pruned {sum(r['kind'] == 'pruned' for r in fa)}")
        print("shortest arc at failing junctions, q10/50/90 =",
              np.round(np.percentile([min(r["arc_lens"]) for r in fa if r["arc_lens"]],
                                     [10, 50, 90]), 1),
              f"  (min_arc_len = {params['min_arc_len']}, merge_radius = {params['merge_radius']:.1f})")
    if args.out:
        json.dump(out, open(args.out, "w"))


if __name__ == "__main__":
    main()
