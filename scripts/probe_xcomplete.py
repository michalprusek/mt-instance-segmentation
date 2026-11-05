#!/usr/bin/env python3
"""Stage 0 of XCOMP: can LOCAL mask repair at crossings pay for itself, at all?

The day's diagnosis: on real data 97.1 % of crossing failures are a mask that breaks the filament,
an oracle mask through the SAME assembler reaches junction identity 0.835 against 0.402, and the
assembler resolves 0.90-0.95 wherever the mask carries the filament through. The obvious response
is to repair the mask locally at crossings -- grow a free end onto a junction under an image test.

Before building any of that, this measures its CEILING with the repair done PERFECTLY: the GT
centerline of the failing filament is painted into the mask, `instance_a` is re-run unchanged, and
junction identity is read over ALL crossings so the number is comparable to the baseline. GT is
used purely as a measuring stick and never as an input to any deployed method.

The repair is `r = 1` by construction, so no gate, filter or geometry can exceed what this
measures. If the ceiling is below the pre-registered line, post-hoc local repair of the
probability map is CLOSED, and that costs two CPU-hours instead of a build.

Pre-registered (docs/TODO.md, docs/research/2026-09-08-xcomp-crossing-gated-growth.md):
    KILL IF delta junction identity < +0.08 on the one-sided repaint.
Two repaint variants are reported; the two-sided one is a strictly LARGER upper bound, so a
failure there is a stronger closure than a failure of the declared one.

    PYTHONPATH=src:scripts python scripts/probe_xcomplete.py --data data/real/roi303_eval \
        --pred-dir <2-channel dump> --params src/instance/params_a.json --thr 0.97 --out probe.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from instance.geometry import arclength, resample  # noqa: E402
from instance.instancer_a import default_params, instance_a  # noqa: E402
from instance.metrics import _covering_instances, _mask_trees  # noqa: E402
from instance.oracle import oracle_instance_masks  # noqa: E402
from instance.skeleton_graph import build_arc_graph  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.gt_stats import count_crossings  # noqa: E402

UP, KAPPA_MAX = 1.5, 0.25


def per_filament_identity(trees, pts, s0):
    """Does ONE predicted instance cover both arm samples of this filament? (diag convention)"""
    L = float(arclength(pts)[-1]) if len(pts) > 1 else 0.0
    if s0 - 15 < 0 or s0 + 15 > L:
        return None
    pb = pts[min(int(round(s0 - 15)), len(pts) - 1)]
    pa = pts[min(int(round(s0 + 15)), len(pts) - 1)]
    return bool(_covering_instances(trees, pb, 5.0) & _covering_instances(trees, pa, 5.0))


def paint(mask, pts, s_lo, s_hi, half_width=1.0):
    """Paint the GT centerline over an arc-length interval into the mask."""
    lo, hi = int(max(round(s_lo), 0)), int(min(round(s_hi), len(pts) - 1))
    if hi <= lo:
        return 0
    seg = pts[lo:hi + 1]
    m = oracle_instance_masks([seg], mask.shape, half_width=half_width, up=1.0)
    if not m:
        return 0
    add = m[0] & ~mask
    mask |= m[0]
    return int(add.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--thr", type=float, default=0.97)
    ap.add_argument("--split", default="val")
    ap.add_argument("--reach", type=float, default=26.0, help="eval px from crossing to a free end")
    ap.add_argument("--junc-radius", type=float, default=8.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    params = {**default_params(), **json.load(open(args.params))}
    params.pop("kappa_max", None); params.pop("prob_thr", None)

    tot = {k: 0 for k in ("n", "base_ok", "one_ok", "two_ok", "pool", "pool_reach",
                          "pool_junc", "pool_disc", "px_one", "px_two")}
    rows = []
    frames = 0
    for p in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        stem = os.path.basename(p)[:-3]
        npz = os.path.join(args.pred_dir, stem + ".npz")
        if not os.path.exists(npz):
            continue
        fr = read_frame_h5(p)
        if str(fr["attrs"].get("split")) != args.split:
            continue
        prob = np.load(npz)["prob"].astype(np.float32)[0]
        gt = [np.asarray(q, float) * UP for q in fr["polylines"]]
        if len(gt) < 2:
            continue
        mask = prob > args.thr
        res = [resample(q, ds=1.0) for q in gt]
        cr = count_crossings(gt, tol=2.0 * UP, endpoint_margin=8.0 * UP)
        if not cr:
            continue

        polys, _ = instance_a(mask, KAPPA_MAX, params, channels=prob[None], prob=prob)
        trees = _mask_trees(oracle_instance_masks(polys, mask.shape, half_width=1.0, up=1.0))

        g = build_arc_graph(mask, merge_radius=params["merge_radius"],
                            min_arc_len=int(params["min_arc_len"]),
                            bridge_max_len=params["bridge_max_len"])
        fends = [a[k] for a, e in zip(g.arcs, g.arc_ends)
                 for k, jj in ((0, e[0]), (-1, e[1])) if jj is None and len(a)]
        ftree = cKDTree(np.asarray(fends)) if fends else None
        jtree = cKDTree(np.asarray(g.junctions)) if g.junctions else None

        m_one, m_two = mask.copy(), mask.copy()
        n_frame = 0
        for c in cr:
            ids = {}
            for ik, sk in (("i", "s_i"), ("j", "s_j")):
                ids[ik] = per_filament_identity(trees, res[c[ik]], c[sk])
            if ids["i"] is None or ids["j"] is None:
                continue
            n_frame += 1
            tot["n"] += 1
            tot["base_ok"] += int(ids["i"] and ids["j"])
            xy = np.array([[c["x"], c["y"]]], float)
            d_free = float(ftree.query(xy)[0][0]) if ftree is not None else float("inf")
            d_junc = float(jtree.query(xy)[0][0]) if jtree is not None else float("inf")
            # pool: EXACTLY one filament fails, a free end is in reach, an arity>=3 junction is near
            n_fail = int(not ids["i"]) + int(not ids["j"])
            if n_fail != 1:
                continue
            if d_free > args.reach:
                continue
            tot["pool_reach"] += 1
            if d_junc > args.junc_radius:
                continue
            tot["pool_junc"] += 1
            tot["pool"] += 1
            ik, sk = ("i", "s_i") if not ids["i"] else ("j", "s_j")
            pts, s0 = res[c[ik]], c[sk]
            # variant ONE-SIDED (declared): from the nearest free end's arc position to the crossing
            if ftree is not None:
                fe = np.asarray(fends)[int(ftree.query(xy)[1][0])]
                s_end = float(np.argmin(np.linalg.norm(pts - fe[None, :], axis=1)))
                tot["px_one"] += paint(m_one, pts, min(s_end, s0), max(s_end, s0))
            # variant TWO-SIDED: the whole +-15 px arm window -- a strictly LARGER upper bound
            tot["px_two"] += paint(m_two, pts, s0 - 15, s0 + 15)
        if n_frame == 0:
            continue

        for tag, mm in (("one", m_one), ("two", m_two)):
            pl, _ = instance_a(mm, KAPPA_MAX, params, channels=prob[None], prob=prob)
            tr = _mask_trees(oracle_instance_masks(pl, mm.shape, half_width=1.0, up=1.0))
            ok = 0
            for c in cr:
                a = per_filament_identity(tr, res[c["i"]], c["s_i"])
                b = per_filament_identity(tr, res[c["j"]], c["s_j"])
                if a is None or b is None:
                    continue
                ok += int(a and b)
            tot[f"{tag}_ok"] += ok
        frames += 1
        rows.append({"frame": stem, "n": n_frame})
        print(f"{stem}: {n_frame} crossings, pool so far {tot['pool']}", flush=True)
        if args.limit and frames >= args.limit:
            break

    n = max(tot["n"], 1)
    summ = {"frames": frames, "thr": args.thr, "n_crossings": tot["n"],
            "baseline_identity": round(tot["base_ok"] / n, 4),
            "repaint_one_sided": round(tot["one_ok"] / n, 4),
            "repaint_two_sided": round(tot["two_ok"] / n, 4),
            "delta_one_sided": round((tot["one_ok"] - tot["base_ok"]) / n, 4),
            "delta_two_sided": round((tot["two_ok"] - tot["base_ok"]) / n, 4),
            "pool": tot["pool"], "pool_share": round(tot["pool"] / n, 4),
            "pool_after_reach": tot["pool_reach"], "pool_after_junction": tot["pool_junc"],
            "painted_px_one": tot["px_one"], "painted_px_two": tot["px_two"],
            "KILL_LINE": 0.08,
            "verdict_declared": ("PASS" if (tot["one_ok"] - tot["base_ok"]) / n >= 0.08 else "KILL")}
    json.dump({"summary": summ, "rows": rows}, open(args.out, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
