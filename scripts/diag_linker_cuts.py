#!/usr/bin/env python3
"""Why does the instancer leave the 0-5 px crossing cuts open?  READ-ONLY diagnosis on cached maps
(declared docs/TODO.md 0.0-CAMPAIGN 2026-09-06 06:20; nothing in src/instance changes, no parameter
is fitted). RUNS ON TULEN (CPU, chunked).

For every MAP_BREAK GT of diag_covered_split.py (a union-covered, unmatched GT whose thresholded
mask splits into >= 2 components along it), each CUT (consecutive GT samples in different mask
components) is located and the instancer's own stages are replayed up to -- but not including --
_link_free_ends, exactly as instance_a does. The cut is then classified by which gate stops it:

  G_BRIDGED    the arc graph already joins the two sides (bridge_max_len) -- the cut is not the failure
  D_CONSUMED   an arc end at the cut is a JUNCTION arm that match_junction paired with another arc:
               the crossing took the arm, so no free end exists to link
  S_SHORT      the chain on one side was dropped before linking (< min_length/2)
  N_NOEND      no chain end within R of one side (the skeleton passes; failure is elsewhere)
  E_FAR        free ends on both sides, tip-to-tip > link_max_gap
  B_EVIDENCE   free ends, mean p along the bridge < bridge_thr
  C_CURVATURE  free ends, evidence ok, implied turn over the floored span > kappa_max
  A_COST       free ends, admissible, evidence ok, but 2*c_open_link - cost <= 0
  A_LINKED     the gap-link matching DID join these two ends -- the GT fails downstream (curvature
               enforcement, min_length, or the 1:1 assignment), not at the cut
  A_ELSEWHERE  one of the two ends was linked to a DIFFERENT partner (the linker chose a wrong join)
  A_UNCHOSEN   positive weight, neither end matched: the max-weight matching left both open

    PYTHONPATH=.:src:scripts python scripts/diag_linker_cuts.py --pred-dir preds/SPARSE35_roi_native \
        --csplit "runs/diag_roi/csplit/SPARSE35_roi_native_*.json" --thr 0.99 --frames a,b --out runs/diag_roi/linker/S35_0.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import label as cc_label
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from diag_fp_anatomy import UP, KAPPA_MAX, K_ORI, TOL, resample as resample_poly  # noqa: E402
from instance.geometry import resample, total_length  # noqa: E402
from instance.instancer_a import _arm_at, _build_arms, _walk, default_params  # noqa: E402
from instance.matching import bridge_evidence, match_junction, pair_cost  # noqa: E402
from instance.skeleton_graph import build_arc_graph  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402

CLASSES = ("G_BRIDGED", "D_CONSUMED", "S_SHORT", "N_NOEND", "E_FAR", "B_EVIDENCE", "C_CURVATURE", "A_COST", "A_LINKED", "A_ELSEWHERE", "A_UNCHOSEN")


def replay(mask, prob, chans, p):
    """instance_a up to _link_free_ends; returns everything the diagnosis needs."""
    g = build_arc_graph(mask, merge_radius=p["merge_radius"], min_arc_len=int(p["min_arc_len"]), bridge_max_len=p["bridge_max_len"])
    if not g.arcs:
        return None
    arcs = [resample(a, ds=p["ds"]) for a in g.arcs]
    arms_by_j = _build_arms(arcs, g.arc_ends, p["window"], chans)
    partner = {}
    for jid, arms in arms_by_j.items():
        for i, j in match_junction(arms, kappa_max=KAPPA_MAX, w_theta=p["w_theta"], w_kappa=p["w_kappa"], w_gap=p["w_gap"],
                                   c_open=p["c_open"], gap_len=None, gap_floor=p["gap_floor"], w_ori=p["w_ori"], p_open=p.get("p_open", 0.5)):
            pa = (arms[i].arc_idx, arms[i].which); pb = (arms[j].arc_idx, arms[j].which)
            partner[pa] = (pb, jid); partner[pb] = (pa, jid)
    chains_all = _walk(arcs, partner, bridge_pts=g.junctions)
    chains = [c for c in chains_all if total_length(c) >= p["min_length"] * 0.5]
    short = [c for c in chains_all if total_length(c) < p["min_length"] * 0.5]
    free_arms = [_arm_at(ch, ci, which, p["window"], chans) for ci, ch in enumerate(chains) for which in ("start", "end")]
    # the gap-link matching itself, exactly as _link_free_ends runs it (geometric cost, no learned linker)
    allow = (lambda a, b: bridge_evidence(prob, a, b) >= p["bridge_thr"]) if (prob is not None and p["bridge_thr"] > 0) else None
    linked = {}
    for i, j in match_junction(free_arms, kappa_max=KAPPA_MAX, w_theta=p["w_theta"], w_kappa=p["w_kappa"], w_gap=p["w_gap"],
                               c_open=p["c_open_link"], gap_len=None, gap_floor=p["gap_floor"], w_ori=p["w_ori"],
                               max_gap=p["link_max_gap"], allow=allow, p_open=p.get("p_open_link", p.get("p_open", 0.5))):
        linked[i] = j; linked[j] = i
    # arc ends that sit at junctions and were consumed by a match
    consumed = [(np.asarray(arcs[ai][0] if which == "start" else arcs[ai][-1], float)) for (ai, which) in partner]
    # component id per arc (for G_BRIDGED: an arc/chain that spans two mask components)
    return dict(arcs=arcs, chains=chains, short=short, free_arms=free_arms, linked=linked, consumed=np.array(consumed) if consumed else np.zeros((0, 2)),
                free_pos=np.array([a.pos for a in free_arms]) if free_arms else np.zeros((0, 2)),
                short_pos=np.array([c[k] for c in short for k in (0, -1)]) if short else np.zeros((0, 2)),
                chain_pts=[np.asarray(c, float) for c in chains])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/roi303_eval")
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--csplit", required=True, help="glob of diag_covered_split outputs for this arm/threshold")
    ap.add_argument("--thr", type=float, default=0.99)
    ap.add_argument("--params", default="src/instance/params_a_model_v5H.json")
    ap.add_argument("--frames", default="")
    ap.add_argument("--radius", type=float, default=10.0, help="eval px: a chain end within R of the cut side counts")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    params = json.load(open(args.params)); params.pop("kappa_max", None); params.pop("prob_thr", None)
    p = {**default_params(), **params}
    want = set(args.frames.split(",")) if args.frames else None
    todo = {}
    for f in glob.glob(args.csplit):
        for rec in json.load(open(f))["frames"]:
            gts = [r["gt"] for r in rec["rows"] if r["cls"] == "MAP_BREAK"]
            if gts and (want is None or rec["frame"] in want):
                todo[rec["frame"]] = gts
    print(len(todo), "frames with MAP_BREAK GT", flush=True)
    out = {"args": vars(args), "cuts": []}
    for stem in sorted(todo):
        fr = read_frame_h5(os.path.join(args.data, stem + ".h5"))
        if str(fr["attrs"].get("split")) == "test":
            raise SystemExit("refusing to touch TEST")
        allch = np.load(os.path.join(args.pred_dir, stem + ".npz"))["prob"].astype(np.float32)
        prob = allch[0]; chans = allch[:K_ORI]; H, W = prob.shape
        mask = prob > args.thr
        lab, _ = cc_label(mask, structure=np.ones((3, 3)))
        mask_yx = np.stack(np.where(mask), 1).astype(np.float32); mtree = cKDTree(mask_yx)
        gt = [np.asarray(q, float) * UP for q in fr["polylines"] if len(q) >= 2]
        rp = replay(mask, prob, chans, p)
        if rp is None:
            continue
        ftree = cKDTree(rp["free_pos"]) if len(rp["free_pos"]) else None
        ctree = cKDTree(rp["consumed"]) if len(rp["consumed"]) else None
        stree = cKDTree(rp["short_pos"]) if len(rp["short_pos"]) else None
        chain_all = np.concatenate(rp["chain_pts"], 0) if rp["chain_pts"] else np.zeros((0, 2))
        chain_id = np.concatenate([np.full(len(c), i) for i, c in enumerate(rp["chain_pts"])]) if rp["chain_pts"] else np.zeros(0, int)
        chtree = cKDTree(chain_all) if len(chain_all) else None
        for j in todo[stem]:
            c = resample_poly(gt[j])                      # (x, y) eval px, 1-px samples
            d, idx = mtree.query(c[:, ::-1]); near = d <= TOL
            comp = np.where(near, lab[mask_yx[idx, 0].astype(int), mask_yx[idx, 1].astype(int)], 0)
            # cuts: transitions between two different non-zero components, possibly across a run of zeros
            runs = []; last_comp, last_i = None, None
            for i, cid in enumerate(comp):
                if cid == 0:
                    continue
                if last_comp is not None and cid != last_comp:
                    runs.append((last_i, i))
                last_comp, last_i = cid, i
            for (ia, ib) in runs:
                pa, pb = c[ia], c[ib]; gap_px = float(np.hypot(*(pb - pa)))
                rec = {"frame": stem, "gt": j, "gap_px": round(gap_px, 1)}
                # both sides on one chain already -> the graph bridged it
                if chtree is not None:
                    da, ka = chtree.query(pa); db, kb = chtree.query(pb)
                    if da <= args.radius and db <= args.radius and chain_id[ka] == chain_id[kb]:
                        rec["cls"] = "G_BRIDGED"; out["cuts"].append(rec); continue
                sides = []
                for pt in (pa, pb):
                    fa = ftree.query(pt) if ftree is not None else (np.inf, -1)
                    if fa[0] <= args.radius:
                        sides.append(("free", int(fa[1]))); continue
                    if ctree is not None and ctree.query(pt)[0] <= args.radius:
                        sides.append(("consumed", None)); continue
                    if stree is not None and stree.query(pt)[0] <= args.radius:
                        sides.append(("short", None)); continue
                    sides.append(("none", None))
                kinds = [s[0] for s in sides]
                if "consumed" in kinds:
                    rec["cls"] = "D_CONSUMED"
                elif "short" in kinds:
                    rec["cls"] = "S_SHORT"
                elif "none" in kinds:
                    rec["cls"] = "N_NOEND"
                else:
                    a, b = rp["free_arms"][sides[0][1]], rp["free_arms"][sides[1][1]]
                    L = float(np.linalg.norm(a.pos - b.pos)); ev = bridge_evidence(prob, a, b)
                    cost, kimp = pair_cost(a, b, None, p["w_theta"], p["w_kappa"], p["w_gap"], p["w_ori"], p["gap_floor"])
                    rec.update({"tip_dist": round(L, 1), "evidence": round(ev, 3), "kappa_implied": round(kimp, 3), "cost": round(cost, 2),
                                "weight": round(2 * p["c_open_link"] - cost, 2)})
                    if a.arc_idx == b.arc_idx:
                        rec["cls"] = "G_BRIDGED"          # same chain's two ends (a loop): not a linking failure
                    elif L > p["link_max_gap"]:
                        rec["cls"] = "E_FAR"
                    elif p["bridge_thr"] > 0 and ev < p["bridge_thr"]:
                        rec["cls"] = "B_EVIDENCE"
                    elif kimp > KAPPA_MAX:
                        rec["cls"] = "C_CURVATURE"
                    elif 2 * p["c_open_link"] - cost <= 0:
                        rec["cls"] = "A_COST"
                    else:
                        ia_, ib_ = sides[0][1], sides[1][1]; lk = rp["linked"]
                        if lk.get(ia_) == ib_:
                            rec["cls"] = "A_LINKED"       # the linker DID join them; the GT fails downstream
                        elif ia_ in lk or ib_ in lk:
                            rec["cls"] = "A_ELSEWHERE"    # one end was linked to a different partner
                            other = lk.get(ia_, lk.get(ib_)); o = rp["free_arms"][other]
                            rec["elsewhere_dist"] = round(float(np.linalg.norm((a if ia_ in lk else b).pos - o.pos)), 1)
                        else:
                            rec["cls"] = "A_UNCHOSEN"     # positive weight, neither end matched: the matching left both open
                out["cuts"].append(rec)
        n = sum(1 for r in out["cuts"] if r["frame"] == stem)
        print(f"{stem}: {len(todo[stem])} MAP_BREAK GT, {n} cuts so far total {len(out['cuts'])}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"))
    cnt = {c: sum(1 for r in out["cuts"] if r["cls"] == c) for c in CLASSES}
    print("TOTAL cuts", len(out["cuts"]), cnt)


if __name__ == "__main__":
    main()
