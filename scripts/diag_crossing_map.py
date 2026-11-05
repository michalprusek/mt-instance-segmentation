#!/usr/bin/env python3
"""WHY does the model's mask lose identity at crossings that an oracle mask keeps?

Junction identity on roi303 VAL is 0.32 with the model mask against 0.75 with the oracle mask
through the SAME instancer, so the loss is in the probability map around the crossing. This
script measures, per GT crossing, the candidate mechanisms:

  dip      -- min probability on each filament within +-`near` px of the crossing, relative to
              the filament's own arm probability 8-20 px away. A dip below the decode threshold
              breaks the filament at the junction; the instancer then sees 4 free ends, not a
              junction, and must gap-link.
  polarity -- image contrast sign of each arm (IRM flips dark<->bright). Opposite-sign arms
              CANCEL where they overlap, which is the physical reason for a dip.
  connected-- whether the two arm samples of each filament (+-15 px) lie in the same connected
              component of the thresholded mask (mask-level identity, before any instancer).
  identity -- whether ONE predicted instance covers both arm samples (instancer-level identity,
              the published metric), reproduced here per crossing.
  blob     -- mask area inside a 7 px disc at the crossing over the same for the oracle mask
              (a fused/bloated junction neighbourhood distorts the skeleton's arms).

plus a montage of the failures so the mechanism can be SEEN, not argued.

Reads cached probability maps in the eval frame (as `eval_v5.py --dump-pred` writes them).
Conventions: polylines are (x=col, y=row) at native scale; prob is [C, H*1.5, W*1.5].

    PYTHONPATH=src:scripts python scripts/diag_crossing_map.py --data data/real/roi303_eval \
        --pred-dir <dir with roi_img_*.npz> --thr 0.98 --out data/enc_sensitivity_testset/crossing_diag
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import label as cc_label, map_coordinates
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from instance.geometry import arclength, resample  # noqa: E402
from instance.instancer_a import instance_a  # noqa: E402
from instance.metrics import _covering_instances, _mask_trees  # noqa: E402
from instance.skeleton_graph import build_arc_graph  # noqa: E402
from instance.oracle import oracle_instance_masks, oracle_mask  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.gt_stats import count_crossings  # noqa: E402

UP, KAPPA_MAX = 1.5, 0.25


def _sample(arr, pts_xy):
    """Bilinear sample of a 2-D array at (x, y) points."""
    return map_coordinates(arr, [pts_xy[:, 1], pts_xy[:, 0]], order=1, mode="nearest")


def _arm_stats(prob, img_eval, pts, s0, near, far_lo, far_hi):
    """Probability + image contrast along one filament around arc position s0."""
    s = arclength(pts)
    # ridge-tolerant probability: max over +-2 px perpendicular to the GT (human GT is ~0.5 px off)
    tang = np.gradient(pts, axis=0)
    nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
    nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-8)
    pmax = np.max(np.stack([_sample(prob, pts + d * nrm) for d in (-2, -1, 0, 1, 2)]), 0)
    win_near = np.abs(s - s0) <= near
    win_far = (np.abs(s - s0) >= far_lo) & (np.abs(s - s0) <= far_hi)
    if not win_far.any() or not win_near.any():
        return None
    # local image contrast sign: centre minus mean of +-5 px flanks (in eval px), on the far arms
    c0 = _sample(img_eval, pts)
    cf = 0.5 * (_sample(img_eval, pts + 5.0 * nrm) + _sample(img_eval, pts - 5.0 * nrm))
    contrast = c0 - cf
    return {
        "p_near_min": float(pmax[win_near].min()),
        "p_far_mean": float(pmax[win_far].mean()),
        "p_far_min": float(pmax[win_far].min()),
        "contrast_far": float(np.median(contrast[win_far])),
        "contrast_near": float(np.median(contrast[win_near])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/roi303_eval")
    ap.add_argument("--split", default="val")
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--topo-radius", type=float, default=8.0,
                    help="eval-frame px (1.5x native) within which a junction cluster counts")
    ap.add_argument("--thr", type=float, default=0.98)
    ap.add_argument("--params", default="src/instance/params_a_model_v5H.json")
    ap.add_argument("--near", type=float, default=4.5, help="eval px around the crossing")
    ap.add_argument("--frames", default="", help="comma list of stems to restrict to")
    ap.add_argument("--out", required=True)
    ap.add_argument("--montage", type=int, default=36)
    ap.add_argument("--channel", type=int, default=0,
                    help="dump channel used as the probability map (0 = existence; a 2-channel "
                         "heat dump must NOT be max-reduced, that would mix the two heads)")
    args = ap.parse_args()
    if args.split == "test":
        raise SystemExit("refusing to touch TEST")
    os.makedirs(args.out, exist_ok=True)
    params = json.load(open(args.params))
    params.pop("kappa_max", None); params.pop("prob_thr", None)
    # `instance_a` merges as {**default_params(), **params}, and default_params() already loads
    # src/instance/params_a.json -- the vector behind every published instancer number. So
    # --params does not SELECT a configuration, it OVERRIDES the published one. Passing a tuned
    # probe vector here once halved in-domain junction identity (0.630 -> 0.311) while the mask
    # was bit-identical, and the only thing that caught it was a pre-declared self-check.
    from instance.instancer_a import default_params as _dp
    _base = _dp()
    _diff = sorted(k for k, v in params.items() if k in _base and _base[k] != v)
    if _diff:
        print(f"WARNING: --params overrides the published instancer on {len(_diff)} keys: "
              f"{', '.join(_diff)} -- identity numbers will NOT match published ones")
    want = set(args.frames.split(",")) if args.frames else None

    rows, crops = [], []
    for p in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        stem = os.path.basename(p)[:-3]
        if want and stem not in want:
            continue
        npz = os.path.join(args.pred_dir, stem + ".npz")
        if not os.path.exists(npz):
            continue
        fr = read_frame_h5(p)
        if str(fr["attrs"].get("split")) != args.split:
            continue
        shape = (int(fr["attrs"]["height"]), int(fr["attrs"]["width"]))
        hi = (int(round(shape[0] * UP)), int(round(shape[1] * UP)))
        gt = [np.asarray(q, float) * UP for q in fr["polylines"]]
        if len(gt) < 2:
            continue
        allch = np.load(npz)["prob"].astype(np.float32)
        prob = allch[min(args.channel, allch.shape[0] - 1)]
        from scipy.ndimage import zoom
        img = np.asarray(fr["image"], float)
        lo, hi_ = np.percentile(img, [1, 99])
        img01 = np.clip((img - lo) / (hi_ - lo + 1e-6), 0, 1)
        img_eval = zoom(img01, UP, order=1)
        mask = prob > args.thr
        omask = oracle_mask(fr["polylines"], shape, half_width=1.0, up=UP)
        lab, _ = cc_label(mask, structure=np.ones((3, 3), int))
        # the arc graph the instancer itself walks -- does a junction EXIST at the crossing?
        g = build_arc_graph(mask, merge_radius=params["merge_radius"],
                            min_arc_len=int(params["min_arc_len"]),
                            bridge_max_len=params["bridge_max_len"])
        jtree = cKDTree(np.asarray(g.junctions)) if g.junctions else None
        # how many arc ends attach to each junction: 4 = the graph already knows two filaments
        # pass through (multiplicity is redundant there), 3 = an arm was pruned or absorbed,
        # 2 = a contracted kink that only looks like a junction.
        n_arms_j = [0] * len(g.junctions)
        for arc_, (j0_, j1_) in zip(g.arcs, g.arc_ends):
            if len(arc_) < 2:
                continue
            for jj in (j0_, j1_):
                if jj is not None:
                    n_arms_j[jj] += 1
        fends = [arc_[k] for arc_, e in zip(g.arcs, g.arc_ends)
                 for k, jj in ((0, e[0]), (-1, e[1])) if jj is None and len(arc_)]
        ftree = cKDTree(np.asarray(fends)) if fends else None
        # instancer on the model mask, published metric's inputs
        polys, _ = instance_a(mask, KAPPA_MAX, params, channels=prob[None], prob=prob)
        pmasks = oracle_instance_masks(polys, hi, half_width=1.0, up=1.0)
        trees = _mask_trees(pmasks)
        # oracle-mask instancer for the paired reference
        opolys, _ = instance_a(omask, KAPPA_MAX, params, channels=None, prob=None)
        otrees = _mask_trees(oracle_instance_masks(opolys, hi, half_width=1.0, up=1.0))

        res = [resample(q, ds=1.0) for q in gt]
        lens = [float(arclength(q)[-1]) if len(q) > 1 else 0.0 for q in res]
        cr = count_crossings(gt, tol=2.0 * UP, endpoint_margin=8.0 * UP)
        yy, xx = np.mgrid[-7:8, -7:8]
        disc = (yy ** 2 + xx ** 2) <= 49
        for c in cr:
            rec = {"frame": stem, "x": float(c["x"]), "y": float(c["y"]),
                   "angle": float(c["angle_deg"])}
            ok_all, conn_all, oracle_ok_all = True, True, True
            usable = True
            for k, (ik, sk) in enumerate((("i", "s_i"), ("j", "s_j"))):
                gi, s = c[ik], c[sk]
                pts = res[gi]
                if s - 15 < 0 or s + 15 > lens[gi]:
                    usable = False
                    break
                st = _arm_stats(prob, img_eval, pts, s, args.near, 8.0, 20.0)
                if st is None:
                    usable = False
                    break
                pb = pts[min(int(round(s - 15)), len(pts) - 1)]
                pa = pts[min(int(round(s + 15)), len(pts) - 1)]
                before = _covering_instances(trees, pb, 5.0)
                after = _covering_instances(trees, pa, 5.0)
                ok = bool(before & after)
                ook = bool(_covering_instances(otrees, pb, 5.0) & _covering_instances(otrees, pa, 5.0))
                # mask-level connectivity: nearest mask pixel within 3 px of each arm sample
                def _lab_at(pt):
                    r, cc_ = int(round(pt[1])), int(round(pt[0]))
                    r0, r1 = max(0, r - 3), min(hi[0], r + 4)
                    c0, c1 = max(0, cc_ - 3), min(hi[1], cc_ + 4)
                    win = lab[r0:r1, c0:c1]
                    vals = win[win > 0]
                    return set(vals.tolist())
                conn = bool(_lab_at(pb) & _lab_at(pa))
                rec[f"f{k}"] = {**st, "identity": ok, "connected": conn, "oracle_identity": ook,
                                "found": bool(before or after)}
                ok_all &= ok; conn_all &= conn; oracle_ok_all &= ook
            if not usable:
                continue
            r, cc_ = int(round(c["y"])), int(round(c["x"]))
            if 7 <= r < hi[0] - 7 and 7 <= cc_ < hi[1] - 7:
                m_area = int(mask[r - 7:r + 8, cc_ - 7:cc_ + 8][disc].sum())
                o_area = int(omask[r - 7:r + 8, cc_ - 7:cc_ + 8][disc].sum())
                rec["blob_ratio"] = m_area / max(o_area, 1)
                rec["p_cross"] = float(prob[r - 1:r + 2, cc_ - 1:cc_ + 2].max())
            pxy = np.array([[rec["x"], rec["y"]]], float)
            d_j = float(jtree.query(pxy)[0][0]) if jtree is not None else float("inf")
            d_f = float(ftree.query(pxy)[0][0]) if ftree is not None else float("inf")
            on = bool(mask[max(r - 1, 0):r + 2, max(cc_ - 1, 0):cc_ + 2].any())
            rec["d_junction"], rec["d_free_end"], rec["mask_on"] = d_j, d_f, on
            if jtree is not None:
                rec["n_arms_junction"] = int(n_arms_j[int(jtree.query(pxy)[1][0])])
            # `conn_all` is per-FILAMENT continuity: false means the mask breaks the filament at
            # the crossing, which is a map failure and must not be pooled with the assembler's.
            rec["topo"] = ("broken" if not conn_all else
                           "node" if d_j <= args.topo_radius else "fused")
            rec["identity"] = ok_all; rec["connected"] = conn_all
            rec["oracle_identity"] = oracle_ok_all
            s0 = np.sign(rec["f0"]["contrast_far"]); s1 = np.sign(rec["f1"]["contrast_far"])
            rec["same_polarity"] = bool(s0 == s1)
            rec["dip"] = float(1.0 - min(rec["f0"]["p_near_min"], rec["f1"]["p_near_min"])
                               / max(1e-6, min(rec["f0"]["p_far_mean"], rec["f1"]["p_far_mean"])))
            rows.append(rec)
            if not ok_all and len(crops) < 4 * args.montage:
                R = 30
                if R <= r < hi[0] - R and R <= cc_ < hi[1] - R:
                    sl = (slice(r - R, r + R), slice(cc_ - R, cc_ + R))
                    inst = np.zeros(hi[:2], np.int32)
                    for n_, m_ in enumerate(pmasks, 1):
                        inst[m_ & (inst == 0)] = n_
                    crops.append({"img": img_eval[sl].copy(), "prob": prob[sl].copy(),
                                  "mask": mask[sl].copy(), "omask": omask[sl].copy(),
                                  "inst": inst[sl].copy(), "rec": rec})
        print(f"{stem}: {len(cr)} crossings, cumulative rows {len(rows)}", flush=True)

    # ---- summary
    def rate(sel):
        sel = list(sel)
        return (float(np.mean([r["identity"] for r in sel])) if sel else float("nan"), len(sel))
    summ = {"n": len(rows), "thr": args.thr,
            "identity": rate(rows), "oracle_identity": (float(np.mean([r["oracle_identity"] for r in rows])) if rows else float("nan")),
            "mask_connected": float(np.mean([r["connected"] for r in rows])) if rows else float("nan")}
    same = [r for r in rows if r["same_polarity"]]; opp = [r for r in rows if not r["same_polarity"]]
    summ["identity_same_polarity"] = rate(same); summ["identity_opposite_polarity"] = rate(opp)
    for nm, lo, hi_ in (("shallow", 0, 30), ("oblique", 30, 60), ("steep", 60, 91)):
        summ[f"identity_{nm}"] = rate(r for r in rows if lo <= r["angle"] < hi_)
    # dip bins
    for nm, lo, hi_ in (("dip<0.02", -1, 0.02), ("dip0.02-0.2", 0.02, 0.2), ("dip0.2-0.5", 0.2, 0.5), ("dip>0.5", 0.5, 2)):
        summ[f"identity_{nm}"] = rate(r for r in rows if lo <= r["dip"] < hi_)
    # failure taxonomy among identity failures
    fails = [r for r in rows if not r["identity"]]
    summ["n_fail"] = len(fails)
    summ["fail_mask_disconnected"] = float(np.mean([not r["connected"] for r in fails])) if fails else float("nan")
    summ["fail_connected_but_split"] = float(np.mean([r["connected"] for r in fails])) if fails else float("nan")
    topo_rows = [r for r in rows if "topo" in r]
    if topo_rows:
        for tag, sel in (("all", lambda r: True), ("fail", lambda r: not r["identity"]),
                         ("ok", lambda r: r["identity"]),
                         ("fail_connected", lambda r: not r["identity"] and r["connected"]),
                         ("fail_disconnected", lambda r: not r["identity"] and not r["connected"])):
            sub = [r for r in topo_rows if sel(r)]
            summ[f"topo_{tag}"] = ({k: round(float(np.mean([r["topo"] == k for r in sub])), 4)
                                    for k in ("node", "fused", "broken")} | {"n": len(sub)}) if sub else {"n": 0}
        for k in ("node", "fused", "broken"):
            sub = [r for r in topo_rows if r["topo"] == k]
            na = [r["n_arms_junction"] for r in sub if "n_arms_junction" in r]
            fa = [r["n_arms_junction"] for r in sub if "n_arms_junction" in r and not r["identity"]]
            if na:
                summ[f"arms_at_junction_{k}"] = {str(v): int(np.sum(np.array(na) == v)) for v in (2, 3, 4)} | \
                    {"5+": int(np.sum(np.array(na) >= 5)), "n": len(na)}
                summ[f"arms_at_junction_{k}_fail"] = {str(v): int(np.sum(np.array(fa) == v)) for v in (2, 3, 4)} | \
                    {"5+": int(np.sum(np.array(fa) >= 5)), "n": len(fa)}
            summ[f"identity_topo_{k}"] = [round(float(np.mean([r["identity"] for r in sub])), 4), len(sub)] if sub else None
            summ[f"oracle_identity_topo_{k}"] = [round(float(np.mean([r["oracle_identity"] for r in sub])), 4), len(sub)] if sub else None
        fr_ = [r for r in topo_rows if not r["identity"]]
        summ["fail_node_share_by_radius"] = {str(R): round(float(np.mean([r["d_junction"] <= R for r in fr_])), 4)
                                             for R in (4, 6, 8, 12, 16)} if fr_ else {}
    summ["fail_dip_gt_thr"] = float(np.mean([min(r["f0"]["p_near_min"], r["f1"]["p_near_min"]) < args.thr for r in fails])) if fails else float("nan")
    summ["fail_oracle_ok"] = float(np.mean([r["oracle_identity"] for r in fails])) if fails else float("nan")
    summ["fail_arm_missing"] = float(np.mean([not (r["f0"]["found"] and r["f1"]["found"]) for r in fails])) if fails else float("nan")
    summ["blob_ratio_p50_ok"] = float(np.median([r["blob_ratio"] for r in rows if r["identity"] and "blob_ratio" in r] or [np.nan]))
    summ["blob_ratio_p50_fail"] = float(np.median([r["blob_ratio"] for r in fails if "blob_ratio" in r] or [np.nan]))
    summ["dip_p50_ok"] = float(np.median([r["dip"] for r in rows if r["identity"]] or [np.nan]))
    summ["dip_p50_fail"] = float(np.median([r["dip"] for r in fails] or [np.nan]))
    print(json.dumps(summ, indent=1))
    with open(os.path.join(args.out, "crossing_diag.json"), "w") as fh:
        json.dump({"summary": summ, "rows": rows}, fh, indent=1)

    # ---- montage of failures
    if crops:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        crops = crops[:args.montage]
        n = len(crops)
        fig, axes = plt.subplots(n, 5, figsize=(12.5, 2.5 * n))
        if n == 1:
            axes = axes[None]
        for a, c in zip(axes, crops):
            r = c["rec"]
            a[0].imshow(c["img"], cmap="gray"); a[0].set_title(f"{r['frame']} ang={r['angle']:.0f} {'same' if r['same_polarity'] else 'OPP'}", fontsize=7)
            a[1].imshow(c["prob"], cmap="magma", vmin=0, vmax=1); a[1].set_title(f"prob  dip={r['dip']:.2f} pX={r.get('p_cross', float('nan')):.2f}", fontsize=7)
            a[2].imshow(c["mask"], cmap="gray"); a[2].set_title(f"mask>{args.thr}  conn={r['connected']}", fontsize=7)
            a[3].imshow(c["inst"] % 20, cmap="tab20", interpolation="nearest"); a[3].set_title("instances", fontsize=7)
            a[4].imshow(c["omask"], cmap="gray"); a[4].set_title(f"oracle mask  or_id={r['oracle_identity']}", fontsize=7)
            for ax in a:
                ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        fig.savefig(os.path.join(args.out, "crossing_failures.png"), dpi=110)
        print("montage:", os.path.join(args.out, "crossing_failures.png"))


if __name__ == "__main__":
    main()
