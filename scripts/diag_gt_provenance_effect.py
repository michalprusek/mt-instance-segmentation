#!/usr/bin/env python3
"""Does our score depend on WHO drew each ground-truth polyline?

26.1 % of the crossing-dense block's ground truth (190 of 728 polylines) is `source="file"` --
unreviewed output of an older v7+PySOAX pipeline that a human never corrected. The rest is
`source="manual"`. The h5 files have carried this split since the benchmark was built and it has
never been read.

Two opposite failures are possible and they need opposite responses:

  * scoring WORSE on the model-drawn polylines means those curves are simply wrong, and the block's
    score is being dragged down by grading against another model's mistakes;
  * scoring BETTER on them is AGREEMENT BIAS -- our instancer is in the same family as the one that
    produced them, so it is being graded against its own ancestor, and the block's score is inflated.

The block in question is the one the headline capability claim rests on, so either way the number
cannot be quoted until this is known.

    python scripts/diag_gt_provenance_effect.py --pred-dir preds/LEN_treatment --thr 0.997 --min-length 20
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.ndimage import zoom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from instance.instancer_a import instance_a  # noqa: E402
from instance.metrics import centerline_f1  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.fov import fov_mask  # noqa: E402

UP = 1.5
KAPPA_MAX = 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--data-dir", default="data/real/mt34_htw_eval")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--params", default="src/instance/params_a_model_v5H.json")
    ap.add_argument("--thr", type=float, default=0.997)
    ap.add_argument("--min-length", type=float, default=20.0)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out", default="gt_provenance_effect.json")
    args = ap.parse_args()
    if args.split == "test":
        raise SystemExit("refusing to touch TEST: it carries a declared exposure budget")

    params = dict(json.load(open(args.params)))
    params["min_length"] = args.min_length
    params["prob_thr"] = args.thr

    rec = defaultdict(list)          # (block, source) -> [is_tp, ...]
    lens = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(args.data_dir, "*.h5"))):
        fr = read_frame_h5(path)
        if str(fr["attrs"].get("split")) != args.split:
            continue
        b = os.path.basename(path)
        block = "htw" if b.startswith("htw") else ("585" if b.startswith("alice") else "586")
        raw = str(fr["attrs"].get("sources", ""))
        srcs = raw.split(",") if raw else []
        polys = fr["polylines"]
        if len(srcs) != len(polys):
            # The provenance list and the polyline list must line up or every label is wrong.
            # Better to refuse this frame loudly than to mislabel a whole block.
            print(f"  SKIP {b}: {len(srcs)} provenance entries for {len(polys)} polylines")
            continue
        keep = [i for i, p in enumerate(polys) if len(p) >= 2]
        gt = [np.asarray(polys[i], float) * UP for i in keep]
        srcs = [srcs[i] for i in keep]
        if not gt:
            continue
        npz = os.path.join(args.pred_dir, b.replace(".h5", ".npz"))
        if not os.path.isfile(npz):
            raise SystemExit(f"missing prediction {npz} for a frame that has ground truth")
        prob = np.load(npz)["prob"].astype(np.float32)
        if prob.ndim == 3:
            prob = prob.max(axis=0)
        img = fr["image"]
        hi = (int(round(img.shape[0] * UP)), int(round(img.shape[1] * UP)))
        fv = zoom(fov_mask(img).astype(np.uint8), UP, order=0).astype(bool)
        fv = np.pad(fv, ((0, max(0, hi[0] - fv.shape[0])), (0, max(0, hi[1] - fv.shape[1]))),
                    constant_values=True)[:hi[0], :hi[1]]
        _p, masks = instance_a((prob > args.thr) & fv, KAPPA_MAX, params=params,
                               channels=prob[None], prob=prob)
        r = centerline_f1(masks, gt, tol=5.0, length_coverage=0.95, precision_coverage=0.95)
        tp_gt = {p["gt"] for p in r["pairs"] if p["tp"]}
        for j, s in enumerate(srcs):
            rec[(block, s)].append(1.0 if j in tp_gt else 0.0)
            lens[(block, s)].append(float(np.sqrt(((gt[j][1:] - gt[j][:-1]) ** 2).sum(1)).sum()))

    print(f"TRUE-POSITIVE RATE BY GROUND-TRUTH PROVENANCE, STRATIFIED BY LENGTH  "
          f"(thr={args.thr}, min_length={args.min_length}, {args.split})\n")
    print('  "manual" = a human drew or corrected it     "file" = unreviewed older-model output')
    print("  Model-drawn polylines are systematically SHORTER than human ones, and length affects")
    print("  difficulty on its own, so the marginal comparison is confounded. Within a length band")
    print("  it is not.\n")
    LEN = [(0, 200), (200, 450), (450, 900), (900, 1e9)]
    out = {}
    rng = np.random.default_rng(20260828)
    for block in ("585", "586", "htw"):
        d = {s: np.array(v) for (bk, s), v in rec.items() if bk == block and len(v)}
        L = {s: np.array(lens[(block, s)]) for s in d}
        if not d:
            continue
        print(f'--- {block} ---')
        print(f'  {"length band":>14s} | {"manual":>18s} {"file (model)":>18s} {"difference":>12s}')
        deltas, weights = [], []
        strat = {}
        for lo, hi_ in LEN:
            cells, means = [], {}
            for s in ("manual", "file"):
                if s not in d:
                    cells.append("-"); continue
                m = (L[s] >= lo) & (L[s] < hi_)
                if m.sum() >= 8:
                    means[s] = float(d[s][m].mean())
                    cells.append(f'{means[s]:.3f} (n={int(m.sum())})')
                else:
                    cells.append(f'- (n={int(m.sum())})')
            lbl = f"{int(lo)}-{'inf' if hi_ > 1e8 else int(hi_)}"
            diff = f'{means["file"] - means["manual"]:+.3f}' if len(means) == 2 else ''
            if len(means) == 2:
                n = min(((L["manual"] >= lo) & (L["manual"] < hi_)).sum(),
                        ((L["file"] >= lo) & (L["file"] < hi_)).sum())
                deltas.append(means["file"] - means["manual"]); weights.append(n)
            strat[lbl] = cells + [diff]
            print(f'  {lbl:>14s} | {cells[0]:>18s} {cells[1]:>18s} {diff:>12s}')
        if deltas:
            w = np.array(weights, float); dd = np.array(deltas)
            pooled = float((dd * w).sum() / w.sum())
            print(f'  length-matched difference (weighted by the smaller cell): {pooled:+.3f}'
                  f'   over {len(dd)} comparable bands')
            out[block] = dict(length_matched_delta=pooled, bands=strat,
                              marginal_manual=float(d["manual"].mean()) if "manual" in d else None,
                              marginal_file=float(d["file"].mean()) if "file" in d else None)
        print()
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
