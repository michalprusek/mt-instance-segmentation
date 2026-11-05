"""Grid the instancer knobs that a THIN mask changes, on cached predictions.

The shipped `params_a_model_v5H` were fitted to a mask with fg 1.54 % whose optimal threshold is
0.99. The thin-target model produces fg 0.82 % and peaks at 0.50 -- wholly different statistics, so
the inherited knobs disadvantage it and the measured +0.0095 is a lower bound.

Knobs chosen because a thin mask changes exactly what they control:
  merge_radius   how far junction pixels are contracted -- a thin skeleton has smaller junction
                 clusters, so the radius fitted to a fat one over-merges.
  link_max_gap   how far a free end may reach for a continuation -- a thin mask breaks more often
                 (fragmentation 1.211 -> 1.253), so the reach that suited a fat mask is too short.
  min_length     the length below which a chain is discarded.

Read TRUE POSITIVES as well as F1: deleting structure raises F1 mechanically.

NOTE ON PROVENANCE: this grids on REAL VAL, which section 17q showed rewards permissive settings
because human GT is incomplete. It is therefore a SCREEN for headroom, an upper bound, not a
shippable fit; anything promising must be refitted on synthetic ground truth before it is quoted.
"""
from __future__ import annotations

import argparse, glob, itertools, json, os, sys, time
import multiprocessing as mp
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from instance.instancer_a import instance_a                      # noqa: E402
from instance.metrics import aggregate_benchmark, centerline_f1  # noqa: E402
from instance.oracle import oracle_instance_masks                # noqa: E402
from mt_bench.cvat_import import read_frame_h5                   # noqa: E402

UP, KAPPA_MAX = 1.5, 0.25
_C: dict = {}


def _init(c):
    _C.update(c)


def one(job):
    path, params = job
    fr = read_frame_h5(path)
    a = fr["attrs"]
    shape = (int(a["height"]), int(a["width"]))
    hi = (int(round(shape[0] * UP)), int(round(shape[1] * UP)))
    gt = [np.asarray(q, float) * UP for q in fr["polylines"]]
    prob = np.load(os.path.join(_C["pred_dir"],
                                os.path.basename(path).replace(".h5", ".npz")))["prob"]
    prob = prob.astype(np.float32).max(axis=0)
    polys, _ = instance_a(prob > _C["thr"], KAPPA_MAX, params, channels=prob[None], prob=prob)
    masks = oracle_instance_masks(polys, hi, half_width=1.0, up=1.0)
    r = centerline_f1(masks, gt, tol=5.0, length_coverage=0.95, precision_coverage=0.95)
    r.pop("pairs", None)
    r["name"] = os.path.basename(path)
    r["task"] = str(a.get("source_task", "?"))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/mt34_eval")
    ap.add_argument("--split", default="val")
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--thr", type=float, required=True)
    ap.add_argument("--base-params", default="src/instance/params_a_model_v5H.json")
    ap.add_argument("--merge", default="2.0,3.5,5.0,7.0,9.0")
    ap.add_argument("--gap", default="20,36,55")
    ap.add_argument("--minlen", default="30,45")
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()
    if args.split == "test":
        raise SystemExit("refusing to touch TEST")

    base = json.load(open(args.base_params))
    base.pop("kappa_max", None); base.pop("prob_thr", None)
    paths = [p for p in sorted(glob.glob(os.path.join(args.data, "*.h5")))
             if str(read_frame_h5(p)["attrs"].get("split")) == args.split]

    grid = list(itertools.product([float(x) for x in args.merge.split(",")],
                                  [float(x) for x in args.gap.split(",")],
                                  [float(x) for x in args.minlen.split(",")]))
    print(f"{len(paths)} frames x {len(grid)} configs", flush=True)

    results = []
    t0 = time.time()
    with mp.Pool(args.workers, initializer=_init,
                 initargs=({"pred_dir": args.pred_dir, "thr": args.thr},)) as pool:
        for mr, gp, ml in grid:
            p = dict(base, merge_radius=mr, link_max_gap=gp, min_length=ml)
            rows = pool.map(one, [(x, p) for x in paths])
            agg = aggregate_benchmark(rows)
            results.append({"merge_radius": mr, "link_max_gap": gp, "min_length": ml,
                            "f1": agg["mean_f1"], "tp": agg["total_tp"],
                            "fp": agg["total_fp"], "fn": agg["total_fn"]})
            print(f"  merge={mr:<4} gap={gp:<4} minlen={ml:<4} "
                  f"F1={agg['mean_f1']:.4f} tp={agg['total_tp']} fp={agg['total_fp']}", flush=True)
    print(f"{time.time()-t0:.0f}s")
    results.sort(key=lambda r: -r["f1"])
    print("\nTOP 5 by F1:")
    for r in results[:5]:
        print(f"  merge={r['merge_radius']:<4} gap={r['link_max_gap']:<4} "
              f"minlen={r['min_length']:<4} F1={r['f1']:.4f} tp={r['tp']}")
    print("\nTOP 5 by TRUE POSITIVES (the honest read):")
    for r in sorted(results, key=lambda r: -r["tp"])[:5]:
        print(f"  merge={r['merge_radius']:<4} gap={r['link_max_gap']:<4} "
              f"minlen={r['min_length']:<4} F1={r['f1']:.4f} tp={r['tp']}")
    json.dump(results, open(args.out_json, "w"), indent=1, default=float)
    print("\nwrote", args.out_json)


if __name__ == "__main__":
    main()
