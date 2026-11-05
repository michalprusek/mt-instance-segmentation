"""How well does the CURRENT system keep two microtubules apart where they cross?

Answers the overlap question with a number instead of an assertion, on whatever benchmark is
pointed at. Three measurements, all already implemented in instance.metrics:

  junction_identity -- for each GT crossing, does ONE predicted instance still cover the same
                       filament on both sides of it? A tracer that breaks there scores 0 even
                       when its centerline-F1 looks fine.
  bundle_recovery   -- do N close parallels come back as N instances rather than one?
  fragmentation     -- predicted instances per GT microtubule.

Reported split by crossing ANGLE, because shallow crossings are the hard case and pooling hides
them, and split by source task, because the combined benchmark holds two different labs.
"""
from __future__ import annotations

import argparse, glob, json, os, sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from instance.instancer_a import instance_a                          # noqa: E402
from instance.metrics import (bundle_recovery, fragmentation,        # noqa: E402
                              junction_identity)
from instance.oracle import oracle_instance_masks, oracle_mask       # noqa: E402
from mt_bench.cvat_import import read_frame_h5                       # noqa: E402
from mt_bench.gt_stats import count_crossings                        # noqa: E402
from run_oracle_eval import parallel_pairs                           # noqa: E402

UP, KAPPA_MAX = 1.5, 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--thr", type=float, default=0.97)
    ap.add_argument("--params", default="src/instance/params_a_model_v5H.json")
    ap.add_argument("--oracle", action="store_true",
                    help="run on the ORACLE mask too, to separate instancer from foreground")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--allow-test", action="store_true",
                    help="permit --split test. The guard below exists so TEST is never touched "
                         "by accident; pass this ONLY as part of a read that is pre-declared in "
                         "docs/TODO.md 0.0-CAMPAIGN (as final read 34 was, 2026-09-08), and "
                         "record the reason there.")
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("refusing to touch TEST (pass --allow-test for a pre-declared read)")

    params = json.load(open(args.params))
    params.pop("kappa_max", None); params.pop("prob_thr", None)

    acc = {}
    for p in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        fr = read_frame_h5(p)
        a = fr["attrs"]
        if str(a.get("split")) != args.split:
            continue
        name = os.path.basename(p)
        task = "htw" if name.startswith("htw_") else str(a.get("source_task", "?"))
        shape = (int(a["height"]), int(a["width"]))
        hi = (int(round(shape[0] * UP)), int(round(shape[1] * UP)))
        gt = [np.asarray(q, float) * UP for q in fr["polylines"]]
        if len(gt) < 2:
            continue
        # Every threshold here carries the x1.5 eval upscale, exactly as run_oracle_eval.py does.
        # With the native-scale defaults this found ZERO crossings on 1.5x GT and the identity
        # metric silently reported nan -- a scale mismatch that looks like "no crossings exist".
        cr = count_crossings(gt, tol=2.0 * UP, endpoint_margin=8.0 * UP)
        par = parallel_pairs(gt)
        npz = os.path.join(args.pred_dir, name.replace(".h5", ".npz"))
        if not os.path.exists(npz):
            continue
        prob = np.load(npz)["prob"].astype(np.float32).max(axis=0)

        sources = [("model", prob > args.thr)]
        if args.oracle:
            sources.append(("oracle", oracle_mask(fr["polylines"], shape, half_width=1.0, up=UP)))
        for src, mask in sources:
            polys, _ = instance_a(mask, KAPPA_MAX, params, channels=prob[None], prob=prob)
            masks = oracle_instance_masks(polys, hi, half_width=1.0, up=1.0)
            # tol / gap thresholds carry the x1.5 eval upscale exactly as run_oracle_eval does,
            # or these numbers cannot be compared with any historical one.
            ji = junction_identity(masks, gt, cr, tol=5.0, arm_offset=15.0 * UP)
            br = bundle_recovery(masks, gt, par, tol=1.5 * UP)
            fg = fragmentation(masks, gt)
            d = acc.setdefault((src, task), {"n_frames": 0, "ji_used": 0, "ji_pres": 0,
                                             "by_angle": {}, "br": [], "frag": []})
            d["n_frames"] += 1
            # Index the contract, never .get() it with a default: junction_identity returns
            # n_crossings / n_preserved / rate / by_angle / by_angle_n, and reading a
            # mis-remembered key through .get(k, 0) turns the mistake into a plausible ZERO
            # instead of a crash. That is exactly how this report first claimed there were no
            # crossings in a benchmark built to be crossing-dense.
            d["ji_used"] += ji["n_crossings"]
            d["ji_pres"] += ji["n_preserved"]
            for k, n_k in ji["by_angle_n"].items():
                if n_k:
                    b = d["by_angle"].setdefault(k, [0, 0])
                    b[0] += n_k
                    b[1] += int(round(ji["by_angle"][k] * n_k))
            if br == br:            # bundle_recovery returns a float, nan when no pairs
                d["br"].append(float(br))
            if fg == fg:            # fragmentation likewise
                d["frag"].append(float(fg))

    print(f"\n{'source':8s} {'task':6s} {'n':>3} {'crossings':>9} {'identity':>9} "
          f"{'bundle':>7} {'frag':>6}")
    out = {}
    for (src, task), d in sorted(acc.items()):
        ji = d["ji_pres"] / d["ji_used"] if d["ji_used"] else float("nan")
        br = float(np.mean(d["br"])) if d["br"] else float("nan")
        fr_ = float(np.mean(d["frag"])) if d["frag"] else float("nan")
        print(f"{src:8s} {task:6s} {d['n_frames']:3d} {d['ji_used']:9d} {ji:9.3f} "
              f"{br:7.3f} {fr_:6.3f}")
        angles = {k: (v[1] / v[0] if v[0] else None, v[0]) for k, v in d["by_angle"].items()}
        for k in ("shallow", "oblique", "steep"):
            if k in angles and angles[k][1]:
                print(f"           {k:9s} n={angles[k][1]:4d}  identity={angles[k][0]:.3f}")
        out[f"{src}|{task}"] = {
            "n_frames": d["n_frames"], "crossings": d["ji_used"],
            "junction_identity": ji, "bundle_recovery": br, "fragmentation": fr_,
            "by_angle": {k: {"n": n_k, "identity": id_k} for k, (id_k, n_k) in angles.items()},
        }
    if args.out_json:
        json.dump(out, open(args.out_json, "w"), indent=1, default=float)
        print("\nwrote", args.out_json)


if __name__ == "__main__":
    main()
