#!/usr/bin/env python3
"""External baselines on MT-34, scored through the project's own strict metric. RUNS ON TULEN.

The reference numbers this project quotes for TARDIS (0.326), ORION (0.519), v19 (0.696) and
SAM 3 (0.712) are all on the OLD 12-frame Alice set. MT-34 has 32.1 crossings per frame against
Alice's 2.2, so those numbers say very little about the benchmark actually in use — and none of
them were measured with each method's own operating point swept, which is the confound that
turned a 0.016 difference into 0.47 elsewhere in this project.

Both baselines here are END-TO-END: image in, instances out, with no step of ours in between.
That is the honest comparison for a whole system. They are given the same 1.5x input, the same
rasterise-and-skeletonise conversion, the same GT and the same tol=5 / 0.95 / 0.95 metric.

    PYTHONPATH=src:third_party ~/sam3_env/bin/python scripts/baseline_mt34.py --method sam3
    PYTHONPATH=src:third_party ~/tardis_env/bin/python scripts/baseline_mt34.py --method tardis
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import zoom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "third_party"))

from instance.metrics import aggregate_benchmark, centerline_f1  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402

UP = 1.5
TOL, LEN_COV, PREC_COV = 5.0, 0.95, 0.95
DUMP_DIR = None   # set from --dump-inst; every method writes its instances the same way


def norm01(a, p=(1, 99)):
    lo, hi = np.percentile(a, p)
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)


def load_frames(data: str, split: str, tif_dir: str):
    out = []
    for p in sorted(glob.glob(os.path.join(data, "*.h5"))):
        fr = read_frame_h5(p)
        if str(fr["attrs"].get("split")) != split or not fr["polylines"]:
            continue
        name = os.path.basename(p).replace(".h5", "")
        tif = os.path.join(tif_dir, name + ".tif")
        out.append({"name": name, "tif": tif, "image": fr["image"],
                    "polylines": fr["polylines"]})
    return out



def dump_instances(out_dir, name, masks):
    """Save a method's predicted instances as sparse pixel indices.

    Every baseline is scored through the same code as our own method, which is the only way the
    provenance-split comparison means anything -- a per-method scoring path would let the
    comparison differ for reasons other than the methods. Masks round-trip exactly: a boolean
    mask rebuilt from these indices yields the same points the metric would have extracted.
    """
    import numpy as _np
    os.makedirs(out_dir, exist_ok=True)
    d = {"shape": _np.asarray(masks[0].shape if masks else (0, 0), _np.int32),
         "n": _np.asarray(len(masks), _np.int32)}
    for k, m in enumerate(masks):
        r, c = _np.where(m)
        d[f"r{k}"] = r.astype(_np.int32)
        d[f"c{k}"] = c.astype(_np.int32)
    _np.savez_compressed(os.path.join(out_dir, name + ".npz"), **d)


def score(masks, polylines):
    gt = [np.asarray(q, float) * UP for q in polylines]
    r = centerline_f1(masks, gt, tol=TOL, length_coverage=LEN_COV,
                      precision_coverage=PREC_COV)
    r["n_pred"] = len(masks)
    return r


# --------------------------------------------------------------------------- SAM 3
def run_sam3(frames, confs, prompt):
    """SAM 3 with a text prompt -- the engine SynthMT reports as its winner."""
    from scipy.ndimage import binary_dilation
    from skimage.morphology import skeletonize
    from skimage.transform import resize as skresize
    from ultralytics.models.sam import SAM3SemanticPredictor

    results = {}
    for conf in confs:
        pred = SAM3SemanticPredictor(overrides=dict(
            conf=conf, task="segment", mode="predict",
            model="/disk1/prusek/sam3/sam3.pt", save=False, verbose=False))
        rows = []
        for fr in frames:
            img = zoom(norm01(np.asarray(fr["image"], float)), UP, order=1)
            rgb = (np.stack([img] * 3, -1) * 255).astype(np.uint8)
            # set_image + text=[...] is the calling convention scripts/sam3_alice.py uses;
            # passing the prompt through __call__ hits a code path that never populates
            # 'language_features' and raises.
            pred.set_image(rgb)
            res = pred(text=[prompt])[0]
            m = getattr(res, "masks", None)
            masks = []
            if m is not None:
                arr = m.data.cpu().numpy() if hasattr(m, "data") else np.asarray(m)
                for k in range(arr.shape[0]):
                    mm = arr[k] > 0.5
                    if mm.shape != img.shape:
                        mm = skresize(mm.astype(float), img.shape, order=0) > 0.5
                    # Skeletonise-then-dilate, exactly as the Alice baseline did: SAM 3 returns
                    # blobs, and the metric compares centerlines. Skipping this would score the
                    # baseline on a different representation from everything else here.
                    sk = skeletonize(mm)
                    if sk.sum() >= 5:
                        masks.append(binary_dilation(sk, iterations=1))
            if DUMP_DIR:
                dump_instances(os.path.join(DUMP_DIR, f"conf{conf}"), fr["name"], masks)
            r = score(masks, fr["polylines"])
            r["name"] = fr["name"]
            rows.append(r)
            print(f"    {fr['name'][:38]:38s} pred={r['n_pred']:3d} tp={r['tp']:3d} "
                  f"fp={r['fp']:3d} f1={r['f1']:.3f}", flush=True)
        results[conf] = rows
        a = aggregate_benchmark(rows)
        print(f"  conf={conf:.2f}  F1={a['mean_f1']:.4f} tp={a['total_tp']} "
              f"fp={a['total_fp']} fn={a['total_fn']}", flush=True)
    return results


# --------------------------------------------------------------------------- TARDIS
def run_tardis(frames, csv_dir, dilate: int):
    """Score TARDIS instance centerline CSVs that its CLI has already written.

    The CLI is run separately (it wants a directory of tifs); this only converts its output.
    Columns are ``[ID, X, Y, Z]`` at NATIVE scale, so they are multiplied into the 1.5x eval
    frame here rather than anywhere else.
    """
    from instance.oracle import oracle_instance_masks

    rows = []
    for fr in frames:
        cands = [os.path.join(csv_dir, fr["name"] + s) for s in
                 ("_instances_filter.csv", "_instances.csv")]
        csv = next((c for c in cands if os.path.exists(c)), None)
        if csv is None:
            print(f"    {fr['name'][:38]:38s} NO CSV -- skipped", flush=True)
            continue
        d = np.loadtxt(csv, delimiter=",", skiprows=1, ndmin=2)
        polys = []
        if len(d):
            for i in np.unique(d[:, 0]):
                pts = d[d[:, 0] == i][:, 1:3]        # (X, Y) = (col, row)
                if len(pts) >= 2:
                    polys.append(pts.astype(float))
        masks = oracle_instance_masks(polys, fr["image"].shape,
                                      half_width=float(dilate), up=UP)
        if DUMP_DIR:
            dump_instances(os.path.join(DUMP_DIR, f"dil{dilate}"), fr["name"], masks)
        r = score(masks, fr["polylines"])
        r["name"] = fr["name"]
        rows.append(r)
        print(f"    {fr['name'][:38]:38s} pred={r['n_pred']:3d} tp={r['tp']:3d} "
              f"fp={r['fp']:3d} f1={r['f1']:.3f}", flush=True)
    return {dilate: rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-inst", default=None,
                    help="write each method's predicted instances here, so all of them "
                         "can be re-scored through one identical path")
    ap.add_argument("--method", required=True, choices=["sam3", "tardis"])
    ap.add_argument("--data", default="data/real/mt34_eval")
    ap.add_argument("--split", default="val")
    ap.add_argument("--tif-dir", default="/home/prusek/mt_enc_exp/mt34/tif")
    ap.add_argument("--confs", default="0.10,0.25,0.40,0.60",
                    help="sam3: confidence sweep -- its own operating point, on VAL only")
    ap.add_argument("--prompt", default="thin line")
    ap.add_argument("--csv-dir", default=None, help="tardis: directory of instance CSVs")
    ap.add_argument("--dilate", type=float, default=1.0,
                    help="tardis: rasterised half-width. NOT a lever -- the metric skeletonises "
                         "the masks, so 1.0 and 2.0 score identically; do not sweep it")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()
    global DUMP_DIR
    DUMP_DIR = args.dump_inst

    frames = load_frames(args.data, args.split, args.tif_dir)
    print(f"{len(frames)} {args.split} frames, method={args.method}", flush=True)

    if args.method == "sam3":
        res = run_sam3(frames, [float(c) for c in args.confs.split(",")], args.prompt)
    else:
        res = run_tardis(frames, args.csv_dir, args.dilate)

    print("\n=== summary ===")
    best = None
    for k, rows in res.items():
        a = aggregate_benchmark(rows)
        print("  setting=%-6s F1=%.4f  tp=%3d fp=%3d fn=%3d  n=%d"
              % (str(k), a["mean_f1"], a["total_tp"], a["total_fp"], a["total_fn"], len(rows)))
        if best is None or a["mean_f1"] > best[1]["mean_f1"]:
            best = (k, a)
    if best:
        print(f"\nBEST {args.method}: {best[1]['mean_f1']:.4f} at setting {best[0]}")
    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump({"method": args.method, "split": args.split,
                       "results": {str(k): {"agg": aggregate_benchmark(v), "_frames": v}
                                   for k, v in res.items()}}, fh, indent=1)


if __name__ == "__main__":
    main()
