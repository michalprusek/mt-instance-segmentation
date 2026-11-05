#!/usr/bin/env python3
"""ARM 52 falsifier 4: with `paint_aa` ON vs OFF, the rendered set's per-frame contrast marginals must
agree within 2 % (same seeds, same backgrounds, same morphology), otherwise the renderer change is a
photometry change and the arm must be read as such. Runs on tulen (needs gen_train + backgrounds).

Per frame (paired by seed): field = img / background - 1; on the union of instance masks:
mean |field|, p90 |field|; over the frame: fraction of pixels with |field| > 0.05 (foreground
extent proxy). The label masks come from the morphology and are asserted identical.

    PYTHONPATH=.:src:scripts:synth python scripts/paint_aa_marginals.py --calib calib_V6_c070_narrow.json \\
        --bg-dir /home/prusek/mt_enc_exp/irm_backgrounds_v2 --n 60 --out marg.json
"""
import argparse
import dataclasses
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "synth"))
from gen_train import build_cfg  # noqa: E402
from mt_generator import generate_frame  # noqa: E402


def load_bg(path):
    import tifffile
    a = tifffile.imread(path).astype(np.float64)
    return a if a.ndim == 2 else a[0]


def stats(img, bg, instances):
    field = img / np.maximum(bg, 1e-6) - 1.0
    H, W = field.shape
    m = np.zeros((H, W), bool)
    for ins in instances:
        mk = ins.get("mask")
        if mk is not None and mk.shape == m.shape:
            m |= mk.astype(bool)
    a = np.abs(field[m]) if m.any() else np.zeros(1)
    return {"mean_abs_on_mask": float(a.mean()), "p90_abs_on_mask": float(np.percentile(a, 90)),
            "frac_gt_005": float((np.abs(field) > 0.05).mean()), "mask_px": int(m.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--bg-dir", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cal = json.load(open(args.calib))
    cfg0 = build_cfg(cal["best_params"], mask_hw=1.0)
    if "gen_overrides" in cal:
        cfg0 = dataclasses.replace(cfg0, **cal["gen_overrides"])
    cfg1 = dataclasses.replace(cfg0, paint_aa=True)
    bgs = sorted(glob.glob(os.path.join(args.bg_dir, "*.tif")))
    rows = []
    for i in range(args.n):
        bg = load_bg(bgs[i % len(bgs)])
        out = {}
        masks = {}
        for lab, cfg in (("off", cfg0), ("on", cfg1)):
            rng = np.random.default_rng(1000 + i)
            img, inst, meta = generate_frame(bg, rng, cfg, modality="irm")
            out[lab] = stats(img, bg, inst)
            masks[lab] = [ins["mask"] for ins in inst if ins.get("mask") is not None]
        same = len(masks["off"]) == len(masks["on"]) and all(np.array_equal(a, b) for a, b in zip(masks["off"], masks["on"]))
        rows.append({"seed": 1000 + i, "off": out["off"], "on": out["on"], "masks_identical": bool(same)})
    keys = ("mean_abs_on_mask", "p90_abs_on_mask", "frac_gt_005")
    summ = {}
    for k in keys:
        a = np.array([r["off"][k] for r in rows]); b = np.array([r["on"][k] for r in rows])
        summ[k] = {"off_median": float(np.median(a)), "on_median": float(np.median(b)), "ratio_of_medians": float(np.median(b) / np.median(a)),
                   "paired_ratio_median": float(np.median(b / np.maximum(a, 1e-9)))}
    summ["masks_identical_all"] = bool(all(r["masks_identical"] for r in rows))
    json.dump({"n": len(rows), "summary": summ, "rows": rows}, open(args.out, "w"), indent=1)
    for k in keys:
        s = summ[k]
        print(f"{k:<18} off {s['off_median']:.4f}  on {s['on_median']:.4f}  ratio {s['ratio_of_medians']:.4f}  paired-ratio median {s['paired_ratio_median']:.4f}")
    print("label masks identical for every frame:", summ["masks_identical_all"])


if __name__ == "__main__":
    main()
