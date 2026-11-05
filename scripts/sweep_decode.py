#!/usr/bin/env python3
"""D3: sweep the DECODING step (prob-map -> binary mask) over saved v4b probability maps.

No training, no GPU. Everything downstream of the mask is held fixed: instancer A with
``params_a_model_synthtuned.json``, strict centerline-F1 (tol=5, 0.95/0.95), the same 1.5x
eval frame and the same FOV masking ``run_oracle_eval.py`` uses. The ONLY thing a setting
changes is the boolean mask handed to ``instance_a``; ``channels`` and ``prob`` stay the raw
identity-transform maps so ``w_ori`` and ``bridge_thr`` see exactly what they saw at the
baseline operating point.

Every row also carries the over-firing gate (``mt_bench.fg_quality``) measured on the SAME
decoded mask, because a continuity/F1 number without it has been misread once already.

    ~/dinov3_env/bin/python scripts/sweep_decode.py \
        --data data/synth_eval --pred-dir /home/prusek/mt_enc_exp/synth_pred_v4b \
        --split val --out-json data/enc_sensitivity_testset/D3_sweep_synthval.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.ndimage import label as cc_label
from scipy.ndimage import zoom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "third_party"))

import run_oracle_eval as roe  # noqa: E402  (reuse score/parallel_pairs verbatim)
from instance.instancer_a import instance_a  # noqa: E402
from instance.metrics import aggregate_benchmark  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.fg_quality import (foreground_quality, mean_properties,  # noqa: E402
                                 passes_overfiring_gate, quality_score)
from mt_bench.fov import fov_mask  # noqa: E402
from mt_bench.gt_stats import count_crossings  # noqa: E402

UP = roe.UP
KAPPA_MAX = roe.KAPPA_MAX
BASE_THR = 0.4418491760570235       # the shipped operating point (params_a_model_synthtuned)


# ----------------------------------------------------------------------------------------
# decoding rules
# ----------------------------------------------------------------------------------------
def _hysteresis(low: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Components of ``low`` that contain at least one ``seed`` pixel (8-connectivity)."""
    lab, n = cc_label(low, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return np.zeros_like(low, dtype=bool)
    keep = np.zeros(n + 1, dtype=bool)
    keep[np.unique(lab[seed & low])] = True
    keep[0] = False
    return keep[lab]


def _ridge(prob: np.ndarray, which: str) -> np.ndarray:
    """Per-frame max-normalised vesselness of the probability map."""
    from skimage.filters import frangi, meijering, sato
    sig = (1.0, 2.0, 3.0)
    f = {"frangi": frangi, "sato": sato, "meijering": meijering}[which]
    r = f(prob.astype(np.float32), sigmas=sig, black_ridges=False)
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    m = float(r.max())
    return (r / m) if m > 0 else r


class Maps:
    """Lazily derived maps for one frame; every setting reuses them."""

    def __init__(self, chans: np.ndarray):
        self.chans = chans
        self.prob = chans.max(axis=0)
        self._cache: dict = {}

    def sum(self) -> np.ndarray:
        if "sum" not in self._cache:
            self._cache["sum"] = self.chans.sum(axis=0)
        return self._cache["sum"]

    def softor(self) -> np.ndarray:
        if "softor" not in self._cache:
            self._cache["softor"] = 1.0 - np.prod(1.0 - np.clip(self.chans, 0, 1), axis=0)
        return self._cache["softor"]

    def top2(self) -> np.ndarray:
        if "top2" not in self._cache:
            s = np.sort(self.chans, axis=0)
            self._cache["top2"] = s[-1] + s[-2]
        return self._cache["top2"]

    def ridge(self, which: str) -> np.ndarray:
        k = "ridge_" + which
        if k not in self._cache:
            self._cache[k] = _ridge(self.prob, which)
        return self._cache[k]


def decode(maps: Maps, spec: dict) -> np.ndarray:
    k = spec["kind"]
    if k == "flat":
        return maps.prob > spec["thr"]
    if k == "hyst":
        return _hysteresis(maps.prob > spec["lo"], maps.prob > spec["hi"])
    if k == "sum":
        return maps.sum() > spec["thr"]
    if k == "softor":
        return maps.softor() > spec["thr"]
    if k == "top2":
        return maps.top2() > spec["thr"]
    if k == "sum_hyst":
        return _hysteresis(maps.sum() > spec["lo"], maps.prob > spec["hi"])
    if k == "ridge":
        return maps.ridge(spec["filt"]) > spec["thr"]
    if k == "ridge_hyst":
        # seed on a CONFIDENT probability ridge, grow into the vesselness response
        return _hysteresis(maps.ridge(spec["filt"]) > spec["thr"], maps.prob > spec["hi"])
    if k == "blur":
        # Control for the multi-scale result: predicting at 1.25x and resampling INTO the
        # 1.5x frame low-pass filters the map. If a plain Gaussian blur of the identity map
        # reproduces the gain, the effect is smoothing, not scale.
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(maps.prob, spec["sigma"]) > spec["thr"]
    if k == "ridge_or":
        return (maps.prob > spec["hi"]) | (maps.ridge(spec["filt"]) > spec["thr"])
    raise SystemExit(f"unknown decode kind {k}")


def build_grid() -> list[dict]:
    g: list[dict] = []
    for t in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, BASE_THR, 0.45, 0.50, 0.55, 0.60]:
        g.append({"name": f"flat/{t:.4f}", "kind": "flat", "thr": t})
    for hi in [0.35, BASE_THR, 0.50]:
        for lo in [0.10, 0.15, 0.20, 0.25, 0.30]:
            g.append({"name": f"hyst/{hi:.3f}->{lo:.2f}", "kind": "hyst", "hi": hi, "lo": lo})
    for t in [0.35, BASE_THR, 0.50, 0.60, 0.70, 0.80, 1.00]:
        g.append({"name": f"sum/{t:.3f}", "kind": "sum", "thr": t})
    for t in [BASE_THR, 0.50, 0.60, 0.70, 0.80]:
        g.append({"name": f"softor/{t:.3f}", "kind": "softor", "thr": t})
    for t in [BASE_THR, 0.50, 0.60, 0.70, 0.80]:
        g.append({"name": f"top2/{t:.3f}", "kind": "top2", "thr": t})
    for lo in [0.50, 0.60, 0.70]:
        g.append({"name": f"sum_hyst/{BASE_THR:.3f}->{lo:.2f}", "kind": "sum_hyst",
                  "hi": BASE_THR, "lo": lo})
    for sg in [0.5, 0.75, 1.0, 1.5, 2.0]:
        for t in [0.35, BASE_THR]:
            g.append({"name": f"blur/{sg:.2f}/{t:.4f}", "kind": "blur",
                      "sigma": sg, "thr": t})
    for f in ["frangi", "sato", "meijering"]:
        for t in [0.02, 0.05, 0.10, 0.20, 0.40]:
            g.append({"name": f"ridge/{f}/{t:.2f}", "kind": "ridge", "filt": f, "thr": t})
        for t in [0.02, 0.05, 0.10]:
            g.append({"name": f"ridgehyst/{f}/{t:.2f}", "kind": "ridge_hyst", "filt": f,
                      "thr": t, "hi": BASE_THR})
            g.append({"name": f"ridgeor/{f}/{t:.2f}", "kind": "ridge_or", "filt": f,
                      "thr": t, "hi": BASE_THR})
    return g


# ----------------------------------------------------------------------------------------
# per-frame work
# ----------------------------------------------------------------------------------------
_CFG: dict = {}


def _init(cfg):
    global _CFG
    _CFG = cfg


def run_frame(path: str) -> dict:
    cfg = _CFG
    fr = read_frame_h5(path)
    attrs = fr["attrs"]
    name = os.path.basename(path)
    shape = (int(attrs["height"]), int(attrs["width"]))
    hi_shape = (int(round(shape[0] * UP)), int(round(shape[1] * UP)))
    gt = [np.asarray(p, float) * UP for p in fr["polylines"]]
    crossings = count_crossings(gt, tol=2.0 * UP, endpoint_margin=8.0 * UP)
    par = roe.parallel_pairs(gt)
    task = str(attrs.get("source_task", "?"))

    npz = np.load(os.path.join(cfg["pred_dir"], name.replace(".h5", ".npz")))
    chans = npz["prob"].astype(np.float32)
    if chans.shape[1:] != hi_shape:
        raise SystemExit(f"{name}: prediction {chans.shape[1:]} != eval frame {hi_shape}")
    if not cfg["no_fov"]:
        fv = zoom(fov_mask(fr["image"]).astype(np.uint8), UP, order=0).astype(bool)
        fv = np.pad(fv, ((0, max(0, hi_shape[0] - fv.shape[0])),
                         (0, max(0, hi_shape[1] - fv.shape[1]))),
                    constant_values=True)[:hi_shape[0], :hi_shape[1]]
        chans = chans * fv[None, :, :]

    maps = Maps(chans)
    prob = maps.prob

    # ---- item 4's independent diagnostic: does a crossing pixel split its probability mass?
    split = None
    if crossings:
        pts = np.asarray([[c["y"], c["x"]] if isinstance(c, dict) else [c[1], c[0]]
                          for c in crossings], dtype=float)
        rr = np.clip(np.rint(pts[:, 0]).astype(int), 0, hi_shape[0] - 1)
        cc = np.clip(np.rint(pts[:, 1]).astype(int), 0, hi_shape[1] - 1)
        m = np.zeros(hi_shape, dtype=bool)
        m[rr, cc] = True
        from scipy.ndimage import binary_dilation
        m = binary_dilation(m, np.ones((7, 7), dtype=bool))          # ~3 px radius
        pv, sv = prob[m], maps.sum()[m]
        thr = cfg["base_thr"]
        split = {"n_px": int(m.sum()),
                 "frac_max_below": float((pv < thr).mean()),
                 "frac_rescued_by_sum": float(((pv < thr) & (sv >= thr)).mean()),
                 "mean_max": float(pv.mean()), "mean_sum": float(sv.mean())}

    out = {"name": name, "task": task, "n_gt": len(gt), "split_diag": split, "rows": {}}
    for spec in cfg["grid"]:
        mask = decode(maps, spec)
        pls, _ = instance_a(mask, KAPPA_MAX, cfg["params_a"], channels=chans, prob=prob)
        row = roe.score(pls, gt, hi_shape, crossings, par)
        props = foreground_quality(mask, fr["polylines"], up=UP, tol=2.0)
        out["rows"][spec["name"]] = {"name": name, "task": task, "n_gt": len(gt),
                                     **row, "props": props}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/mt34_eval")
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--params-a", default="src/instance/params_a_model_synthtuned.json")
    ap.add_argument("--no-fov", action="store_true")
    ap.add_argument("--jobs", type=int, default=17)
    ap.add_argument("--only", default=None, help="comma-separated setting-name prefixes")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    pa = json.load(open(args.params_a))
    pa.pop("kappa_max", None)
    pa.pop("prob_thr", None)      # inert here: the mask is built by the decode rule

    grid = build_grid()
    if args.only:
        pref = tuple(s.strip() for s in args.only.split(","))
        grid = [g for g in grid if g["name"].startswith(pref)]

    paths = []
    for p in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
        import h5py
        with h5py.File(p) as f:
            if args.split == "all" or str(f.attrs.get("split")) == args.split:
                paths.append(p)
    print(f"{len(paths)} frames x {len(grid)} settings", flush=True)

    cfg = {"pred_dir": args.pred_dir, "no_fov": args.no_fov, "params_a": pa,
           "grid": grid, "base_thr": BASE_THR}
    t0 = time.time()
    with Pool(min(args.jobs, len(paths)), initializer=_init, initargs=(cfg,)) as pool:
        frames = pool.map(run_frame, paths)
    print(f"done in {time.time() - t0:.0f}s", flush=True)

    report = {"settings": {}, "split_diag": {}}
    for fr in frames:
        if fr["split_diag"]:
            report["split_diag"][fr["name"]] = fr["split_diag"]
    for spec in grid:
        rows = [fr["rows"][spec["name"]] for fr in frames]
        agg = aggregate_benchmark(rows)
        props = mean_properties([r["props"] for r in rows])
        entry = {"spec": spec, "pooled": agg, "props": props,
                 "quality": quality_score(props),
                 "gate": bool(passes_overfiring_gate(props)),
                 "frag": float(np.nanmean([r["fragmentation"] for r in rows])),
                 "n_pred": float(np.mean([r["n_pred"] for r in rows])),
                 "by_task": {}}
        for task in sorted({r["task"] for r in rows}):
            entry["by_task"][task] = aggregate_benchmark(
                [r for r in rows if r["task"] == task])
        entry["frames"] = [{k: v for k, v in r.items() if k != "props"} for r in rows]
        report["settings"][spec["name"]] = entry
        g = "PASS" if entry["gate"] else "FAIL"
        print(f"{spec['name']:28s} F1={agg['mean_f1']:.3f} micro={agg['micro_f1']:.3f} "
              f"TP={agg['total_tp']} FP={agg['total_fp']} FN={agg['total_fn']} "
              f"fg%={100 * props.get('fg', float('nan')):.2f} "
              f"rec2={props.get('rec2', float('nan')):.3f} "
              f"cc/gt={props.get('cc_per_gt', float('nan')):.2f} gate={g}", flush=True)

    with open(args.out_json, "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(f"wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
