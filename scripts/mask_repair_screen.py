"""Which ORACLE-FREE mask repairs recover the +93 TP the factorial says are on the table?

The factorial's C/E/CE variants define what to fill and what to delete BY REFERENCE TO THE ORACLE
MASK, so they are a budget, not a method: "far from any ground-truth microtubule" is not computable
without ground truth. This screen tests operations that use only the image and the predicted
probability map, and scores them against the same 17 real VAL frames and the same instancer, so the
numbers sit directly beside the factorial's.

Read TRUE POSITIVES, not F1. Deleting foreground removes false positives mechanically, so F1 rises
even when assembly got no better; only a rise in TP means more microtubules were recovered.

Two hints from the factorial shape the candidates:
  * F20/F50 (drop small components) bought exactly ZERO extra TP, so the excess is not islands.
  * removing excess takes cc/MT from 2.10 to 1.20, so the excess is BRIDGING -- spurious structure
    that fuses separate microtubules into one component and shatters the arc graph.
So the candidates below attack weak bridges and weak branches, not small blobs.

    PYTHONPATH=src:scripts python scripts/mask_repair_screen.py --pred-dir <npz> --out-json <f>
"""
from __future__ import annotations

import argparse, glob, json, os, sys, time
import multiprocessing as mp

import numpy as np
from scipy.ndimage import (binary_closing, binary_dilation, binary_opening,
                           generate_binary_structure, label)
from skimage.morphology import skeletonize

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from instance.instancer_a import instance_a                      # noqa: E402
from instance.metrics import aggregate_benchmark, centerline_f1  # noqa: E402
from instance.oracle import oracle_instance_masks, oracle_mask   # noqa: E402
from mt_bench.cvat_import import read_frame_h5                   # noqa: E402

UP, KAPPA_MAX, TOL, LCOV, PCOV = 1.5, 0.25, 5.0, 0.95, 0.95
_CFG: dict = {}


def _disk(r: float) -> np.ndarray:
    r = max(1, int(round(r)))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def hysteresis(prob: np.ndarray, hi: float, lo: float) -> np.ndarray:
    """Seed at `hi`, grow into connected `lo` territory.

    The pipeline currently uses ONE flat threshold, which forces a single compromise: high enough
    to suppress texture and therefore high enough to punch holes through faint stretches of real
    filaments. Hysteresis separates the two decisions -- a component is kept only if it contains
    strong evidence somewhere, but once kept it may extend through weak evidence.
    """
    seed, grow = prob > hi, prob > lo
    lab, n = label(grow, structure=np.ones((3, 3), int))
    if n == 0:
        return seed
    keep = np.zeros(n + 1, bool)
    keep[np.unique(lab[seed])] = True
    keep[0] = False
    return keep[lab]


def prune_weak_branches(mask: np.ndarray, prob: np.ndarray, thr: float,
                        max_len: int = 40) -> np.ndarray:
    """Delete short skeleton branches whose evidence is weak.

    Targets the bridging that the factorial fingerprinted: a spur or a faint link between two real
    filaments creates a junction the arc graph then splits on. Length-capped so a genuine faint
    microtubule is never removed wholesale.
    """
    skel = skeletonize(mask)
    if not skel.any():
        return mask
    nb = np.zeros_like(skel, dtype=np.uint8)
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
    from scipy.ndimage import convolve
    nb = convolve(skel.astype(np.uint8), k, mode="constant") * skel
    junc = skel & (nb >= 3)
    arcs = skel & ~binary_dilation(junc, structure=np.ones((3, 3), bool))
    lab, n = label(arcs, structure=np.ones((3, 3), int))
    if n == 0:
        return mask
    drop = np.zeros(n + 1, bool)
    for i in range(1, n + 1):
        sel = lab == i
        size = int(sel.sum())
        if size == 0 or size > max_len:
            continue
        ends = int((nb[sel] <= 1).sum())
        if ends == 0:                      # interior link, not a spur -> only cut if very weak
            if float(prob[sel].mean()) < thr * 0.75:
                drop[i] = True
        elif float(prob[sel].mean()) < thr:
            drop[i] = True
    if not drop.any():
        return mask
    kill = binary_dilation(drop[lab], structure=_disk(2))
    return mask & ~kill


def variants(prob: np.ndarray, base_thr: float, mask_o: np.ndarray) -> dict:
    b = prob > base_thr
    v = {"B_model": b}
    for lo in (0.5, 0.7, 0.9):
        v[f"H_hyst_{lo}"] = hysteresis(prob, base_thr, lo)
    for r in (2, 3):
        v[f"CL_close_r{r}"] = binary_closing(b, structure=_disk(r))
    for t in (0.3, 0.5):
        v[f"PR_prune_{t}"] = prune_weak_branches(b, prob, t)
    v["HP_hyst0.7_prune0.5"] = prune_weak_branches(hysteresis(prob, base_thr, 0.7), prob, 0.5)
    v["HPC_hyst0.7_prune0.5_close2"] = binary_closing(v["HP_hyst0.7_prune0.5"], structure=_disk(2))
    # reference ceilings from the factorial -- ORACLE-BASED, not achievable, for scale only
    v["ref_CE4_oracle"] = (b | mask_o) & binary_dilation(mask_o, structure=_disk(4))
    v["ref_A_oracle"] = mask_o
    return v


def _init(cfg):
    _CFG.update(cfg)


def run_frame(path):
    fr = read_frame_h5(path)
    a = fr["attrs"]
    name = os.path.basename(path)
    shape = (int(a["height"]), int(a["width"]))
    hi = (int(round(shape[0] * UP)), int(round(shape[1] * UP)))
    gt = [np.asarray(p, float) * UP for p in fr["polylines"]]
    prob = np.load(os.path.join(_CFG["pred_dir"], name.replace(".h5", ".npz")))["prob"]
    prob = prob.astype(np.float32).max(axis=0)
    mask_o = oracle_mask(fr["polylines"], shape, half_width=1.0, up=UP)
    out = {}
    for vn, m in variants(prob, _CFG["thr"], mask_o).items():
        polys, _ = instance_a(m, KAPPA_MAX, _CFG["params"], channels=prob[None], prob=prob)
        masks = oracle_instance_masks(polys, hi, half_width=1.0, up=1.0)
        r = centerline_f1(masks, gt, tol=TOL, length_coverage=LCOV, precision_coverage=PCOV)
        r.pop("pairs", None)
        r.update({"name": name, "task": str(a.get("source_task", "?")),
                  "fg": float(m.mean()), "n_pred": len(polys)})
        out[vn] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/mt34_eval")
    ap.add_argument("--split", default="val")
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--thr", type=float, default=0.97)
    ap.add_argument("--params", default="src/instance/params_a_model_v5H.json")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()
    if args.split == "test":
        raise SystemExit("refusing to touch TEST")

    params = json.load(open(args.params))
    params.pop("kappa_max", None); params.pop("prob_thr", None)
    paths = [p for p in sorted(glob.glob(os.path.join(args.data, "*.h5")))
             if str(read_frame_h5(p)["attrs"].get("split")) == args.split]
    print(f"{len(paths)} {args.split} frames", flush=True)

    t0 = time.time()
    with mp.Pool(args.workers, initializer=_init,
                 initargs=({"pred_dir": args.pred_dir, "thr": args.thr, "params": params},)) as pool:
        res = pool.map(run_frame, paths)
    print(f"{time.time() - t0:.0f}s", flush=True)

    keys = sorted({k for r in res for k in r})
    agg = {}
    print(f"\n{'variant':32s} {'F1':>7} {'TP':>5} {'FP':>5} {'FN':>5} {'dTP':>5}  fg%")
    base_tp = None
    rows_out = {}
    for k in ["B_model"] + [x for x in keys if x != "B_model"]:
        rows = [r[k] for r in res]
        a = aggregate_benchmark(rows)
        a["fg"] = float(np.mean([r["fg"] for r in rows]))
        agg[k] = a
        rows_out[k] = rows
        if base_tp is None:
            base_tp = a["total_tp"]
        d = a["total_tp"] - base_tp
        print(f"{k:32s} {a['mean_f1']:7.4f} {a['total_tp']:5d} {a['total_fp']:5d} "
              f"{a['total_fn']:5d} {d:+5d}  {100*a['fg']:.2f}")
    json.dump({"agg": agg, "frames": rows_out}, open(args.out_json, "w"), indent=1, default=float)
    print("\nwrote", args.out_json)


if __name__ == "__main__":
    main()
