#!/usr/bin/env python3
"""ADDRESSABILITY of a two-sheet (overlap-permitting) emission, from GT ALONE. No model, no GPU.

For every GT polyline, densified to 1 eval px, the distance to the nearest point of any OTHER GT
polyline in the same frame. Reports, per provenance block:
  * share of total GT arc length whose nearest-other-curve distance is <= R  (the support on which
    a SECOND sheet has any target at all, for a target band of radius R);
  * number / share of GT curves carrying >= 20 eval px in the close-parallel band [2, 6] eval px
    (the pool a parallel-separating mechanism could convert);
  * the same for the wider band [2, 11] eval px.
All lengths in EVAL pixels (native x 1.5), the same convention as scripts/diag_model_only.py.
"""
import glob, os, sys, json
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
sys.path.insert(0, "src")
from mt_bench.cvat_import import read_frame_h5

UP = 1.5
RS = (2.0, 4.0, 6.0, 8.0, 11.0)


def source_of(name):
    if name.startswith("htw_"): return "htw"
    if name.startswith("alice_"): return "alice"
    if name.startswith("training_img_"): return "roi303-derived"
    if name.startswith("roi_img_"): return "roi303"
    if name.startswith("synth_"): return "synth"
    return "other"


def densify(poly, step=1.0):
    d = np.hypot(*np.diff(poly, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 0: return poly[:1]
    t = np.arange(0.0, s[-1] + 1e-9, step)
    return np.stack([np.interp(t, s, poly[:, 0]), np.interp(t, s, poly[:, 1])], axis=1)


def main(dirs, split):
    acc = {}
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "*.h5"))):
            fr = read_frame_h5(p)
            if split != "all" and str(fr["attrs"].get("split", "")) != split:
                continue
            src = source_of(os.path.basename(p))
            a = acc.setdefault(src, {"frames": 0, "curves": 0, "L": 0.0,
                                     "Lr": {r: 0.0 for r in RS},
                                     "n26": 0, "n211": 0, "L26": 0.0, "L211": 0.0,
                                     "curve_len": []})
            a["frames"] += 1
            polys = [np.asarray(q, float) * UP for q in fr["polylines"] if len(q) >= 2]
            dense = [densify(q) for q in polys]
            trees = [cKDTree(q) if len(q) > 1 else None for q in dense]
            for i, q in enumerate(dense):
                if len(q) < 2: continue
                dmin = np.full(len(q), np.inf)
                for j, t in enumerate(trees):
                    if j == i or t is None: continue
                    dd, _ = t.query(q, k=1)
                    dmin = np.minimum(dmin, dd)
                L = float(len(q))          # 1 px spacing -> length in eval px
                a["curves"] += 1; a["L"] += L
                a["curve_len"].append(L)
                for r in RS:
                    a["Lr"][r] += float((dmin <= r).sum())
                l26 = float(((dmin >= 2.0) & (dmin <= 6.0)).sum())
                l211 = float(((dmin >= 2.0) & (dmin <= 11.0)).sum())
                a["L26"] += l26; a["L211"] += l211
                a["n26"] += int(l26 >= 20.0); a["n211"] += int(l211 >= 20.0)
                a.setdefault("rel6", []).append(float((dmin <= 6.0).sum()) / L)
                a.setdefault("rel11", []).append(float((dmin <= 11.0).sum()) / L)
    out = {}
    for src, a in sorted(acc.items()):
        cl = np.array(a["curve_len"])
        out[src] = {
            "frames": a["frames"], "curves": a["curves"],
            "median_curve_len_evalpx": float(np.median(cl)) if len(cl) else 0.0,
            "total_len_evalpx": a["L"],
            "frac_len_within_R": {str(r): a["Lr"][r] / max(a["L"], 1) for r in RS},
            "frac_len_band_2_6": a["L26"] / max(a["L"], 1),
            "frac_len_band_2_11": a["L211"] / max(a["L"], 1),
            "curves_with_20px_in_2_6": a["n26"],
            "curves_with_20px_in_2_11": a["n211"],
            "share_curves_2_6": a["n26"] / max(a["curves"], 1),
            "share_curves_2_11": a["n211"] / max(a["curves"], 1),
            "share_curves_rel20_R6": float(np.mean(np.array(a["rel6"]) >= 0.20)),
            "share_curves_rel20_R11": float(np.mean(np.array(a["rel11"]) >= 0.20)),
            "n_curves_rel20_R6": int(np.sum(np.array(a["rel6"]) >= 0.20)),
            "n_curves_rel20_R11": int(np.sum(np.array(a["rel11"]) >= 0.20)),
        }
        print(f"\n=== {src}  frames {a['frames']}  curves {a['curves']}  "
              f"median len {out[src]['median_curve_len_evalpx']:.0f} eval px ===")
        print("  frac of GT LENGTH with another centerline within R eval px:")
        print("   " + "  ".join(f"R={r:<4.0f} {out[src]['frac_len_within_R'][str(r)]:.4f}" for r in RS))
        print(f"  frac of length in close-parallel band [2,6]: {out[src]['frac_len_band_2_6']:.4f}"
              f"   [2,11]: {out[src]['frac_len_band_2_11']:.4f}")
        print(f"  curves with >=20 eval px in [2,6]: {a['n26']} ({out[src]['share_curves_2_6']:.3f})"
              f"   in [2,11]: {a['n211']} ({out[src]['share_curves_2_11']:.3f})")
        print(f"  curves with >=20% of OWN length within R=6: {out[src]['n_curves_rel20_R6']} "
              f"({out[src]['share_curves_rel20_R6']:.3f})   within R=11: "
              f"{out[src]['n_curves_rel20_R11']} ({out[src]['share_curves_rel20_R11']:.3f})")
    print("\n" + json.dumps(out, indent=1)[:0])
    return out


if __name__ == "__main__":
    split = sys.argv[1]
    main(sys.argv[2:], split)
