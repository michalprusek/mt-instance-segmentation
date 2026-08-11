"""Ground-truth statistics for the MT-34 benchmark.

Two jobs:

1. **Estimate kappa_max**, the hard curvature bound the instancer enforces. It comes from the
   annotations themselves: microtubules physically cannot kink, so the high quantile of the
   observed |dtheta/ds| is an empirical ceiling on how sharply a *correct* centerline bends.
2. **Characterize the benchmark** -- crossings (with their angles) and close parallel bundles
   -- so the crossings/parallels claim can be measured rather than asserted. The 12 Alice
   frames are mostly well-separated near-horizontal filaments and cannot test it.

**Measurement scale matters.** Human polylines are coarse (median 5 vertices in task 586), so
vertex-to-vertex turn angles are dominated by the polygonal approximation, not by real bending.
Curvature is therefore measured over a BASELINE of ``ds`` px (default 8): resample at that
spacing and take the turn per unit arc length. That is also the scale at which the instancer's
window-fitted tangents operate.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
from scipy.spatial import cKDTree

from instance.geometry import (arclength, polyline_curvature, resample,
                               segment_angles, total_length, wrap_angle)
from mt_bench.cvat_import import read_frame_h5


def curvature_samples(polylines, ds: float = 8.0) -> np.ndarray:
    """Pool |curvature| (rad/px) over all polylines, measured at baseline ``ds``."""
    vals = []
    for p in polylines:
        p = np.asarray(p, dtype=float)
        if total_length(p) < 3 * ds:
            continue
        k = polyline_curvature(resample(p, ds=ds), ds=ds)
        if len(k):
            vals.append(np.abs(k))
    return np.concatenate(vals) if vals else np.zeros(0)


def curvature_quantile(polylines, ds: float = 8.0, q: float = 99.5) -> float:
    v = curvature_samples(polylines, ds=ds)
    return float(np.percentile(v, q)) if len(v) else 0.0


def _local_tangent(pts: np.ndarray, idx: int, half: int = 6) -> float:
    lo = max(0, idx - half)
    hi = min(len(pts) - 1, idx + half)
    if hi <= lo:
        return 0.0
    d = pts[hi] - pts[lo]
    return float(np.arctan2(d[1], d[0]))


def count_crossings(polylines, tol: float = 2.0, ds: float = 1.0,
                    endpoint_margin: float = 8.0,
                    self_min_separation: float = 20.0) -> list[dict]:
    """Find places where two GT centerlines (or one with itself) cross.

    A contact within ``endpoint_margin`` arc length of an endpoint of either polyline is a
    *touch*, not a crossing, and is skipped -- otherwise every T-shaped abutment would be
    counted as a junction the instancer must pass through.

    Returns dicts with ``i, j, x, y, s_i, s_j, angle_deg`` where ``angle_deg`` is folded to
    ``[0, 90]`` (filaments have no head or tail).
    """
    res = [resample(np.asarray(p, dtype=float), ds=ds) for p in polylines]
    lens = [arclength(p)[-1] if len(p) > 1 else 0.0 for p in res]
    trees = [cKDTree(p) if len(p) else None for p in res]

    out = []
    n = len(res)
    for i in range(n):
        if trees[i] is None or len(res[i]) < 3:
            continue
        for j in range(i, n):
            if trees[j] is None or len(res[j]) < 3:
                continue
            pairs = trees[i].query_ball_tree(trees[j], r=tol)
            hits = [(a, b) for a, lst in enumerate(pairs) for b in lst]
            if i == j:
                hits = [(a, b) for a, b in hits
                        if abs(a - b) * ds > self_min_separation]
            if not hits:
                continue
            # Cluster contacts that are contiguous along polyline i.
            hits.sort()
            clusters, cur = [], [hits[0]]
            for h in hits[1:]:
                if h[0] - cur[-1][0] <= 3 and abs(h[1] - cur[-1][1]) <= 6:
                    cur.append(h)
                else:
                    clusters.append(cur)
                    cur = [h]
            clusters.append(cur)

            for cl in clusters:
                a = int(np.median([c[0] for c in cl]))
                b = int(np.median([c[1] for c in cl]))
                s_i, s_j = a * ds, b * ds
                if min(s_i, lens[i] - s_i) < endpoint_margin:
                    continue
                if min(s_j, lens[j] - s_j) < endpoint_margin:
                    continue
                t_i = _local_tangent(res[i], a)
                t_j = _local_tangent(res[j], b)
                ang = abs(np.rad2deg(wrap_angle(t_j - t_i))) % 180.0
                ang = min(ang, 180.0 - ang)
                out.append({"i": i, "j": j,
                            "x": float(res[i][a, 0]), "y": float(res[i][a, 1]),
                            "s_i": s_i, "s_j": s_j, "angle_deg": float(ang)})
    return out


def count_parallel_pairs(polylines, gap_lo: float = 2.0, gap_hi: float = 6.0,
                         min_len: float = 20.0, ds: float = 2.0) -> int:
    """Number of polyline pairs running side by side at a ``[gap_lo, gap_hi]`` px gap.

    This is the "close parallels" stress statistic: pairs a fixed-radius embedding would
    intrinsically merge.
    """
    res = [resample(np.asarray(p, dtype=float), ds=ds) for p in polylines]
    trees = [cKDTree(p) if len(p) > 1 else None for p in res]
    count = 0
    for i in range(len(res)):
        if trees[i] is None:
            continue
        for j in range(i + 1, len(res)):
            if trees[j] is None:
                continue
            d, _ = trees[j].query(res[i], k=1)
            inband = float(np.sum((d >= gap_lo) & (d <= gap_hi)) * ds)
            if inband >= min_len:
                count += 1
    return count


def characterize(h5_dir: str, ds_curv: float = 8.0,
                 out_png: str | None = None) -> dict:
    """Per-source benchmark statistics + a pooled curvature histogram."""
    per_source: dict[str, dict] = {}
    all_curv: dict[str, np.ndarray] = {}

    for path in sorted(glob.glob(os.path.join(h5_dir, "*.h5"))):
        fr = read_frame_h5(path)
        task = str(fr["attrs"].get("source_task", "?"))
        pls = fr["polylines"]
        acc = per_source.setdefault(task, {
            "frames": 0, "mts": 0, "crossings": 0, "parallels": 0,
            "angles": [], "lengths": [], "curv": [],
        })
        acc["frames"] += 1
        acc["mts"] += len(pls)
        cr = count_crossings(pls)
        acc["crossings"] += len(cr)
        acc["angles"].extend(c["angle_deg"] for c in cr)
        acc["parallels"] += count_parallel_pairs(pls)
        acc["lengths"].extend(total_length(np.asarray(p, float)) for p in pls)
        acc["curv"].append(curvature_samples(pls, ds=ds_curv))

    summary = {}
    for task, acc in per_source.items():
        curv = np.concatenate([c for c in acc["curv"] if len(c)]) if acc["curv"] else np.zeros(0)
        all_curv[task] = curv
        ang = np.array(acc["angles"])
        summary[task] = {
            "frames": acc["frames"],
            "mts_per_frame": acc["mts"] / max(acc["frames"], 1),
            "crossings_per_frame": acc["crossings"] / max(acc["frames"], 1),
            "parallel_pairs_per_frame": acc["parallels"] / max(acc["frames"], 1),
            "shallow_crossings_frac": float(np.mean(ang < 30.0)) if len(ang) else 0.0,
            "median_length_px": float(np.median(acc["lengths"])) if acc["lengths"] else 0.0,
            "kappa_p99_5": float(np.percentile(curv, 99.5)) if len(curv) else 0.0,
            "kappa_p99": float(np.percentile(curv, 99.0)) if len(curv) else 0.0,
            "kappa_median": float(np.median(curv)) if len(curv) else 0.0,
        }

    pooled = np.concatenate([c for c in all_curv.values() if len(c)]) if all_curv else np.zeros(0)
    summary["_pooled"] = {
        "kappa_p99_5": float(np.percentile(pooled, 99.5)) if len(pooled) else 0.0,
        "kappa_p99": float(np.percentile(pooled, 99.0)) if len(pooled) else 0.0,
        "kappa_p99_9": float(np.percentile(pooled, 99.9)) if len(pooled) else 0.0,
        "n_samples": int(len(pooled)),
    }

    if out_png and len(pooled):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        for task, curv in all_curv.items():
            if len(curv):
                ax.hist(curv, bins=120, range=(0, 0.5), histtype="step",
                        density=True, label=f"task {task} (n={len(curv)})")
        k995 = summary["_pooled"]["kappa_p99_5"]
        ax.axvline(k995, color="red", ls="--",
                   label=f"pooled p99.5 = {k995:.3f} rad/px")
        ax.set_xlabel(f"|dtheta/ds| at {ds_curv:.0f} px baseline  [rad/px]")
        ax.set_ylabel("density")
        ax.set_title("MT-34 ground-truth curvature: microtubules do not kink")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_png, dpi=130)
        plt.close(fig)

    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("h5_dir", nargs="?", default="data/real/mt34_eval")
    ap.add_argument("--ds", type=float, default=8.0)
    ap.add_argument("--out-png",
                    default="data/enc_sensitivity_testset/mt34_overlays/_kappa_hist.png")
    args = ap.parse_args()

    s = characterize(args.h5_dir, ds_curv=args.ds, out_png=args.out_png)
    for task in sorted(k for k in s if k != "_pooled"):
        v = s[task]
        print(f"task {task}: {v['frames']} frames | {v['mts_per_frame']:.1f} MT/frame | "
              f"{v['crossings_per_frame']:.1f} crossings/frame "
              f"({100 * v['shallow_crossings_frac']:.0f}% below 30 deg) | "
              f"{v['parallel_pairs_per_frame']:.1f} close-parallel pairs/frame | "
              f"median length {v['median_length_px']:.0f} px | "
              f"kappa med {v['kappa_median']:.4f} p99 {v['kappa_p99']:.3f} "
              f"p99.5 {v['kappa_p99_5']:.3f}")
    p = s["_pooled"]
    print(f"POOLED kappa (n={p['n_samples']}): p99 {p['kappa_p99']:.4f}  "
          f"p99.5 {p['kappa_p99_5']:.4f}  p99.9 {p['kappa_p99_9']:.4f} rad/px "
          f"(baseline {args.ds:.0f} px)")


if __name__ == "__main__":
    main()
