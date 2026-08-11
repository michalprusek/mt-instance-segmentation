"""Evaluation metrics for microtubule instance segmentation.

``rasterise_polyline``, ``centerline_f1`` and ``aggregate_f1`` are ported VERBATIM from
``tulen:~/mt_enc_exp/scripts/centerline_f1.py`` so numbers stay comparable with the project's
existing bars (TARDIS 0.326, ORION 0.519, v19 0.696, ours 0.697). Do not change their
semantics.

The rest are new diagnostics that measure the two documented bottlenecks directly, because
a single F1 cannot say *why* it dropped:

* :func:`junction_identity`  -- does each microtubule keep ONE identity through a crossing?
* :func:`fragmentation`      -- how many predicted pieces cover one real microtubule?
* :func:`bundle_recovery`    -- are close parallels kept apart?
* :func:`max_curvature`      -- does the output honour the physical no-kink constraint?
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from skimage.draw import line as draw_line
from skimage.morphology import skeletonize

from instance.geometry import arclength, max_abs_curvature, resample

# --------------------------------------------------------------------------------------
# Ported verbatim -- do not change semantics.
# --------------------------------------------------------------------------------------


def rasterise_polyline(points: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Bresenham-rasterise a polyline; return (H, W) bool mask."""
    H, W = shape
    out = np.zeros((H, W), dtype=bool)
    if len(points) < 1:
        return out
    rr_int = np.clip(np.rint(points[:, 1]).astype(int), 0, H - 1)
    cc_int = np.clip(np.rint(points[:, 0]).astype(int), 0, W - 1)
    if len(points) == 1:
        out[rr_int[0], cc_int[0]] = True
        return out
    for i in range(len(points) - 1):
        rr, cc = draw_line(rr_int[i], cc_int[i], rr_int[i + 1], cc_int[i + 1])
        out[rr, cc] = True
    return out


def _skel_points(mask: np.ndarray) -> np.ndarray:
    skel = skeletonize(mask)
    ys, xs = np.where(skel)
    return np.stack([ys, xs], axis=1).astype(np.float32) if len(ys) else \
        np.zeros((0, 2), dtype=np.float32)


def _polyline_pixels(points: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    mask = rasterise_polyline(points, shape)
    ys, xs = np.where(mask)
    return np.stack([ys, xs], axis=1).astype(np.float32) if len(ys) else \
        np.zeros((0, 2), dtype=np.float32)


def _pair_score(pred_pts: np.ndarray, gt_pts: np.ndarray,
                tol: float) -> Tuple[float, float, float]:
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return 0.0, 0.0, float("inf")
    tree_pred = cKDTree(pred_pts)
    tree_gt = cKDTree(gt_pts)
    d_pred_to_gt, _ = tree_gt.query(pred_pts, k=1)
    d_gt_to_pred, _ = tree_pred.query(gt_pts, k=1)
    prec = float((d_pred_to_gt <= tol).mean())
    rec = float((d_gt_to_pred <= tol).mean())
    haus = float(0.5 * (d_pred_to_gt.mean() + d_gt_to_pred.mean()))
    return prec, rec, haus


def centerline_f1(pred_masks: Sequence[np.ndarray],
                  gt_polylines: Sequence[np.ndarray],
                  tol: float = 5.0,
                  length_coverage: float = 0.80,
                  precision_coverage: float | None = None) -> dict:
    """Strict centerline-F1 for one image. See the module docstring for provenance."""
    pred_list = list(pred_masks)
    gt_list = list(gt_polylines)
    if not pred_list and not gt_list:
        return {"tp": 0, "fp": 0, "fn": 0, "precision": 1.0,
                "recall": 1.0, "f1": 1.0, "pairs": []}

    if pred_list:
        H, W = pred_list[0].shape
    else:
        all_pts = np.concatenate([p for p in gt_list if len(p) > 0], axis=0) \
            if any(len(p) > 0 for p in gt_list) else None
        if all_pts is not None and len(all_pts):
            H = int(all_pts[:, 1].max() + 50)
            W = int(all_pts[:, 0].max() + 50)
        else:
            H = W = 1024

    pred_pts_list = [_skel_points(m) for m in pred_list]
    gt_pts_list = [_polyline_pixels(p, (H, W)) for p in gt_list]

    Np, Ng = len(pred_pts_list), len(gt_pts_list)
    if Np == 0:
        return {"tp": 0, "fp": 0, "fn": Ng, "precision": 0.0,
                "recall": 0.0, "f1": 0.0, "pairs": []}
    if Ng == 0:
        return {"tp": 0, "fp": Np, "fn": 0, "precision": 0.0,
                "recall": 0.0, "f1": 0.0, "pairs": []}

    cost = np.ones((Np, Ng), dtype=np.float32)
    score_grid: List[List[Tuple[float, float, float]]] = [
        [(0.0, 0.0, float("inf"))] * Ng for _ in range(Np)
    ]
    for i in range(Np):
        for j in range(Ng):
            p, r, h = _pair_score(pred_pts_list[i], gt_pts_list[j], tol)
            score_grid[i][j] = (p, r, h)
            f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
            cost[i, j] = 1.0 - f1

    row_idx, col_idx = linear_sum_assignment(cost)

    tp = 0
    matched_preds, matched_gts, pairs = set(), set(), []
    for i, j in zip(row_idx, col_idx):
        prec, rec, haus = score_grid[i][j]
        if precision_coverage is None:
            is_tp = (haus <= tol) and (rec >= length_coverage)
        else:
            is_tp = (rec >= length_coverage) and (prec >= precision_coverage)
        pairs.append({"pred": int(i), "gt": int(j), "precision": prec,
                      "recall": rec, "hausdorff": haus, "tp": bool(is_tp)})
        if is_tp:
            tp += 1
            matched_preds.add(int(i))
            matched_gts.add(int(j))

    fp = Np - len(matched_preds)
    fn = Ng - len(matched_gts)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) \
        if (precision + recall) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "f1": f1, "pairs": pairs}


def aggregate_f1(per_image: Iterable[dict]) -> dict:
    """Aggregate per-image centerline-F1 results into mean / micro F1."""
    items = list(per_image)
    if not items:
        return {"mean_f1": 0.0, "micro_precision": 0.0,
                "micro_recall": 0.0, "micro_f1": 0.0, "n_images": 0}
    mean_f1 = float(np.mean([r["f1"] for r in items]))
    tp = sum(r["tp"] for r in items)
    fp = sum(r["fp"] for r in items)
    fn = sum(r["fn"] for r in items)
    micro_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) \
        if (micro_p + micro_r) > 0 else 0.0
    return {"mean_f1": mean_f1, "micro_precision": micro_p,
            "micro_recall": micro_r, "micro_f1": micro_f1,
            "n_images": len(items), "total_tp": tp, "total_fp": fp, "total_fn": fn}


# --------------------------------------------------------------------------------------
# New diagnostics.
# --------------------------------------------------------------------------------------


def aggregate_benchmark(per_image: Iterable[dict]) -> dict:
    """Aggregate MT-34 results under the benchmark's empty-frame policy.

    ``training_img_102`` has zero ground-truth microtubules, and it lands in VAL. Per-frame
    F1 is undefined there, and ``centerline_f1`` hands out a free 1.0 when both GT and
    prediction are empty -- which would inflate every method's macro mean and the tuning
    objective alike. Such frames are therefore EXCLUDED from the macro mean but keep
    contributing their false positives to the micro totals, where they belong: an empty field
    is a real test of whether an instancer invents microtubules.

    Rows must carry ``n_gt``.
    """
    items = list(per_image)
    if not items:
        return {"mean_f1": 0.0, "micro_f1": 0.0, "n_images": 0, "n_scored": 0}
    scored = [r for r in items if r.get("n_gt", 1) > 0]
    tp = sum(r["tp"] for r in items)
    fp = sum(r["fp"] for r in items)
    fn = sum(r["fn"] for r in items)
    micro_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) \
        if (micro_p + micro_r) > 0 else 0.0
    return {
        "mean_f1": float(np.mean([r["f1"] for r in scored])) if scored else float("nan"),
        "micro_precision": micro_p, "micro_recall": micro_r, "micro_f1": micro_f1,
        "n_images": len(items), "n_scored": len(scored),
        "total_tp": tp, "total_fp": fp, "total_fn": fn,
    }


def _mask_trees(masks: Sequence[np.ndarray]) -> list:
    """KD-tree of (x=col, y=row) foreground pixels for each predicted instance."""
    trees = []
    for m in masks:
        ys, xs = np.where(m)
        trees.append(cKDTree(np.stack([xs, ys], axis=1)) if len(ys) else None)
    return trees


def _covering_instances(trees, point: np.ndarray, tol: float) -> set:
    return {k for k, t in enumerate(trees)
            if t is not None and t.query(point[None, :], k=1)[0][0] <= tol}


def _coverage_fraction(tree, gt_pts: np.ndarray, tol: float) -> float:
    if tree is None or len(gt_pts) == 0:
        return 0.0
    d, _ = tree.query(gt_pts, k=1)
    return float(np.mean(d <= tol))


def junction_identity(pred_masks: Sequence[np.ndarray],
                      gt_polylines: Sequence[np.ndarray],
                      crossings: Sequence[dict],
                      tol: float = 5.0,
                      arm_offset: float = 15.0) -> dict:
    """Fraction of GT crossings where BOTH microtubules keep a single identity through it.

    For each of the two filaments, sample the GT centerline ``arm_offset`` px before and
    after the crossing; the filament survives if the SAME predicted instance covers both
    samples. This is the direct measurement of bottleneck #2 (junctions): a tracer that
    breaks a microtubule at the crossing scores 0 here even when its centerline-F1 looks
    respectable.

    Crossings whose arms fall off the end of a GT polyline are skipped.
    """
    trees = _mask_trees(pred_masks)
    res = [resample(np.asarray(p, dtype=float), ds=1.0) for p in gt_polylines]
    lens = [float(arclength(p)[-1]) if len(p) > 1 else 0.0 for p in res]

    n_used = 0
    n_pres = 0
    by_angle: dict[str, list] = {"shallow": [], "oblique": [], "steep": []}

    for cr in crossings:
        ok = True
        usable = True
        for idx_key, s_key in (("i", "s_i"), ("j", "s_j")):
            gi = cr[idx_key]
            s = cr[s_key]
            if gi >= len(res) or lens[gi] <= 0:
                usable = False
                break
            s_before, s_after = s - arm_offset, s + arm_offset
            if s_before < 0 or s_after > lens[gi]:
                usable = False
                break
            pts = res[gi]
            p_before = pts[min(int(round(s_before)), len(pts) - 1)]
            p_after = pts[min(int(round(s_after)), len(pts) - 1)]
            before = _covering_instances(trees, p_before, tol)
            after = _covering_instances(trees, p_after, tol)
            if not (before & after):
                ok = False
        if not usable:
            continue
        n_used += 1
        n_pres += int(ok)
        ang = cr.get("angle_deg", 90.0)
        key = "shallow" if ang < 30.0 else ("oblique" if ang < 60.0 else "steep")
        by_angle[key].append(int(ok))

    return {
        "n_crossings": n_used,
        "n_preserved": n_pres,
        "rate": (n_pres / n_used) if n_used else float("nan"),
        "by_angle": {k: (float(np.mean(v)) if v else float("nan"))
                     for k, v in by_angle.items()},
        "by_angle_n": {k: len(v) for k, v in by_angle.items()},
    }


def fragmentation(pred_masks: Sequence[np.ndarray],
                  gt_polylines: Sequence[np.ndarray],
                  tol: float = 5.0, min_frac: float = 0.2) -> float:
    """Mean number of predicted instances covering at least ``min_frac`` of a GT centerline.

    1.0 is ideal. Values above 1 mean microtubules are being broken into pieces -- the
    failure PySOAX exhibits at crossings. GT microtubules that no prediction reaches at all
    are skipped (they are recall failures, measured by centerline-F1, not fragmentation).
    """
    trees = _mask_trees(pred_masks)
    counts = []
    for p in gt_polylines:
        gt_pts = resample(np.asarray(p, dtype=float), ds=1.0)
        if len(gt_pts) < 2:
            continue
        c = sum(1 for t in trees if _coverage_fraction(t, gt_pts, tol) >= min_frac)
        if c > 0:
            counts.append(c)
    return float(np.mean(counts)) if counts else float("nan")


def bundle_recovery(pred_masks: Sequence[np.ndarray],
                    gt_polylines: Sequence[np.ndarray],
                    pairs: Sequence[Tuple[int, int]],
                    tol: float = 1.5, min_cov: float = 0.7) -> float:
    """Fraction of close-parallel GT pairs recovered as two DIFFERENT instances.

    Measures bottleneck #1. A merged bundle (one instance covering both filaments) scores 0
    -- the failure mode of fixed-radius embedding methods.

    ``tol`` MUST be smaller than half the bundle gap. The pairs this metric is built for sit
    2-6 px apart, so at ``tol >= 3`` a prediction lying on one filament is also "covering"
    the other and the two are indistinguishable *to the metric*, regardless of how well the
    instancer did. Hence the default of 1.5 px.

    The two filaments are assigned to instances with a one-to-one Hungarian matching rather
    than an independent argmax per filament: with argmax, two equally good candidates are a
    tie that numpy silently breaks toward the lower index, which reports a perfectly
    separated bundle as merged.
    """
    trees = _mask_trees(pred_masks)
    if not pairs:
        return float("nan")
    good = 0
    for i, j in pairs:
        gi = resample(np.asarray(gt_polylines[i], dtype=float), ds=1.0)
        gj = resample(np.asarray(gt_polylines[j], dtype=float), ds=1.0)
        cov = np.array([[_coverage_fraction(t, g, tol) for t in trees]
                        for g in (gi, gj)])
        if cov.size == 0 or cov.shape[1] < 2:
            continue
        rows, cols = linear_sum_assignment(1.0 - cov)
        assigned = cov[rows, cols]
        if len(set(cols)) == 2 and assigned.min() >= min_cov:
            good += 1
    return good / len(pairs)


def _strata_index(rows: Sequence[dict], stratum_key: str | None) -> List[np.ndarray]:
    """Row indices grouped by stratum, or one group holding everything."""
    if stratum_key is None:
        return [np.arange(len(rows))]
    groups: dict = {}
    for i, r in enumerate(rows):
        groups.setdefault(r.get(stratum_key), []).append(i)
    return [np.asarray(v) for _, v in sorted(groups.items(), key=lambda kv: str(kv[0]))]


def bootstrap_ci(rows: Sequence[dict], stat: str = "mean_f1", n_boot: int = 10000,
                 stratum_key: str | None = "task", seed: int = 0,
                 alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI for one method's aggregate, resampling FRAMES.

    ``rows`` are per-frame dicts as produced by :func:`centerline_f1` plus ``n_gt`` (and the
    stratum field). Each replicate re-runs :func:`aggregate_benchmark`, so the empty-frame
    policy -- exclude from the macro mean, keep the false positives in micro -- is applied
    inside the resample rather than to a pre-computed number.

    Resampling is **stratified by source task** by default. Pooled MT-34 TEST is a fixed
    6 (Alice) + 11 (new-22) design, and the two halves differ enormously (2.2 vs 32.1
    crossings per frame); resampling them jointly would inject composition variance that the
    measurement does not have and inflate every interval.
    """
    rng = np.random.default_rng(seed)
    strata = _strata_index(rows, stratum_key)
    vals = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in strata])
        vals[b] = aggregate_benchmark([rows[i] for i in idx]).get(stat, float("nan"))
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": aggregate_benchmark(rows).get(stat, float("nan")),
            "lo": float(lo), "hi": float(hi), "n_frames": len(rows),
            "n_boot": n_boot, "stat": stat}


def paired_bootstrap(rows_a: Sequence[dict], rows_b: Sequence[dict],
                     stat: str = "mean_f1", n_boot: int = 10000,
                     stratum_key: str | None = "task", frame_key: str = "frame",
                     seed: int = 0, alpha: float = 0.05) -> dict:
    """CI on the DIFFERENCE (a - b) between two methods scored on the same frames.

    The comparison this project actually needs. With n = 17 TEST frames, two marginal
    intervals overlap for almost any pair of methods and say nothing about whether one beats
    the other; the difference is what has to be bounded. Each replicate draws ONE frame
    multiset and scores both methods on it, so the frame-difficulty variance the two methods
    share cancels instead of being counted twice.

    ``rows_a`` and ``rows_b`` must describe the same frames. When both carry ``frame_key``
    they are aligned by it and a mismatch raises -- silently comparing different frame sets
    would produce a confident, meaningless interval.

    Returns the point difference, its interval, and ``p_two_sided``, the usual bootstrap
    approximation ``2 * min(P(diff <= 0), P(diff >= 0))``.
    """
    if len(rows_a) != len(rows_b):
        raise ValueError(f"paired bootstrap needs the same frames: "
                         f"{len(rows_a)} vs {len(rows_b)}")
    if all(frame_key in r for r in rows_a) and all(frame_key in r for r in rows_b):
        order = {r[frame_key]: i for i, r in enumerate(rows_b)}
        if set(order) != {r[frame_key] for r in rows_a}:
            raise ValueError("paired bootstrap got two different frame sets")
        rows_b = [rows_b[order[r[frame_key]]] for r in rows_a]

    rng = np.random.default_rng(seed)
    strata = _strata_index(rows_a, stratum_key)
    diffs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in strata])
        va = aggregate_benchmark([rows_a[i] for i in idx]).get(stat, float("nan"))
        vb = aggregate_benchmark([rows_b[i] for i in idx]).get(stat, float("nan"))
        diffs[b] = va - vb
    lo, hi = np.nanpercentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    point = (aggregate_benchmark(rows_a).get(stat, float("nan"))
             - aggregate_benchmark(rows_b).get(stat, float("nan")))
    finite = diffs[np.isfinite(diffs)]
    p = 2 * min((finite <= 0).mean(), (finite >= 0).mean()) if len(finite) else float("nan")
    return {"diff": float(point), "lo": float(lo), "hi": float(hi),
            "p_two_sided": float(min(p, 1.0)), "n_frames": len(rows_a),
            "n_boot": n_boot, "stat": stat,
            "significant": bool(np.isfinite(lo) and np.isfinite(hi) and lo * hi > 0)}


def max_curvature(pred_polylines: Sequence[np.ndarray], ds: float = 2.0) -> float:
    """Largest |dtheta/ds| over all predicted polylines, in rad/px.

    A curvature-bounded instancer must keep this at or below its kappa_max; anything above
    is a kink the physics forbids.
    """
    vals = [max_abs_curvature(resample(np.asarray(p, dtype=float), ds=ds), ds=ds)
            for p in pred_polylines if len(np.asarray(p)) >= 3]
    return float(max(vals)) if vals else 0.0
