"""Foreground QUALITY as a checkpoint-selection gate for the semantic model.

Why this module exists
----------------------
The semantic model is trained and selected on a tolerant centerline-coverage F1, and that
metric **cannot rank foregrounds by the thing we care about**. Measured on MT-34: nnU-Net and
v4b score almost identically on coverage F1 (Alice tol2 0.947 vs 0.950) yet nnU-Net's
foreground yields markedly better instances (pooled 0.418 vs 0.379). Over 198 same-frame
model pairs across four foregrounds, the *pairwise ranking accuracy* -- how often a property
ordered two foregrounds the same way their downstream instance F1 did -- was:

=====================  ========  ===============================================
property               accuracy  what it measures
=====================  ========  ===============================================
``cc_per_gt``              0.82  connected components per GT microtubule
``gaps_per_mt``            0.80  foreground dropouts along a real microtubule
``endp_per_kpx``           0.79  skeleton endpoints per 1000 skeleton px
``skel_px_per_gt_px``      0.58  near chance
``prec2``                  0.58  the control -- **chance**, despite rho = +0.87
=====================  ========  ===============================================

``prec2`` is the cautionary result: a property can correlate strongly across frames (both it
and F1 track frame difficulty) and still rank *models* at chance. Correlation was the wrong
diagnostic; paired ranking accuracy -- which cancels frame difficulty by construction -- is
the right one. :func:`ranking_accuracy` reproduces this table from the saved measurements.

The over-firing trap
--------------------
All three winners improve MONOTONICALLY as the foreground dilates -- a mask that floods the
frame merges every component (``cc_per_gt`` -> 0) and fills every dropout (``gaps_per_mt``
-> 0). The trap is not hypothetical: on this same battery, **raw foreground fraction ranks at
0.75** (more foreground = better instances) across four *calibrated* models, so an
unconstrained search would happily dilate its way to a better score. A training trajectory
contains checkpoints that are not calibrated at all, and selecting on continuity alone would
walk straight into the failure mode the project's standing constraint calls fatal ("a
synth-trained foreground that over-fires on real breaks every downstream step"). Selection
here is consequently **constrained minimisation**: minimise the continuity score SUBJECT TO
an over-firing ceiling, which is why :func:`foreground_quality` returns the full battery
(``fg``, ``rec2``, ``prec2``, ...) and not just the three winners.

Where to run the gate
---------------------
On the **real VAL split** -- that is where the ranking accuracy was measured, and selecting
hyperparameters/checkpoints on a real VAL split is explicitly inside the synth-only rule
(training data stays synthetic; TEST stays sealed). Running it additionally on a synthetic
VAL split is a useful no-domain-gap control, not a substitute.

Coordinates
-----------
GT polyline vertices are ``(x=col, y=row)`` -- the transpose of NumPy indexing. Everything
inside this module works in ``(row, col)`` and converts at the boundary. This has caused two
silent bugs elsewhere in the project; ``tests/test_fg_quality.py`` pins it.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve, label
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from instance.geometry import resample

_K8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

#: Properties that rank foregrounds better than chance, lower being better for each.
CONTINUITY_KEYS = ("cc_per_gt", "endp_per_kpx", "gaps_per_mt")

#: Which direction is "better" per property. Needed because :func:`ranking_accuracy` is a
#: directed test: read it with the wrong sign and a chance-level property (``prec2``, 0.58)
#: looks like an informative one inverted (0.42).
LOWER_IS_BETTER = {"cc_per_gt": True, "endp_per_kpx": True, "gaps_per_mt": True,
                   "junc_per_kpx": True, "skel_px_per_gt_px": True,
                   "prec2": False, "rec2": False, "cov_per_mt": False, "fg": False}

#: Pairwise ranking accuracy measured over 198 same-frame model pairs on MT-34 (module docs).
RANKING_ACCURACY = {"cc_per_gt": 0.82, "gaps_per_mt": 0.80, "endp_per_kpx": 0.79,
                    "prec2": 0.58, "fg": 0.75}

#: v4b's measured values on MT-34 (whole-frame norm, thr 0.35). Used only as the SCALE ANCHOR
#: for :func:`quality_score`, so a score of 1.0 reads as "v4b-level" and 0.5 as "twice as
#: continuous as v4b". Changing these rescales the score; it never changes an ordering.
REFERENCE = {"cc_per_gt": 10.689, "endp_per_kpx": 23.568, "gaps_per_mt": 1.559}

#: Mean predicted foreground fraction of in-domain synthetic frames (~1.6%). The over-firing
#: ceiling is a multiple of this; v4b sits at 1.93% on MT-34, i.e. 1.2x -- not over-firing.
SYNTH_FG_REFERENCE = 0.016


def foreground_quality(mask: np.ndarray, gt_polylines, up: float = 1.0,
                       tol: float = 2.0) -> dict | None:
    """Descriptors of foreground quality for one frame. No instancer, no tuned parameters.

    ``mask`` is a boolean foreground at ``up`` scale; ``gt_polylines`` are NATIVE-scale
    ``(x, y)`` vertices and are scaled by ``up`` here. Returns ``None`` for an empty mask or
    empty GT -- callers aggregate with :func:`mean_properties`, which skips them.

    Deliberately cheap: it is meant to run every epoch over a VAL split inside a training
    loop, so it must not invoke the instancer (which is seconds per frame, not milliseconds).
    """
    gt_polylines = [np.asarray(p, dtype=float) for p in gt_polylines
                    if len(np.asarray(p)) >= 2]
    ys, xs = np.where(mask)
    if len(ys) == 0 or not gt_polylines:
        return None

    fg_tree = cKDTree(np.stack([ys, xs], axis=1))                    # (row, col)
    gt_pts = np.concatenate([resample(p * up, ds=1.0) for p in gt_polylines])
    gt_rc = np.stack([gt_pts[:, 1], gt_pts[:, 0]], axis=1)           # (x, y) -> (row, col)
    gt_tree = cKDTree(gt_rc)

    skel = skeletonize(mask)
    sy, sx = np.where(skel)
    n_skel = max(len(sy), 1)
    nb = convolve(skel.astype(np.uint8), _K8, mode="constant") * skel

    # How often the foreground DROPS OUT along a real microtubule. Every dropout is a break
    # the instancer must bridge from image evidence or lose -- this is the property that most
    # directly explains v4b's 1.56 gaps/MT against nnU-Net's 0.37.
    gaps, cov_frac = [], []
    for p in gt_polylines:
        q = resample(p * up, ds=1.0)
        cov = fg_tree.query(np.stack([q[:, 1], q[:, 0]], axis=1), k=1)[0] <= tol
        if len(cov) < 5:
            continue
        gaps.append(int((np.diff(cov.astype(int)) == -1).sum()))
        cov_frac.append(float(cov.mean()))

    _, n_cc = label(mask, structure=np.ones((3, 3)))
    skel_rc = np.stack([sy, sx], axis=1)
    return {
        # ranks foregrounds correctly (see RANKING_ACCURACY)
        "cc_per_gt": n_cc / len(gt_polylines),
        "endp_per_kpx": float(1000 * (nb == 1).sum() / n_skel),
        "gaps_per_mt": float(np.mean(gaps)) if gaps else float("nan"),
        # the over-firing guard -- without these the three above are trivially gameable
        "fg": float(mask.mean()),
        "prec2": float((gt_tree.query(skel_rc, k=1)[0] <= tol).mean()),
        "rec2": float((fg_tree.query(gt_rc, k=1)[0] <= tol).mean()),
        # context, not selection criteria
        "junc_per_kpx": float(1000 * (nb >= 3).sum() / n_skel),
        "cov_per_mt": float(np.mean(cov_frac)) if cov_frac else float("nan"),
        "skel_px_per_gt_px": n_skel / max(len(gt_pts), 1),
        "n_gt": len(gt_polylines),
    }


def mean_properties(per_frame) -> dict:
    """Mean of :func:`foreground_quality` over frames, skipping ``None`` and NaN entries."""
    rows = [r for r in per_frame if r is not None]
    if not rows:
        return {}
    keys = {k for r in rows for k in r}
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows if k in r], dtype=float)
        vals = vals[np.isfinite(vals)]
        out[k] = float(vals.mean()) if len(vals) else float("nan")
    out["n_frames"] = len(rows)
    return out


def quality_score(props: dict, reference: dict = REFERENCE) -> float:
    """Continuity score, **lower is better**, normalised so 1.0 is the v4b baseline.

    A plain mean of the three winners' reference-normalised values. They are weighted equally
    rather than by their ranking accuracies: 0.82 vs 0.79 is well inside the noise of 174
    paired comparisons, and pretending otherwise would fit the weights to that noise.

    Returns ``nan`` if no component is finite -- an unusable checkpoint, not a perfect one.
    """
    vals = [props[k] / reference[k] for k in CONTINUITY_KEYS
            if k in props and np.isfinite(props.get(k, np.nan))]
    return float(np.mean(vals)) if vals else float("nan")


def passes_overfiring_gate(props: dict, synth_fg: float = SYNTH_FG_REFERENCE,
                           factor: float = 3.0, min_rec2: float = 0.90) -> bool:
    """Is this foreground sane enough for its continuity score to mean anything?

    Two ways to cheat the continuity metrics, both blocked here:

    * **dilate** -- a flooded mask has no gaps and one component. Bounded by
      ``fg <= factor * synth_fg``, the same over-firing gate used at prediction time (v4b sits
      at 1.2x, and >3x is the established "over-firing" line).
    * **collapse** -- a nearly EMPTY mask also has few components and, because a microtubule
      it misses entirely contributes no dropouts, few gaps. Bounded by ``rec2 >= min_rec2``.

    ``min_rec2`` is a floor, not an objective: every sane foreground measured here sits at
    0.97-0.99, so it rejects collapse without competing with the continuity score.
    """
    fg = props.get("fg", np.nan)
    rec2 = props.get("rec2", np.nan)
    if not np.isfinite(fg) or not np.isfinite(rec2):
        return False
    return bool(fg <= factor * synth_fg and rec2 >= min_rec2)


def select_checkpoint(candidates, synth_fg: float = SYNTH_FG_REFERENCE,
                      factor: float = 3.0, min_rec2: float = 0.90,
                      min_frames: int | None = None) -> int | None:
    """Index of the best checkpoint: ``argmin quality_score`` among those passing the gate.

    ``candidates`` is a sequence of aggregated property dicts, one per checkpoint (each the
    :func:`mean_properties` of that checkpoint over the VAL split). Returns ``None`` when no
    checkpoint passes -- a genuine outcome that must be reported, never silently replaced by
    the least-bad over-firing model.

    ``min_frames`` closes a **partial-collapse** hole and callers with a fixed VAL split should
    always pass it. :func:`foreground_quality` returns ``None`` for an empty mask and
    :func:`mean_properties` skips those, so a checkpoint that fires on 3 of 16 VAL frames and
    nothing on the other 13 is averaged over just those 3: ``rec2`` can clear the floor, ``fg``
    looks sane, and if the 3 happen to be easy frames its continuity score can even win. The
    all-empty case fails on its own (nothing is measurable); the partial case does not, and
    only the frame count reveals it.
    """
    scored = []
    for i, c in enumerate(candidates):
        if min_frames is not None and c.get("n_frames", 0) < min_frames:
            continue
        if not passes_overfiring_gate(c, synth_fg, factor, min_rec2):
            continue
        s = quality_score(c)
        if np.isfinite(s):
            scored.append((s, i))
    return min(scored)[1] if scored else None


def ranking_accuracy(rows, key: str, score_key: str = "inst_f1",
                     model_key: str = "model", frame_key: str = "frame",
                     lower_is_better: bool | None = None) -> float:
    """Fraction of same-frame model pairs that ``key`` orders as ``score_key`` does.

    This is the diagnostic that produced the table at the top of the module, kept here so the
    claim is reproducible rather than quoted. Pairs are formed WITHIN a frame, so frame
    difficulty -- the confound that let ``prec2`` reach rho = +0.87 while ranking at chance --
    cancels. Ties in either quantity are skipped.

    Direction comes from :data:`LOWER_IS_BETTER` unless ``lower_is_better`` says otherwise;
    it matters, because the test is directed and 1 - accuracy is the accuracy of the opposite
    claim.
    """
    if lower_is_better is None:
        lower_is_better = LOWER_IS_BETTER.get(key, True)
    by_frame: dict = {}
    for r in rows:
        by_frame.setdefault(r[frame_key], []).append(r)
    hits = total = 0
    for group in by_frame.values():
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                ra, rb = group[a], group[b]
                if ra[model_key] == rb[model_key]:
                    continue
                x, y = ra.get(key, np.nan), rb.get(key, np.nan)
                fa, fb = ra[score_key], rb[score_key]
                if not (np.isfinite(x) and np.isfinite(y)) or x == y or fa == fb:
                    continue
                total += 1
                better = (x < y) if lower_is_better else (x > y)
                hits += better == (fa > fb)
    return hits / total if total else float("nan")
