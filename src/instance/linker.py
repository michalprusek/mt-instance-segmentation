"""Learned arm-pairing cost for junction matching.

``matching.pair_cost`` decides which arms of a junction continue into each other from BINARY
geometry alone -- two turn angles, two curvatures and a gap length. Everything the image knew
was discarded at the binarisation threshold. The measured consequence is stark: given a
rasterised-ground-truth mask instancer A reaches 0.920 on MT-34 TEST with junction identity
0.965, and given its own predicted mask the same code reaches 0.457 with junction identity
~0.5. The geometry is sufficient when the skeleton is clean and fails when it is not, and the
mask factorial (docs/F1_PUSH_PLAN.md) shows the dominant term is exactly that: 88 of the
microtubules lost on real VAL are lost to mask defects WITHIN 4 px of the ground truth --
spurs, blobs and width irregularity that corrupt the tangents the cost is computed from.

External evidence that this is the right place to intervene: on a synthetic vessel benchmark
with exact graph ground truth, "U-Net + skeletonise + heuristics" scores edge mAP 17.9 while
predicting the graph with a learned edge head scores 78.1 on identical data (Vesselformer,
MIDL 2023) -- the same oracle-versus-model gap this project measures, localised in the
assembly heuristics rather than in the features.

So the cost becomes learned, while the two things that are known to be right stay hard:

* ``kappa_max`` remains a FORBIDDEN-region constraint, not a learned preference. It is derived
  from the 957 annotated microtubules, not tuned, and a learned scorer must not be able to
  buy its way through a physically impossible kink.
* the pairing is still a global minimum-cost matching over the whole junction, so an X still
  resolves into two consistent through-paths.

The forward pass is plain NumPy on weights loaded from an ``.npz`` so ``src/instance`` keeps
its numpy/scipy/networkx-only dependency surface; training lives in ``scripts/train_linker.py``.
"""
from __future__ import annotations

import os

import numpy as np

from instance.geometry import turn_penalty
from instance.matching import ArmEnd, _ori_mismatch

#: Order is part of the file format -- a checkpoint records ``n_features`` and is rejected if
#: it disagrees, because a silent re-ordering would train one thing and serve another.
FEATURE_NAMES = (
    "total_turn",        # turn charged through the gap direction (the displacement-aware form)
    "direct_turn",       # turn ignoring displacement; differs exactly for close parallels
    "kappa_sum_abs",     # |k_a + k_b| -- zero for a smooth continuation
    "kappa_abs_sum",     # |k_a| + |k_b| -- how bent the two arms are at all
    "gap_len",
    "kappa_implied",
    "ori_mismatch",
    "bridge_mean",       # image evidence along the straight tip-to-tip bridge
    "bridge_min",
    "bridge_frac",
    "arc_len_a",
    "arc_len_b",
    "n_arms",            # crowding of the junction cluster
    "is_gap_link",
)
N_FEATURES = len(FEATURE_NAMES)


def _bridge_stats(prob: np.ndarray | None, a: ArmEnd, b: ArmEnd,
                  n_samples: int = 24, thr: float = 0.2) -> tuple[float, float, float]:
    """Mean, min and above-threshold fraction of the probability along the bridge.

    ``bridge_evidence`` returns the mean only. The MINIMUM is what separates "a microtubule
    that dipped below the binarisation threshold" from "two filaments with background between
    them": a real continuation is weak everywhere along the bridge, a spurious one is strong at
    both ends and empty in the middle, and both have similar means.
    """
    if prob is None:
        return 0.0, 0.0, 0.0
    h, w = prob.shape
    t = np.linspace(0.0, 1.0, max(n_samples, 2))[:, None]
    pts = np.asarray(a.pos, float)[None, :] * (1 - t) + np.asarray(b.pos, float)[None, :] * t
    cc = np.rint(pts[:, 0]).astype(int)
    rr = np.rint(pts[:, 1]).astype(int)
    ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
    if not ok.any():
        return 0.0, 0.0, 0.0
    v = prob[rr[ok], cc[ok]]
    return float(v.mean()), float(v.min()), float((v >= thr).mean())


def edge_features(a: ArmEnd, b: ArmEnd, *, prob: np.ndarray | None = None,
                  gap_floor: float = 4.0, n_arms: int = 2,
                  is_gap_link: bool = False) -> np.ndarray:
    """Feature vector for the candidate continuation ``a -> b``.

    Symmetric in ``a`` and ``b`` by construction: the only asymmetric quantities (the two arc
    lengths) are sorted, so the scorer cannot depend on the order the junction happened to
    enumerate its arms in.
    """
    d = np.asarray(b.pos, float) - np.asarray(a.pos, float)
    L = float(np.linalg.norm(d))
    span = max(L, gap_floor)
    direct = turn_penalty(a.theta + np.pi, b.theta)
    if L >= gap_floor:
        ang_d = float(np.arctan2(d[1], d[0]))
        total = turn_penalty(a.theta + np.pi, ang_d) + turn_penalty(ang_d, b.theta)
    else:
        total = direct
    bm, bmin, bfrac = _bridge_stats(prob, a, b)
    la = float(getattr(a, "arc_len", 0.0) or 0.0)
    lb = float(getattr(b, "arc_len", 0.0) or 0.0)
    lo, hi = (la, lb) if la <= lb else (lb, la)
    return np.array([
        total, direct, abs(a.kappa + b.kappa), abs(a.kappa) + abs(b.kappa),
        L, total / max(span, 1e-6), _ori_mismatch(a, b),
        bm, bmin, bfrac, lo, hi, float(n_arms), float(is_gap_link),
    ], dtype=np.float32)


class Linker:
    """Two-hidden-layer MLP with a sigmoid output: P(the two arms are the same microtubule)."""

    def __init__(self, weights: dict):
        self.W = [weights[f"W{i}"] for i in range(3)]
        self.b = [weights[f"b{i}"] for i in range(3)]
        self.mu = weights["mu"]
        self.sd = weights["sd"]
        n = int(weights["n_features"][0]) if "n_features" in weights else N_FEATURES
        if n != N_FEATURES:
            raise ValueError(f"linker checkpoint has {n} features, code expects {N_FEATURES}; "
                             "the feature order changed and the two must not be mixed")

    @classmethod
    def load(cls, path: str) -> "Linker":
        with np.load(path) as z:
            return cls({k: z[k] for k in z.files})

    def score(self, feats: np.ndarray) -> np.ndarray:
        """``feats`` is (N, N_FEATURES); returns (N,) probabilities."""
        x = (np.atleast_2d(feats) - self.mu) / self.sd
        for i in range(2):
            x = np.maximum(x @ self.W[i] + self.b[i], 0.0)
        z = (x @ self.W[2] + self.b[2]).ravel()
        return 1.0 / (1.0 + np.exp(-z))


_CACHE: dict = {}


def load_default(path: str | None = None) -> "Linker | None":
    """Load the shipped linker, or return None so callers fall back to the geometric cost."""
    p = path or os.path.join(os.path.dirname(__file__), "linker_a.npz")
    if p in _CACHE:
        return _CACHE[p]
    m = Linker.load(p) if os.path.exists(p) else None
    _CACHE[p] = m
    return m
