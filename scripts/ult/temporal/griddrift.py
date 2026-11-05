"""Best translation between two frames' centerlines by exhaustive search; the independent check on estimate_drift (docs/TODO.md 2026-11-03 14:45).
Maximises symmetric overlap at tol on distance transforms: A + t against B, B - t against A. Ties go to the smallest shift, so an empty field stays at zero."""
import numpy as np
from scipy import ndimage as ndi


def _dt(P, lo, shape):
    r = np.ones(shape, bool); q = np.rint(P - lo).astype(int)
    r[q[:, 0], q[:, 1]] = False
    return ndi.distance_transform_edt(r)


def best_shift(A, B, tol=5 / 1.5, R=60.0):
    if len(A) == 0 or len(B) == 0: return np.zeros(2)
    A, B = np.asarray(A, float), np.asarray(B, float)
    M = int(R) + 8; lo = np.floor(np.minimum(A.min(0), B.min(0))) - M; hi = np.ceil(np.maximum(A.max(0), B.max(0))) + M
    shape = tuple((hi - lo).astype(int) + 1); dB, dA = _dt(B, lo, shape), _dt(A, lo, shape)
    a, b = A[::3] - lo, B[::3] - lo
    def score(t):
        qa = np.rint(a + t).astype(int); qb = np.rint(b - t).astype(int)
        return ((dB[qa[:, 0], qa[:, 1]] <= tol).mean() + (dA[qb[:, 0], qb[:, 1]] <= tol).mean()) / 2 - 1e-6 * np.hypot(*t)
    def search(cy, cx, half, step):
        best, arg = -1.0, np.zeros(2)
        for y in np.arange(cy - half, cy + half + step / 2, step):
            for x in np.arange(cx - half, cx + half + step / 2, step):
                s = score(np.array([y, x]))
                if s > best: best, arg = s, np.array([y, x])
        return arg
    t = search(0.0, 0.0, R, 2.0)
    return search(t[0], t[1], 2.0, 0.5)
