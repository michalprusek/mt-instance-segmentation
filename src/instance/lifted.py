"""The orientation-lifted (x, y, theta) representation.

The K=6 amodal "overpass" channels of ``dino_seg_ori_v4b.pth`` are a learned, discretised
orientation score: at a crossing the two filaments respond in DIFFERENT channels, so the
shared pixel is no longer a single ambiguous site. Lifting them into a graph over
``(pixel, direction)`` is what lets two microtubules occupy one pixel without competing --
the amodal property a single-layer skeleton structurally cannot have.

Directions are stored DOUBLED: ``2 * K_out`` directed bins spanning [0, 360), two per
undirected orientation. A traced path has a heading, so it must not silently reverse; the
probability of a directed bin is simply that of its undirected parent.

This is NOT the earlier per-layer approach that scored F1 0.11. That segmented each
orientation bin independently, so a wavy microtubule sweeping its tangent through all bins
shattered into ~25 arcs. Here a bin transition is a legal, priced edge in ONE joint graph, so
the cut moves from *between bins* to *between instances*.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from instance.geometry import wrap_angle


def refine_theta(channels: np.ndarray, K_out: int = 12) -> np.ndarray:
    """Interpolate a ``(K, H, W)`` orientation stack to ``K_out`` bins.

    K=6 means 30-degree bins, too coarse to separate a shallow crossing: both filaments can
    land in the same bin and the amodal split never happens. Interpolation is circular with
    period 180 degrees (a filament has no head or tail), done on bin CENTRES so the peak
    orientation is preserved.
    """
    K = channels.shape[0]
    if K_out == K:
        return np.asarray(channels, dtype=np.float32)

    in_centres = (np.arange(K) + 0.5) * (180.0 / K)
    out_centres = (np.arange(K_out) + 0.5) * (180.0 / K_out)

    out = np.zeros((K_out, *channels.shape[1:]), dtype=np.float32)
    for m, oc in enumerate(out_centres):
        # Circular distance on a 180-degree period, then linear weights over the two
        # nearest input centres.
        d = np.abs(((in_centres - oc + 90.0) % 180.0) - 90.0)
        order = np.argsort(d)
        i0, i1 = order[0], order[1]
        d0, d1 = d[i0], d[i1]
        if d0 < 1e-9:
            out[m] = channels[i0]
        else:
            w0, w1 = 1.0 / d0, 1.0 / d1
            out[m] = (w0 * channels[i0] + w1 * channels[i1]) / (w0 + w1)
    return out


class LiftedGraph:
    """Nodes ``(row, col, d)`` with ``d`` a directed bin in ``[0, 2*K)``.

    Spatial nodes live on the skeleton of the union foreground, which keeps the graph small
    enough to trace repeatedly. Steps of up to ``max_step`` px are allowed so a path can cut
    straight across the small deformation skeletonisation introduces at a crossing instead of
    being forced around it.
    """

    def __init__(self, channels: np.ndarray, kappa_max: float,
                 prob_thr: float = 0.3, max_step: float = 2.5,
                 dir_tol_deg: float = 65.0, lam: float = 4.0):
        self.K = channels.shape[0]
        self.n_dir = 2 * self.K
        self.prob = np.asarray(channels, dtype=np.float32)
        self.kappa_max = kappa_max
        self.prob_thr = prob_thr
        self.lam = lam
        self.phi = np.arange(self.n_dir) * (2 * np.pi / self.n_dir)
        # Quantisation slack. The smallest non-zero turn the graph can represent is one bin
        # (180/K degrees). At kappa_max=0.25 and a 1 px step the bound allows only 14.3
        # degrees, which is LESS than a 15-degree bin -- so without this slack a curving
        # filament cannot turn at all and shatters into straight fragments. The hard
        # guarantee is not weakened: instancer_b.enforce_curvature still forces the emitted
        # polyline under kappa_max at the 8 px baseline the bound was measured at.
        self.bin_width = np.pi / self.K

        union = self.prob.max(axis=0) > prob_thr
        self.skel = skeletonize(union)
        self.H, self.W = union.shape

        # Which undirected bins are lit at each skeleton pixel.
        rr, cc = np.where(self.skel)
        self.pixels = np.stack([rr, cc], axis=1)
        self.pixel_index = {(int(r), int(c)): i for i, (r, c) in enumerate(self.pixels)}
        lit = self.prob[:, rr, cc] > prob_thr           # (K, N)
        # A skeleton pixel whose union exceeds the threshold but whose individual bins do not
        # would have no node at all; give it its argmax bin so the chain never breaks.
        empty = ~lit.any(axis=0)
        if empty.any():
            lit[self.prob[:, rr, cc].argmax(axis=0)[empty], np.where(empty)[0]] = True
        self.lit = lit

        # Neighbour offsets within max_step, with their length and direction.
        r = int(np.floor(max_step))
        offs = []
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                if dr == 0 and dc == 0:
                    continue
                L = float(np.hypot(dr, dc))
                if L <= max_step:
                    offs.append((dr, dc, L, float(np.arctan2(dr, dc))))
        self.offsets = offs
        self.dir_tol = np.deg2rad(dir_tol_deg)
        self.tree = cKDTree(self.pixels) if len(self.pixels) else None

    def node_prob(self, pix_i: int, d: int) -> float:
        b = d % self.K
        r, c = self.pixels[pix_i]
        return float(self.prob[b, r, c])

    def successors(self, pix_i: int, d: int, alive: np.ndarray):
        """Valid ``(pix_j, d_next, step_len, dtheta, cost)`` continuations."""
        r, c = self.pixels[pix_i]
        phi = self.phi[d]
        out = []
        for dr, dc, L, ang in self.offsets:
            if abs(wrap_angle(ang - phi)) > self.dir_tol:
                continue
            j = self.pixel_index.get((int(r + dr), int(c + dc)))
            if j is None:
                continue
            for b in np.where(self.lit[:, j])[0]:
                for cand in (b, b + self.K):
                    dtheta = abs(wrap_angle(self.phi[cand] - phi))
                    if dtheta > self.kappa_max * L + self.bin_width:
                        continue
                    if not alive[j, cand]:
                        continue
                    p = float(self.prob[b, self.pixels[j][0], self.pixels[j][1]])
                    cost = -np.log(p + 1e-6) * L + self.lam * (dtheta ** 2) / L
                    out.append((j, int(cand), L, dtheta, cost))
        return out

    def alive_array(self) -> np.ndarray:
        """Boolean ``(n_pixels, n_dir)`` of nodes still available for tracing."""
        alive = np.zeros((len(self.pixels), self.n_dir), dtype=bool)
        for b in range(self.K):
            idx = np.where(self.lit[b])[0]
            alive[idx, b] = True
            alive[idx, b + self.K] = True
        return alive
