"""Instancer B: curvature-constrained beam tracing in the orientation-lifted graph.

Where instancer A works on a single-layer skeleton and therefore has to ASSIGN each shared
pixel to one microtubule, B traces in ``(pixel, direction)`` space, so a crossing pixel exists
once per orientation slice. Consuming a traced path removes only the orientation slices it
actually used -- the crossing filament keeps its own slice and survives at full length. That
is the one failure cause A structurally cannot fix.

Three design decisions worth stating, because each replaces something more obvious:

* **Beam search, not Dijkstra.** In 2D a greedy tracer fails because a junction offers several
  plausible continuations. In the lifted graph it does not: successors are restricted to
  ``|dtheta| <= kappa_max * step``, which already excludes the crossing filament. The
  ambiguity Dijkstra would pay for was removed by the representation. A full Dijkstra over
  ~500k nodes costs ~60 s per frame; this runs in seconds.
* **Bidirectional tracing, not tip detection.** Deciding whether a node is a filament tip
  means probing its predecessors, which is O(all nodes) of Python-level work per frame.
  Tracing forward from ``(p, d)`` AND from the reversed ``(p, d + K)`` and joining the two
  makes the seed's position along the filament irrelevant, so any node will do.
* **Consumption by ANGLE, not by bin index.** After ``refine_theta`` a single filament lights
  2-3 adjacent bins, so consuming only the exact traced bin leaves the same microtubule
  traceable again in its neighbour -- duplicate instances. Everything within ``consume_deg``
  of the local traced heading is consumed instead. This sets the method's honest resolution
  limit: two filaments crossing at less than ``consume_deg`` are absorbed into one, the same
  shallow-angle wall instancer A hits from the other side.
"""
from __future__ import annotations

import json
import os

import numpy as np

from instance.geometry import total_length, wrap_angle
from instance.instancer_a import enforce_curvature
from instance.lifted import LiftedGraph, refine_theta
from instance.oracle import oracle_instance_masks

DEFAULTS = {
    "K_out": 18,
    "prob_thr": 0.3,
    "max_step": 2.5,
    "dir_tol_deg": 65.0,
    "lam": 4.0,
    "beam": 4,
    "consume_deg": 25.0,
    "consume_radius": 2.0,
    "min_length": 15.0,
    "ds": 2.0,
    "half_width": 1.0,
    "max_instances": 600,
}


def _trace_forward(graph: LiftedGraph, seed: tuple[int, int], alive: np.ndarray,
                   beam: int, max_steps: int):
    """Beam-search the longest valid continuation from one lifted node.

    Every beam entry has taken the same number of steps, so comparing accumulated cost
    compares like with like. Returns the node list of the best path (including the seed).
    """
    beams = [(0.0, 0.0, [seed], {seed})]
    finished = []
    for _ in range(max_steps):
        nxt = []
        for cost, length, path, used in beams:
            pix_i, d = path[-1]
            extended = False
            for j, d2, L, _dth, c in graph.successors(pix_i, d, alive):
                if (j, d2) in used:
                    continue
                extended = True
                nxt.append((cost + c, length + L, path + [(j, d2)], used | {(j, d2)}))
            if not extended:
                finished.append((cost, length, path))
        if not nxt:
            break
        nxt.sort(key=lambda t: t[0])
        beams = nxt[:beam]
    finished.extend((c, ln, p) for c, ln, p, _ in beams)
    if not finished:
        return [seed]
    # Longest wins; cost only breaks ties. A filament is defined by extent, and the beam has
    # already discarded the expensive ways of reaching that extent.
    return max(finished, key=lambda t: (t[1], -t[0]))[2]


def _consume(graph: LiftedGraph, path, alive: np.ndarray, consume_deg: float,
             consume_radius: float) -> None:
    """Retire the orientation slices this trace explains, around the whole traced curve.

    Consumption is SPATIAL as well as angular. The beam maximises length, so it happily takes
    2-px steps and skips intervening skeleton pixels; retiring only the visited nodes leaves
    those skipped pixels alive and the same microtubule gets traced a second time as a
    duplicate instance. Every skeleton pixel within ``consume_radius`` of a traced node is
    therefore retired at the traced orientation.
    """
    tol = np.deg2rad(consume_deg)
    for pix_i, d in path:
        phi = graph.phi[d]
        near = (graph.tree.query_ball_point(graph.pixels[pix_i], r=consume_radius)
                if graph.tree is not None else [pix_i])
        for cand in range(graph.n_dir):
            dth = abs(wrap_angle(graph.phi[cand] - phi))
            # Undirected: a heading and its reverse describe the same filament.
            if min(dth, abs(np.pi - dth)) <= tol:
                alive[near, cand] = False


_PARAM_FILE = os.path.join(os.path.dirname(__file__), "params_b.json")


def default_params() -> dict:
    """Defaults, overridden by ``params_b.json`` if a tuning run has written one.

    ``kappa_max`` is dropped if present -- it is an explicit argument to :func:`instance_b`.
    """
    p = dict(DEFAULTS)
    if os.path.exists(_PARAM_FILE):
        with open(_PARAM_FILE) as fh:
            loaded = json.load(fh)
        loaded.pop("kappa_max", None)
        p.update(loaded)
    return p


def instance_b(channels: np.ndarray, kappa_max: float,
               params: dict | None = None) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Instance-segment an orientation-channel stack. Returns ``(polylines, masks)``."""
    p = {**default_params(), **(params or {})}
    chans = refine_theta(np.asarray(channels, dtype=np.float32), int(p["K_out"]))

    graph = LiftedGraph(chans, kappa_max=kappa_max, prob_thr=p["prob_thr"],
                        max_step=p["max_step"], dir_tol_deg=p["dir_tol_deg"],
                        lam=p["lam"])
    if len(graph.pixels) == 0:
        return [], []

    alive = graph.alive_array()
    max_steps = int(3.0 * max(graph.H, graph.W))

    # Seed order: highest-probability nodes first. Because tracing is bidirectional the seed
    # may sit anywhere along a filament, so no tip detection is needed.
    node_p = np.where(alive, graph.prob[np.arange(graph.n_dir) % graph.K][
        :, graph.pixels[:, 0], graph.pixels[:, 1]].T, -1.0)
    order = np.argsort(-node_p, axis=None)

    polylines = []
    for flat in order:
        if len(polylines) >= p["max_instances"]:
            break
        pix_i, d = int(flat // graph.n_dir), int(flat % graph.n_dir)
        if not alive[pix_i, d]:
            continue

        fwd = _trace_forward(graph, (pix_i, d), alive, int(p["beam"]), max_steps)
        back_seed = (pix_i, (d + graph.K) % graph.n_dir)
        bwd = _trace_forward(graph, back_seed, alive, int(p["beam"]), max_steps) \
            if alive[back_seed[0], back_seed[1]] else [back_seed]
        path = [(i, (dd + graph.K) % graph.n_dir) for i, dd in reversed(bwd[1:])] + fwd

        _consume(graph, path, alive, p["consume_deg"], p["consume_radius"])

        rc = graph.pixels[[n[0] for n in path]]
        poly = np.stack([rc[:, 1], rc[:, 0]], axis=1).astype(float)   # -> (x, y)
        if total_length(poly) < p["min_length"]:
            continue
        poly = enforce_curvature(poly, kappa_max, ds=p["ds"])
        if total_length(poly) >= p["min_length"]:
            polylines.append(poly)

    masks = oracle_instance_masks(polylines, (graph.H, graph.W),
                                  half_width=p["half_width"], up=1.0)
    return polylines, masks
