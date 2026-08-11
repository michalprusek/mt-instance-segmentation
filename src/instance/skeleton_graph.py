"""Skeleton -> junction-contracted arc graph.

The representation PySOAX lacks. Three things happen here that its tracer does not do:

1. **Junction-cluster contraction.** ``skeletonize`` rarely turns an X into one degree-4
   node; it usually produces two degree-3 nodes joined by a short bridge (a Y-Y pattern).
   Tracing through that bridge systematically biases the pairing and injects a kink exactly
   where the physics forbids one. Dilating the junction pixels by ``merge_radius`` merges
   them into a single junction, and the distorted neighbourhood is removed from the arcs.
2. **Arcs are extracted as whole degree-2 chains**, never consumed pixel by pixel, so there
   is no first-come-first-served race for junction pixels.
3. **Spurs are pruned** by keeping only the diameter path of each arc component -- short
   skeleton hairs off a filament would otherwise become spurious arms.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

_NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


@dataclass
class ArcGraph:
    """Arcs (maximal degree-2 chains) plus the junctions they attach to.

    ``arcs[k]`` is an ``(N, 2)`` array of ``(x=col, y=row)``.
    ``junctions[j]`` is the ``(x, y)`` centroid of junction cluster ``j``.
    ``arc_ends[k]`` is ``(j_start, j_end)``; ``None`` means a free endpoint.
    """
    arcs: list = field(default_factory=list)
    junctions: list = field(default_factory=list)
    arc_ends: list = field(default_factory=list)


def _disk(radius: float) -> np.ndarray:
    r = int(np.ceil(radius))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (yy ** 2 + xx ** 2) <= radius ** 2 + 1e-9


def _component_path(coords: np.ndarray) -> np.ndarray:
    """Order one thin connected component into a path, dropping spurs.

    Uses the classic double sweep: the longest shortest-path between two far-apart pixels.
    Any side branch is simply not on that path, which is how spurs get pruned.
    """
    if len(coords) == 1:
        return coords.astype(float)
    index = {tuple(c): i for i, c in enumerate(coords)}
    g = nx.Graph()
    g.add_nodes_from(range(len(coords)))
    for i, (r, c) in enumerate(coords):
        for dr, dc in _NB8:
            j = index.get((r + dr, c + dc))
            if j is not None and j > i:
                g.add_edge(i, j, weight=float(np.hypot(dr, dc)))
    if g.number_of_edges() == 0:
        return coords[:1].astype(float)

    def farthest(src):
        dist = nx.single_source_dijkstra_path_length(g, src)
        return max(dist.items(), key=lambda kv: kv[1])[0]

    a = farthest(0)
    b = farthest(a)
    path = nx.shortest_path(g, a, b, weight="weight")
    return coords[path].astype(float)


def build_arc_graph(mask: np.ndarray, merge_radius: float = 3.0,
                    min_arc_len: int = 3,
                    bridge_max_len: float = 18.0) -> ArcGraph:
    """Build the arc graph of a binary foreground mask.

    ``merge_radius`` contracts nearby junction pixels into one junction; ``bridge_max_len``
    additionally merges junctions joined by a short stub (see
    :func:`_absorb_crossing_bridges`). ``min_arc_len`` drops arc components shorter than this
    many pixels.
    """
    skel = skeletonize(np.asarray(mask, dtype=bool))
    if not skel.any():
        return ArcGraph()

    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    nb = ndimage.convolve(skel.astype(np.uint8), kernel, mode="constant") * skel
    junction_px = skel & (nb >= 3)

    graph = ArcGraph()
    junction_labels = np.zeros(skel.shape, dtype=np.int32)

    if junction_px.any():
        grown = ndimage.binary_dilation(junction_px, structure=_disk(merge_radius))
        junction_labels, n_j = ndimage.label(grown & skel,
                                             structure=np.ones((3, 3), dtype=int))
        for jid in range(1, n_j + 1):
            core = junction_px & (junction_labels == jid)
            src = core if core.any() else (junction_labels == jid)
            rr, cc = np.where(src)
            graph.junctions.append(np.array([cc.mean(), rr.mean()], dtype=float))
    else:
        n_j = 0

    arc_px = skel & (junction_labels == 0)
    lab, n_arc = ndimage.label(arc_px, structure=np.ones((3, 3), dtype=int))

    # KD-tree over junction-region pixels, so an arc end can be attached to the junction
    # it actually touches rather than to the nearest centroid (which for a long, curved
    # junction cluster can be a different one).
    if n_j:
        jr, jc = np.where(junction_labels > 0)
        jtree = cKDTree(np.stack([jc, jr], axis=1))
        jids = junction_labels[jr, jc]
    else:
        jtree, jids = None, None

    attach_radius = merge_radius + 1.5

    for aid in range(1, n_arc + 1):
        coords = np.argwhere(lab == aid)          # (row, col)
        if len(coords) < min_arc_len:
            continue
        path_rc = _component_path(coords)
        if len(path_rc) < min_arc_len:
            continue
        arc_xy = np.stack([path_rc[:, 1], path_rc[:, 0]], axis=1)

        ends = []
        for endpoint in (arc_xy[0], arc_xy[-1]):
            j = None
            if jtree is not None:
                d, k = jtree.query(endpoint[None, :], k=1)
                if float(d[0]) <= attach_radius:
                    j = int(jids[int(k[0])]) - 1
            ends.append(j)

        graph.arcs.append(arc_xy)
        graph.arc_ends.append((ends[0], ends[1]))

    graph = _absorb_crossing_bridges(graph, bridge_max_len)

    # Drop junctions no surviving arc attaches to, and renumber.
    used = sorted({e for ends in graph.arc_ends for e in ends if e is not None})
    if len(used) != len(graph.junctions):
        remap = {old: new for new, old in enumerate(used)}
        graph.junctions = [graph.junctions[o] for o in used]
        graph.arc_ends = [(remap.get(a), remap.get(b)) for a, b in graph.arc_ends]
    return graph


def _absorb_crossing_bridges(graph: ArcGraph, bridge_max_len: float) -> ArcGraph:
    """Merge junctions joined by a SHORT junction-to-junction arc.

    Radius-based contraction handles the Y-Y bridge of a near-perpendicular crossing, but at
    shallow angles the two filaments stay skeletally FUSED over a long stretch that no sane
    merge radius reaches. The fused length has a closed form: two bands of half-width ``r``
    crossing at angle ``alpha`` overlap while their centerline separation is under ``2r``,
    i.e. over

        L_fuse ~= 4 * r / sin(alpha)

    -- 15.5 px for ``r=1`` at 15 degrees (measured: 14.4). Such a stub is a crossing bridge,
    not a microtubule segment: it has a junction at BOTH ends and is shorter than any real
    filament worth instancing, so its two junctions are one crossing and get merged.

    ``bridge_max_len`` should therefore be set to ``~4 * half_width / sin(alpha_min)`` for the
    shallowest crossing angle to be resolved. It is a genuine trade-off, not a free
    parameter: raising it also absorbs real short segments between two nearby crossings,
    which matters in dense frames (the MT-34 task-586 frames average 32 crossings each).
    A LONG junction-to-junction arc is a genuine segment and is always kept.
    """
    if not graph.junctions:
        return graph

    parent = list(range(len(graph.junctions)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    absorbed = []
    for k, (j0, j1) in enumerate(graph.arc_ends):
        if j0 is None or j1 is None:
            continue
        length = float(np.linalg.norm(np.diff(graph.arcs[k], axis=0), axis=1).sum())
        if length <= bridge_max_len:
            absorbed.append(k)
            if j0 != j1:
                union(j0, j1)

    if not absorbed:
        return graph

    groups: dict[int, list[int]] = {}
    for j in range(len(graph.junctions)):
        groups.setdefault(find(j), []).append(j)
    order = sorted(groups)
    new_id = {root: i for i, root in enumerate(order)}

    merged = ArcGraph()
    for root in order:
        pts = [graph.junctions[j] for j in groups[root]]
        merged.junctions.append(np.mean(pts, axis=0))
    for k, arc in enumerate(graph.arcs):
        if k in absorbed:
            continue
        j0, j1 = graph.arc_ends[k]
        merged.arcs.append(arc)
        merged.arc_ends.append((None if j0 is None else new_id[find(j0)],
                                None if j1 is None else new_id[find(j1)]))
    return merged
