import numpy as np

from instance.oracle import oracle_mask
from instance.skeleton_graph import build_arc_graph


def _mask(polylines, shape=(80, 80)):
    return oracle_mask(polylines, shape, half_width=1.0, up=1.0)


def test_isolated_line_yields_one_arc_and_no_junctions():
    g = build_arc_graph(_mask([np.array([[10.0, 40.0], [70.0, 40.0]])]))
    assert len(g.arcs) == 1 and len(g.junctions) == 0
    assert g.arc_ends[0] == (None, None)


def test_two_separate_lines_yield_two_arcs():
    g = build_arc_graph(_mask([np.array([[10.0, 20.0], [70.0, 20.0]]),
                               np.array([[10.0, 60.0], [70.0, 60.0]])]))
    assert len(g.arcs) == 2 and len(g.junctions) == 0


def test_x_crossing_yields_one_junction_and_four_arms():
    h = np.array([[10.0, 40.0], [70.0, 40.0]])
    v = np.array([[40.0, 10.0], [40.0, 70.0]])
    g = build_arc_graph(_mask([h, v]))
    assert len(g.junctions) == 1, "the Y-Y bridge must be contracted into ONE junction"
    arms = sum(1 for ends in g.arc_ends for e in ends if e == 0)
    assert arms == 4


def test_shallow_crossing_also_contracts_to_one_junction():
    ang = np.deg2rad(20.0)
    c = np.array([40.0, 40.0])
    a = np.array([c - [30, 0], c + [30, 0]])
    b = np.array([c - [30 * np.cos(ang), 30 * np.sin(ang)],
                  c + [30 * np.cos(ang), 30 * np.sin(ang)]])
    g = build_arc_graph(_mask([a, b]), merge_radius=4.0)
    assert len(g.junctions) == 1


def test_arc_vertices_are_x_col_y_row():
    g = build_arc_graph(_mask([np.array([[10.0, 40.0], [70.0, 40.0]])]))
    arc = g.arcs[0]
    assert abs(arc[:, 1].mean() - 40.0) < 2.0   # y (row) constant at 40
    assert np.ptp(arc[:, 0]) > 50.0             # x (col) spans the line


def test_long_segment_between_two_crossings_is_kept():
    # Two crossings 40 px apart on one horizontal filament: the segment between them is a
    # real microtubule stretch and must NOT be absorbed as a crossing bridge.
    h = np.array([[5.0, 40.0], [115.0, 40.0]])
    v1 = np.array([[40.0, 10.0], [40.0, 70.0]])
    v2 = np.array([[80.0, 10.0], [80.0, 70.0]])
    g = build_arc_graph(_mask([h, v1, v2], shape=(120, 120)))  # default bridge_max_len
    assert len(g.junctions) == 2, "two distinct crossings must stay distinct"
    middles = [k for k, (a, b) in enumerate(g.arc_ends)
               if a is not None and b is not None]
    assert len(middles) == 1
    seg = g.arcs[middles[0]]
    assert np.linalg.norm(np.diff(seg, axis=0), axis=1).sum() > 25.0


def test_t_junction_yields_three_arms():
    stem = np.array([[40.0, 40.0], [40.0, 72.0]])
    through = np.array([[8.0, 40.0], [72.0, 40.0]])
    g = build_arc_graph(_mask([through, stem]))
    assert len(g.junctions) == 1
    arms = sum(1 for ends in g.arc_ends for e in ends if e == 0)
    assert arms == 3


def test_short_spur_is_dropped():
    line = np.array([[10.0, 40.0], [70.0, 40.0]])
    spur = np.array([[40.0, 40.0], [40.0, 45.0]])
    g = build_arc_graph(_mask([line, spur]), min_arc_len=8)
    total = sum(len(a) for a in g.arcs)
    assert total > 50 and all(len(a) >= 8 for a in g.arcs)
