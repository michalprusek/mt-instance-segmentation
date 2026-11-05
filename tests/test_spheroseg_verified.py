import numpy as np

from mt_bench.spheroseg_verified import (canonical_frame, classify, id_class, pick_proposals,
                                         to_xy)


def _poly(pid, pts):
    return {"id": pid, "points": [{"x": float(x), "y": float(y)} for x, y in pts]}


def _line(x0, x1, y, n=21):
    return np.stack([np.linspace(x0, x1, n), np.full(n, float(y))], axis=1)


def test_id_class_reads_the_hub_id_grammar():
    assert id_class("polyline_7") == "model"
    assert id_class("polyline_1728374650123_a_x9k") == "split"
    assert id_class("polygon_1728374650123_nkp9mx") == "drawn"
    assert id_class("0f3a9c1e-12ab-4cde-9f00-0123456789ab") == "copy"
    assert id_class("something_else") == "other"


def test_to_xy_keeps_x_col_y_row_order():
    a = to_xy(_poly("polyline_1", [(10, 2), (30, 4)]))
    # x is the COLUMN and comes first; a transpose here silently moves every curve
    assert a[0].tolist() == [10.0, 2.0] and a[1].tolist() == [30.0, 4.0]


def test_every_fate_is_told_apart():
    props = [_line(0, 100, 10), _line(0, 100, 30), _line(0, 100, 50), _line(0, 100, 70),
             _line(0, 100, 90), _line(0, 100, 110), _line(0, 40, 130)]
    bent = _line(0, 100, 70)
    bent[15:, 1] += 6.0                                              # one end dragged 6 px away
    stored = [
        _poly("polyline_1", props[0]),                               # untouched
        _poly("polyline_2", _line(0, 60, 30)),                       # end pulled back
        _poly("polyline_3", _line(0, 160, 50, n=33)),                # end pulled on, past the proposal
        _poly("polyline_4", bent),                                   # vertices moved
        _poly("polyline_1728374650123_a_abc", _line(0, 45, 90)),     # split in two
        _poly("polyline_1728374650123_b_def", _line(55, 100, 90)),
        _poly("polygon_1728374650999_zzz", _line(0, 100, 150)),      # drawn where nothing was proposed
        _poly("polygon_1728374651999_yyy", _line(0, 40, 131)),       # drawn over a proposal -> replaces it
    ]
    s, p = classify(stored, props)
    assert [x["fate"] for x in s] == ["kept", "trimmed", "extended", "reshaped",
                                      "split", "split", "drawn", "drawn"]
    assert s[0]["exact"] and not s[1]["exact"]
    assert [x["fate"] for x in p] == ["kept", "trimmed", "extended", "reshaped",
                                      "split", "deleted", "absorbed"]
    assert s[6]["stored_on_proposal"] == 0.0 and s[7]["stored_on_proposal"] > 0.95


def test_an_index_shift_does_not_turn_kept_curves_into_edits():
    # the regenerated list has one extra marginal fragment at the front: every id is off by one
    stored = [_poly(f"polyline_{i + 1}", _line(0, 100, 10 + 20 * i)) for i in range(4)]
    props = [_line(200, 215, 5)] + [_line(0, 100, 10 + 20 * i) + 0.2 for i in range(4)]
    s, p = classify(stored, props)
    assert [x["fate"] for x in s] == ["kept"] * 4
    assert [x["proposal"] for x in s] == [1, 2, 3, 4]
    assert p[0]["fate"] == "deleted"          # which is why an inexact re-inference is a caveat


def test_pick_proposals_prefers_the_list_that_reproduces_vertices():
    stored = [_poly(f"polyline_{i + 1}", _line(0, 100, 10 + 20 * i)) for i in range(3)]
    right = [_line(0, 100, 10 + 20 * i) for i in range(3)]
    wrong = [_line(0, 100, 11 + 20 * i) for i in range(3)]
    assert pick_proposals(stored, {"wrong": wrong, "right": right}) == ("right", 3)


def test_canonical_frame_is_the_edited_one_not_the_richest_copy():
    edited = {"frame_index": 0, "polygons": [_poly("polyline_1", _line(0, 9, 1)),
                                             _poly("polygon_1728374650999_zzz", _line(0, 9, 5))]}
    # the untouched proposal plus one propagated copy: more curves, more editor ids, not the edit
    sibling = {"frame_index": 1, "polygons": [
        _poly("polyline_1", _line(0, 9, 1)), _poly("polyline_2", _line(0, 9, 3)),
        _poly("polyline_3", _line(0, 9, 7)),
        _poly("0f3a9c1e-12ab-4cde-9f00-0123456789ab", _line(0, 9, 5))]}
    assert canonical_frame([sibling, edited]) is edited
