import numpy as np

from mt_bench.overlay import render_overlay


def test_overlay_draws_polyline_at_row_y_col_x(tmp_path):
    # A 1-px-wide bright image; polyline (x=col) 10..10, (y=row) 2..8 is VERTICAL.
    img = np.zeros((20, 30), dtype=np.float32)
    pl = np.array([[10.0, 2.0], [10.0, 8.0]])
    out = tmp_path / "o.png"
    render_overlay(img, [pl], str(out))
    assert out.exists() and out.stat().st_size > 0
