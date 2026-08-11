import textwrap

import h5py
import numpy as np

from mt_bench.cvat_import import assign_split, parse_cvat_xml, write_frame_h5

XML = textwrap.dedent("""\
<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta><task><id>1</id></task></meta>
  <image id="0" name="a.tif" width="100" height="50">
    <polyline label="microtubule" source="manual" points="1.00,2.00;3.00,4.00"/>
    <polyline label="microtubule" source="file" points="5.50,6.50;7.00,8.00;9.00,10.00"/>
  </image>
  <image id="1" name="b.tif" width="100" height="50"/>
</annotations>
""")


def test_parse_keeps_x_col_y_row_order(tmp_path):
    p = tmp_path / "ann.xml"
    p.write_text(XML)
    frames = parse_cvat_xml(str(p))
    assert len(frames) == 2
    f0 = frames[0]
    assert f0["name"] == "a.tif" and f0["width"] == 100 and f0["height"] == 50
    assert len(f0["polylines"]) == 2
    # first vertex is (x=1, y=2): x is the COLUMN, y is the ROW
    np.testing.assert_allclose(f0["polylines"][0], [[1.0, 2.0], [3.0, 4.0]])
    assert f0["sources"] == ["manual", "file"]
    assert frames[1]["polylines"] == []


def test_write_frame_h5_roundtrip(tmp_path):
    img = np.arange(50 * 100, dtype=np.float32).reshape(50, 100)
    pls = [np.array([[1.0, 2.0], [3.0, 4.0]])]
    out = tmp_path / "f.h5"
    write_frame_h5(str(out), img, pls, {"split": "val"})
    with h5py.File(out, "r") as h:
        assert h.attrs["height"] == 50 and h.attrs["width"] == 100
        assert h.attrs["n_polylines"] == 1 and h.attrs["split"] == "val"
        np.testing.assert_allclose(h["polylines/pl_0000"][:], pls[0])
        assert h["image"].shape == (50, 100)


def test_assign_split_alternates_over_sorted_names():
    got = assign_split(["c", "a", "b", "d"])
    assert got == {"a": "val", "b": "test", "c": "val", "d": "test"}
