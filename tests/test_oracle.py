import numpy as np

from instance.oracle import (oracle_instance_masks, oracle_mask,
                             oracle_ori_channels)


def test_oracle_mask_is_in_upscaled_frame():
    m = oracle_mask([np.array([[0.0, 5.0], [20.0, 5.0]])], (30, 40), up=1.5)
    assert m.shape == (45, 60) and m.any()


def test_oracle_mask_marks_row_y_times_up():
    m = oracle_mask([np.array([[0.0, 5.0], [39.0, 5.0]])], (30, 40), half_width=1.0, up=1.5)
    rows = np.where(m.any(axis=1))[0]
    assert abs(rows.mean() - 7.5) < 1.6   # y=5 -> row 7.5 after 1.5x


def test_oracle_mask_half_width_one_gives_a_three_pixel_band():
    m = oracle_mask([np.array([[2.0, 20.0], [58.0, 20.0]])], (40, 60),
                    half_width=1.0, up=1.0)
    col = m[:, 30]
    assert col.sum() == 3


def test_instance_masks_are_disjoint_for_separated_lines():
    a = np.array([[2.0, 10.0], [58.0, 10.0]])
    b = np.array([[2.0, 30.0], [58.0, 30.0]])
    masks = oracle_instance_masks([a, b], (40, 60), up=1.0)
    assert len(masks) == 2
    assert not (masks[0] & masks[1]).any()


def test_crossing_writes_into_two_orientation_channels():
    horiz = np.array([[0.0, 25.0], [49.0, 25.0]])
    vert = np.array([[25.0, 0.0], [25.0, 49.0]])
    ch = oracle_ori_channels([horiz, vert], (50, 50), K=6, up=1.5)
    r = int(round(25 * 1.5))
    c = int(round(25 * 1.5))
    lit = (ch[:, r - 2:r + 3, c - 2:c + 3].max(axis=(1, 2)) > 0.5).sum()
    assert lit >= 2


def test_orientation_channels_union_equals_the_binary_mask():
    pls = [np.array([[2.0, 10.0], [58.0, 30.0]]), np.array([[10.0, 2.0], [30.0, 38.0]])]
    ch = oracle_ori_channels(pls, (40, 60), K=6, up=1.0)
    m = oracle_mask(pls, (40, 60), up=1.0)
    assert np.array_equal(ch.max(axis=0) > 0.5, m)
