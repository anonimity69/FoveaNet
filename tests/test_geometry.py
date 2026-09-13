import numpy as np
import pytest

import evaluate as ev
import foveanet as fn


@pytest.mark.parametrize("metric", [ev.iou, fn.box_iou])
def test_iou_identity_disjoint_and_partial(metric):
    assert metric((0, 0, 2, 2), (0, 0, 2, 2)) == 1
    assert metric((0, 0, 2, 2), (3, 3, 4, 4)) == 0
    assert metric((0, 0, 2, 2), (1, 0, 3, 2)) == pytest.approx(1 / 3)
    assert metric((0, 0, 0, 0), (0, 0, 0, 0)) == 0


def test_quadrilateral_ceiling_matches_axis_aligned_case():
    quad = [(0, 0), (4, 0), (4, 2), (0, 2)]
    assert ev.max_iou_quad(quad, 4, 2) == pytest.approx(1)
    assert ev.max_iou_aabb((0, 0, 4, 2), 2, 2) == pytest.approx(0.5)


def test_absent_annotation_and_degenerate_ceiling():
    assert not ev.is_present((np.nan, 0, 1, 1))
    value, mask = ev.relative_iou([0.5, 0], [1, 0])
    assert value == 0.5
    np.testing.assert_array_equal(mask, [True, False])


def test_frame_windows_do_not_duplicate_boundary_events():
    t = np.array([0, 9, 10, 19, 20, 21])
    x = np.arange(len(t))
    frames = list(fn.iter_frames(x, x, t, window=10, min_events=1))
    assert [int(f[0]) for f in frames] == [0, 10, 20]
    np.testing.assert_array_equal(np.concatenate([f[3] for f in frames]), t)
