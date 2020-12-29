import pytest

from tofa.bbox_utils import bbox_area, bbox_intersection, bbox_iou

import numpy as np


def test_bbox_intersection():
    boxes = [(1, 1, 3, 3), (2, 3, 5, 7), (3, 2, 7, 5)]
    nd_boxes = np.array(boxes)
    int_mat = bbox_intersection(nd_boxes, nd_boxes)
    assert int_mat.shape == (3, 3, 4)
    assert bbox_intersection(nd_boxes, nd_boxes[1:]).shape == (3, 2, 4)

    int_mat2 = bbox_intersection(nd_boxes, boxes[1])
    assert int_mat2.shape == (3, 1, 4)
    np.testing.assert_allclose(int_mat2[:, 0, :], int_mat[:, 1, :])

    int_mat3 = bbox_intersection(boxes[1], nd_boxes)
    assert int_mat3.shape == (1, 3, 4)
    np.testing.assert_allclose(int_mat3[0, :, :], int_mat[1, :, :])


def test_bbox_area():
    boxes = [(1, 1, 3, 3), (2, 3, 5, 7), (3, 2, 7, 5), (6, 6, 8, 8)]
    areas = bbox_area(boxes)

    assert areas.shape == (4,)
    np.testing.assert_allclose(areas, [4, 12, 12, 4])

    area = bbox_area(boxes[0])
    assert isinstance(area, float)


def test_bbox_iou():
    boxes = [(1, 1, 3, 3), (2, 3, 5, 7), (3, 2, 7, 5), (6, 6, 8, 8)]
    nd_boxes = np.array(boxes)

    # all vs all
    iou_mat = bbox_iou(nd_boxes, nd_boxes)
    expect = np.array([[1, 0, 0, 0], [0, 1, 0.2, 0], [0, 0.2, 1, 0], [0, 0, 0, 1]])
    np.testing.assert_almost_equal(iou_mat, expect)
    assert iou_mat.shape == (4, 4)

    # all vs one
    iou_mat2 = bbox_iou(nd_boxes, boxes[1])
    assert iou_mat2.shape == (4, 1)
    np.testing.assert_almost_equal(iou_mat2[:, 0], [0, 1.0, 0.2, 0])

    iou_mat3 = bbox_iou(boxes[1], nd_boxes)
    assert iou_mat3.shape == (1, 4)
    np.testing.assert_allclose(iou_mat3[0, :], [0, 1, 0.2, 0])

    # one vs one
    iou = bbox_iou(boxes[1], boxes[2])
    assert isinstance(iou, float)
    assert pytest.approx(0.2) == iou
