import pytest

from tofa.bbox_utils import (
    bbox_area,
    bbox_center_to_corner,
    bbox_corner_to_center,
    bbox_intersection,
    bbox_iou,
    clip_bbox,
    scale_bbox,
    shift_bbox,
)

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


def test_bbox_corner_to_center():
    boxes = [(1, 1, 3, 3), (2, 3, 5, 7)]
    boxes_center = bbox_corner_to_center(boxes)
    expect = np.array([[2, 2, 2, 2], [3.5, 5, 3, 4]])
    assert boxes_center.shape == (2, 4)
    np.testing.assert_almost_equal(boxes_center, expect)

    assert bbox_corner_to_center(boxes[1]) == (3.5, 5, 3, 4)


def test_bbox_center_to_corner():
    boxes = [(2, 2, 2, 2), (3.5, 5, 3, 4)]
    boxes_corner = bbox_center_to_corner(boxes)
    expect = np.array([[1, 1, 3, 3], [2, 3, 5, 7]])
    assert boxes_corner.shape == (2, 4)
    np.testing.assert_almost_equal(boxes_corner, expect)

    assert bbox_center_to_corner(boxes[1]) == (2, 3, 5, 7)


def test_shift_bbox():
    boxes = [(1, 1, 3, 3), (2, 3, 5, 7)]
    boxes_shifted = shift_bbox(boxes, 2, 3)
    expect = np.array([[3, 4, 5, 6], [4, 6, 7, 10]])
    np.testing.assert_almost_equal(boxes_shifted, expect)
    assert boxes_shifted.shape == (2, 4)


def test_scale_bbox():
    boxes = [(1, 1, 3, 3), (2, 3, 5, 7)]
    boxes_scaleed = scale_bbox(boxes, 2)
    expect = np.array([[0, 0, 4, 4], [0.5, 1, 6.5, 9]])
    np.testing.assert_almost_equal(boxes_scaleed, expect)
    assert boxes_scaleed.shape == (2, 4)


def test_clip_bbox():
    boxes = [(1, 1, 3, 3), (2, 3, 5, 7)]
    boxes_cliped = clip_bbox(boxes, 4, 4)
    expect = np.array([[1, 1, 3, 3], [2, 3, 4, 4]])
    np.testing.assert_almost_equal(boxes_cliped, expect)
    assert boxes_cliped.shape == (2, 4)
