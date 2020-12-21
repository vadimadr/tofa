from tofa.bbox_utils import bbox_area, bbox_intersection, bbox_iou

import numpy as np


def test_bbox_intersection():
    boxes = [(0, 1, 2, 3), (3, 2.5, 4, 3.5), (3.5, 3, 5, 6)]
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

    bbox_intersection(boxes[1], boxes[2])


def test_bbox_area():
    boxes = [(0, 1, 2, 3), (3, 2.5, 4, 3.5), (3.5, 3, 5, 6)]
    assert bbox_area(boxes).shape == (len(boxes),)
    area = bbox_area(boxes[0])
    assert isinstance(area, float)


def test_bbox_iou():
    boxes = [(0, 1, 2, 3), (3, 2.5, 4, 3.5), (3.5, 3, 5, 6)]
    nd_boxes = np.array(boxes)

    iou_mat = bbox_iou(nd_boxes, nd_boxes)
    assert iou_mat.shape == (3, 3)
    assert bbox_iou(nd_boxes, nd_boxes[1:]).shape == (3, 2)

    iou_mat2 = bbox_iou(nd_boxes, boxes[1])
    assert iou_mat2.shape == (3, 1)
    np.testing.assert_allclose(iou_mat2[:, 0], iou_mat[:, 1])

    iou_mat3 = bbox_iou(boxes[1], nd_boxes)
    assert iou_mat3.shape == (1, 3)
    np.testing.assert_allclose(iou_mat3[0, :], iou_mat[1, :])

    iou = bbox_iou(boxes[1], boxes[2])
    assert isinstance(iou, float)
