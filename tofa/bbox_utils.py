import numpy as np

from tofa._typing import array
from tofa.torch_utils import as_numpy

GREEN_COLOR = (255, 0, 0)


def bbox_intersection(bbox1, bbox2):
    nd_boxes1, nd_boxes2 = _bboxes_as_array(bbox1, bbox2)

    minimums = np.minimum(nd_boxes1, nd_boxes2)
    maximums = np.maximum(nd_boxes1, nd_boxes2)
    x1 = minimums[..., 2]
    y1 = minimums[..., 3]
    x0 = np.minimum(x1, maximums[..., 0])
    y0 = np.minimum(y1, maximums[..., 1])

    nd_inters = np.stack([x0, y0, x1, y1], axis=2)
    return nd_inters


def bbox_area(bbox):
    nd_boxes = as_numpy(bbox)
    input_shape = nd_boxes.shape
    nd_boxes = nd_boxes.reshape((-1, 4))
    areas = (nd_boxes[:, 2] - nd_boxes[:, 0]) * (nd_boxes[:, 3] - nd_boxes[:, 1])

    if not isinstance(bbox, array) and len(input_shape) == 1:
        return float(areas)
    return areas.reshape(input_shape[:-1])


def bbox_iou(bbox1, bbox2):
    nd_boxes1, nd_boxes2 = _bboxes_as_array(bbox1, bbox2)

    bbox_inter = bbox_intersection(bbox1, bbox2)
    inter_area = bbox_area(bbox_inter)

    union_area = bbox_area(nd_boxes1) + bbox_area(nd_boxes2) - inter_area
    ious = inter_area / union_area

    inputs_are_arrays = isinstance(bbox1, array) and isinstance(bbox2, array)
    if not inputs_are_arrays and ious.shape == (1, 1):
        return float(ious)
    return ious


def bbox_overlap(bbox1, bbox2):
    bbox_inter = bbox_intersection(bbox1, bbox2)
    return bbox_area(bbox_inter) / bbox_area(bbox1)


def bbox_nms(
    bboxes,
    ranks=None,
    intersection_function=bbox_iou,
    intersection_matrix=None,
    max_intersection=0.5,
):
    if ranks is None:
        ranks = np.arange(bboxes.shape[0])[::-1]
    ranks = sorted(ranks, reverse=True)
    ord = np.argsort(ranks)[::-1]

    if intersection_function in (bbox_iou, bbox_overlap):
        intersection_matrix = intersection_function(bboxes, bboxes)

    selected_boxes = []
    for i in ord:
        for j in selected_boxes:
            if intersection_matrix is not None:
                w = intersection_matrix[i, j]
            else:
                w = intersection_function(bboxes[i], bboxes[j])

            if w > max_intersection:
                break
        else:
            selected_boxes.append(i)

    return [bboxes[i] for i in selected_boxes]


def bbox_corner_to_center(bbox_corner):
    np_bbox_corner = as_numpy(bbox_corner)

    x0, y0, x1, y1 = np_bbox_corner[..., :4].T
    xc_yc_w_h = ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)

    if np_bbox_corner.ndim == 1:
        if isinstance(bbox_corner, array):
            return np.array(xc_yc_w_h)
        return type(bbox_corner)(xc_yc_w_h)
    return np.stack(xc_yc_w_h, axis=1)


def bbox_center_to_corner(bbox_center):
    np_bbox_center = as_numpy(bbox_center)

    xc, yc, w, h = np_bbox_center[..., :4].T
    x0_y0_x1_y1 = (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)

    if np_bbox_center.ndim == 1:
        if isinstance(bbox_center, array):
            return np.array(x0_y0_x1_y1)
        return type(bbox_center)(x0_y0_x1_y1)
    return np.stack(x0_y0_x1_y1, axis=1)


def shift_bbox(boxes, dx, dy=None):
    if dy is None:
        dy = dx
    return as_numpy(boxes) + (dx, dy, dx, dy)


def scale_bbox(bbox, sx, sy=None):
    if sy is None:
        sy = sx
    boxes_center = bbox_corner_to_center(bbox)
    boxes_center[..., 2] *= sx
    boxes_center[..., 3] *= sy
    return bbox_center_to_corner(boxes_center)


def clip_bbox(bbox, iw, ih, as_int=True):
    bbox = as_numpy(bbox)
    bbox_clip = bbox.clip(0, (iw, ih, iw, ih))
    if as_int:
        bbox_clip = bbox_clip.astype(int)
    return bbox_clip


def _bboxes_as_array(bbox1, bbox2):
    nd_boxes1 = as_numpy(bbox1)
    nd_boxes2 = as_numpy(bbox2)

    if nd_boxes1.ndim == 1:
        nd_boxes1 = nd_boxes1[np.newaxis, np.newaxis, :]
    else:
        nd_boxes1 = nd_boxes1[:, np.newaxis, :]

    if nd_boxes2.ndim == 1:
        nd_boxes2 = nd_boxes2[np.newaxis, np.newaxis, :]
    else:
        nd_boxes2 = nd_boxes2[np.newaxis, :, :]

    return nd_boxes1, nd_boxes2
