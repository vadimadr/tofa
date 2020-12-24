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


def nms_by_area(bboxes, iou_threshold=0.5):
    bboxes = sorted(bboxes, key=bbox_area, reverse=True)
    keep_bboxes = []
    for bb in bboxes:
        for other in keep_bboxes:
            m = bbox_area(bbox_intersection(bb, other)) / bbox_area(bb)
            if m > iou_threshold:
                break
        else:
            keep_bboxes.append(bb)
    return keep_bboxes


def bbox_corner_to_center(bbox_corner):
    x0, y0, x1, y1 = bbox_corner[:4]
    return (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0


def bbox_center_to_corner(bbox_center):
    xc, yc, w, h = bbox_center[:4]
    return xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2


def scale_bbox(bbox, scale_factor):
    if isinstance(scale_factor, (int, float)):
        scale_factor = (scale_factor, scale_factor)
    xc, yc, w, h = bbox_corner_to_center(bbox)
    return bbox_center_to_corner((xc, yc, w * scale_factor[0], h * scale_factor[1]))


def shift_bbox(bbox, shift):
    if isinstance(shift, (int, float)):
        shift = (shift, shift)
    xc, yc, w, h = bbox_corner_to_center(bbox)
    return bbox_center_to_corner((xc + shift[0], yc + shift[1], w, h))


def clip_bbox(bbox, iw, ih):
    bbox = as_numpy(bbox)
    bbox_clip = bbox.clip(0, (iw, ih))
    bbox_int = bbox_clip.astype(int)
    return bbox_int


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


def inside_bbox(point, bbox, margin=0):
    x, y = point
    x0, y0, x1, y1 = bbox
    return x0 - margin <= x <= x1 + margin and y0 - margin <= y <= y1 + margin
