from collections import namedtuple
from enum import Enum

import cv2
import numpy as np
from PIL import Image

from tofa._typing import image_like
from tofa.bbox_utils import clip_bbox
from tofa.torch_utils import as_numpy

InterpolationValue = namedtuple("InterpolationValue", ("str", "opencv", "pil"))


OPENCV_COLOR_TRANSFORM_MATRIX = {
    "BGR": {
        "RGB": cv2.COLOR_BGR2RGB,
        "HLS": cv2.COLOR_BGR2HLS,
        "HSV": cv2.COLOR_BGR2HSV,
        "LAB": cv2.COLOR_BGR2LAB,
        "LUV": cv2.COLOR_BGR2LUV,
        "GRAY": cv2.COLOR_BGR2GRAY,
    },
    "RGB": {
        "BGR": cv2.COLOR_RGB2BGR,
        "HLS": cv2.COLOR_RGB2HLS,
        "HSV": cv2.COLOR_RGB2HSV,
        "LAB": cv2.COLOR_RGB2LAB,
        "LUV": cv2.COLOR_RGB2LUV,
        "GRAY": cv2.COLOR_RGB2GRAY,
    },
    "HLS": {"RGB": cv2.COLOR_HLS2RGB, "BGR": cv2.COLOR_HLS2BGR},
    "HSV": {"RGB": cv2.COLOR_HSV2RGB, "BGR": cv2.COLOR_HSV2BGR},
    "LAB": {"RGB": cv2.COLOR_LAB2RGB, "BGR": cv2.COLOR_LAB2BGR},
    "LUV": {"RGB": cv2.COLOR_LUV2RGB, "BGR": cv2.COLOR_LUV2BGR},
    "GRAY": {"RGB": cv2.COLOR_GRAY2RGB, "BGR": cv2.COLOR_GRAY2BGR},
}


class Interpolation(Enum):
    NEAREST = InterpolationValue("nearest", cv2.INTER_NEAREST, Image.NEAREST)
    LINEAR = InterpolationValue("linear", cv2.INTER_LINEAR, Image.LINEAR)
    CUBIC = InterpolationValue("cubic", cv2.INTER_CUBIC, Image.CUBIC)
    LANCZOS = InterpolationValue("lanczos", cv2.INTER_LANCZOS4, Image.LANCZOS)

    DEAFULT = LINEAR

    @property
    def str(self):
        return self.value.str

    @property
    def opencv(self):
        return self.value.opencv

    @property
    def pil(self):
        return self.value.pil


def image_size(image):
    """Returns image size (w x h)"""
    if _is_array(image):
        h, w = image.shape[:2]
        return w, h
    return image.size


def color_convert(image: image_like, to_colorspace="RGB", from_colorspace=None):
    original_pil = _is_pil_image(image)
    if from_colorspace is None:
        if original_pil and image.mode in ("RGB", "HSV", "LAB"):
            from_colorspace = image.mode
        elif original_pil:
            raise ValueError(f"Unsupported conversion from PIL mode: {image.mode}")
        else:
            # default color space for OpenCV is BGR
            from_colorspace = "BGR"
    else:
        from_colorspace = from_colorspace.upper()

    to_colorspace = to_colorspace.upper()
    # PIL gray
    if to_colorspace == "L":
        to_colorspace = "GRAY"

    if to_colorspace == from_colorspace:
        return image
    image_array = np.array(image)

    if to_colorspace not in OPENCV_COLOR_TRANSFORM_MATRIX[from_colorspace]:
        code_bgr = OPENCV_COLOR_TRANSFORM_MATRIX[from_colorspace]["BGR"]
        image_array = cv2.cvtColor(image_array, code_bgr)
        from_colorspace = "BGR"
    code = OPENCV_COLOR_TRANSFORM_MATRIX[from_colorspace][to_colorspace]
    image_transformed = cv2.cvtColor(image_array, code)

    if original_pil:
        return Image.fromarray(image_transformed)
    return image_transformed


def resize(
    image: image_like, size, keep_aspect=False, interpolation=Interpolation.DEAFULT
):
    tw, th = _size_as_tuple(size)

    if keep_aspect:
        iw, ih = image_size(image)
        scale_factor = tw / iw if iw < ih else th / ih
        tw, th = tw * scale_factor, th * scale_factor
    return _resize_impl(image, (tw, th), interpolation=interpolation)


def rescale(image: image_like, scale_factor, interpolation=Interpolation.DEAFULT):
    w, h = image_size(image)
    size_scaled = w * scale_factor, h * scale_factor
    return _resize_impl(image, size_scaled, interpolation=interpolation)


def crop(image, position):
    iw, ih = image_size(image)
    x1, y1, x2, y2 = clip_bbox(position, iw, ih)
    if _is_array(image):
        return image[y1:y2, x1:x2]
    return image.crop((x1, y1, x2, y2))


def flip(img, horizontal=False):
    if isinstance(img, np.ndarray):
        if horizontal:
            return img[:, ::-1, :]
        return img[::-1, ...]
    if horizontal:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img.transpose(Image.FLIP_TOP_BOTTOM)


def hflip(img):
    return flip(img, horizontal=True)


def vflip(img):
    return flip(img, horizontal=False)


def crop_resize(img, crop_position, crop_size, interpolation=Interpolation.DEAFULT):
    image_crop = crop(img, crop_position)
    return resize(image_crop, crop_size, interpolation=interpolation)


def perspective_crop(img, crop_points, crop_size, interpolation=Interpolation.DEAFULT):
    assert isinstance(img, np.ndarray)
    w, h = crop_size

    output_position = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(as_numpy(crop_points, np.float32), output_position)
    crop = cv2.warpPerspective(img, H, crop_size, flags=interpolation.opencv)
    return crop


def _resize_impl(image, size, interpolation=Interpolation.DEAFULT):
    w, h = _size_as_tuple(size)
    if _is_array(image):
        return cv2.resize(image, (int(w), int(h)), interpolation=interpolation.opencv)
    return image.resize((int(h), int(w)), interpolation=interpolation.pil)


def _is_array(image):
    return isinstance(image, np.ndarray)


def _is_pil_image(image):
    return isinstance(image, Image.Image)


def _size_as_tuple(size):
    if not isinstance(size, (list, tuple)):
        size = (size, size)
    assert len(size) == 2
    return int(size[0]), int(size[1])
