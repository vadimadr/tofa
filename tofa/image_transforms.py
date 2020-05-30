from collections import namedtuple
from enum import Enum

import cv2
import numpy as np
from PIL import Image

from tofa._typing import image_like

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


def color_convert(image: image_like, colorspace="RGB", from_colorspace=None):
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

    colorspace = colorspace.upper()
    # PIL gray
    if colorspace == "L":
        colorspace = "GRAY"

    if colorspace == from_colorspace:
        return image
    image_array = np.array(image)

    if colorspace not in OPENCV_COLOR_TRANSFORM_MATRIX[from_colorspace]:
        code_bgr = OPENCV_COLOR_TRANSFORM_MATRIX[from_colorspace]["BGR"]
        image_array = cv2.cvtColor(image_array, code_bgr)
        from_colorspace = "BGR"
    code = OPENCV_COLOR_TRANSFORM_MATRIX[from_colorspace][colorspace]
    image_transformed = cv2.cvtColor(image_array, code)

    if original_pil:
        return Image.fromarray(image_transformed)
    return image_transformed


def _is_array(image):
    return isinstance(image, np.ndarray)


def _is_pil_image(image):
    return isinstance(image, Image.Image)


def _size_as_tuple(size):
    if not isinstance(size, (list, tuple)):
        size = (size, size)
    assert len(size) == 2
    return int(size[0]), int(size[1])