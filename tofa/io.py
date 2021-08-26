import json
import os
import pickle
from pathlib import Path
from typing import Union
import warnings

import numpy as np
import PIL
import yaml
from PIL import Image

from tofa._typing import path_classes, path_like
from tofa.filesystem import as_path, existing_path, file_extension, prepare_directory
from tofa.image_transforms import color_convert
from tofa.visualization import imshow_debug

try:
    import cv2

    OPENCV_AVAILABE = True
except ImportError:
    OPENCV_AVAILABE = False

try:
    import jpeg4py

    JPEG4PY_AVAILABLE = True
except ImportError:
    JPEG4PY_AVAILABLE = False


PIL_SIMD_AVAILABLE = "post" in PIL.__version__


YAML_EXTENSIONS = (".yaml", ".yml")
PICKLE_EXTENSIONS = (".pckl", ".pkl", ".pickle")
JSON_EXTENSIONS = (".json",)


def load(path: path_like):
    """Loads ab arbitrary file. Tries to guess correct loader"""

    extension = file_extension(path).lower()

    if extension in PICKLE_EXTENSIONS:
        return load_pickle(path)
    if extension in JSON_EXTENSIONS:
        return load_json(path)
    if extension in YAML_EXTENSIONS:
        return load_yaml(path)

    raise ValueError(f"File {path} extension ({extension}) is not recognized.")


def save(obj, path: path_like):
    """Saves an arbitrary file. Tries to guess correct
    save function by file extension"""

    extension = file_extension(path).lower()

    if extension in PICKLE_EXTENSIONS:
        return save_pickle(obj, path)
    if extension in JSON_EXTENSIONS:
        return save_json(obj, path)
    if extension in YAML_EXTENSIONS:
        return save_yaml(obj, path)

    raise ValueError(f"File {path} extension ({extension}) is not recognized.")


def load_pickle(path: path_like, fix_imports=None, from_py2=False):
    if from_py2:
        encoding = "latin1"
    else:
        encoding = "ascii"
    if fix_imports is None:
        fix_imports = from_py2

    with existing_path(path).open("rb") as f:
        return pickle.load(f, fix_imports=fix_imports, encoding=encoding)


def load_yaml(path, loader_cls=yaml.FullLoader):
    with existing_path(path).open("r") as f:
        return yaml.load(f, Loader=loader_cls)


def save_yaml(obj, path, dumper=yaml.Dumper, **kwargs):
    with as_path(path).open("w") as f:
        yaml.dump(obj, f, dumper, **kwargs)


def save_pickle(obj, path):
    with open(str(path), "wb") as f:
        return pickle.dump(obj, f)


def load_json(path):
    with open(str(path), "r") as f:
        return json.load(f)


def save_json(obj, path, serializer=None, **json_options):
    if serializer is None:
        serializer = _json_serializer
    with open(str(path), "w") as f:
        return json.dump(obj, f, default=serializer, **json_options)


def imread(
    image_path: path_like, pil=False, grayscale=False, rgb=True, backend=None
) -> np.ndarray:
    """Loads an image with best available image loader"""
    image_path = existing_path(image_path).as_posix()
    extension = file_extension(image_path)
    backend = _get_image_read_backed(extension, backend=backend)

    if backend == "opencv":
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Opencv can not open file: {}".format(image_path))
        image_mode = "BGR"
    if backend == "jpeg4py":
        image = jpeg4py.JPEG(image_path).decode()
        image_mode = "RGB"
    if backend == "PIL":
        image = Image.open(image_path)
        image_mode = "RGB"

    if grayscale:
        image = color_convert(image, to_colorspace="GRAY", from_colorspace=image_mode)
    elif rgb:
        image = color_convert(image, to_colorspace="RGB", from_colorspace=image_mode)
    else:
        image = color_convert(image, to_colorspace="BGR", from_colorspace=image_mode)

    if pil and not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    if not pil and not isinstance(image, np.ndarray):
        image = np.asarray(image)

    return image


def imwrite(image_path: path_like, image, create_parent=True, image_bgr=False):
    image_path = str(image_path)

    if create_parent:
        prepare_directory(image_path)

    if not isinstance(image, np.ndarray):
        image = np.asarray(image)

    if not image_bgr:
        image = color_convert(image, to_colorspace="BGR", from_colorspace="RGB")
    return_code = cv2.imwrite(image_path, image)
    if not return_code:
        raise ValueError("OpenCV could not write image file to {}".format(image_path))


def iterate_video(video: Union[path_like, cv2.VideoCapture]):
    """Iterate over video stream"""
    if isinstance(video, path_classes):
        video = cv2.VideoCapture(str(video))
    while True:
        ok, frame = video.read()
        if not ok:
            break
        yield frame


class VideoReader:
    """Wrapper for OpenCV VideoCapture"""

    def __init__(self, video_path, resolution=None, rgb=True) -> None:
        if isinstance(video_path, (Path, str)):
            video_path = str(video_path)

        self.video_cap = cv2.VideoCapture(video_path)
        if resolution is not None:
            w, h = resolution
            self.video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        self.rgb = rgb
        self._current_frame = None
        self._current_frame_index = -1

    def __iter__(self):
        if self._current_frame_index != 0:
            self._set_frame_pos(0)
            self._next_frame()
        while self._current_frame is not None:
            yield self.get_current_frame()
            self._next_frame()

    def __len__(self):
        frame_count = self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        return int(frame_count)

    def get_frame(self, frame_index=None, iterative=False):
        if frame_index is None:
            frame_index = max(self._current_frame_index, 0)

        if iterative:
            if frame_index < self._current_frame_index:
                self._set_frame_pos(0)
            while self._current_frame_index < frame_index:
                self._next_frame()
        else:
            self._set_frame_pos(frame_index)
            self._next_frame()
        return self.get_current_frame()

    def get_current_frame(self):
        if self._current_frame is None:
            return None
        if self.rgb:
            return cv2.cvtColor(self._current_frame, cv2.COLOR_BGR2RGB)
        return self._current_frame

    @property
    def shape(self):
        if self._current_frame is None:
            self._set_frame_pos(0)
            self._next_frame()
        return self._current_frame.shape

    def _next_frame(self):
        status, frame = self.video_cap.read()
        if not status:
            self._current_frame = None
            return
        self._current_frame = frame
        self._current_frame_index += 1

    def _set_frame_pos(self, frame_index):
        if frame_index != 0:
            warnings.warn(
                "OpenCV frame seeking functionality is inexact: "
                " (https://github.com/opencv/opencv/issues/9053)"
            )
        if frame_index == self._current_frame_index:
            return
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        self._current_frame_index = frame_index - 1
        self._current_frame = None


class VideoWriter:
    """Wrapper for cv2.VideoWriter"""

    CODECS = ("MP4V", "MJPG", "RAW", "RGBA", "H264", "AVC", "X264")

    def __init__(self, destination, resolution=None, fps=30, codec="MP4V", rgb=True):
        assert codec.upper() in VideoWriter.CODECS, f"Unknown codec value: {codec}"

        self.destination = destination
        self.next_video_fragment = 0
        self.cv_writer = None
        self.resolution = resolution
        self.fps = fps
        self.rgb = rgb
        self.codec = codec.upper()

        if self.resolution:
            self._open_video_writer()

    def get_video_file_name(self, file_ext):
        file_name, ext = os.path.splitext(str(self.destination))

        if ext and ext.lower() != file_ext.lower():
            raise ValueError(
                f"Unsupported file extension {ext} "
                f"for codec {self.codec} expected: {file_ext}"
            )
        return file_name + file_ext

    def write(self, frame):
        if self.cv_writer is None:
            ih, iw = frame.shape[:2]
            self.resolution = (iw, ih)
            self._open_video_writer()

        if self.rgb:
            frame = color_convert(frame, "RGB")

        self.cv_writer.write(frame)

    def release(self):
        if self.cv_writer is not None:
            self.cv_writer.release()

    def _open_video_writer(self):
        self.release()
        file_ext, codec_fourcc = self._get_resolution_and_codec()
        video_path = self.get_video_file_name(file_ext)
        self.cv_writer = cv2.VideoWriter(
            video_path, codec_fourcc, self.fps, self.resolution
        )

        if not self.cv_writer.isOpened():
            raise RuntimeError(
                f"Stream {self.destination} for VideoWriter is not opened."
            )

    def _get_resolution_and_codec(self):
        if self.codec == "MP4V":
            return ".mp4", cv2.VideoWriter_fourcc(*"mp4v")
        elif self.codec == "MJPG":
            return ".avi", cv2.VideoWriter_fourcc(*"MJPG")
        elif self.codec in ("RAW", "RGBA"):
            return ".avi", cv2.VideoWriter_fourcc(*"RGBA")
        elif self.codec in ("H264", "AVC", "X264"):
            return ".mp4", cv2.VideoWriter_fourcc(*"X264")


class DisplayVideoWriter(VideoWriter):
    """Class that emulates cv2.VideoWriter but shows preview of the video"""

    def __init__(self, window_name="playback", wait_space=True, resolution=None):
        self.wait_space = wait_space
        self.resolution = resolution
        self.window_name = window_name
        self._window_created = False

    def write(self, frame):
        self._window_created = True
        imshow_debug(
            frame,
            self.window_name,
            wait_space=self.wait_space,
            resolution=self.resolution,
        )

    def release(self):
        if self._window_created:
            cv2.destroyWindow(self.window_name)


def _get_image_read_backed(extension, backend=None):
    if backend is None:
        if extension in ("jpeg", "jpg") and JPEG4PY_AVAILABLE:
            return "jpeg4py"
        return "opencv"

    if backend == "jpeg4py":
        if not JPEG4PY_AVAILABLE:
            raise ValueError(
                "jpeg4py is not available, please install jpeg4py and libturbojpeg",
            )
        if extension not in ("jpeg", "jpg"):
            raise ValueError("Only jpeg images can be opened with jpeg4py")
        return "jpeg4py"
    if backend == "opencv":
        if not OPENCV_AVAILABE:
            raise ValueError("opencv is not available")
        return "opencv"
    if backend == "PIL":
        return backend
    raise ValueError("Unknown image io backed type: {}".format(backend))


def _json_serializer(obj):
    if isinstance(obj, Path):
        return obj.as_posix()
    if hasattr(obj, "__dict__"):
        return obj.__dict__

    # try serialize obj, save as null if not serializable
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return None
