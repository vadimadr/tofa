import collections.abc
from pathlib import Path
from typing import Union

import numpy as np
from PIL.Image import Image

Sequence = collections.abc.Sequence
Iterable = collections.abc.Iterable
Mapping = collections.abc.Mapping

array = np.ndarray
path_like = Union[Path, str]
array_like = Union[np.ndarray, Iterable, int, float]
image_like = Union[np.ndarray, Image]


string_classes = (str, bytes)
path_classes = (Path, str)
array_classes = (Sequence, array)
