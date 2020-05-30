from pathlib import Path
from typing import Iterable, Union

import PIL
import numpy as np
import collections.abc

array = np.ndarray
path_like = Union[Path, str]
array_like = Union[np.ndarray, Iterable, int, float]
image_like = Union[np.ndarray, PIL.Image.Image]


Sequence = collections.abc.Sequence
Iterable = collections.abc.Iterable
Mapping = collections.abc.Mapping

string_classes = (str, bytes)
path_classes = (Path, str)
array_classes = (Sequence, array)
