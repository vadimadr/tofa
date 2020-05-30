from pathlib import Path
from typing import Iterable, Union

import PIL
import numpy as np

array = np.ndarray
path_like = Union[Path, str]
array_like = Union[np.ndarray, Iterable, int, float]
image_like = Union[np.ndarray, PIL.Image.Image]
