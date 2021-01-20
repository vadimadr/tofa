import os.path as osp
import shutil
from pathlib import Path

from tofa._typing import path_like

try:
    import jstyleson as json
except ImportError:
    import json


IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def as_path(path: path_like, absolute=False, expanduser=True) -> Path:
    path = Path(path)
    if expanduser:
        path = path.expanduser()
    if absolute:
        path = path.resolve()
    return path


def existing_path(path: path_like, absolute=False, expanduser=True) -> Path:
    path = as_path(path, absolute=absolute, expanduser=expanduser)
    if not path.exists():
        raise FileNotFoundError(f"File {path} not found")
    return path


def make_path(path, absolute=False, expanduser=True) -> Path:
    path = as_path(path, absolute=absolute, expanduser=expanduser)
    path.mkdir(exist_ok=True, parents=True)
    return path


def file_extension(path: path_like, lowercase=True) -> str:
    path = as_path(path)
    _, extension = osp.splitext(path.name)
    if lowercase:
        return extension.lower()
    return extension


def prepare_directory(path: path_like, exist_ok=True, parents=True) -> Path:
    path = as_path(path)
    if file_extension(path):
        # file has an extension => assuming path is not a directory
        path = path.parent
    path.mkdir(exist_ok=exist_ok, parents=parents)
    return path


def find_files(path: path_like, pattern="*", max_depth=None, extensions=None):
    path = as_path(path)
    result = []
    for p in path.rglob(pattern):
        if max_depth is not None:
            depth = len(p.relative_to(path).parents) - 1
            if depth > max_depth:
                continue
        if extensions and file_extension(p) not in extensions:
            continue
        result.append(p)
    return result


def find_images(
    path: path_like, pattern="*", max_depth=None, extensions=IMG_EXTENSIONS
):
    return find_files(path, pattern, max_depth, extensions)


def copy_file(src: path_like, dst: path_like, create_parent=True):
    src = as_path(src)
    dst = as_path(dst)
    if create_parent:
        prepare_directory(dst, exist_ok=True, parents=True)
    shutil.copy2(src.as_posix(), dst.as_posix())
