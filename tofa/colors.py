import colorsys
from collections import namedtuple

from tofa._typing import Sequence

ColorRGB = namedtuple("ColorRGB", "r g b")
ColorBGR = namedtuple("ColorBGR", "b g r")


def make_n_colors(n, bright=True):
    """Generates n visually separable colors"""
    brightness = 1.0 if bright else 0.7
    hsv = [(i / n, 1, brightness) for i in range(n)]
    colors = [colorsys.hls_to_rgb(*c) for c in hsv]
    return colors


def color_to_rgba(color):
    if len(color) == 3:
        if not _is_normalized(color):
            return (*_normalize_color(color), 1)
        return (*color, 1)
    return color


def color_to_rgb(color):
    if isinstance(color, ColorBGR):
        return color.value
    if _is_normalized(color):
        return _normalize_color(color, inverse=True)
    return color


def rgb(*color_components, normalize=False) -> ColorRGB:
    """Tries to construct a color tuple from arbitrary input"""
    assert len(color_components) in (1, 3)

    if len(color_components) == 1:
        color = color_components[0]
        if isinstance(color, ColorRGB):
            return color
        if isinstance(color, str):
            assert color.lower() in COLOR_MAP, "Unknown color name: %s" % color
            color = COLOR_MAP[color.lower()]
        if isinstance(color, ColorBGR):
            color = _color_from_tuple_denorm(tuple(reversed(color)))
        if isinstance(color, Sequence):
            color = _color_from_tuple_denorm(color)
    if len(color_components) == 3:
        color = _color_from_tuple_denorm(color_components)
    if normalize:
        return ColorRGB(*_normalize_color(color))
    return ColorRGB(*color)


def bgr(*color_components, normalize=False):
    # use rgb() and reverse order
    color = rgb(*reversed(color_components), normalize=normalize)
    return ColorBGR(*reversed(color))


def _color_from_tuple_denorm(color: Sequence):
    has_floats = any(isinstance(c, float) for c in color)
    below_one = all(c <= 1 for c in color)
    if has_floats and below_one:
        return _normalize_color(color, inverse=True)
    return tuple(color)


def _normalize_color(color, inverse=False):
    if inverse:
        return tuple(c * 255 for c in color)
    return tuple(c / 255 for c in color)


def _is_normalized(color):
    return all(c <= 1 for c in color)


# Some predefined colors:
# 20 colors with evenly distributed hue
# better to pick colors as far as possible to have nice color separation
# or use make_n_colors function to pick best separated n colors
BRIGHT_COLORS = make_n_colors(20, bright=True)
DIM_COLORS = make_n_colors(20, bright=False)

COLOR_MAP = {
    "red": rgb(255, 0, 0),
    "green": rgb(0, 255, 0),
    "blue": rgb(0, 0, 255),
    "white": rgb(255, 255, 255),
    "black": rgb(0, 0, 0),
    "aqua": rgb(0, 255, 255),
    "yellow": rgb(255, 255, 0),
    "fuchsia": rgb(255, 0, 255),
}
