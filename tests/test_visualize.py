import numpy as np

from tofa.colors import bgr, rgb
from tofa.io import imread, imwrite
from tofa.visualization import (
    draw_bbox,
    draw_landmarks,
    draw_mask,
    draw_text,
    draw_text_multiline,
    imshow_debug,
)

landmarks = [
    (318, 273),
    (327, 263),
    (337, 262),
    (342, 268),
    (338, 275),
    (328, 276),
    (255, 267),
    (265, 261),
    (278, 262),
    (285, 272),
    (275, 275),
    (263, 274),
]


def test_visualize_smoke(lena_rgb, get_asset):
    # draw some primitives

    bbox = (228, 228, 377, 377)
    draw_bbox(lena_rgb, bbox)
    draw_bbox(lena_rgb, (350, 350, 480, 480), color="red", text="bbox text")

    # sample mask
    mask = np.full((512, 512), False)
    mask[130:260, 130:260] = True
    draw_mask(lena_rgb, mask)

    draw_landmarks(lena_rgb, landmarks)

    draw_text(lena_rgb, "Sample text")
    draw_text(lena_rgb, "Shadowed text", shadow=True, position=(0, 55))

    draw_text_multiline(lena_rgb, "line 1\nline 2", position=(415, 25))

    ref_image = imread(get_asset("lena_visualization.png"))
    np.testing.assert_allclose(lena_rgb, ref_image)


def test_color__rgb():
    assert rgb("red") == (255, 0, 0)
    assert rgb(rgb("red")) == (255, 0, 0)
    assert rgb(bgr("red")) == (255, 0, 0)
    assert rgb(255, 0, 0) == (255, 0, 0)
    assert rgb(1.0, 0, 0) == (255, 0, 0)
    assert rgb("red").r == 255 and rgb("red").g == 0 and rgb("red").b == 0


def test_color__bgr():
    assert bgr("red") == (0, 0, 255)
    assert bgr(rgb("red")) == (0, 0, 255)
    assert bgr(bgr("red")) == (0, 0, 255)
    assert bgr(255, 0, 0) == (255, 0, 0)
    assert bgr(1.0, 0, 0) == (255, 0, 0)
    assert bgr("red").r == 255 and bgr("red").g == 0 and bgr("red").b == 0
