import numpy as np

from tofa.io import imread, imwrite
from tofa.visualization import (draw_bbox, draw_landmarks, draw_mask, draw_text,
                                imshow_debug)

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

    # sample mask
    mask = np.full((512, 512), False)
    mask[130:260, 130:260] = True
    draw_mask(lena_rgb, mask)

    draw_landmarks(lena_rgb, landmarks)

    draw_text(lena_rgb, "Sample text")

    ref_image = imread(get_asset("lena_visualization.png"))
    np.testing.assert_allclose(lena_rgb, ref_image)
