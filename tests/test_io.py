from pathlib import Path

from tofa.io import imread, imwrite, load, save

import cv2
import numpy as np
from PIL import Image

TEST_FILE_CONTENTS = {
    "american": ["Boston Red Sox", "Detroit Tigers", "New York Yankees"],
    "national": ["New York Mets", "Chicago Cubs", "Atlanta Braves"],
}


class TestImread:
    def test_imread(self, get_asset):
        lena_png = get_asset("lena.png")

        image = imread(lena_png)

        assert image.shape == (512, 512, 3)

    def test_imread_jpg(self, get_asset):
        lena_png = get_asset("lena.jpg")

        image = imread(lena_png)

        assert image.shape == (512, 512, 3)

    def test_correct_color(self, get_asset):
        lena_png = get_asset("lena.png")

        # reads as rgb by default
        image_rgb = imread(lena_png)
        image_bgr = imread(lena_png, rgb=False)

        image_cv_bgr = cv2.imread(lena_png.as_posix())
        image_cv_rgb = cv2.cvtColor(image_cv_bgr, cv2.COLOR_BGR2RGB)

        assert np.allclose(image_cv_rgb, image_rgb)
        assert np.allclose(image_cv_bgr, image_bgr)

    def test_correct_pil(self, get_asset):
        lena_png = get_asset("lena.png")

        # reads as rgb by default
        image_rgb = imread(lena_png, pil=True)
        image_bgr = imread(lena_png, rgb=False, pil=True)

        image_cv_bgr = cv2.imread(lena_png.as_posix())
        image_cv_rgb = cv2.cvtColor(image_cv_bgr, cv2.COLOR_BGR2RGB)

        assert isinstance(image_rgb, Image.Image)
        assert isinstance(image_bgr, Image.Image)
        assert np.allclose(image_cv_rgb, np.asarray(image_rgb))
        assert np.allclose(image_cv_bgr, np.asarray(image_bgr))


class TestImwrite:
    def test_imwrite(self, get_asset, tmpdir):
        lena_png = get_asset("lena.png")

        image_rgb = imread(lena_png)
        imwrite(Path(tmpdir) / "lena.png", image_rgb)
        image_rgb = imread(lena_png)

        image_cv_bgr = cv2.imread(lena_png.as_posix())
        image_cv_rgb = cv2.cvtColor(image_cv_bgr, cv2.COLOR_BGR2RGB)

        assert np.allclose(image_cv_rgb, image_rgb)

    def test_imwrite_bgr(self, get_asset, tmpdir):
        lena_png = get_asset("lena.png")

        image_bgr = imread(lena_png, rgb=False)
        imwrite(Path(tmpdir) / "lena.png", image_bgr, image_bgr=True)
        image_bgr = imread(lena_png, rgb=False)

        image_cv_bgr = cv2.imread(lena_png.as_posix())

        assert np.allclose(image_cv_bgr, image_bgr)


def test_read_json(get_asset):
    json_file = get_asset("json_file.json")

    json_data = load(json_file)

    assert TEST_FILE_CONTENTS == json_data


def test_read_yaml(get_asset):
    yaml_file = get_asset("yaml_file.yaml")

    yaml_data = load(yaml_file)

    assert TEST_FILE_CONTENTS == yaml_data


def test_write_yaml(tmpdir):
    yaml_out = Path(tmpdir).joinpath("yaml_out.yaml")

    save(TEST_FILE_CONTENTS, yaml_out)
    yaml_data = load(yaml_out)

    assert TEST_FILE_CONTENTS == yaml_data


def test_read_pickle(get_asset):
    pickle_file = get_asset("pickle_file.pkl")

    pickle_data = load(pickle_file)

    assert TEST_FILE_CONTENTS == pickle_data


def test_write_pickle(tmpdir):
    pickle_out = Path(tmpdir).joinpath("pickle_out.pickle")

    save(TEST_FILE_CONTENTS, pickle_out)
    pickle_data = load(pickle_out)

    assert TEST_FILE_CONTENTS == pickle_data
