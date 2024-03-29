import numpy as np
from torchvision.transforms import Normalize, ToTensor

from tofa.torch_utils import (
    IMAGENET_MEAN_RGB,
    IMAGENET_STD_RGB,
    as_numpy,
    tensor_to_image,
)


def test_as_numpy():
    list_ = [1, 2, 3]
    list_arr_ = as_numpy(list_)
    assert list_arr_.shape == (3,)

    scalar = 42.5
    scalar_arr = as_numpy(scalar)

    assert isinstance(scalar_arr, np.ndarray)
    assert scalar_arr.shape == ()


def test_tensor_to_image(lena_rgb):
    normalize = Normalize(IMAGENET_MEAN_RGB, IMAGENET_STD_RGB)
    to_tensor = ToTensor()

    tensor = normalize(to_tensor(lena_rgb))
    tensor2image = tensor_to_image(tensor)

    np.testing.assert_allclose(lena_rgb, tensor2image, atol=1)
