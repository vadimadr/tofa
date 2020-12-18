from tofa.torch_utils import as_numpy


def test_as_numpy():
    list_ = [1, 2, 3]
    list_arr_ = as_numpy(list_)
    assert list_arr_.shape == (3,)


    scalar = 42.5
    scalar_arr = as_numpy(scalar)

    print(scalar)
