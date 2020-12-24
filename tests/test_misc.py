import pytest

from tofa.misc import AttribDict


def test_attrib_dict():
    d = AttribDict()

    with pytest.raises(KeyError):
        _ = d.non_existent_prop

    d.some_prop = 123
    assert "some_prop" in d
    assert d.some_prop == d["some_prop"]
