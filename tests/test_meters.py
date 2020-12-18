import numpy as np
import pytest
import torch

from tofa.meters import MeanMeter, MedianMeter, StdMeter


def test_mean_meter():
    xs = []
    meter = MeanMeter()
    assert meter.n == 0
    with pytest.warns(RuntimeWarning):
        assert _feed_meter(meter, xs) is None

    xs = [5]
    meter = MeanMeter()
    assert _feed_meter(meter, xs) == pytest.approx(np.mean(xs))

    xs = [7.5, 9, 11]
    meter = MeanMeter()
    assert _feed_meter(meter, xs) == pytest.approx(np.mean(xs))

    meter.reset()
    assert meter.n == 0
    with pytest.warns(RuntimeWarning):
        assert meter.value is None


def test_mean_meter__numpy():
    xs = np.array([8.5, 9.5, 11.5])
    meter = MeanMeter()
    ret = _feed_meter(meter, xs)
    assert type(ret) is float
    assert ret == pytest.approx(xs.mean())


def test_mean_meter__torch():
    xs = torch.tensor([8.5, 9.5, 11.5])
    meter = MeanMeter()
    ret = _feed_meter(meter, xs)

    # returned type must be plain python float
    # isinstance not works because np.float64 is subtype of float
    assert type(ret) is float
    assert ret == pytest.approx(xs.mean().item())


def test_std_meter():
    xs = []
    meter = StdMeter()
    assert meter.n == 0
    with pytest.warns(RuntimeWarning):
        assert _feed_meter(meter, xs) is None

    xs = [5]
    meter = StdMeter()
    assert _feed_meter(meter, xs) == np.std(xs)

    xs = [7.5, 9, 11]
    meter = StdMeter()
    ret = _feed_meter(meter, xs)
    assert type(ret) is float
    assert ret == pytest.approx(np.std(xs))

    meter.reset()
    assert meter.n == 0
    with pytest.warns(RuntimeWarning):
        assert meter.value is None


def test_median_meter():
    xs = []
    meter = MedianMeter()
    assert meter.n == 0
    with pytest.warns(RuntimeWarning):
        assert _feed_meter(meter, xs) is None

    xs = [5]
    meter = MedianMeter()
    assert _feed_meter(meter, xs) == np.median(xs)

    xs = [7.5, 9, 11]
    meter = MedianMeter()
    ret = _feed_meter(meter, xs)
    assert type(ret) is float
    assert ret == pytest.approx(np.median(xs))

    xs = [7.5, 9, 10, 11]
    meter = MedianMeter()
    ret = _feed_meter(meter, xs)
    assert type(ret) is float
    assert ret == pytest.approx(np.median(xs))

    meter.reset()
    assert meter.n == 0
    with pytest.warns(RuntimeWarning):
        assert meter.value is None


def _feed_meter(meter, values):
    for val in values:
        meter.update(val)
    return meter.value
