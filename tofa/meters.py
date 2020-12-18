import warnings
from math import sqrt

import numpy as np

from tofa.torch_utils import as_numpy


class Meter:
    def reset(self):
        raise NotImplementedError

    def update(self, values):
        raise NotImplementedError

    @property
    def value(self):
        raise NotImplementedError


class MeanMeter(Meter):
    def __init__(self):
        self.mean = None
        self.n = None
        self.reset()

    def update(self, values):
        values = _as_numpy_flat(values)

        if self.mean is None:
            self.mean = float(values.mean())
            self.n = values.size
        else:
            sum_prev = self.mean * self.n
            self.n += values.size
            self.mean = (sum_prev + float(values.sum())) / self.n

    def reset(self):
        self.mean = None
        self.n = 0

    @property
    def value(self):
        if self.n == 0:
            warnings.warn(
                "Value of MeanMeter in initial state is None.", RuntimeWarning
            )
            # Returns None instead of nan because:
            # 1. None is easier to work with (no need to import math)
            # 2. There are multiple nan types (e.g. numpy.float64 nans)
            return None

        return self.mean


class StdMeter(Meter):
    def __init__(self):
        self.sq_mean = MeanMeter()
        self.mean = MeanMeter()
        self.n = 0

    def update(self, values):
        values = _as_numpy_flat(values)
        self.sq_mean.update(values ** 2)
        self.mean.update(values)
        self.n += values.size

    def reset(self):
        self.n = 0
        self.mean.reset()
        self.sq_mean.reset()

    @property
    def value(self):
        if self.n == 0:
            warnings.warn("Value of StdMeter in initial state is None.", RuntimeWarning)
            return None

        m1 = self.sq_mean.value
        m2 = self.mean.value
        return sqrt(m1 - m2 ** 2)


class MedianMeter(Meter):
    def __init__(self):
        self.values = None
        self.n = None
        self.reset()

    def reset(self):
        self.values = []
        self.n = 0

    def update(self, values):
        values = _as_numpy_flat(values).astype(float)
        self.values.extend(values)
        self.n = len(self.values)

    @property
    def value(self):
        if self.n == 0:
            warnings.warn(
                "Value of MedianMeter in initial state is nan.", RuntimeWarning
            )
            return None
        return float(np.median(self.values))


def _as_numpy_flat(values):
    # turns scalar inputs to arr with shape (1, )
    return as_numpy(values).flatten()
