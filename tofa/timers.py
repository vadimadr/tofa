import time
from contextlib import contextmanager

from tofa.meters import MeanMeter, StdMeter


class IncrementalTimer:
    def __init__(self):
        self._tick_time = None
        self.total_elapsed_ms = 0
        self.last_elapsed_ms = 0

        self._mean = MeanMeter()
        self._std = StdMeter()

        self.tick()

    def tick(self):
        self._tick_time = time.perf_counter()

    def tock(self):
        tock_time = time.perf_counter()
        elapsed_ms = (tock_time - self._tick_time) * 1000.0
        self.total_elapsed_ms += elapsed_ms

    @property
    def fps(self):
        return 1000 / self.avg

    @property
    def avg(self):
        """Returns average time in ms"""
        return self._mean.value

    @property
    def std(self):
        return self._std.value

    def __repr__(self):
        return "{:.2f}ms (±{:.2f}) {:.2f}fps".format(self.avg, self.std, self.fps)


@contextmanager
def benchmark_section():
    timer = IncrementalTimer()
    timer.tick()
    yield timer
    timer.tock()
