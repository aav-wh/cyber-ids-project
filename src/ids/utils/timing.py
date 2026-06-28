"""
ids.utils.timing
----------------
Performance timing utilities.

Used to measure and log inference latency, pipeline step durations,
and batch throughput throughout the project.
"""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager


class Timer:
    """
    Context manager and utility class for high-resolution timing.

    Usage
    -----
        with Timer("RF inference") as t:
            result = rf_model.predict(X)
        print(f"Elapsed: {t.elapsed_ms:.2f}ms")

        # Or manual usage:
        t = Timer()
        t.start()
        ...
        t.stop()
        print(t.elapsed_ms)
    """

    def __init__(self, label: str = "", verbose: bool = False) -> None:
        self.label       = label
        self.verbose     = verbose
        self._start: float | None = None
        self._end:   float | None = None

    def start(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def stop(self) -> "Timer":
        self._end = time.perf_counter()
        if self.verbose and self.label:
            print(f"[Timer] {self.label}: {self.elapsed_ms:.2f}ms")
        return self

    @property
    def elapsed_s(self) -> float:
        if self._start is None:
            return 0.0
        end = self._end or time.perf_counter()
        return end - self._start

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_s * 1000

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()


def timed(label: str = "", verbose: bool = True):
    """
    Decorator that measures and optionally prints the execution time of a function.

    Usage
    -----
        @timed("Model training")
        def train_model(X, y):
            ...
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t = Timer(label or fn.__name__, verbose=verbose)
            with t:
                result = fn(*args, **kwargs)
            return result
        return wrapper
    return decorator


@contextmanager
def measure(label: str = ""):
    """
    Convenience context manager that yields elapsed time in ms.

    Usage
    -----
        with measure("batch inference") as t:
            results = [model.predict(x) for x in batch]
        print(f"Throughput: {len(batch) / (t.elapsed_ms/1000):.0f} flows/sec")
    """
    t = Timer(label)
    t.start()
    try:
        yield t
    finally:
        t.stop()


class ThroughputMeter:
    """
    Measures and reports inference throughput (flows per second).

    Usage
    -----
        meter = ThroughputMeter()
        for flow in stream:
            with meter.tick():
                classify_flow(flow, ...)
        print(f"Throughput: {meter.flows_per_second:.0f} fps")
    """

    def __init__(self) -> None:
        self._count    = 0
        self._start    = time.perf_counter()
        self._last_ms  = 0.0

    @contextmanager
    def tick(self):
        t0 = time.perf_counter()
        yield
        self._last_ms = (time.perf_counter() - t0) * 1000
        self._count  += 1

    @property
    def flows_per_second(self) -> float:
        elapsed = time.perf_counter() - self._start
        return self._count / elapsed if elapsed > 0 else 0.0

    @property
    def last_latency_ms(self) -> float:
        return self._last_ms

    @property
    def total_flows(self) -> int:
        return self._count
