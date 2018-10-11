#!/usr/bin/python3

import numpy as np
import contextlib
import logging
import time


def get_rng(rng=None, seed=None):
    "Hepler returning given `rng`, new one based on `seed` or `np.random`."
    if rng is not None and seed is not None:
        raise ValueError("provide ony one of `seed` and `rng`.")
    if seed is not None:
        rng = np.random.RandomState(seed)
    if rng is None:
        rng = np.random.RandomState()
    assert isinstance(rng, np.random.RandomState)
    return rng


class ProgressReporter:
    def __init__(self, msg, limit, loglevel=logging.INFO, log=logging.root, exponent=1.5, min_time=5):
        self.msg = msg
        self.limit = limit
        self.loglevel = loglevel
        self.log = log
        self.exponent = exponent
        self.min_time = min_time

    def __enter__(self):
        self.start_time = time.clock()
        self.last_time = time.clock()
        self.last_val = 0
        return self

    def update(self, value):
        t = time.clock()
        dt = t - self.start_time
        self.last_val = value
        if dt > self.min_time and dt > self.exponent * (self.last_time - self.start_time):
            eta = 1.0 * (self.limit - value) * dt / value
            self.log.log(self.loglevel, "{}: {:5.2f}% ({} of {}), ETA +{:.2f}s".format(
                self.msg, 100.0 * value / self.limit, value, self.limit, eta))
            self.last_time = t

    def __exit__(self, exception_type, exception_value, traceback):
        t = time.clock()
        dt = t - self.start_time
        self.log.log(self.loglevel, "{} done in {:.2}s ({:.2f} / s)".format(
            self.msg, dt, self.last_val / dt))
