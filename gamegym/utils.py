#!/usr/bin/python3

import numpy as np


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
