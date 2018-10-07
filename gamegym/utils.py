#!/usr/bin/python3

import random


def get_rng(rng=None, seed=None):
    "Hepler returning given `rng`, new one based on `seed` or `random`."
    if rng is not None and seed is not None:
        raise ValueError("provide ony one of `seed` and `rng`.")
    if seed is not None:
        rng = random.Random(seed)
    if rng is None:
        rng = random
    return rng
