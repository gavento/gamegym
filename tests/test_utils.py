from gamegym.utils import debug_assert, get_rng, ProgressReporter
import numpy as np
import pytest
import random


def test_rng():
    get_rng(seed=42)
    get_rng(rng=np.random.RandomState(43))
    with pytest.raises(TypeError):
        get_rng(rng=random)
    with pytest.raises(TypeError):
        get_rng(rng=random.Random(41))


def test_debug_assert():
    debug_assert(lambda: True)
    with pytest.raises(AssertionError):
        debug_assert(lambda: False)
    