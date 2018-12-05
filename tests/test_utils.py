from gamegym.utils import debug_assert, get_rng, uniform, np_uniform
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


def test_uniform():
    assert uniform(1) == (1.0, )
    assert uniform(2) == (0.5, 0.5)
    assert len(uniform(10)) == 10
    assert sum(uniform(10)) == pytest.approx(1.0)
    assert (np_uniform(2) == np.array([0.5, 0.5])).all()
    assert len(np_uniform(10)) == 10
    assert np.sum(np_uniform(10)) == pytest.approx(1.0)
