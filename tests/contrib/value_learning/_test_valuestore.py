from gamegym.game import Game, Situation
from gamegym.utils import get_rng
from gamegym.distribution import Explicit
from gamegym.value_learning.valuestore import LinearValueStore
import numpy as np
import pytest
from scipy.sparse import csr_matrix


def test_init():
    LinearValueStore(shape=(3, 3))
    LinearValueStore(np.zeros((4, 3)))
    LinearValueStore(np.zeros((4, 3)), shape=(4, 3))
    with pytest.raises(Exception):
        LinearValueStore((3, 3))
    with pytest.raises(Exception):
        LinearValueStore(np.zeros((4, 3)), shape=(4, 4))


def test_value_update():
    a = np.ones((4, ))
    vs = LinearValueStore(a)
    f = [0, 2, -1, 3]
    assert vs.get(f) == pytest.approx(4.0)
    assert vs.get(np.array(f)) == pytest.approx(4.0)
    #assert vs.get(csr_matrix(f)) == pytest.approx(4.0)
    vs.update(f, -0.5)
    assert vs.values == pytest.approx([1, 0, 1.5, -0.5])
    assert vs.get(f) == pytest.approx(-3.0)


def test_norm():
    vs = LinearValueStore(shape=(2, 3), fix_mean=1.0)
