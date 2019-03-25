import numpy as np
import pytest

from gamegym.algorithms.mcts import search, buffer
from gamegym.utils import Distribution
from gamegym.games import Gomoku, gomoku


def test_search():
    g = Gomoku(3, 3, 3)
    s = g.start()
    def estimator(situation):
        return np.array((0, 0)), Distribution(situation.state[1], None)
    sh = search.MctSearch(s, estimator)
    sh.search(10)

    assert list(sh.root.value) == [0.0, 0.0]
    assert sh.root.value[1] == 0.0
    assert sh.root.visit_count == 10

    t = s.play((0, 0))
    t = t.play((1, 0))
    t = t.play((2, 2))
    sh = search.MctSearch(t, estimator)
    sh.search(10000)

    n = sh.root
    others = [c.visit_count for a, c in n.children.items() if a != (1, 1)]
    assert max(others) * 10 < n.children[(1, 1)].visit_count