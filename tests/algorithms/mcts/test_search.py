import numpy as np
import pytest

from gamegym.algorithms.mcts import search, buffer
from gamegym.games import Gomoku, gomoku


def test_search():
    g = Gomoku(3, 3, 3)
    rb = buffer.ReplayBuffer(10)
    ad = gomoku.GomokuAdaptor(g)
    s = g.initial_state()
    def estimator(features):
        return (np.zeros(2), np.ones((3,3)))
    srch = search.MCTSearch(s, ad, estimator)
    srch.search(10)
