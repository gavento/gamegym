from gamegym.games import Gomoku, TicTacToe
from gamegym.algorithms.stats import sample_payoff
import numpy as np
import pytest


def test_features():
    for g in [
        TicTacToe(),
        Gomoku(3, 3, 2),
        Gomoku(1, 5, 10)]:
        s = g.start()
        fs = g.get_features(s)
        fss = g.get_features_shape()
        for features, shape in zip(fs, fss):
            assert features.shape == shape
            assert (features == np.zeros(shape)).all()

        s = s.play((0, 0))
        fs = g.get_features(s)
        assert fs[0][0, 0] == 0.0
        assert fs[1][0, 0] == 1.0
        assert fs[2] == 1.0

        s = s.play((1, 0))
        fs = g.get_features(s)
        assert fs[0][0, 0] == 1.0
        assert fs[0][1, 0] == 0.0
        assert fs[1][0, 0] == 0.0
        assert fs[1][1, 0] == 1.0
        assert fs[2] == 0.0


def test_gomoku():
    g = Gomoku(10, 10, 1)
    s = g.start()
    s = s.play((2, 3))
    assert s.is_terminal()
    assert (s.payoff == (1.0, -1.0)).all()
    g.show_board(s)

    g = Gomoku(4, 2, 5)
    s = g.start()
    for a in g.actions:
        s = s.play(a)
        g.show_board(s)
    assert s.is_terminal()
    assert (s.payoff == (0.0, 0.0)).all()

    g = Gomoku(2, 5, 2)
    s = g.start()
    for a in [(0, 0), (1, 0), (3, 0), (0, 1)]:
        s = s.play(a)
        g.show_board(s)
    assert s.is_terminal()
    assert (s.payoff == (-1.0, 1.0)).all()


