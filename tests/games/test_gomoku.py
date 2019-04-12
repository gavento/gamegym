from gamegym.games import Gomoku, TicTacToe
from gamegym.algorithms.stats import sample_payoff
from gamegym.strategy import UniformStrategy
from gamegym.algorithms.stats import play_strategies


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


def test_gomoku_text_adapter():
    g = Gomoku(3, 4, 3)
    a = Gomoku.TextAdapter(g)
    s = g.start()

    obs = a.get_observation(s)
    assert obs.player == s.player
    assert (obs.data ==
        ("  123\n"
         "1 ...\n"
         "2 ...\n"
         "3 ...\n"
         "4 ..."))

    s1 = s.play((1, 2)).play((3, 0))
    obs = a.get_observation(s1)
    assert (obs.data ==
        ("  123\n"
         "1 ...\n"
         "2 ..x\n"
         "3 ...\n"
         "4 o.."))

    assert a.decode_actions(a.get_observation(s1), "2 1").vals == [(1, 0)]
    assert a.decode_actions(a.get_observation(s1), "1 3").vals == [(0, 2)]
    assert a.decode_actions(a.get_observation(s1), "xxx") is None
    assert a.decode_actions(a.get_observation(s1), "4 1") is None


def test_gomoku_tensor_adapter():
    g = Gomoku(3, 4, 3)
    a = Gomoku.TensorAdapter(g)
    s = g.start()

    obs = a.get_observation(s)
    assert obs.player == s.player

    board = np.zeros((2, 4, 3), dtype=np.bool)
    assert np.all(obs.data == board)

    s1 = s.play((1, 2)).play((3, 0))
    obs = a.get_observation(s1)

    board[0, 1, 2] = True
    board[1, 3, 0] = True

    assert np.all(obs.data == board)

    ac = np.zeros((3, 4))
    ac[1, 3] = 1
    ac[0, 1] = 1

    d = a.decode_actions(a.get_observation(s1), (ac,))
    assert tuple(d.probs) == (
        (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    )
    assert tuple(d.vals) == (
        (0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 1), (0, 2), (2, 2), (3, 2)
    )

    assert (a.encode_actions(d) == ac).all()


def test_gomoku_play_strategies():
    g = Gomoku(4, 4, 3)
    s = UniformStrategy()
    result = play_strategies(g, [s, s])