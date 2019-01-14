import numpy as np
import pytest

from gamegym.games.matrix import (GameOfChicken, MatchingPennies, MatrixGame, MatrixZeroSumGame,
                                  PrisonersDilemma, RockPaperScissors)
from gamegym.strategy import ConstStrategy, UniformStrategy
from gamegym.utils import get_rng
from gamegym.algorithms.stats import sample_payoff


def test_base():
    gs = [
        PrisonersDilemma(),
        GameOfChicken(),
        RockPaperScissors(),
        MatchingPennies(),
        MatrixZeroSumGame([[1, 3], [3, 2], [0, 0]], [["A", "B", "C"], [0, 1]]),
        MatrixGame([[1], [2], [3]], [["A1", "A2", "A3"]]),
        MatrixGame(np.zeros([2, 4, 5, 3], dtype=np.int32)),
    ]
    for g in gs:
        s = g.start()
        assert not s.is_terminal()
        assert s.player == 0
        assert len(s.actions) == g.m.shape[0]
        repr(s)
        repr(g)
        s = s.play(g.actions[0])
    g = RockPaperScissors()
    s = g.start()
    assert s.observations == ((), (), ())
    s = s.play("R")
    assert s.observations == ("R", (), ())
    s = s.play("P")
    assert s.is_terminal()
    assert s.observations == (("R", "P"), ("R", "P"), ("R", "P"))
    assert ((-1, 1) == s.payoff).all()


def test_strategies():
    g = RockPaperScissors()
    rng = get_rng(seed=41)

    s1 = [UniformStrategy(), UniformStrategy()]
    v1 = sample_payoff(g, s1, 300, rng=rng)
    assert sum(v1[0]) == pytest.approx(0.0)
    assert v1[0] == pytest.approx([0.0, 0.0], abs=0.1)

    s2 = [
        ConstStrategy((1.0, 0.0, 0.0)),
        ConstStrategy((0.5, 0.5, 0.0)),
    ]
    v2 = sample_payoff(g, s2, 300, rng=rng)
    assert sum(v2[0]) == pytest.approx(0.0)
    assert v2[0] == pytest.approx([-0.5, 0.5], abs=0.1)
