import pytest
from gamegym.strategy import UniformStrategy, ConstStrategy
from gamegym.utils import get_rng
from gamegym.games.matrix import *


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
        assert not s.active.is_terminal()
        assert s.active.player == 0
        assert len(s.active.actions) == g.m.shape[0]
        repr(s)
        repr(g)
    g = RockPaperScissors()
    s = g.start()
    s = g.play(s, "R")
    s = g.play(s, "P")
    assert s.active.is_terminal()
    assert ((-1, 1) == s.active.payoff).all()


def test_strategies():
    g = RockPaperScissors()
    rng = get_rng(seed=41)
    s1 = [UniformStrategy(), UniformStrategy()]
    v1 = np.mean([g.play_strategies(s1, rng=rng)[-1].active.payoff for i in range(300)], 0)
    assert sum(v1) == pytest.approx(0.0)
    assert v1[0] == pytest.approx(0.0, abs=0.1)
    s2 = [
        ConstStrategy((1.0, 0.0, 0.0)),
        ConstStrategy((0.5, 0.5, 0.0)),
    ]
    v2 = np.mean([g.play_strategies(s2, rng=rng)[-1].active.payoff for i in range(300)], 0)
    assert sum(v2) == pytest.approx(0.0)
    assert v2 == pytest.approx([-0.5, 0.5], abs=0.1)
