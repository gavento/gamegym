import pytest
from gamegym.strategy import UniformStrategy, FixedStrategy
from gamegym.distribution import Explicit
from gamegym.utils import get_rng
from gamegym.games.matrix import *


def test_base():
    gs = [
        PrisonersDilemma(),
        GameOfChicken(),
        RockPaperScissors(),
        MatchingPennies(),
        MatrixZeroSumGame([[1, 3], [3, 2], [0, 0]], ["A", "B", "C"], [0, 1]),
        MatrixGame([[1], [2], [3]], [["A1", "A2", "A3"]]),
        MatrixGame(np.zeros([2, 4, 5, 3], dtype=np.int32)),
    ]
    for g in gs:
        s = g.initial_state()
        assert not s.is_terminal()
        assert s.player() == 0
        assert len(s.actions()) == g.m.shape[0]
        repr(s)
        repr(g)
    g = RockPaperScissors()
    s = g.initial_state().play("R").play("P")
    assert s.is_terminal()
    print(s.history, s.values())
    assert ((-1, 1) == s.values()).all()


def test_strategies():

    g = RockPaperScissors()
    rng = get_rng(seed=41)
    s1 = [UniformStrategy(), UniformStrategy()]
    v1 = np.mean([g.play_strategies(s1, rng=rng)[-1].values() for i in range(300)], 0)
    assert sum(v1) == pytest.approx(0.0)
    assert v1[0] == pytest.approx(0.0, abs=0.1)
    s2 = [
        FixedStrategy(Explicit({
            "R": 1.0,
            "P": 0.0,
            "S": 0.0
        })),
        FixedStrategy(Explicit({
            "R": 0.5,
            "P": 0.5,
            "S": 0.0
        }))
    ]
    v2 = np.mean([g.play_strategies(s2, rng=rng)[-1].values() for i in range(300)], 0)
    assert sum(v2) == pytest.approx(0.0)
    assert v2[0] == pytest.approx(-0.5, abs=0.1)
