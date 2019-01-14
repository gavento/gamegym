import pytest
from gamegym.games import Goofspiel
from gamegym.strategy import UniformStrategy
from gamegym.algorithms.stats import play_strategies

def test_goofspiel():
    g = Goofspiel(7, Goofspiel.Scoring.ZEROSUM)
    s = g.start()

    assert s.is_chance()
    assert s.actions == tuple(range(1, 8))
    assert (s.chance == (pytest.approx(1 / 7), ) * 7)

    for i, a in enumerate([4, 2, 1, 5, 4, 6, 6, 3, 3, 2, 5, 4, 3, 7]):
        s = s.play(a)

    assert s.player == 1
    assert s.actions == (2, 5, 7)
    assert s.observations[2] == (4, 1, 5, -1, 6, 0, 2, 1, 3)
    assert s.state[1] == pytest.approx([6, 5])
    assert s.state[0][0] == (1, 6)
    assert s.state[0][1] == (2, 5, 7)
    assert s.state[0][2] == (1, 7)

    for a in [2, 7, 6, 7, 1, 1, 5]:
        s = s.play(a)

    assert s.is_terminal()
    assert s.state[1] == pytest.approx([9, 13])
    assert s.payoff == pytest.approx([-4.0, 4.0])


def test_goofspiel_rewards():
    us = [UniformStrategy(), UniformStrategy()]
    g = Goofspiel(2, Goofspiel.Scoring.ZEROSUM, rewards=[100, 11])
    for i in range(50):
        s = play_strategies(g, us, seed=i)
        assert tuple(s.payoff) in ((0.0, 0.0), (-89.0, 89.0), (89.0, -89.0))

    g = Goofspiel(2, Goofspiel.Scoring.ABSOLUTE, rewards=[100, 11])
    for i in range(50):
        s = play_strategies(g, us, seed=i)
        assert tuple(s.payoff) in ((0.0, 0.0), (100.0, 11.0), (11.0, 100.0))
