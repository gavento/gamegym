import pytest
from gamegym.games import Goofspiel
from gamegym.strategy import UniformStrategy


def test_goofspiel():
    g = Goofspiel(7)
    s = g.start()

    assert s.active.is_chance()
    assert s.active.actions == tuple(range(1, 8))
    assert (s.active.chance == (pytest.approx(1 / 7), ) * 7)

    for i, a in enumerate([4, 2, 1, 5, 4, 6, 6, 3, 3, 2, 5, 4, 3, 7]):
        s = g.play(s, a)

    assert s.active.player == 1
    assert s.active.actions == (2, 5, 7)
    assert [o.obs for o in s.observations[2]] == [4, 1, 5, -1, 6, 0, 2, 1, 3]
    assert s.state[1] == pytest.approx([6, 5])
    assert s.state[0][0] == (1, 6)
    assert s.state[0][1] == (2, 5, 7)
    assert s.state[0][2] == (1, 7)

    for a in [2, 7, 6, 7, 1, 1, 5]:
        s = g.play(s, a)

    assert s.active.is_terminal()
    assert s.state[1] == pytest.approx([9, 13])
    assert s.active.payoff == pytest.approx([-1.0, 1.0])


def test_goofspeil_rewards():
    us = [UniformStrategy(), UniformStrategy()]
    g = Goofspiel(2, Goofspiel.Scoring.ZEROSUM, rewards=[100, 11])
    for i in range(50):
        s = g.play_strategies(us, seed=i)[-1]
        assert s.active.payoff in ((0.0, 0.0), (-89.0, 89.0), (89.0, -89.0))

    g = Goofspiel(2, Goofspiel.Scoring.ABSOLUTE, rewards=[100, 11])
    for i in range(50):
        s = g.play_strategies(us, seed=i)[-1]
        assert s.active.payoff in ((0.0, 0.0), (100.0, 11.0), (11.0, 100.0))
