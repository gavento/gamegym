import pytest
from gamegym.games import Goofspiel
from gamegym.strategy import UniformStrategy


def test_goofspiel():
    g = Goofspiel(7)
    s = g.initial_state()

    assert s.player() == s.P_CHANCE
    assert s.score(0) == 0
    assert s.score(1) == 0
    assert s.actions() == list(range(7))
    assert (s.chance_distribution().probabilities() == (pytest.approx(1 / 7), ) * 7).all()

    for i, a in enumerate([3, 1, 0, 4, 3, 5, 5, 2, 2, 1, 4, 3, 2, 6]):
        s = s.play(a)
        assert s.player() == (i + 1) % 3 - 1

    assert s.round() == 4
    assert s.player() == 1
    assert s.actions() == [1, 4, 6]
    assert s.winners() == [0, 1, -1, 0]
    assert (s.chance_distribution().probabilities() == (pytest.approx(1.0 / 3), ) * 3).all()
    assert s.score(0) == 6
    assert s.score(1) == 5

    assert s.cards_in_hand(-1) == [0, 6]
    assert s.cards_in_hand(0) == [0, 5]
    assert s.cards_in_hand(1) == [1, 4, 6]

    for a in [1, 6, 5, 6, 0, 0, 4]:
        s = s.play(a)

    assert s.is_terminal()
    assert s.score(0) == 9
    assert s.score(1) == 13

    assert s.values() == (-1, 1)


def test_goofspeil_rewards():
    g = Goofspiel(2, Goofspiel.Scoring.ZEROSUM, rewards=[100, 11])
    for _ in range(10):
        history = g.play_strategies([UniformStrategy(), UniformStrategy()])
        t = history[-1]
        assert t.values() in ([0.0, 0.0], [-89.0, 89.0], [89.0, -89.0])
