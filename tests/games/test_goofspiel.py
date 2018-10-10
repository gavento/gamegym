import pytest
from gamegym.games.goofspiel import Goofspiel

def test_goofspeil():
    g = Goofspiel(7)
    s = g.initial_state()

    assert s.player() == s.P_CHANCE
    assert s.score(0) == 0
    assert s.score(1) == 0
    assert s.actions() == list(range(1, 8))
    assert (s.chance_distribution().probabilities() == (pytest.approx(1 / 7),) * 7).all()

    for i, a in enumerate(
                [4, 2, 1,
                 5, 4, 6,
                 6, 3, 3,
                 2, 5, 4,
                 3, 7]):
        s = s.play(a)
        assert s.player() == (i + 1) % 3 - 1

    assert s.round() == 4
    assert s.player() == 1
    assert s.actions() == [2, 5, 7]
    assert s.winners() == [0, 1, -1, 0]
    assert (s.chance_distribution().probabilities() == (pytest.approx(1.0 / 3),) * 3).all()
    assert s.score(0) == 6
    assert s.score(1) == 5

    assert s.cards_in_hand(-1) == [1, 7]
    assert s.cards_in_hand(0) == [1, 6]
    assert s.cards_in_hand(1) == [2, 5, 7]

    for a in [2,
              7, 6, 7,
              1, 1, 1]:
        s = s.play(a)

    assert s.is_terminal()
    assert s.score(0) == 9
    assert s.score(1) == 12

    assert s.values() == (-1, 1)