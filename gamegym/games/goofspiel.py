from ..game import Game, GameState
from ..distribution import Uniform

import enum




class Goofspiel(Game):
    class Scoring(enum.Enum):

        ZEROSUM_BINARY = 0
        ZEROSUM = 1
        ABSOLUTE = 2

    def __init__(self, n_cards, scoring=None):
        self.n = n_cards
        self.cards = tuple(range(1, n_cards + 1))
        self.scoring = self.Scoring.ZEROSUM_BINARY if scoring is None else scoring

    def initial_state(self):
        return GoofspielState(None, None, game=self)

    def players(self):
        return 2


def determine_winner(card1, card2):
    if card1 > card2:
        return 0
    if card2 > card1:
        return 1
    return -1


class GoofspielState(GameState):

    def player(self):
        if len(self.history) == len(self.game.cards) * 3:
            return self.P_TERMINAL
        if len(self.history) % 3 == 0:
            return self.P_CHANCE
        return len(self.history) % 3 - 1

    def round(self):
        return len(self.history) // 3

    def cards_in_hand(self, player):
        cards = list(self.game.cards)
        for c in self.played_cards(player):
            cards.remove(c)
        return cards

    def played_cards(self, player):
        return self.history[player + 1::3]

    def actions(self):
        return self.cards_in_hand(self.player())

    def chance_distribution(self):
        return Uniform(self.actions())

    def winners(self):
        cards0 = self.played_cards(0)
        cards1 = self.played_cards(1)
        return [determine_winner(c0, c1) for c0, c1 in zip(cards0, cards1)]

    def score(self, player):
        return sum(v for w, v in zip(self.winners(), self.played_cards(-1))
                   if w == player)

    def values(self):
        s1 = self.score(0)
        s2 = self.score(1)
        if self.game.scoring == Goofspiel.Scoring.ZEROSUM:
            return [s1 - s2, s2 - s1]
        if self.game.scoring == Goofspiel.Scoring.ZEROSUM_BINARY:
            if s1 < s2:
                return (-1, 1)
            elif s1 > s2:
                return (1, -1)
            else:
                return (0, 0)
        if self.game.scoring == Goofspiel.Scoring.ABSOLUTE:
            return (s1, s2)

    def player_information(self, player):
        return (len(self.history),
                tuple(self.winners()),
                tuple(self.played_cards(-1)),
                tuple(self.played_cards(player)))


def test_goofspeil():
    import pytest
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