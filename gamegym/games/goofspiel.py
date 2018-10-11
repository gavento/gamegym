from ..game import Game, GameState
from ..distribution import Uniform

import enum


class Goofspiel(Game):

    class Scoring(enum.Enum):
        ZEROSUM_BINARY = 0
        ZEROSUM = 1
        ABSOLUTE = 2

    def __init__(self, n_cards, scoring=None, rewards=None):
        self.cards = tuple(range(n_cards))
        if rewards is None:
            rewards = tuple(range(1, n_cards + 1))
        else:
            assert len(rewards) == n_cards
        self.rewards = tuple(rewards)
        self.scoring = self.Scoring.ZEROSUM_BINARY if scoring is None else scoring

    @property
    def n_cards(self):
        return len(self.cards)

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
        return sum(self.game.rewards[v]
                   for w, v in zip(self.winners(), self.played_cards(-1))
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
        return (tuple(self.winners()),
                tuple(self.played_cards(-1)),
                tuple(self.played_cards(player)))


