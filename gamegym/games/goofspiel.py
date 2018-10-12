from ..game import Game, GameState
from ..distribution import Uniform
from ..server.ui import CardBuilder, Screen

import enum
import numpy as np


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

    def name(self):
        return "Goofspiel-{}".format(self.n_cards)


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
        return [self.determine_winner(c0, c1) for c0, c1 in zip(cards0, cards1)]

    def score(self, player):
        return sum(self.game.rewards[v]
                   for w, v in zip(self.winners(), self.played_cards(-1))
                   if w == player)

    def determine_winner(self, card1, card2):
        if card1 > card2:
            return 0
        if card2 > card1:
            return 1
        return -1

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

    def make_screen(self, player, live):
        # Root element
        screen = Screen(1000, 1000)
        cb = CardBuilder()

        # Rewards
        screen.add("text", "Rewards", x=10, y=50, font_size=40)

        # Reward cards
        cards = self.played_cards(-1)
        winners = self.winners()
        for i, c in enumerate(cards):
            cb.build(screen, 40 + 100 * i, 80, self.game.rewards[c])

        # Reward deck
        for i in range(self.game.n_cards - len(cards)):
            cb.build(screen, 40 + 100 * len(cards) + i * 30, 80, "")

        # Win/Loss/Draw labels
        for i, w in enumerate(winners):
            if w == -1:
                text, color = "draw", "black"
            elif w == player:
                text, color = "win", "green"
            else:
                text, color = "lost", "red"
            screen.add("text", text=text, x=50 + i * 100, y=220,
                       font_size=20, fill=color)

        # Played cards
        screen.add("text", "Played cards",
                   x=10, y=270, font_size=40)

        for i, c in enumerate(self.played_cards(player)):
            cb.build(screen, 40 + 100 * i, 300, c + 1)

        # Hand
        screen.add("text", "Cards in hand",
                   x=10, y=480, font_size=40)

        for i, c in enumerate(self.cards_in_hand(player)):
            cb.build(screen, 40 + 100 * i, 520, c + 1,
                     callback=(lambda state, action=c: state.play(action))
                              if live else None)
        return screen


def goofspiel_feaures_cards(state, sparse=False):
    """
    Goofspiel final state features for card value learning.

    Return a np.array containing, for every card:
    * 1 if player 0 won it
    * -1 if player 1 won it
    * 0 otherwise

    For nonterminal state, returns zero array of the appropriate size.
    """
    assert not sparse
    features = np.zeros(state.game.n_cards, dtype=np.float32)
    if state.is_terminal():
        card_seq = state.played_cards(-1)
        winners = state.winners()
        for i in range(len(features)):
            if winners[i] == 0:
                features[card_seq[i]] = 1.0
            elif winners[i] == 1:
                features[card_seq[i]] = -1.0
    return features
