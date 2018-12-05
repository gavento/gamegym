from ..game import Game, GameState, Active
from ..utils import uniform
from ..server.ui import CardBuilder, Screen

from typing import Any, Tuple
import enum
import numpy as np


class Goofspiel(Game):
    EPS = 1e-6

    class Scoring(enum.Enum):
        WINLOSS = 0
        ZEROSUM = 1
        ABSOLUTE = 2

    def __init__(self, cards: int, scoring=None, rewards=None):
        assert cards >= 1
        self.cards = cards
        self.custom_rewards = rewards is not None
        if rewards is None:
            rewards = range(1, self.cards + 1)
        self.rewards = np.array(rewards, dtype=float)
        assert len(self.rewards) == self.cards
        self.scoring = self.Scoring.WINLOSS if scoring is None else scoring
        self.players = 2

    def initial_state(self) -> Tuple[Any, Active]:
        """
        Return the initial internal state and active player.
        """
        cset = list(range(1, self.cards + 1))
        return (([tuple(cset)] * 3, (0.0, 0.0)), Active.new_chance(None, tuple(cset)))

    def update_state(self, state: GameState, action: Any) -> Tuple[Any, Active, tuple]:
        """
        Return the updated internal state, active player and per-player observations.
        """
        csets, scores = state.state
        player = (len(state) - 1) % 3  # players=0,1 chance=2
        new_csets = list(csets)
        nst = list(csets[player])
        nst.remove(action)
        new_csets[player] = tuple(nst)

        # First player just bid `action`
        if player == 0:
            return ((new_csets, scores), Active.new_player(1, new_csets[1]), ())

        # Chance just drew the prize number `action`
        if player == 2:
            return ((new_csets, scores), Active.new_player(0, new_csets[0]), (action, ) * 3)

        # Otherwise, the second player just bid `action`
        prize = self.rewards[state.history[-2] - 1]
        first_action = state.history[-1]
        if first_action > action:
            new_obs = (1, -1, 1)
            new_scores = (scores[0] + prize, scores[1])
        elif first_action < action:
            new_obs = (-1, 1, -1)
            new_scores = (scores[0], scores[1] + prize)
        else:
            new_obs = (0, 0, 0)
            new_scores = scores

        # If fhis was not the last turn
        if len(state) + 1 < self.cards * 3:
            return ((new_csets, new_scores), Active.new_chance(None, new_csets[2]), new_obs)

        # This was the last turn
        assert len(state) + 1 == self.cards * 3
        if self.scoring == self.Scoring.WINLOSS:
            if new_scores[0] - new_scores[1] > self.EPS:
                tscore = (1.0, -1.0)
            elif new_scores[0] - new_scores[1] < -self.EPS:
                tscore = (-1.0, 1.0)
            else:
                tscore = (0.0, 0.0)
        elif self.scoring == self.Scoring.ZEROSUM:
            tscore = (new_scores[0] - new_scores[1], new_scores[1] - new_scores[0])
        else:
            tscore = new_scores
        return ((new_csets, new_scores), Active.new_terminal(tscore), new_obs)

    def __repr__(self):
        return "<Goofspiel({}, {}{})>".format(
            self.cards, self.scoring.name,
            ", {}".format(self.rewards) if self.custom_rewards else "")


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


if 0:

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
            return sum(self.game.rewards[v] for w, v in zip(self.winners(), self.played_cards(-1))
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
                    return (-1.0, 1.0)
                elif s1 > s2:
                    return (1.0, -1.0)
                else:
                    return (0.0, 0.0)
            if self.game.scoring == Goofspiel.Scoring.ABSOLUTE:
                return (s1, s2)

        def player_information(self, player):
            return (tuple(self.winners()), tuple(self.played_cards(-1)),
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
                screen.add("text", text=text, x=50 + i * 100, y=220, font_size=20, fill=color)

            # Played cards
            screen.add("text", "Played cards", x=10, y=270, font_size=40)

            for i, c in enumerate(self.played_cards(player)):
                cb.build(screen, 40 + 100 * i, 300, c + 1)

            # Hand
            screen.add("text", "Cards in hand", x=10, y=480, font_size=40)

            for i, c in enumerate(self.cards_in_hand(player)):
                cb.build(
                    screen,
                    40 + 100 * i,
                    520,
                    c + 1,
                    callback=(lambda state, action=c: state.play(action)) if live else None)
            return screen
