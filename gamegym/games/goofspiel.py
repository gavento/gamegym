from ..game import Game, Situation, StateInfo
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

    def initial_state(self) -> Tuple[Any, StateInfo]:
        """
        Return the initial internal state and active player.
        """
        cset = list(range(1, self.cards + 1))
        return (([tuple(cset)] * 3, (0.0, 0.0)), StateInfo.new_chance(None, tuple(cset)))

    def update_state(self, state: Situation, action: Any) -> Tuple[Any, StateInfo, tuple]:
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
            return ((new_csets, scores), StateInfo.new_player(1, new_csets[1]), ())

        # Chance just drew the prize number `action`
        if player == 2:
            return ((new_csets, scores), StateInfo.new_player(0, new_csets[0]), (action, ) * 3)

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
            return ((new_csets, new_scores), StateInfo.new_chance(None, new_csets[2]), new_obs)

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
        return ((new_csets, new_scores), StateInfo.new_terminal(tscore), new_obs)

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
