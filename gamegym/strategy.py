#!/usr/bin/python3

import random
from .game import Game, GameState


class Strategy:
    def __init__(self, game, seed=None, rng=None):
        assert isinstance(game, Game)
        self.game = game
        self.rng = rng or random.Random(seed)

    def make_normalized_epsilon_greedy(self, actions, epsilon=0.0):
        """
        Given a list/tuple of `Game.NextAction`, normalize their probalilities,
        optionally making them epsilon-greedy.
        """
        assert isinstance(actions, (list, tuple))
        assert isinstance(actions[0], Game.NextAction)
        s = sum(a.probability for a in actions)
        if s <= 1e-30: # In case all probs are 0
            s = 1.0
            epsilon = 1.0
        return [Game.NextAction(
            label, st, (1.0 - epsilon) * prob / s + epsilon / len(actions))
            for label, st, prob in actions]

    def distribution(self, state, epsilon=0.0):
        """
        Returns a list of `Game.NextAction` with the action probabilities.
        Optionally make the distribution epsilon-greedy.
        """
        raise NotImplemented

    def sample(self, state, *, epsilon=0.0, rng=None):
        """
        Return one randomly chosen `Game.NextAction`, optionally using
        provided rng and optionally chosing epsilon-greedily.
        """
        dist = self.distribution(state, epsilon=epsilon)
        p = (rng or self.rng).random()
        s = 0.0
        for na in dist:
            s += na.probability
            if p < s:
                return na
        assert s > 0.999
        return na


#class EpsilonProxy(Strategy):
#    def __init__(self, strategy, epsilon=0.0):
#        assert isinstance(strategy, Strategy)
#        super().__init__(strategy.game, rng=strategy.rng)
#        self.epsilon = epsilon
#        self.strategy = straregy
#
#    def distribution(self, state, epsilon=0.0):
#        """
#        Returns a list of `Game.NextAction` with the action probabilities.
#        Optionally make the distribution epsilon-greedy.
#        """
#        raise NotImplemented
