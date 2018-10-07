#!/usr/bin/python3

import random
import game
import distribution


class Strategy:
    def __init__(self, game_):
        assert isinstance(game_, game.Game)
        self.game = game_

    def distribution(self, state):
        """
        Returns a `Discrete` distribution on actions for the current state.
        """
        raise NotImplemented


class UniformStrategy(Strategy):
    def distribution(self, state):
        """
        Returns a `Uniform` distribution on actions for the current state.
        """
        return distribution.Uniform(state.actions())


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
