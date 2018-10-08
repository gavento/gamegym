#!/usr/bin/python3

import random
from . import game, distribution


class Strategy:
    """
    Base class for a strategy (and strategy-computing algorithms).
    """
    def distribution(self, state):
        """
        Returns a `Discrete` distribution on actions of the given state.
        Must not be called for terminal states or chance nodes.
        """
        raise NotImplementedError


class UniformStrategy(Strategy):
    """
    Strategy that plays uniformly random action from those avalable.
    """
    def distribution(self, state):
        """
        Returns a `Uniform` distribution on actions for the current state.
        """
        return distribution.Uniform(state.actions())


class EpsilonUniformProxy(Strategy):
    """
    Proxy for a strategy that plays uniformly random action with prob. `epsilon`
    and the original strategy otherwise.
    """
    def __init__(self, strategy, epsilon):
        assert isinstance(strategy, Strategy)
        self.strategy = strategy
        self.epsilon = epsilon

    def distribution(self, state):
        return distribution.EpsilonUniformProxy(
            self.strategy.distribution(state), self.epsilon)


class FixedStrategy(Strategy):
    """
    A strategy that always returns a single distribution.
    (Useful e.g. for matrix games.)
    """
    def __init__(self, dist):
        assert isinstance(dist, distribution.Discrete)
        self.dist = dist

    def distribution(self, state):
        return self.dist


class DictStrategy(Strategy):
    """
    A strategy that plays according to a given dictionary
    `(player, player_information): distribution`.
    """
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def distribution(self, state):
        p = state.player()
        return self.dictionary[(p, state.player_information(p))]
