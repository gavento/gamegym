#!/usr/bin/python3

import random
import game
import distribution


class Strategy:
    """
    Base class for a strategy (and strategy-computing algorithms).
    """
    def distribution(self, state):
        """
        Returns a `Discrete` distribution on actions of the given state.
        """
        raise NotImplemented


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
