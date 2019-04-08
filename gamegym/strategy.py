from typing import Any, Union

import numpy as np

from .game import Game
from .situation import Situation, StateInfo
from .utils import uniform, Distribution
from .observation import Observation, ActionData


class StrategyBase:
    """
    Base class for a strategy.
    """

    def distribution(self, situation: Situation) -> Distribution:
        """
        Returns a distribution vector on actions for given situation
        """
        raise NotImplementedError()


class Strategy(StrategyBase):

    def __init__(self, observation_class):
        self.observation_class = observation_class

    def compute_action_data(self, observation: Observation) -> ActionData:
        raise NotImplementedError()

    def distribution(self, situation: Situation) -> Distribution:
        observation = self.observation_class.new_observation(situation)
        return observation.decode_actions(compute_action_data(observation))


class UniformStrategy(StrategyBase):
    """
    Strategy that plays uniformly random action from those available.
    """

    def distribution(self, situation: Situation) -> Distribution:
        """
        Returns a uniform distribution on actions for the current observation.
        """
        return Distribution(situation.actions)


class ConstStrategy(StrategyBase):
    """
    A strategy that always returns a single distribution.

    Note that all received action sets must have the same size.
    Useful e.g. for testing and matrix games.
    """

    def __init__(self, distibution: Distribution):
        self.distribution = distribution

    def distribution(self, situation: Situation) -> Distribution:
        assert set(situation.actions) == set(self.distribution.vals)
        return self.distribution


class DictStrategy(Strategy):
    """
    A strategy that plays according to a given dictionary.

    The dictionary has the form `observations: distribution` where `distribution`
    is a tuple or a numpy array.
    If `default_uniform` is set, uniform strategy is returned for unknown observations,
    otherwise `KeyError` is raised.
    """

    def __init__(self, dictionary: dict, default_uniform: bool = False):
        self.dictionary = dictionary
        self.default_uniform = default_uniform

    def _strategy(self, observation: Any, n_actions: int, situation: Situation = None) -> tuple:
        if self.default_uniform:
            dist = self.dictionary.get(observation, None)
            if dist is None:
                dist = uniform(len(n_actions))
        else:
            dist = self.dictionary[observation]
        assert n_actions == len(dist)
        return dist
