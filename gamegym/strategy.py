from typing import Any, Union

import numpy as np

from .game import Game
from .situation import Situation, StateInfo
from .utils import uniform, Distribution
from .observation import Observation
from .adapter import BlindAdapter, Adapter


class Strategy:

    def __init__(self, adapter):
        self.adapter = adapter

    def make_policy(self, observation: Observation) -> Distribution:
        raise NotImplementedError()

    def get_policy(self, situation: Situation) -> Distribution:
        return self.make_policy(self.adapter.get_observation(situation))


class BlindStrategy(Strategy):

    def __init__(self):
        super().__init__(BlindAdapter(None))


class UniformStrategy(BlindStrategy):
    """
    Strategy that plays uniformly random action from those available.
    """

    def make_policy(self, observation: Observation) -> Distribution:
        """
        Returns a uniform distribution on actions for the current observation.
        """
        return Distribution(observation.actions, None)


class ConstStrategy(BlindStrategy):
    """
    A strategy that always returns a single distribution.

    Note that all received action sets must have the same size.
    Useful e.g. for testing and matrix games.
    """

    def __init__(self, distribution: Distribution):
        super().__init__()
        assert isinstance(distribution, Distribution) or isinstance(distribution, tuple)
        self.distribution = distribution

    def make_policy(self, observation: Observation) -> Distribution:
        distribution = self.distribution
        if isinstance(distribution, Distribution):
            assert set(observation.actions).issuperset(set(self.distribution.vals))
            return distribution
        else:
            assert len(observation.actions) == len(self.distribution)
            return Distribution(observation.actions, distribution)

    @classmethod
    def single_action(cls, action):
        return cls(Distribution.single_value(action))


class DictStrategy(Strategy):
    """
    A strategy that plays according to a given dictionary.

    The dictionary has the form `observations: distribution` where `distribution`
    is a tuple or a numpy array.
    If `default_uniform` is set, uniform strategy is returned for unknown observations,
    otherwise `KeyError` is raised.
    """

    def __init__(self, adapter: Adapter, dictionary: dict, default_uniform: bool = False):
        super().__init__(adapter)
        self.dictionary = dictionary
        self.default_uniform = default_uniform

    def make_policy(self, observation: Observation) -> Distribution:
        if self.default_uniform:
            dist = self.dictionary.get(observation)
            if dist is None:
                return Distribution(observation.actions, None)
            else:
                return dist
        else:
            return self.dictionary[observation]