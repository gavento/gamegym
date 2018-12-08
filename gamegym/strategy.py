#!/usr/bin/python3

import numpy as np
from typing import Union, Any

from .game import Game, Situation, ActivePlayer
from .utils import uniform


class Strategy:
    """
    Base class for a strategy.
    """

    def _distribution(self, observation: tuple, n_actions: int, state: Situation = None) -> tuple:
        """
        To be implemented by strategies.
        
        Return a distribution vector on action indexes.
        Never called for terminal states or chance nodes.

        Should not generally depend on `state`, this is provided for
        e.g. debugging and may be `None` in some situations.
        """
        raise NotImplementedError

    def distribution(self, observation_or_state: Union[Situation, tuple],
                     n_actions: int = None) -> Union[tuple, np.ndarray]:
        """
        Returns a distribution vector on action indexes for given observation or state.

        If called on an observation, the number of actions must be also provided.
        Must not be called for terminal states or chance nodes.
        """
        s = observation_or_state
        if isinstance(s, Situation):
            assert n_actions is None
            p = s.active.player
            assert p >= 0
            d = self._distribution(s.observations[p], len(s.active.actions), s)
        elif isinstance(s, tuple):
            assert n_actions is not None
            assert isinstance(n_actions, int)
            d = self._distribution(s, n_actions, None)
        else:
            raise TypeError("Provide GameState or observation tuple")
        assert isinstance(d, (tuple, np.ndarray))
        return d


class UniformStrategy(Strategy):
    """
    Strategy that plays uniformly random action from those avalable.
    """

    def _distribution(self, observation: Any, n_actions: int, state: Situation = None) -> tuple:
        """
        Returns a uniform distribution on actions for the current state.
        """
        return uniform(n_actions)


class ConstStrategy(Strategy):
    """
    A strategy that always returns a single distribution.

    (Useful e.g. for matrix games.)
    """

    def __init__(self, dist):
        self.dist = dist

    def _distribution(self, observation: Any, n_actions: int, state: Situation = None) -> tuple:
        assert n_actions == len(self.dist)
        return self.dist


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

    def _distribution(self, observation: Any, n_actions: int, state: Situation = None) -> tuple:
        if self.default_uniform:
            dist = self.dictionary.get(observation, None)
            if dist is None:
                dist = uniform(len(n_actions))
        else:
            dist = self.dictionary[observation]
        assert n_actions == len(dist)
        return dist
