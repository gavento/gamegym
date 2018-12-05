#!/usr/bin/python3

from .game import Game, GameState, Active
from .utils import uniform


class Strategy:
    """
    Base class for a strategy.
    """

    def distribution(self, observation: tuple, active: Active, state: GameState=None) -> tuple:
        """
        Returns a distribution vector on action indexes.

        Must not be called for terminal states or chance nodes.
        Should not generally depend on `state`, this is provided for e.g. debugging.
        """
        raise NotImplementedError


class UniformStrategy(Strategy):
    """
    Strategy that plays uniformly random action from those avalable.
    """

    def distribution(self, observation, active: Active, state: GameState=None) -> tuple:
        """
        Returns a uniform distribution on actions for the current state.
        """
        assert active.player >= 0
        return uniform(len(active.actions))


class ConstStrategy(Strategy):
    """
    A strategy that always returns a single distribution.

    (Useful e.g. for matrix games.)
    """

    def __init__(self, dist):
        self.dist = dist

    def distribution(self, observation, active: Active, state: GameState=None) -> tuple:
        assert active.player >= 0
        assert len(active.actions) == len(self.dist)
        return self.dist


class DictStrategy(Strategy):
    """
    A strategy that plays according to a given dictionary.

    The dictionary has the form `observations: distribution` where `distribution`
    is a tuple or a numpy array.
    If `default_uniform` is set, uniform strategy is returned for unknown observations,
    otherwise `KeyError` is raised.
    """

    def __init__(self, dictionary: dict, default_uniform: bool=False):
        self.dictionary = dictionary
        self.default_uniform = default_uniform

    def distribution(self, observation, active: Active, state: GameState=None) -> tuple:
        assert active.player >= 0
        if self.default_uniform:
            dist = self.dictionary.get(observation, None)
            if dist is None:
                dist = uniform(len(active.actions))
        else:
            dist = self.dictionary[observation]
        assert len(active.actions) == len(dist)
        return dist
