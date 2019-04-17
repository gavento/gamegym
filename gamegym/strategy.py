from typing import Any, Union

import numpy as np

from .game import Game
from .situation import Situation, StateInfo
from .utils import uniform, Distribution
from .observation import Observation
from .adapter import BlindAdapter, Adapter
from .utils import debug_assert


class Strategy:

    # Either a name of adapter class within a game class
    DEFAULT_ADAPTER = None

    def __init__(self, game: Game, adapter: Adapter = None):
        assert isinstance(game, Game)
        self.game = game
        if adapter is None:
            if self.DEFAULT_ADAPTER is None:
                raise ValueError(
                    "Strategy {!r} has no default adapter type, provide one explicitly.".format(
                        self.__class__.__name__))
            try:
                adapter_type = game.__getattribute__(self.DEFAULT_ADAPTER)
            except AttributeError:
                raise ValueError(
                    "Game {!r} does not have the default adapter {!r} (for strategy {!r}, provide one explicitly."
                    .format(game.__class__.__name__, self.DEFAULT_ADAPTER,
                            self.__class__.__name__))
            assert issubclass(adapter_type, Adapter)
            self.adapter = adapter_type(self.game)
        else:
            assert isinstance(adapter, Adapter)
            self.adapter = adapter

    def make_policy(self, observation: Observation) -> Distribution:
        """
        Create a policy (action distribution) for the given `Observation`.

        This needs to be overriden by subclasses of `Strategy`.
        """
        raise NotImplementedError()

    def get_policy(self, situation: Situation) -> Distribution:
        """
        Return a policy (action distribution) for the given `Situation`.
        """
        return self.make_policy(self.adapter.get_observation(situation))


class BlindStrategy(Strategy):
    """
    Strategy that ignores any observations or the game.
    """
    _EMPTY_GAME = Game(1, ())
    _BLIND_ADAPTER = BlindAdapter(_EMPTY_GAME)

    def __init__(self):
        super().__init__(self._EMPTY_GAME, self._BLIND_ADAPTER)


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
    Useful e.g. for testing and one-round / matrix games.
    """

    def __init__(self, distribution: Distribution):
        super().__init__()
        assert isinstance(distribution, Distribution) or isinstance(distribution, tuple)
        self.distribution = distribution

    def make_policy(self, observation: Observation) -> Distribution:
        distribution = self.distribution
        if isinstance(distribution, Distribution):
            debug_assert(lambda: set(observation.actions).issuperset(set(self.distribution.vals)))
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

    def __init__(self,
                 game: Game,
                 dictionary: dict,
                 *,
                 adapter: Adapter = None,
                 default_uniform: bool = False):
        super().__init__(game, adapter)
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
