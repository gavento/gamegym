
import numpy as np

from .game import Game
from .situation import Situation
from .utils import Distribution


class EstimatorAdaptor:
    """
    Base class for adaptors connecting a given game to a neural network.
    """
    def __init__(self, game: Game):
        assert isinstance(game, Game)
        self.game = game

    def state_features(self, situation: Situation, player=None) -> NestedArrays:
        """
        Extract features from a given game situation
        from the point of view of the given player (default is active player).
        """
        raise NotImplementedError

    def nested_actions(self) -> NestedArray:
        """
        Get a `NestedArray` of all the actions (as Python objects).

        This is used to decode the policy output of the neural network to action distribution.
        """
        return np.array(self.actions, dtype=object)

    def action_values(self, situation, action_logits):
        """
        Return a dict of `valid actions: estimated values`

#TODO(gavento):