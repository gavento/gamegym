
import numpy as np

from .game import Game
from .situation import Situation
from .utils import Distribution
from .nested import NestedArray


class EstimatorAdaptor:
    """
    Base class for adaptors connecting a given game to a neural network.
    """
    def __init__(self, game: Game):
        assert isinstance(game, Game)
        self.game = game

    def state_features(self, situation: Situation, player=None) -> NestedArray:
        """
        Extract features from a given game situation from the point of view of the active player.
        """
        raise NotImplementedError

    def nested_actions(self) -> NestedArray:
        """
        Get a `NestedArray` of all the actions (as Python objects).

        This is used to decode the policy output of the neural network to action distribution.
        """
        return np.array(self.actions, dtype=object)

    def action_policy(self, situation, action_likelihoods):
        """
        Return a Distribution of `valid actions: estimated values`
        """
        raise NotImplementedError
        # TODO: Implement here
