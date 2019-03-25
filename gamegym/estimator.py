
import numpy as np

from .game import Game
from .situation import Situation
from .utils import Distribution
from .nested import NestedArray


class EstimatorAdapter:
    """
    Base class for Adapters connecting a given game to a neural network.
    """
    def __init__(self, game: Game):
        assert isinstance(game, Game)
        self.game = game

    def state_features(self, situation: Situation, player=None) -> NestedArray:
        """
        Extract features from a given game situation from the point of view of the active player.
        """
        raise NotImplementedError()

    def policy_logits_from_distribution(self, distribution: Distribution):
        raise NotImplementedError()

    def distribution_from_policy_logits(self, logits):
        raise NotImplemented()


class SimpleEstimatorAdapter(EstimatorAdapter):

    def __init__(self, game: Game):
        super().__init__(game)

    def policy_logits_from_distribution(self, distribution: Distribution):
        result = np.zeros(len(self.game.actions))
        actions_index = self.game.actions_index
        for action, prob in distribution.items():
            result[actions_index[action]] = prob
        return result

    def distribution_from_policy_logits(self, situation: Situation, logits):
        game = situation.game
        actions = tuple(self.actions)
        probs = [np.exp(game.actions_index[action]) for action in actions]
        return Distribution(actions, probs, norm=True)
