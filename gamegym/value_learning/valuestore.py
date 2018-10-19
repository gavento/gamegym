from ..game import Game, GameState
from ..utils import get_rng
from ..distribution import Explicit
import numpy as np

class LinearZeroSumValueStore:
    """
    Currently assumes that the game is 2-player zero sum.
    """

    def __init__(self, game, feature_extractor, initializer=None,
                 normalize_mean=None, force_mean=None, normalize_l2=None):
        self.normalize_mean = normalize_mean
        self.force_mean = force_mean
        self.normalize_l2 = normalize_l2
        self.feature_extractor = feature_extractor
        if isinstance(initializer, np.ndarray):
            self.parameters = initializer.copy()
        else:
            if initializer is None:
                initializer = normalize_mean or 0.0
            self.parameters = np.full_like(self.feature_extractor(game.initial_state()),
                                           initializer)

    def get_values(self, state, sparse=False):
        """
        Return estimated terminal state value for all players.
        """
        assert isinstance(state, GameState)
        features = self.feature_extractor(state, sparse=sparse)
        value = np.sum(features * self.parameters)
        return np.array((value, -value))

    def update_values(self, state, gradient):
        """
        Update estimated terminal state given the value gradient for all players.
        """
        assert isinstance(state, GameState)
        assert gradient.shape == (2, )
        sparse = False  # TODO(gavento): check for scipy.sparse.spmatrix instance
        features = self.feature_extractor(state, sparse=sparse)
        #print(features, (gradient[0] - gradient[1]))
        self.parameters += features * (gradient[0] - gradient[1])
        #print(self.parameters)

    def regularize(self, step_size=1e-3):
        # normalize the mean to the right value
        if self.force_mean is not None:
            self.parameters += (self.force_mean - np.mean(self.parameters))

        # normalize the mean - by step size
        if self.normalize_mean is not None:
            self.parameters += (self.normalize_mean - np.mean(self.parameters)) * step_size

        # normalize the L2 norm - by step size
        if self.normalize_l2 is not None:
            l2 = np.linalg.norm(self.parameters)
            self.parameters *= 1.0 + (self.normalize_l2 - l2) * step_size

