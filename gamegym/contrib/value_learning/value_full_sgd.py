from ..game import Game, Situation
from ..strategy import Strategy
from ..utils import get_rng
from .expected_features import InfoSetExpectedFeatures
import numpy as np
import scipy.optimize
import scipy as sp

# TODO: Update this to new Game API


class FullSGDZeroSumValueLearning:
    def __init__(self,
                 game,
                 feature_extractor,
                 strategies,
                 expected_features=None,
                 sparse=False,
                 seed=None,
                 rng=None):
        self.rng = get_rng(rng=rng, seed=seed)
        self.game = game
        self.feature_extractor = feature_extractor
        self.sparse = sparse
        if expected_features is None:
            self.expected_features = InfoSetExpectedFeatures(
                game, feature_extractor, strategies, sparse=sparse)
        else:
            self.expected_features = expected_features
        if isinstance(strategies, Strategy):
            self.strategies = (strategies, ) * game.players()
        else:
            self.strategies = tuple(strategies)

        self.p0values = feature_extractor(self.game.initial_state())

    def compute(self):
        pass
