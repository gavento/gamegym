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


class SSValueLearning:
    def __init__(self, game, value_store, infosetsampler, rng=None, seed=None):
        self.rng = get_rng(rng=rng, seed=seed)
        self.game = game
        self.store = value_store
        self.infosetsampler = infosetsampler

    def iteration(self, strategies, alpha=0.01, regularize_step=1e-3):
        # sample a player, info and actions
        player, info, _ = self.infosetsampler.sample_info(rng=self.rng)
        _, _, s1, _ = self.infosetsampler.sample_state(player=player, info=info, rng=self.rng)
        _, _, s2, _ = self.infosetsampler.sample_state(player=player, info=info, rng=self.rng)
        assert s1.actions() == s2.actions()
        a1 = strategies[player].distribution(s1).sample(rng=self.rng)
        a2 = strategies[player].distribution(s1).sample(rng=self.rng)
        if a1 == a2:
            return
        z1val = self.game.play_strategies(strategies, rng=self.rng, state0=s1)[-1]
        z2val = self.game.play_strategies(strategies, rng=self.rng, state0=s2)[-1]
        val1 = z1val.values()[player]
        val2 = z2val.values()[player]
        z1up = self.game.play_strategies(strategies, rng=self.rng, state0=s1)[-1]
        z2up = self.game.play_strategies(strategies, rng=self.rng, state0=s2)[-1]

        if True:
            up = np.zeros(2)
            up[player] = alpha * (val2 - val1)
            self.store.update_values(z1up, up)
            self.store.update_values(z2up, -up)

        self.store.regularize(step_size=regularize_step)

    def compute(self, strategies, iterations, alpha=0.01, regularize_step=1e-3, record_every=None):
        params = []
        for i in range(iterations):
            self.iteration(strategies, alpha=alpha, regularize_step=regularize_step)
            if record_every and i % record_every == 0:
                params.append(self.store.parameters.copy())
            if i % (iterations // 20) == 0:
                print(i, self.store.parameters)
        if record_every:
            return np.array(params)
