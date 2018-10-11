from ..game import Game, GameState
from ..utils import get_rng
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
        print(features, (gradient[0] - gradient[1]))
        self.parameters += features * (gradient[0] - gradient[1])
        print(self.parameters)

    def regularize(self, step_size=1e-3):
        # renormalize the mean - by step size
        if self.force_mean is not None:
            self.parameters += (self.force_mean - np.mean(self.parameters))

        # renormalize the mean - by step size
        if self.normalize_mean is not None:
            self.parameters += (self.normalize_mean - np.mean(self.parameters)) * step_size

        # renormalize the L2 norm - by step size
        if self.normalize_l2 is not None:
            l2 = np.linalg.norm(self.parameters)
            self.parameters *= 1.0 + (self.normalize_l2 - l2) * step_size


class SSValueLearning:
    def __init__(self, game, value_store, rng=None, seed=None):
        self.rng = get_rng(rng=rng, seed=seed)
        self.game = game
        self.store = value_store

    def history_probability(self, state0, action_seq, strategies):
        state = state0
        p = 1.0
        for a in action_seq:
            assert not state.is_terminal()
            if state.is_chance():
                d = state.chance_distribution()
            else:
                d = strategies[state.player()].distribution(state)
            p *= d.probability(a)
            state = state.play(a)
        return p

    def iteration(self, strategies, alpha=0.01, regularize_step=1e-3):
        # sample a play
        seq = self.game.play_strategies(strategies, rng=self.rng)
        # sample a depth at which to operate, extract info
        depth = self.rng.choice([i for i, s in enumerate(seq) if s.player() >= 0])
        state = seq[depth]
        history = seq[-1].history
        actions = state.actions()
        if len(actions) <= 1:
            return
        player = state.player()
        dist = strategies[player].distribution(state).probabilities()
        action = history[depth]
        action_idx = actions.index(action)
        val = self.store.get_values(seq[-1])[player]
        p_pre = self.history_probability(seq[0], history[:depth], strategies)
        p_tail = self.history_probability(seq[depth + 1], history[depth + 1:], strategies)
        # sample a different action uniformly
        action2_idx = self.rng.choice([i for i in range(len(actions)) if i != action_idx])
        action2 = actions[action2_idx]
        seq2 = self.game.play_strategies(strategies, rng=self.rng, state0=state.play(action2))
        history2 = seq2[-1].history
        val2 = self.store.get_values(seq2[-1])[player]
        p_tail2 = self.history_probability(seq2[0], history2[depth + 1:], strategies)
        # consider equlity or inequality
        print(depth, history, seq2[-1].history, val, val2, p_tail, p_tail2)

        if dist[action2_idx] > 1e-9 or val2 > val:
            up = np.zeros(2)
            up[player] = alpha * (val2 - val)
            self.store.update_values(seq[-1], up * p_tail * p_pre)
            self.store.update_values(seq2[-1], -up * p_tail2 * p_pre)
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
