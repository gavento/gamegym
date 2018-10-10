from ..games.goofspiel import Goofspiel, GoofspielState
from ..utils import get_rng
import numpy as np


class GoofSpielCardsValueStore:
    def __init__(self, game):
        assert isinstance(game, Goofspiel)
        self.mean_val = (game.n + 1.0) / 2.0
        print(self.mean_val)
        self.values = np.zeros(game.n) + self.mean_val

    def features(self, state):
        "Return vector who won each card: player0=1, player1=-1, tie=0" 
        features = np.zeros_like(self.values)
        points = state.played_cards(-1)
        winners = state.winners()
        for i in range(len(features)):
            if winners[i] >= 0:
                features[points[i] - 1] = 1 - winners[i] * 2
        return features

    def get_values(self, state):
        assert isinstance(state, GoofspielState)
        val = self.features(state).dot(self.values)
        return np.array((val, -val))
        
    def update_values(self, state, gradient):
        assert isinstance(state, GoofspielState)
        assert gradient.shape == (2,)
        f = self.features(state)
        self.values += f * (gradient[0] - gradient[1])
        # renormalize to the mean
        #self.values += self.mean_val - np.mean(self.values)
        self.values *= self.mean_val / np.mean(self.values)
        

class SparseStochasticValueLearning:
    def __init__(self, game, value_store, rng=None, seed=None):
        assert isinstance(game, Goofspiel)
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

    def iteration(self, strategies, alpha=0.01):
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
        p_tail = self.history_probability(seq[depth + 1], history[depth + 1:], strategies)
        # sample a different action uniformly
        action2_idx = self.rng.choice([i for i in range(len(actions)) if i != action_idx])
        action2 = actions[action2_idx]
        seq2 = self.game.play_strategies(strategies, rng=self.rng, state0=state.play(action2))
        history2 = seq2[-1].history
        val2 = self.store.get_values(seq2[-1])[player]
        p_tail2 = self.history_probability(seq2[0], history2[depth + 1:], strategies)
        # consider equlity or inequality
        if dist[action2_idx] > 1e-9 or val2 > val:
            up = np.zeros(2)
            up[player] = alpha * (val2 - val)
            self.store.update_values(seq[-1], up * p_tail)
            self.store.update_values(seq2[-1], -up * p_tail2)

    def compute(self, strategies, iterations, alpha=0.01):
        for _i in range(iterations):
            self.iteration(strategies, alpha=alpha)
            if _i % (iterations // 20) == 0:
                print(_i, self.store.values)

