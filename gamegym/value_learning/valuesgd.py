from ..game import Game, Situation
from ..utils import get_rng
import numpy as np

# TODO: Update this to new Game API


class SparseSGDLinearValueLearning:
    def __init__(self, game, feature_extractor, value_store, infosetsampler, rng=None, seed=None):
        self.rng = get_rng(rng=rng, seed=seed)
        self.game = game
        self.store = value_store
        self.feature_extractor = feature_extractor
        self.infosetsampler = infosetsampler

    def iteration(self, strategies, step=1e-3, regularize_step=1e-3, val_samples=1,
                  grad_samples=1):
        # sample a player, info and actions
        player, info, _ = self.infosetsampler.sample_info(rng=self.rng)
        _, _, s0, _ = self.infosetsampler.sample_state(player=player, info=info, rng=self.rng)
        # Sample actions to equalize
        a1 = strategies[player].distribution(s0).sample(rng=self.rng)
        a2 = strategies[player].distribution(s0).sample(rng=self.rng)
        if a1 == a2:
            return

        def sample_term_features(a):
            _, _, s, _ = self.infosetsampler.sample_state(player=player, info=info, rng=self.rng)
            s = s.play(a)
            z = self.game.play_strategies(strategies, rng=self.rng, state0=s)[-1]
            return self.feature_extractor(z)

        def sample_features(a, samples):
            return np.mean([sample_term_features(a) for i in range(samples)], axis=0)

        a1val = self.store.get(sample_features(a1, val_samples))
        a2val = self.store.get(sample_features(a2, val_samples))
        d = (a1val - a2val) * step
        self.store.update(sample_features(a1, grad_samples), -d)
        self.store.update(sample_features(a2, grad_samples), d)
        self.store.regularize(regularize_step)

    def compute(self,
                strategies,
                iterations,
                step=1e-3,
                regularize_step=1e-3,
                record_every=None,
                val_samples=1,
                grad_samples=1):
        params = []
        for i in range(iterations):
            self.iteration(
                strategies,
                step=step,
                regularize_step=regularize_step,
                val_samples=val_samples,
                grad_samples=grad_samples)
            if record_every and i % record_every == 0:
                params.append(self.store.values.copy())
        if record_every:
            return np.array(params)
