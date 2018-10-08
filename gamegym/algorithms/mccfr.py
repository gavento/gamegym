#!/usr/bin/python3

import collections
from .. import strategy, distribution
from ..utils import get_rng


class MCCFR(strategy.Strategy):
    Regret = collections.namedtuple("Regret", ("sum", "cnt"))

    def __init__(self, game, seed=None, rng=None):
        self.game = game
        self.rng = get_rng(rng, seed)
        self.regret = {}  # (player, infoset, action_label) -> Regret

    def get_regret(self, player, infoset, action):
        return self.regret.setdefault((player, infoset, action), self.Regret(0.0, 0))

    def add_regret(self, player, infoset, action, r):
        r0 = self.get_regret(player, infoset, action)
        self.regret[(player, infoset, action)] = self.Regret(r0.sum + r, r0.cnt + 1)

    def distribution(self, state):
        p = state.player()
        infoset = state.information_set(p)
        assert not (state.is_terminal() or state.is_chance())
        res = []
        for a in state.actions():
            if state.player() >= 0: # update for non-chance nodes
                reg = self.get_regret(p, infoset, a)
                prob = max(reg.sum / max(reg.cnt, 1), 0.0)
            res.append(prob)
        return distribution.Explicit(res, state.actions(), normalize=True)

    def compute(self, iterations, epsilon=0.1):
        for _i in range(iterations):
            for p in range(self.game.players()):
                play = self.generate_play(epsilon=epsilon)
                pass # TODO
            
    def generate_play(self, epsilon=0.0):
        "Return simulated game state sequence (with exploration)."
        prox = strategy.EpsilonUniformProxy(self, epsilon=epsilon)
        return self.game.play_strategies(
            [prox] * self.game.players(), rng=self.rng)
