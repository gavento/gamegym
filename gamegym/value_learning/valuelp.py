from ..game import Game, GameState
from ..utils import get_rng
from ..distribution import Explicit
import numpy as np
import scipy as sp


class LPValueLearning:
    EPS = 1e-6

    def __init__(self, game, infosetsampler, strategies, rng=None, seed=None):
        self.rng = get_rng(rng=rng, seed=seed)
        self.game = game
        self.infosetsampler = infosetsampler
        self.variables = {}
        self.strategies = strategies
        self.construct_lp()

    def get_var(self, ident):
        if ident not in self.variables:
            self.variables[ident] = len(self.variables)
        return self.variables[ident]

    def construct_lp(self):
        for player in range(self.game.players()):
            info_dist = self.infosetsampler.info_distribution(player)
            for info in info_dist.values():
                value_var = self.get_var(("val", info))
                state_dist = self.infosetsampler.state_distribution(player, info)
                state0 = state_dist.values()[0]
                strategy = self.strategies[player].distribution(state0)
                for a in state0.actions():
                    print("Conditions for player {} in infoset {}, action {}, var {}, terminals:".format(
                        player, info, a, value_var))
#                    if strategy.probability(a) > self.EPS:
#                for state, state_p in state_dist.items():
#                   for a in action

    def terminals_under(self, state, p0):
        if state.is_terminal():
            yield (state, p0)
        else:
            if state.is_chance():
                dist = state.chance_distribution()
            else:
                dist = self.strategies[state.player()]
            for a, ap in dist.items():
                yield from self.terminals_under(state.play(a), p0 * ap)
