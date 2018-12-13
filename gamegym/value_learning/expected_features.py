from ..game import Game, Situation
from ..strategy import Strategy
from ..utils import get_rng
from ..algorithms.infosets import InformationSetSampler
import numpy as np
import scipy.optimize
import scipy as sp

# TODO: Update this to new Game API


class InfoSetExpectedFeatures:
    def __init__(self, game, feature_extractor, strategies, infosetsampler=None, sparse=False):
        """
        """
        self.game = game
        self.feature_extractor = feature_extractor
        self.sparse = sparse
        if infosetsampler is None:
            self.infosetsampler = InformationSetSampler(game, strategies)
        else:
            self.infosetsampler = infosetsampler
        if isinstance(strategies, Strategy):
            self.strategies = (strategies, ) * game.players()
        else:
            self.strategies = tuple(strategies)

        # Zero feature array and feature indices
        self.feature_0 = feature_extractor(self.game.initial_state(), sparse=self.sparse)
        # dict {(player, infoset): Explicit(action)}
        self.info_strategy = {}
        # dict {(player, infoset): {action: (player, infoset)}}
        self.info_next = {}
        # dict {(player, infoset): expected_features}
        self.info_features = {}

        self._construct()

    def _construct(self):
        """
        """
        for player in range(self.game.players()):
            info_dist = self.infosetsampler.info_distribution(player)
            for info in info_dist.values():
                state_dist = self.infosetsampler.state_distribution(player, info)
                state0 = state_dist.values()[0]
                assert state0.player() == player
                ins = {}
                for a in state0.actions():
                    sa = state0.play(a)
                    ins[a] = (sa.player(), sa.player_information(sa.player()))
                self.info_next[(player, info)] = ins
                self.info_strategy[(player, info)] = self.strategies[player].distribution(state0)
                fs = self.feature_0.copy()
                totp = 0.0
                for state, state_p in state_dist.items():
                    for ts, tp in self._terminals_under(state, state_p):
                        totp += tp
                        fs += tp * self.feature_extractor(ts, sparse=self.sparse)
                if totp > 0.0:
                    fs = fs / totp
                self.info_features[(player, info)] = fs

    def _terminals_under(self, state, p0=1.0):
        """
        Iterate over terminal nodes under `state`. Generates `(term_state, p_reach)`
        where the reach corresponds to self.strategies.
        """
        if state.is_terminal():
            yield (state, p0)
        else:
            if state.is_chance():
                dist = state.chance_distribution()
            else:
                dist = self.strategies[state.player()].distribution(state)
            for a, ap in dist.items():
                yield from self._terminals_under(state.play(a), p0 * ap)
