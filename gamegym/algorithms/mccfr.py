#!/usr/bin/python3

import collections
from .. import strategy, distribution
from ..utils import get_rng
import numpy as np


class MCCFR(strategy.Strategy):
    EPS = 1e-12
    Infoset = collections.namedtuple("Infoset", ("regret", "strategy"))

    def __init__(self, game, seed=None, rng=None):
        self.game = game
        self.rng = get_rng(rng, seed)
        self.iss = {} # (player, infoset) -> Infoset(np.array[action], np.array[action])

    def get_infoset(self, player, info, num_actions):
        if (player, info) not in self.iss:
            self.iss[(player, info)] = self.Infoset(
                np.zeros(num_actions, dtype=np.float32),
                np.zeros(num_actions, dtype=np.float32) + 1.0 / num_actions)
        return self.iss[(player, info)]

    def update_infoset(self, player, info, delta_r, delta_s):
        infoset = self.get_infoset(player, info, len(delta_r))
        self.iss[(player, info)] = self.Infoset(
            infoset.regret + delta_r,
            infoset.strategy + delta_s)

    def regret_matching(self, player, info, num_actions):
        "Return strategy based on regret in the state"
        infoset = self.get_infoset(player, info, num_actions)
        regplus = np.max(infoset.regret, 0.0)
        s = np.sum(regplus)
        if s > self.EPS:
            return regplus / s
        else:
            return np.zeros(num_actions, dtype=np.float32) + 1.0 / num_actions

    def distribution(self, state):
        "Return a distribution for playing in the given state."
        assert not (state.is_terminal() or state.is_chance())
        player = state.player()
        info = state.player_information(player)
        actions = state.actions()
        infoset = self.get_infoset(player, info, len(actions))
        return distribution.Explicit(infoset.strategy, actions)

    def generate_play_and_probabilities(self, for_player, epsilon=0.6):
        """
        Return simulated game state sequence (with exploration) and
        true reach prob, counterfactual reach prob and sampling reach prob.
        """
        s = self.game.initial_state()
        seq = [s]
        p_sampled = [1.0]
        p_counter = [1.0]
        p_strategy = [1.0]
        while not s.is_terminal():
            if s.is_chance():
                d = s.chance_distribution()
                a = d.sample(rng=self.rng)
                dp = d.probability(a)
                p_sampled.append(p_sampled[-1] * dp)
                p_strategy.append(p_strategy[-1] * dp)
                p_counter.append(p_counter[-1] * dp)
            else:
                d = self.distribution(s)
                d_eps = distribution.EpsilonUniformProxy(d, epsilon)
                a = d_eps.sample(rng=self.rng)
                dp = d.probability(a)
                dp_eps = d_eps.probability(a)
                p_sampled.append(p_sampled[-1] * dp_eps)
                p_strategy.append(p_strategy[-1] * dp)
                p_counter.append(p_counter[-1] * (1.0 if s.player() == for_player else dp))
            s = s.play(a)
            seq.append(s)
        return (seq, np.array(p_strategy), np.array(p_counter), np.array(p_sampled))

    def compute(self, iterations, epsilon=0.6):
        for _i in range(iterations):
            for player in range(self.game.players()):
                play, p_strategy, p_counter, p_sampled = \
                    self.generate_play_and_probabilities(player, epsilon)
                hist = play[-1].history
                values = play[-1].values()
                for si, s in enumerate(play):
                    if s.player() == player:
                        actions = s.actions()
                        info = s.player_information(player)
                        rs = self.regret_matching(player, info, len(actions))
                        rI = np.zeros(len(actions), dtype=np.float32)
                        for ai, a in enumerate(actions):
                            W = values[player] * p_counter[i] / p_sampled[-1]
                            if a == hist[i]:
                                rI[ai] = W * (p_strategy[-1] / p_strategy[i + 1] - p_strategy[-1] / p_strategy[i])
                            else:
                                rI[ai] = -W * p_strategy[-1] / p_strategy[i]
                        s_update = 0 # TODO
                        self.update_infoset(player, info, rI, s_update)
