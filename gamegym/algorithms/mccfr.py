#!/usr/bin/python3

import collections
from .. import strategy, distribution
from ..utils import get_rng
import numpy as np


class MCCFR(strategy.Strategy):
    EPS = 1e-12
    Infoset = collections.namedtuple("Infoset", ("regret", "strategy", "last_update"))

    def __init__(self, game, seed=None, rng=None):
        self.game = game
        self.rng = get_rng(rng, seed)
        self.iss = {} # (player, infoset) -> Infoset(np.array[action], np.array[action], 0)
        self.iteration = 0

    def get_infoset(self, player, info, num_actions):
        if (player, info) not in self.iss:
            self.iss[(player, info)] = self.Infoset(
                np.zeros(num_actions, dtype=np.float32),
                np.zeros(num_actions, dtype=np.float32) + 1.0 / num_actions,
                0)
        return self.iss[(player, info)]

    def update_infoset(self, player, info, delta_r, delta_s):
        infoset = self.get_infoset(player, info, len(delta_r))
        new_s = infoset.strategy + delta_s * (self.iteration - infoset.last_update) / self.iteration
        assert abs(np.sum(new_s) - 1.0) < 0.1
        new_s = np.maximum(new_s, 0.0)
        self.iss[(player, info)] = self.Infoset(
            infoset.regret + delta_r,
            new_s / np.sum(new_s),
            self.iteration)

    def regret_matching(self, player, info, num_actions):
        "Return strategy based on regret in the state"
        infoset = self.get_infoset(player, info, num_actions)
        regplus = np.maximum(infoset.regret, 0.0)
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

    def generate_play_and_probabilities(self, for_player, strategies, epsilon=0.6):
        """
        Simulate game state sequence (with exploration).
        Return: (state seq, true reach probs, counterfactual reach probs,
                 sampling reach probs, action distributions).
        """
        s = self.game.initial_state()
        seq = [s]
        p_sampled = [1.0]
        p_counter = [1.0]
        p_strategy = [1.0]
        distribs = []
        while not s.is_terminal():
            player = s.player()
            if s.is_chance():
                # Chance node sampling according to chance distribution
                d = s.chance_distribution()
                a = d.sample(rng=self.rng)
                dp = d.probability(a)
                p_sampled.append(p_sampled[-1] * dp)
                p_strategy.append(p_strategy[-1] * dp)
                p_counter.append(p_counter[-1] * dp)
                distribs.append(d)
            else:
                d = strategies[player].distribution(s)
                # Sampling of the current player is exploratory, other players exact
                if player == for_player:
                    d_sample = distribution.EpsilonUniformProxy(d, epsilon)
                else:
                    d_sample = d
                a = d_sample.sample(rng=self.rng)
                dp = d.probability(a)
                dp_sample = d_sample.probability(a)
                p_sampled.append(p_sampled[-1] * dp_sample)
                p_strategy.append(p_strategy[-1] * dp)
                p_counter.append(p_counter[-1] * (1.0 if player == for_player else dp))
                distribs.append(d)
            s = s.play(a)
            seq.append(s)
        assert len(distribs) == len(seq) - 1
        return (seq, np.array(p_strategy), np.array(p_counter),
                np.array(p_sampled), distribs)

    def compute(self, iterations, epsilon=0.6):
        for _i in range(iterations):
            for player in range(self.game.players()):
                self.iteration_for_player(player, [self] * self.game.players(),
                                          epsilon=epsilon)

    def iteration_for_player(self, player, strategies, epsilon):
        self.iteration += 1
        play, p_strategy, p_counter, p_sampled, dists = \
            self.generate_play_and_probabilities(player, strategies, epsilon)
        hist = play[-1].history
        values = play[-1].values()
        for si, s in enumerate(play):
            if s.player() == player:
                dist = dists[si]
                actions = s.actions()
                info = s.player_information(player)
                rs = self.regret_matching(player, info, len(actions))
                rI = np.zeros(len(actions), dtype=np.float32)
                for ai, a in enumerate(actions):
                    W = values[player] * p_counter[si] / p_sampled[-1]
                    if a == hist[si]:
                        rI[ai] = W * (p_strategy[-1] / p_strategy[si + 1] -
                                      p_strategy[-1] / p_strategy[si])
                    else:
                        rI[ai] = -W * p_strategy[-1] / p_strategy[si]
                s_update = (rs - dist.probabilities()) * p_counter[si]
                self.update_infoset(player, info, rI, s_update)

def test_pennies():
    from ..games.matrix import MatrixGame, MatchingPennies
    g = MatchingPennies()
    print(g.m)
    mc = MCCFR(g, seed=42)
    mc.compute(1000)
    s = g.initial_state()
    print(mc.distribution(s))
    s = s.play("H")
    print(mc.distribution(s))
    print(np.mean([g.play_strategies([mc, mc], seed=i)[-1].values()[0] for i in range(1000)]))

    assert False

