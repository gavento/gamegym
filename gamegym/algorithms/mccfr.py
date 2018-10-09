#!/usr/bin/python3

import collections
from .. import strategy, distribution
from ..utils import get_rng
import numpy as np


class OutcomeMCCFR(strategy.Strategy):
    EPS = 1e-30
    Infoset = collections.namedtuple("Infoset", ("regret", "strategy", "last_update"))

    def __init__(self, game, seed=None, rng=None):
        self.game = game
        self.rng = get_rng(rng, seed)
        self.iss = {} # (player, infoset) -> Infoset(np.array[action], np.array[action], int)
        self.iteration = 0

    def get_infoset(self, player, info, num_actions):
        if (player, info) not in self.iss:
            self.iss[(player, info)] = self.Infoset(
                np.zeros(num_actions, dtype=np.float32),
                np.zeros(num_actions, dtype=np.float32) + 1.0 / num_actions,
                0)
        return self.iss[(player, info)]

    def update_infoset(self, player, info, infoset, delta_r=None, delta_s=None):
        self.iss[(player, info)] = self.Infoset(
            infoset.regret + delta_r if delta_r is not None else infoset.regret,
            infoset.strategy + delta_s if delta_s is not None else infoset.strategy,
            self.iteration if delta_s is not None else infoset)

    def regret_matching(self, regret):
        "Return strategy based on the regret vector"
        regplus = np.maximum(regret, 0.0)
        s = np.sum(regplus)
        if s > self.EPS:
            return regplus / s
        else:
            return np.zeros_like(regret) + 1.0 / len(regret)

    def distribution(self, state):
        "Return a distribution for playing in the given state."
        assert not (state.is_terminal() or state.is_chance())
        player = state.player()
        info = state.player_information(player)
        actions = state.actions()
        infoset = self.get_infoset(player, info, len(actions))
        return distribution.Explicit(infoset.strategy, actions, normalize=True)

    def outcome_sampling(self, state, player_updated, p_reach_updated,
                         p_reach_others, p_sample, epsilon):
        """
        Based on Alg 3 from PhD_Thesis_MarcLanctot.pdf.
        Returns (utility, p_tail, p_sample_leaf).
        """

        if state.is_terminal():
            #print("\n### {}: player {}, history {}, payoff {}".format(
            #    self.iteration, player_updated, state.history, state.values()[player_updated]))
            return (state.values()[player_updated], 1.0, p_sample)

        if state.is_chance():
            d = state.chance_distribution()
            a = d.sample(rng=self.rng)
            state2 = state.play(a)
            # No need to factor in the chances in Outcome sampling
            return self.outcome_sampling(state2, player_updated, p_reach_updated,
                                         p_reach_others, p_sample, epsilon)

        # Extract misc, read infoset from storage
        player = state.player()
        info = state.player_information(player)
        actions = state.actions()
        infoset = self.get_infoset(player, info, len(actions))

        # Create dists, sample the action
        dist = self.regret_matching(infoset.regret)
        if player == player_updated:
            dist_sample = dist * (1.0 - epsilon) + 1.0 * epsilon / len(actions)
        else:
            dist_sample = dist
        assert np.abs(np.sum(dist) - 1.0) < 1e-3
        assert np.abs(np.sum(dist_sample) - 1.0) < 1e-3
        action_idx = self.rng.choice(len(actions), p=dist_sample)
        action = actions[action_idx]

        # Future state
        state2 = state.play(action)

        if state.player() == player_updated:
            payoff, p_tail, p_sample_leaf = self.outcome_sampling(
                state2, player_updated, p_reach_updated * dist[action_idx],
                p_reach_others, p_sample * dist_sample[action_idx], epsilon)
            dr = np.zeros_like(infoset.regret)
            U = payoff * p_reach_others / p_sample_leaf
            #print(U, payoff, p_reach_others, p_sample_leaf, p_tail, dist)
            for ai in range(len(actions)):
                if ai == action_idx:
                    dr[ai] = U * (p_tail - p_tail * dist[action_idx])
                else:
                    dr[ai] = -U * p_tail * dist[action_idx]
            self.update_infoset(player, info, infoset, delta_r=dr)
        else:
            payoff, p_tail, p_sample_leaf = self.outcome_sampling(
                state2, player_updated, p_reach_updated,
                p_reach_others * dist[action_idx], p_sample * dist_sample[action_idx], epsilon)
            self.update_infoset(player, info, infoset,
                                delta_s=p_reach_updated / p_sample_leaf * dist)

        return (payoff, p_tail * dist[action_idx], p_sample_leaf)

    def compute(self, iterations, epsilon=0.6):
        for _i in range(iterations):
            for player in range(self.game.players()):
                s0 = self.game.initial_state()
                self.outcome_sampling(s0, player, 1.0, 1.0, 1.0, epsilon=epsilon)


from ..games.matrix import MatchingPennies, RockPaperScissors, ZeroSumMatrixGame
from ..games.goofspiel import Goofspiel
from .bestresponse import BestResponse

def test_regret():
    import pytest
    g = MatchingPennies()
    mc = OutcomeMCCFR(g, seed=42)
    rs = mc.regret_matching(mc.Infoset([-1.0, 0.0, 1.0, 2.0], None, None))
    assert rs == pytest.approx([0.0, 0.0, 1.0 / 3, 2.0 / 3])


def test_pennies():
    np.set_printoptions(precision=3)
    g = MatchingPennies()
    g = RockPaperScissors()
    g = ZeroSumMatrixGame([[1, 0], [0, 1]])
    print(g.m)
    mc = OutcomeMCCFR(g, seed=None)
    mc.compute(100)
        #s1 = g.initial_state()
        #s2 = s1.play("H")
        #print(i, mc.distribution(s1).probabilities(), mc.distribution(s2).probabilities())
    #print(np.mean([g.play_strategies([mc, mc], seed=i)[-1].values()[0] for i in range(1000)]))

    #assert False

def test_exploit_mccfr():
    #g = RockPaperScissors()
    g = Goofspiel(3)
    mc = OutcomeMCCFR(g, seed=None)
    mc.compute(10000)
    br = BestResponse(g, 0, {1:mc})
    print(np.mean([g.play_strategies([br, mc], seed=i)[-1].values()[0] for i in range(1000)]))

    #assert False