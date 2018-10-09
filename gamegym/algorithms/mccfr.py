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

    def iteration_for_player(self, player, strategies, epsilon, alpha=1):
        self.iteration += 1
        play, p_strategy, p_counter, p_sampled, dists = \
            self.generate_play_and_probabilities(player, strategies, epsilon)
        hist = play[-1].history
        values = play[-1].values()
        print(hist, values)
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
                self.update_infoset(player, info, rI * alpha, s_update * alpha)





class OutcomeMCCFR(strategy.Strategy):
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

    def update_infoset(self, player, info, infoset, delta_r=None, delta_s=None):
        #if delta_r is not None:
        #    print("Updating regret {} {} from r={} s={} by r+={}".format(
        #        player, info, infoset.regret, infoset.strategy, delta_r))
        #if delta_s is not None:
        #    print("Updating strat. {} {} from r={} s={} by s+={}".format(
        #        player, info, infoset.regret, infoset.strategy, delta_s))
        self.iss[(player, info)] = self.Infoset(
            infoset.regret + delta_r if delta_r is not None else infoset.regret,
            infoset.strategy + delta_s if delta_s is not None else infoset.strategy,
            self.iteration if delta_s is not None else infoset)

    def regret_matching(self, infoset):
        "Return strategy based on regret in the state"
        regplus = np.maximum(infoset.regret, 0.0)
        s = np.sum(regplus)
        if s > self.EPS:
            return regplus / s
        else:
            return np.zeros_like(infoset.regret) + 1.0 / len(infoset.regret)

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
        dist = self.regret_matching(infoset)
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
    mc.compute(100)
    br = BestResponse(g, 1, {0:mc})
    #print(br.best_responses)
    #print(np.mean([g.play_strategies([mc, br], seed=i)[-1].values()[0] for i in range(1000)]))
