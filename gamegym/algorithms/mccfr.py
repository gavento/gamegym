#!/usr/bin/python3

import collections
from .. import strategy, distribution
from ..utils import get_rng
import numpy as np
import pickle
import logging


MCCFRInfoset = collections.namedtuple("MCCFRInfoset", ("regret", "strategy", "last_update"))


class MCCFRBase(strategy.Strategy):
    """
    Common base for Outcome and External sampling MC CFR.
    """
    EPS = 1e-30

    def __init__(self, game, seed=None, rng=None):
        self.game = game
        self.rng = get_rng(rng, seed)
        self.iss = {} # (player, infoset) -> Infoset(np.array[action], np.array[action], int)
        self.iterations = 0
        self.nodes_traversed = 0

    def get_infoset(self, player, info, num_actions):
        if (player, info) not in self.iss:
            self.iss[(player, info)] = MCCFRInfoset(
                np.zeros(num_actions, dtype=np.float32),
                np.zeros(num_actions, dtype=np.float32) + 1.0 / num_actions,
                0)
        return self.iss[(player, info)]

    def update_infoset(self, player, info, infoset, delta_r=None, delta_s=None):
        self.iss[(player, info)] = MCCFRInfoset(
            infoset.regret + delta_r if delta_r is not None else infoset.regret,
            infoset.strategy + delta_s if delta_s is not None else infoset.strategy,
            self.iterations if delta_s is not None else infoset)

    def regret_matching(self, regret):
        "Return strategy based on the regret vector"
        regplus = np.maximum(regret, np.zeros_like(regret))
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

    def persist(self, fname, iterations=None, *, nodes=None, epsilon=0.6):
        """
        If file exists, read the strategy from the file.
        If it does not, 
        TODO: rename :-)
        Returns True on succesfull load, False if not found, recomputed and stored.
        Exception raised on any loading or storing error.
        """
        assert (nodes is None) != (iterations is None)
        s = None
        try:
            with open(fname, 'rb') as f:
                s = pickle.load(f)
            if s.__class__ != self.__class__:
                raise TypeError("Loaded an incompatible object type")
            if self.game.__class__ != s.game.__class__:  # TODO: better check
                raise TypeError("Loaded strategy for a different game")
            if s is not None:
                self.iss = s.iss
                self.iterations = s.iterations
                self.nodes_traversed = s.nodes_traversed
                self.rng = s.rng
                return True
        except FileNotFoundError:
            pass

        self.compute(nodes=nodes, iterations=iterations, epsilon=epsilon)
        with open(fname, 'wb') as f:
            pickle.dump(self, f)
        return False


class OutcomeMCCFR(MCCFRBase):
    def outcome_sampling(self, state, player_updated, p_reach_updated,
                         p_reach_others, p_sample, epsilon):
        """
        Based on Alg 3 from PhD_Thesis_MarcLanctot.pdf and cfros.cpp from his bluff11.zip.
        Returns `(utility, p_tail, p_sample_leaf)`.
        """
        self.nodes_traversed += 1

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

    def compute(self, iterations=None, *, nodes=None, epsilon=0.6):
        """
        Run Outcome sampling MC CFR, traversing at most `max_nodes` nodes.
        """
        assert (nodes is None) != (iterations is None)
        old_nodes = self.nodes_traversed
        old_iterations = self.iterations
        while True:
            if nodes is not None and self.nodes_traversed >= old_nodes + nodes:
                break
            if iterations is not None and self.iterations >= old_iterations + iterations:
                break
            self.iterations += 1
            for player in range(self.game.players()):
                s0 = self.game.initial_state()
                self.outcome_sampling(s0, player, 1.0, 1.0, 1.0, epsilon=epsilon)

