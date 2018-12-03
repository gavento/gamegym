#!/usr/bin/python3

import collections
from .. import strategy, distribution
from ..utils import get_rng, ProgressReporter, debug_assert
import numpy as np
import pickle
import logging
import time
import bz2
import sys

MCCFRInfoset = collections.namedtuple("MCCFRInfoset", ("regret", "strategy", "last_update"))


class MCCFRBase(strategy.Strategy):
    """
    Common base for Outcome and External sampling MC CFR.
    """
    EPS = 1e-30

    def __init__(self, game, seed=None, rng=None):
        self.game = game
        self.rng = get_rng(rng, seed)
        self.iss = {}  # (player, infoset) -> Infoset(np.array[action], np.array[action], int)
        self.iterations = 0
        self.nodes_traversed = 0

    def get_infoset(self, player, info, num_actions):
        if (player, info) not in self.iss:
            self.iss[(player, info)] = MCCFRInfoset(
                np.zeros(num_actions, dtype=np.float32),
                np.zeros(num_actions, dtype=np.float32) + 1.0 / num_actions, 0)
        return self.iss[(player, info)]

    def reset_after_burnin(self):
        for k, v in self.iss.items():
            num_actions = len(v[0])
            self.iss[k] = MCCFRInfoset(v[0],
                                       np.zeros(num_actions, dtype=np.float32) + 1.0 / num_actions,
                                       v[2])

    def update_infoset(self, player, info, infoset, delta_r=None, delta_s=None):
        self.iss[(player, info)] = MCCFRInfoset(
            infoset.regret + delta_r if delta_r is not None else infoset.regret,
            infoset.strategy + delta_s if delta_s is not None else infoset.strategy,
            self.iterations if delta_s is not None else infoset.last_update)

    def regret_matching(self, regret):
        "Return strategy based on the regret vector"
        regplus = np.clip(regret, 0, None)
        s = np.sum(regplus)
        if s > self.EPS:
            return regplus / s
        else:
            return np.full(regret.shape, 1.0 / len(regret))

    def distribution(self, state):
        "Return a distribution for playing in the given state."
        assert not (state.is_terminal() or state.is_chance())
        player = state.player()
        info = state.player_information(player)
        actions = state.actions()
        infoset = self.get_infoset(player, info, len(actions))
        return distribution.Explicit(infoset.strategy, actions, normalize=True)

    def persist(self, basename, iterations, epsilon=0.6):
        """
        If file exists, read the strategy from the file with given base name, bz2-compressed.
        If it does not, compute to obtain `iterations` total.

        Returns `True` on succesfull load,
        False if not found, recomputed and stored.

        Exception raised on any loading or storing error (incl. game mismatch)
        or if already overcomputed.
        """
        iterations = int(iterations)
        if self.iterations > iterations:
            raise ValueError(
                "Already computed {} iterations, more than {} requested to persist.".format(
                    self.iterations, iterations))
        fname = "{}-I{:07}.mccfr.bz2".format(basename, iterations)
        s = None
        try:
            with bz2.open(fname, 'rb') as f:
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

        if iterations > self.iterations:
            self.compute(iterations=iterations - self.iterations, epsilon=epsilon)
        with bz2.open(fname, 'wb') as f:
            oldlim = sys.getrecursionlimit()
            sys.setrecursionlimit(max(oldlim, 10000))
            try:
                pickle.dump(self, f)
            finally:
                sys.setrecursionlimit(oldlim)
        return False

    def compute(self, iterations, epsilon=0.6):
        """
        Run Outcome sampling MC CFR.
        """
        with ProgressReporter("OuterMCCFR", iterations) as pr:
            for it in range(iterations):
                pr.update(it)
                for player in range(self.game.players()):
                    self.sampling(player, epsilon=epsilon)
                self.iterations += 1

    def sampling(self, player, epsilon):
        "Run one sampling run for the given player."
        raise NotImplementedError


class OutcomeMCCFR(MCCFRBase):
    def outcome_sampling(self, state, player_updated, p_reach_updated, p_reach_others, p_sample,
                         epsilon):
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
            return self.outcome_sampling(state2, player_updated, p_reach_updated, p_reach_others,
                                         p_sample, epsilon)

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
        debug_assert(lambda: np.abs(np.sum(dist) - 1.0) < 1e-3)
        debug_assert(lambda: np.abs(np.sum(dist_sample) - 1.0) < 1e-3)
        action_idx = self.rng.choice(len(actions), p=dist_sample)
        action = actions[action_idx]

        # Future state
        state2 = state.play(action)

        if state.player() == player_updated:
            payoff, p_tail, p_sample_leaf = self.outcome_sampling(
                state2, player_updated, p_reach_updated * dist[action_idx], p_reach_others,
                p_sample * dist_sample[action_idx], epsilon)
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
                state2, player_updated, p_reach_updated, p_reach_others * dist[action_idx],
                p_sample * dist_sample[action_idx], epsilon)
            self.update_infoset(
                player, info, infoset, delta_s=p_reach_updated / p_sample_leaf * dist)

        return (payoff, p_tail * dist[action_idx], p_sample_leaf)

    def sampling(self, player, epsilon):
        s0 = self.game.initial_state()
        self.outcome_sampling(s0, player, 1.0, 1.0, 1.0, epsilon=epsilon)
