import logging
from typing import Any

import numpy as np
from attr import attrib, attrs

from ..observation import Observation
from ..situation import Action, Situation, StateInfo
from ..strategy import Strategy
from ..utils import Distribution, debug_assert, get_rng, np_uniform, uniform
from ..game import Game
from ..adapter import Adapter

class RegretStrategy(Strategy):
    """
    Average strategy and regret storage for one player.

    Dictionary based, any unknown observations return uniform distributions.
    """
    DEFAULT_ADAPTER = "HashableAdapter"
    EPS = 1e-30

    def __init__(self, game: Game, adapter: Adapter = None):
        super().__init__(game, adapter)
        # (player, observation): (regrets, strategy)
        self.table = {}
        # usage statistics
        self.queries = 0
        self.misses = 0
        # training statistics
        self.updates = 0
        self.iterations = 0

    def get_entry(self, observation_data: Any, actions: int) -> tuple:
        entry = self.table.get(observation_data, None)
        if entry is None:
            entry = (np.zeros(actions), np.zeros(actions))
        else:
            assert len(entry[0]) == actions
            assert len(entry[1]) == actions
        return entry

    def update_entry(self, observation_data: Any, actions: int, dr=None, ds=None) -> tuple:
        entry = self.table.get(observation_data, None)
        if entry is None:
            entry = (np.zeros(actions), np.zeros(actions))
        nr = (entry[0] + dr) if dr is not None else entry[0]
        ns = (entry[1] + ds) if ds is not None else entry[1]
        self.table[observation_data] = (nr, ns)
        self.updates += 1

    def regret_matching(self, regret):
        "Return stratefy distribution based on the regret vector"
        regplus = np.clip(regret, 0, None)
        s = np.sum(regplus)
        if s > self.EPS:
            return regplus / s
        else:
            return np_uniform(len(regret))

    def make_policy(self, observation: Observation) -> Distribution:
        self.queries += 1
        entry = self.table.get(observation.data, None)
        if entry is not None and np.sum(entry[1]) > self.EPS:
            return Distribution(observation.actions, entry[1] / np.sum(entry[1]))
        else:
            self.misses += 1
            return Distribution(observation.actions, None)


class MCCFRBase:
    """
    Common base for Outcome and External sampling MC CFR.
    """

    def __init__(self, game: Game, strategies=None, update=None, *, seed=None, rng=None):
        assert isinstance(game, Game)
        self.game = game
        self.rng = get_rng(rng, seed)
        self.strategies = strategies
        if self.strategies is None:
            self.strategies = tuple(RegretStrategy(self.game) for i in range(self.game.players))
        assert len(self.strategies) == self.game.players
        self.update = update
        if self.update is None:
            self.update = tuple(range(self.game.players))
        for i in self.update:
            assert isinstance(self.strategies[i], RegretStrategy)
        # stats
        self.iterations = 0
        self.nodes_traversed = 0

    def compute(self, iterations, epsilon=0.6, weight=1.0, progress=True, burn=0.0):
        """
        Run MC CFR for given iterations.

        Optionally uses a progress bar (default on).

        Updates to the cummulative strategy and regret are weighted by `weight`, this
        allows you to discount early iterations.
        If `burn > 0.0`, perform smooth burn-in by multiplying `weight` by a coefficient
        going from 0.03 to 1.0 in the first `burn`-fraction of iterations (off by default,
        a sensible choice is e.g. 0.3).
        """
        log = logging.getLogger('gamegym.MCCFR')
        log.debug("Computing {} for {} (iterations={}, weight={:.4g}, epsilon={:.4g})".format(
            self.__class__.__name__, repr(self.game), iterations, weight, epsilon))
        its = range(iterations)
        if progress:
            import tqdm
            its = tqdm.tqdm(its, desc="MCCFR")
        if burn > 0.0:
            assert burn <= 1.0
        for i in its:
            self.iterations += 1
            if i < burn * iterations:
                # This was tuned on Goofspiel(4); range 0.01-0.1 seems to be reasonable.
                w = 0.03**(1.0 - float(i) / iterations / burn)
            else:
                w = 1.0
            if progress:
                r = "nodes: {}".format(self.nodes_traversed)
                if burn > 0.0:
                    r += ", burn-in: {:.2f}".format(w)
                its.set_postfix_str(r)
            for player in self.update:
                self.sampling(player, epsilon=epsilon, weight=w * weight)
            # When only one player is updated, we need to traverse as a dummy player
            # to update the cumulative strategies.
            # Using player -1 as the updated one is a bit hacky but works.
            if len(self.update) == 1:
                self.sampling(-1, epsilon=epsilon, weight=w * weight)

    def sampling(self, player, epsilon=0.6, weight=1.0):
        "Run one sampling run for the given player."
        raise NotImplementedError


class OutcomeMCCFR(MCCFRBase):
    def _outcome_sampling(self, situation, player_updated, p_reach_updated, p_reach_others,
                          p_sample, epsilon, weight):
        """
        Based on Alg 3 from PhD_Thesis_MarcLanctot.pdf and cfros.cpp from his bluff11.zip.
        Returns `(utility, p_tail, p_sample_leaf)`.
        """
        self.nodes_traversed += 1

        if situation.is_terminal():
            return (situation.payoff[player_updated], 1.0, p_sample)

        if situation.is_chance():
            ai = self.rng.choice(len(situation.actions), p=situation.chance)
            sit2 = self.game.play(situation, situation.actions[ai])
            # No need to factor in the chances in Outcome sampling
            return self._outcome_sampling(sit2, player_updated, p_reach_updated, p_reach_others,
                                          p_sample, epsilon, weight)

        # Extract misc, read entry from storage
        player = situation.player
        strat = self.strategies[player]
        obs = strat.adapter.get_observation(situation, player)
        actions = situation.actions

        # Treat static players as chance nodes
        if player not in self.update:
            dist = strat.make_policy(obs).probs
            ai = self.rng.choice(len(situation.actions), p=dist)
            sit2 = self.game.play(situation, situation.actions[ai])
            # No need to factor in the chances in Outcome sampling
            return self._outcome_sampling(sit2, player_updated, p_reach_updated, p_reach_others,
                                          p_sample, epsilon, weight)

        # Create dists, sample the action
        entry = strat.get_entry(obs.data, len(actions))
        dist = strat.regret_matching(entry[0])
        # exploration in self-actions
        if player == player_updated:
            dist_sample = dist * (1.0 - epsilon) + 1.0 * epsilon / len(actions)
        else:
            dist_sample = dist
        debug_assert(lambda: np.abs(np.sum(dist) - 1.0) < 1e-3)
        debug_assert(lambda: np.abs(np.sum(dist_sample) - 1.0) < 1e-3)
        action_idx = self.rng.choice(len(actions), p=dist_sample)
        action = actions[action_idx]

        # Future situation
        sit2 = self.game.play(situation, action)

        if player == player_updated:
            # Update regret / current strategy
            payoff, p_tail, p_sample_leaf = self._outcome_sampling(
                sit2, player_updated, p_reach_updated * dist[action_idx], p_reach_others,
                p_sample * dist_sample[action_idx], epsilon, weight)
            dr = np.zeros_like(entry[0])
            U = payoff * p_reach_others / p_sample_leaf
            for ai in range(len(actions)):
                if ai == action_idx:
                    dr[ai] = U * (p_tail - p_tail * dist[action_idx])
                else:
                    dr[ai] = -U * p_tail * dist[action_idx]
            strat.update_entry(obs.data, len(actions), dr=dr * weight)
        else:
            # Update cumulative strategy
            payoff, p_tail, p_sample_leaf = self._outcome_sampling(
                sit2, player_updated, p_reach_updated, p_reach_others * dist[action_idx],
                p_sample * dist_sample[action_idx], epsilon, weight)
            ds = (p_reach_others / p_sample) * dist
            strat.update_entry(obs.data, len(actions), ds=ds * weight)

        return (payoff, p_tail * dist[action_idx], p_sample_leaf)

    def sampling(self, updated_player, epsilon=0.6, weight=1.0):
        """
        Run one outcome sampling for the given player.
        """
        if updated_player >= 0:
            self.strategies[updated_player].iterations += 1
        s0 = self.game.start()
        self._outcome_sampling(s0, updated_player, 1.0, 1.0, 1.0, epsilon=epsilon, weight=weight)
