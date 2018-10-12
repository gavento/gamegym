from ..game import Game, GameState
from ..utils import get_rng
from ..distribution import Explicit
from ..strategy import Strategy
import numpy as np


class InformationSetSampler:
    """
    A helper class to sample information sets and their elements (histories) with the
    probabilities proportional to given strategies.

    This algorithm traverses all possible game histories so it may be very slow for
    medium to large games. Once precomputed, sampling is very fast: O(log |Actions|).

    Read-only attributes:

    * `game`: The `Game` instance.
    * `nodes`: The number of nodes (histories) traversed.
    * `strategies`: The strategies this was computed for.
    """
    def __init__(self, game, strategies, for_players=None, max_nodes=None):
        """
        Compute the information sets for given game and strategies.
        Optionally, you may limit the players this is computed for and
        the number of nodes traversed.
        """
        self.game = game
        if for_players is None:
            self.players = tuple(range(self.game.players()))
        else:
            self.players = tuple(for_players)
        self.strategies = strategies
        if isinstance(self.strategies, Strategy):
            self.strategies = [self.strategies] * self.game.players()
        assert len(self.strategies) == self.game.players()
        self.nodes = 0
        self.max_nodes = max_nodes

        # temporary, {player: { player_info: [RecState(prev_rec_state, prev_action, p_reach)] }}
        self._tmp_infoset_history_dist = {p: {} for p in self.players}
        # temporary, {player: { player_info: p_reach }}
        self._tmp_infoset_dist = {p: {} for p in self.players}
        # temporary, {player: p_total }
        self._tmp_player_dist = {p: 0.0 for p in self.players}

        # Run the trace
        self._trace(self.game.initial_state(), 1.0, None, None)

        # Finalize the sets
        self._infoset_history_dist = {
            p: {
                info: Explicit([i[2] for i in isets], isets, normalize=True)
                for info, isets in self._tmp_infoset_history_dist[p].items()}
            for p in self.players}
        # final, {player: { player_info: Explicit[(prev_rec_state, prev_action, p_reach)] }}
        self._infoset_dist = {p: Explicit(self._tmp_infoset_dist[p], normalize=True)
            for p in self.players}
        # final, {player: Explicit[player_info] }
        self._player_dist = Explicit(self._tmp_player_dist, normalize=True)
        # final, Explicit[player]

    def _trace(self, state, p_reach, prev_rec_state, prev_action):
        "Internal recursive history tracer."
        player = state.player()
        info = state.player_information(player)
        rec_state = (prev_rec_state, prev_action, p_reach)
        self.nodes += 1
        if self.max_nodes is not None and self.nodes > self.max_nodes:
            raise Exception("InformationSetSampler computation reached node limit")

        if player in self.players:
            p_ihd = self._tmp_infoset_history_dist[player]
            p_ihd_set = p_ihd.setdefault(info, list())
            p_ihd_set.append(rec_state)
            p_id = self._tmp_infoset_dist[player]
            p_id[info] = p_id.get(info, 0.0) + p_reach
            self._tmp_player_dist[player] += p_reach

        if state.is_terminal():
            return
        if state.is_chance():
            dist = state.chance_distribution()
        else:
            dist = self.strategies[player].distribution(state)
        for a in state.actions():
            self._trace(state.play(a), p_reach * dist.probability(a), rec_state, a)

    def sample_player(self, rng=None):
        """
        Return `(player, p_sampled)`.
        """
        player = self._player_dist.sample(rng=rng)
        return (player, self._player_dist.probability(player))

    def player_distribution(self):
        """
        Return distribution over proportional to reach prob. of their active states.

        NOTE: The distribution is explicitly precomputed so this is instantaneus.
        """
        return self._player_dist

    def sample_info(self, player=None, rng=None):
        """
        Return `(player, info, p_sampled)`.
        
        Here `p_sampled=P(info|player)`, or `p_sampled=P(info, player)` if `player` is not given.
        """
        p_sample = 1.0
        if player is None:
            player = self._player_dist.sample(rng=rng)
            p_sample *= self._player_dist.probability(player)
        info = self._infoset_dist[player].sample(rng=rng)
        return (player, info, self._infoset_dist[player].probability(info))

    def info_distribution(self, player):
        """
        Return distribution over player informations (information sets) of the given player.

        NOTE: The distribution is explicitly precomputed so this is instantaneus.
        """
        return self._infoset_dist[player]

    def sample_state(self, player=None, info=None, rng=None):
        """
        Return `(player, info, state, p_sampled)`.

        Here `p_sampled=P(state|info, player)`, or `p_sampled=P(state, info|player)` if `player` is not given,
        or `p_sampled=P(state, info, player)` if none are given.
        """
        p_sample = 1.0
        if player is None:
            player = self._player_dist.sample(rng=rng)
            p_sample *= self._player_dist.probability(player)
        if info is None:
            info = self._infoset_dist[player].sample(rng=rng)
            p_sample *= self._infoset_dist[player].probability(info)
        rec_state = self._infoset_history_dist[player][info].sample(rng=rng)
        p_sample *= self._infoset_history_dist[player][info].probability(rec_state)
        state = self._reconstruct_state(rec_state)
        return (player, info, state, p_sample)

    def state_distribution(self, player, info):
        """
        Return distribution over the `GameState`s of the given information set.

        WARNING: This constucts all the states of the information set every time
        you call it. If you only want to sample the states, use `self.sample_state`.
        """
        dist = self._infoset_history_dist[player][info]
        return Explicit(dist.probabilities(),
                        values=[self._reconstruct_state(rec) for rec in dist.values()])

    def _reconstruct_state(self, rec_state):
        "Internal, reconstructs the GameState from given history."
        if rec_state[0] is None:
            return self.game.initial_state()
        prev_state = self._reconstruct_state(rec_state[0])
        return prev_state.play(rec_state[1])

