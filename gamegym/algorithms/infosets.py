from ..game import Game
from ..situation import Situation
from ..strategy import Strategy
from ..utils import get_rng, Distribution
from ..strategy import Strategy
from ..errors import LimitExceeded
import numpy as np
from typing import Iterable


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

    def __init__(self,
                 game: Game,
                 strategies: Iterable[Strategy],
                 for_players: Iterable[int] = None,
                 max_nodes: int = 1e6):
        """
        Compute the information sets for given game and strategies.
        Optionally, you may limit the players this is computed for and
        the number of nodes traversed.
        """
        self.game = game
        if for_players is None:
            self.players = tuple(range(self.game.players))
        else:
            self.players = tuple(for_players)
        self.strategies = strategies
        assert len(self.strategies) == game.players
        self.nodes = 0
        self.max_nodes = max_nodes

        # temporary, {player: { observation: [RecState(prev_rec_state, prev_action, p_reach)] }}
        self._tmp_infoset_history_dist = [{} for p in range(self.game.players)]
        # temporary, {player: { observation: p_reach }}
        self._tmp_infoset_dist = [{} for p in range(self.game.players)]
        # temporary, {player: p_total }
        self._tmp_player_dist = [0.0 for p in range(self.game.players)]

        # Run the trace
        self._trace(self.game.start(), 1.0, None, None)

        # distributions are pairs: (values, probs)
        def _dist(values, probs):
            if not isinstance(values, tuple):
                values = tuple(values)
            assert len(values) == len(probs)
            s = np.sum(probs)
            assert s > 1e-6
            return (values, np.array(probs) / s)

        # Finalize the sets
        # [{ observation: Distribution( (prev_rec_state, prev_action, p_reach) ) }]
        self._infoset_history_dist = [{
            obs: Distribution(support, np.fromiter((i[2] for i in support), float), norm=True)
            for obs, support in self._tmp_infoset_history_dist[p].items()
        } for p in range(self.game.players)]
        # [Distribution(observation)]
        self._infoset_dist = [
            Distribution(
                tuple(self._tmp_infoset_dist[p].keys()),
                np.fromiter(self._tmp_infoset_dist[p].values(), float),
                norm=True) for p in range(self.game.players)
        ]
        # Distribution(player)
        self._player_dist = Distribution(None, self._tmp_player_dist, norm=True)

    def _trace(self, state, p_reach, prev_rec_state, prev_action):
        "Internal recursive history tracer."
        player = state.player
        rec_state = (prev_rec_state, prev_action, p_reach)
        self.nodes += 1
        if self.nodes > self.max_nodes:
            raise LimitExceeded("InformationSetSampler computation reached node limit {}.".format(
                self.max_nodes))

        if player in self.players:
            obs = state.observations[player]
            p_ihd = self._tmp_infoset_history_dist[player]
            p_ihd_set = p_ihd.setdefault(obs, list())
            p_ihd_set.append(rec_state)
            p_id = self._tmp_infoset_dist[player]
            p_id[obs] = p_id.get(obs, 0.0) + p_reach
            self._tmp_player_dist[player] += p_reach

        if state.is_terminal():
            return
        if state.is_chance():
            dist = state.chance
        else:
            dist = self.strategies[player].strategy(state)
        assert len(dist) == len(state.actions)
        for a, p_a in zip(state.actions, dist):
            self._trace(self.game.play(state, a), p_reach * p_a, rec_state, a)

    def sample_player(self, rng=None) -> int:
        """
        Return `(player, p_sampled)`.
        """
        return self._player_dist.sample_with_p(rng)

    def player_distribution(self) -> Distribution:
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
            player, p = self.sample_player(rng)
            p_sample *= p
        info, p = self._infoset_dist[player].sample_with_p(rng)
        p_sample *= p
        return (player, info, p_sample)

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
            player, p = self._player_dist.sample_with_p(rng=rng)
            p_sample *= p
        if info is None:
            info, p = self._infoset_dist[player].sample_with_p(rng=rng)
            p_sample *= p
        rec_state, p = self._infoset_history_dist[player][info].sample_with_p(rng=rng)
        p_sample *= p
        state = self._reconstruct_state(rec_state)
        return (player, info, state, p_sample)

    def state_distribution(self, player, info):
        """
        Return distribution over the `GameState`s of the given information set.

        WARNING: This constucts all the states of the information set every time
        you call it. If you only want to sample the states, use `self.sample_state`.
        """
        dist = self._infoset_history_dist[player][info]
        return Distribution([self._reconstruct_state(rec) for rec in dist.vals], dist.probs)

    def _reconstruct_state(self, rec_state):
        "Internal, reconstructs the GameState from given history."
        if rec_state[0] is None:
            return self.game.start()
        prev_state = self._reconstruct_state(rec_state[0])
        return self.game.play(prev_state, rec_state[1])
