from ..game import Game, GameState
from ..utils import get_rng
from ..distribution import Explicit
from ..strategy import Strategy
import numpy as np


class InformationSetSampler:
    def __init__(self, game, strategies, for_players=None):
        self.game = game
        if for_players is None:
            self.players = tuple(range(self.game.players()))
        else:
            self.players = tuple(for_players)
        self.strategies = strategies
        if isinstance(self.strategies, Strategy):
            self.strategies = [self.strategies] * self.game.players()
        assert len(self.strategies) == self.game.players()

        # temporary, {player: { player_info: [RecState(prev_rec_state, prev_action, p_reach)] }}
        self._infoset_history_dist = {p: {} for p in self.players}
        # temporary, {player: { player_info: p_reach }}
        self._infoset_dist = {p: {} for p in self.players}
        # temporary, {player: p_total }
        self._player_dist = {p: 0.0 for p in self.players}

        # Run the trace
        self.trace(self.game.initial_state(), 1.0, None, None)

        # Finalize the sets
        self.infoset_history_dist = {
            p: {
                info: Explicit([i[2] for i in isets], isets, normalize=True)
                for info, isets in self._infoset_history_dist[p].items()}
            for p in self.players}
        # final, {player: { player_info: Explicit[(prev_rec_state, prev_action, p_reach)] }}
        self.infoset_dist = {p: Explicit(self._infoset_dist[p], normalize=True)
            for p in self.players}
        # final, {player: Explicit[player_info] }
        self.player_dist = Explicit(self._player_dist, normalize=True)
        # final, Explicit[player]

    def trace(self, state, p_reach, prev_rec_state, prev_action):
        player = state.player()
        info = state.player_information(player)
        rec_state = (prev_rec_state, prev_action, p_reach)

        if player in self.players:
            p_ihd = self._infoset_history_dist[player]
            p_ihd_set = p_ihd.setdefault(info, list())
            p_ihd_set.append(rec_state)
            p_id = self._infoset_dist[player]
            p_id[info] = p_id.get(info, 0.0) + p_reach
            self._player_dist[player] += p_reach

        if state.is_terminal():
            return
        if state.is_chance():
            dist = state.chance_distribution()
        else:
            dist = self.strategies[player].distribution(state)
        for a in state.actions():
            self.trace(state.play(a), p_reach * dist.probability(a), rec_state, a)

    def sample_player(self, rng=None):
        "Return (player, p_sampled)"
        player = self.player_dist.sample(rng=rng)
        return (player, self.player_dist.probability(player))

    def sample_info(self, player=None, rng=None):
        "Return (player, info, p_sampled)"
        p_sample = 1.0
        if player is None:
            player = self.player_dist.sample(rng=rng)
            p_sample *= self.player_dist.probability(player)
        info = self.infoset_dist[player].sample(rng=rng)
        return (player, info, self.infoset_dist[player].probability(info))

    def sample_state(self, player=None, info=None, rng=None):
        "Return (player, info, state, p_sampled)"
        p_sample = 1.0
        if player is None:
            player = self.player_dist.sample(rng=rng)
            p_sample *= self.player_dist.probability(player)
        if info is None:
            info = self.infoset_dist[player].sample(rng=rng)
            p_sample *= self.infoset_dist[player].probability(info)
        rec_state = self.infoset_history_dist[player][info].sample(rng=rng)
        p_sample *= self.infoset_history_dist[player][info].probability(rec_state)
        state = self.reconstruct_state(rec_state)
        return (player, info, state, p_sample)

    def reconstruct_state(self, rec_state):
        if rec_state[0] is None:
            return self.game.initial_state()
        prev_state = self.reconstruct_state(rec_state[0])
        return prev_state.play(rec_state[1])

