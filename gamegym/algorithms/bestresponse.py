from ..strategy import Strategy
from ..distribution import Explicit
from ..game import GameState, Game

import collections
import numpy as np

SupportItem = collections.namedtuple("SupportItem", ["state", "probability"])


class BestResponse(Strategy):

    def __init__(self, game, player, other_strategies):

        # DFS for from state to terminal state or stata of "player"
        def trace(state, probability, supports):
            # Just to get rid of nodes where distrbution returned pure zero
            if probability == 0.0:
                return 0.0
            p = state.player()
            if p == player:
                pi = state.player_information(player)
                s = supports.get(pi)
                if s is None:
                    s = []
                    supports[pi] = s
                s.append(SupportItem(state, probability))
                return 0
            if p == GameState.P_TERMINAL:
                return state.values()[player] * probability
            if p == GameState.P_CHANCE:
                distribution = state.chance_distribution()
            else:
                distribution = other_strategies[p].distribution(state)
            return sum(trace(state.play(action), pr * probability, supports)
                       for pr, action in zip(distribution.probabilities(),
                       distribution.values()))

        # DFS from isets to other isets of "player"
        def traverse(iset, support):
            actions = support[0].state.actions()
            values = []
            br_list = []
            for action in actions:
                new_supports = {}
                value = 0
                best_responses = {}
                br_list.append(best_responses)

                for s in support:
                    value += trace(s.state.play(action), s.probability, new_supports)
                for iset2, s in new_supports.items():
                    v, br = traverse(iset2, s)
                    value += v
                    best_responses.update(br)

                values.append(value)

            values = np.array(values)
            mx = values.max()
            is_best = values >= (mx - mx * 0e-6)
            br_result = {}
            br_result[iset] = Explicit(
                is_best.astype(np.float), actions, normalize=True)
            for br, is_b in zip(br_list, is_best):
                if is_b:
                    br_result.update(br)
            return mx, br_result

        supports = {}
        self.best_responses = {}
        value = trace(game.initial_state(), 1.0, supports)
        for iset2, s in supports.items():
            v, br = traverse(iset2, s)
            value += v
            self.best_responses.update(br)
        self.value = value

    def distribution(self, state):
        return self.best_responses[state.player_information(state.player())]


def exploitability(game, measured_player, strategy):
    """
    Exact exploitability of a player strategy in a two player ZERO-SUM game.
    """
    assert measured_player in (0, 1)
    assert isinstance(game, Game)
    assert isinstance(strategy, Strategy)
    BR = BestResponse(game, 1 - measured_player, [strategy, strategy])
    return BR.value