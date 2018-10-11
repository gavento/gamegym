from ..strategy import Strategy
from ..distribution import Explicit
from ..game import GameState

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


class Exploitability:
    def __init__(self, game, strategies):
        self.game = game
        if isinstance(strategies, Strategy):
            strategies = (strategies, strategies)
        assert game.players() == 2 and len(strategies) == 2

        self.BRvsP0 = BestResponse(self.game, 1, strategies)
        self.BRvsP1 = BestResponse(self.game, 0, strategies)

        self.value = self.BRvsP0.value - self.BRvsP1.value

    def __repr__(self):
        return "<Exploitability of {}: {} (BR val vs p0: {}, BR val vs p1: {})>".format(
            self.game, self.value, self.BRvsP0.value, self.BRvsP1.value)