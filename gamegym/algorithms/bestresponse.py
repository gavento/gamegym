from ..strategy import Strategy
from ..game import GameState, Game, Active

import collections
import numpy as np

SupportItem = collections.namedtuple("SupportItem", ["state", "probability"])


class BestResponse(Strategy):
    def __init__(self, game, player, strategies, max_nodes=1e6):
        assert isinstance(game, Game)
        assert player < game.players
        assert len(strategies) == game.players
        nodes = 0

        # DFS for from state to terminal state or stata of "player"
        def trace(state, probability, supports):
            nonlocal nodes
            nodes += 1
            if nodes > max_nodes:
                raise Exception(
                    "BestResponse traversed more than allowed {} nodes.".format(max_nodes) +
                    "Either increase the limit or consider using approximate best response.")
            # Just to get rid of nodes where distrbution returned pure zero
            if probability == 0.0:
                return 0.0
            p = state.active.player
            if p == player:
                pi = state.observations[player]
                s = supports.setdefault(pi, list())
                s.append(SupportItem(state, probability))
                return 0
            if p == Active.TERMINAL:
                return state.active.payoff[player] * probability
            if p == Active.CHANCE:
                distribution = state.active.chance
            else:
                distribution = strategies[p].distribution(state.observations[p], state.active)
            return sum(
                trace(game.play(state, action), pr * probability, supports)
                for pr, action in zip(distribution, state.active.actions))

        # DFS from isets to other isets of "player"
        def traverse(iset, support):
            actions = support[0].state.active.actions
            values = []
            br_list = []
            for action in actions:
                new_supports = {}
                value = 0
                best_responses = {}
                br_list.append(best_responses)

                for s in support:
                    value += trace(game.play(s.state, action), s.probability, new_supports)
                for iset2, s in new_supports.items():
                    v, br = traverse(iset2, s)
                    value += v
                    best_responses.update(br)

                values.append(value)

            values = np.array(values)
            mx = values.max()
            is_best = values >= (mx - mx * 0e-6)
            br_result = {}
            bdist = is_best.astype(np.float)
            br_result[iset] = bdist / sum(bdist)
            for br, is_b in zip(br_list, is_best):
                if is_b:
                    br_result.update(br)
            return mx, br_result

        supports = {}
        self.best_responses = {}
        value = trace(game.start(), 1.0, supports)
        for iset2, s in supports.items():
            v, br = traverse(iset2, s)
            value += v
            self.best_responses.update(br)
        self.value = value

    def distribution(self, observation, _active):
        return self.best_responses[observation]


def exploitability(game, measured_player, strategy, max_nodes=1e6):
    """
    Exact exploitability of a player strategy in a two player ZERO-SUM game.
    """
    assert measured_player in (0, 1)
    assert isinstance(game, Game)
    assert game.players == 2
    assert isinstance(strategy, Strategy)
    br = BestResponse(game, 1 - measured_player, [strategy, strategy], max_nodes)
    return br.value
