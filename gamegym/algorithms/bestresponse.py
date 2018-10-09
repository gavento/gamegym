from ..strategy import Strategy
from ..distribution import Explicit
from ..game import GameState

import collections
import numpy as np

SupportItem = collections.namedtuple("SupportItem", ["state", "probability"])


class BestResponse(Strategy):

    def __init__(self, game, player, other_strategies):

        def trace(state, probability, supports):
            if probability < 0e-30:
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

        def traverse(iset, support):
            actions = support[0].state.actions()
            values = []
            br_list = []
            for action in actions:
                new_supports = {}
                value = 0
                for s in support:
                    value += s.probability * trace(s.state.play(action), 1.0, new_supports)
                for iset2, s in new_supports.items():
                    v, br = traverse(iset2, s)
                    value += v
                    br_list.append(br)

                values.append(value)

            values = np.array(values)
            mx = values.max()
            m = mx - mx * 0e-6
            is_best = values >= m
            br_result = {}
            br_result[iset] = Explicit(
                is_best.astype(np.float), actions, normalize=True)
            for br, is_b in zip(br_list, is_best):
                if is_b:
                    br_result.update(br)
            return mx, br_result

        supports = {}
        self.best_responses = {}
        trace(game.initial_state(), 1.0, supports)
        for iset2, s in supports.items():
            _, br = traverse(iset2, s)
            self.best_responses.update(br)

    def distribution(self, state):
        return self.best_responses[state.player_information(state.player())]


def test_best_response_rps():
    from ..games.rps import RockPaperScissors
    from ..strategy import FixedStrategy

    bart_simpson_strategy = FixedStrategy(Explicit([1, 0, 0], values=["R", "P", "S"]))
    game = RockPaperScissors()
    strategy = BestResponse(game, 0, {1: bart_simpson_strategy})
    #print(strategy.best_responses)
    assert list(strategy.best_responses.values())[0].probability("R") == 0.0
    assert list(strategy.best_responses.values())[0].probability("P") == 1.0
    assert list(strategy.best_responses.values())[0].probability("S") == 0.0

    strategy = BestResponse(game, 1, {0: bart_simpson_strategy})
    #print(strategy.best_responses)
    assert list(strategy.best_responses.values())[0].probability("R") == 0.0
    assert list(strategy.best_responses.values())[0].probability("P") == 1.0
    assert list(strategy.best_responses.values())[0].probability("S") == 0.0


def test_best_response_goofspiel():
    from ..games.goofspiel import Goofspiel, GoofspielScoring
    from ..strategy import UniformStrategy

    for n_cards in [4]:
        game = Goofspiel(n_cards, GoofspielScoring.ZEROSUM)
        strategy = BestResponse(game, 0, {1: UniformStrategy()})
        for k, v in strategy.best_responses.items():
            reward = k[2][-1]
            assert reward not in v.values() or v.probability(reward) == 1.0