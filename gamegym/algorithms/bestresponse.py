from ..strategy import Strategy
from ..game import GameState, Game, Active
from .mccfr import OutcomeMCCFR, RegretStrategy
from ..utils import get_rng

import collections
import numpy as np

SupportItem = collections.namedtuple("SupportItem", ["state", "probability"])


class BestResponse(Strategy):
    """
    Compute a best-response strategy by game tree traversal.

    May be very computationaly demanding as it traverses the whole tree on creation.
    `strategies[player]` is ignored and may be e.g. `None`.
    """

    def __init__(self, game: Game, player: int, strategies, max_nodes=1e6):
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
                dist = state.active.chance
            else:
                dist = strategies[p].distribution(state)
            return sum(
                trace(game.play(state, action), pr * probability, supports)
                for pr, action in zip(dist, state.active.actions))

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

    def _distribution(self, observation, n_active, state=None):
        return self.best_responses[observation]


class ApproxBestResponse(Strategy):
    """
    Compute an approximate best-response strategy using MCCFR.

    Uses given number of iterations of OutcomeMCCFR.
    `strategies[player]` is ignored and may be e.g. `None`.
    """

    def __init__(self, game: Game, player: int, strategies, iterations, *, seed=None, rng=None):
        self.rng = get_rng(seed=seed, rng=rng)
        self.player = player
        self.game = game
        self.strategies = list(strategies)
        self.strategies[self.player] = RegretStrategy()
        self.mccfr = OutcomeMCCFR(game, self.strategies, [self.player], rng=self.rng)
        self.mccfr.compute(iterations, burn=0.5)

    def _distribution(self, observation, n_active, state=None):
        return self.strategies[self.player]._distribution(observation, n_active, state)

    def sample_value(self, iterations):
        val = self.game.sample_payoff(self.strategies, iterations, rng=self.rng)[0][self.player]
        return val


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


def approx_exploitability(game, measured_player, strategy, iterations, seed=None, rng=None):
    """
    Approximate exploitability of a player strategy in a two player ZERO-SUM game.

    Uses given number of iterations of OutcomeMCCFR.
    The value is then taken from a mean of `iterations / 4` plays.
    Note that the "best-response" strategy may be worse than the original if the
    iteration number is too small.
    """
    assert isinstance(game, Game)
    assert game.players == 2
    assert measured_player in (0, 1)
    assert isinstance(strategy, Strategy)
    rng = get_rng(seed=seed, rng=rng)
    br = ApproxBestResponse(game, 1 - measured_player, [strategy, strategy], iterations, rng=rng)
    return br.sample_value(iterations // 2)
