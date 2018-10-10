from gamegym.games.rps import RockPaperScissors
from gamegym.games.goofspiel import Goofspiel
from gamegym.strategy import FixedStrategy, UniformStrategy
from gamegym.algorithms.bestresponse import BestResponse
from gamegym.distribution import Explicit
import pytest


def test_best_response_rps():

    bart_simpson_strategy = FixedStrategy(Explicit([1, 0, 0], values=["R", "P", "S"]))
    game = RockPaperScissors()
    strategy = BestResponse(game, 0, {1: bart_simpson_strategy})
    assert list(strategy.best_responses.values())[0].probability("R") == 0.0
    assert list(strategy.best_responses.values())[0].probability("P") == 1.0
    assert list(strategy.best_responses.values())[0].probability("S") == 0.0
    assert strategy.value == pytest.approx(1.0)

    strategy = BestResponse(game, 1, {0: bart_simpson_strategy})
    assert list(strategy.best_responses.values())[0].probability("R") == 0.0
    assert list(strategy.best_responses.values())[0].probability("P") == 1.0
    assert list(strategy.best_responses.values())[0].probability("S") == 0.0
    assert strategy.value == pytest.approx(1.0)


def test_best_response_goofspiel():

    for n_cards, br_value in [(3, pytest.approx(4/3)), (4, pytest.approx(2.5))]:
        game = Goofspiel(n_cards, Goofspiel.Scoring.ZEROSUM)
        strategy = BestResponse(game, 0, {1: UniformStrategy()})
        for k, v in strategy.best_responses.items():
            reward = k[2][-1]
            assert reward not in v.values() or v.probability(reward) == 1.0
        assert strategy.value == br_value