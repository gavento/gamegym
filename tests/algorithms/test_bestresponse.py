from gamegym.games import RockPaperScissors, Goofspiel
from gamegym.strategy import ConstStrategy, UniformStrategy
from gamegym.algorithms import BestResponse
import pytest


def test_best_response_rps():

    bart_simpson_strategy = ConstStrategy((1, 0, 0))
    game = RockPaperScissors()
    strategy = BestResponse(game, 0, [bart_simpson_strategy] * 2)
    assert tuple(strategy.best_responses.values())[0] == pytest.approx((0.0, 1.0, 0.0))
    assert strategy.value == pytest.approx(1.0)

    strategy = BestResponse(game, 1, [bart_simpson_strategy] * 2)
    assert tuple(strategy.best_responses.values())[0] == pytest.approx((0.0, 1.0, 0.0))
    assert strategy.value == pytest.approx(1.0)


@pytest.mark.slow
def test_best_response_goofspiel():

    for n_cards, br_value in [(3, pytest.approx(4 / 3)), (4, pytest.approx(2.5))]:
        game = Goofspiel(n_cards, Goofspiel.Scoring.ZEROSUM)
        strategy = BestResponse(game, 0, [UniformStrategy()] * 2)
        for k, v in strategy.best_responses.items():
            reward = k[-1].obs
            played_cards = [o.obs for o in k[0::2]]
            idx = len([i for i in range(n_cards) if i < reward and i not in played_cards])
            assert reward in played_cards or v[idx] == 1.0
        assert strategy.value == br_value
