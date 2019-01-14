import pytest

from gamegym import LimitExceeded
from gamegym.algorithms import ApproxBestResponse, BestResponse
from gamegym.games import Goofspiel, RockPaperScissors
from gamegym.strategy import ConstStrategy, UniformStrategy


def test_best_response_rps():
    bart_simpson_strategy = ConstStrategy((1, 0, 0))
    game = RockPaperScissors()
    for p in [0, 1]:
        strategy = BestResponse(game, p, [bart_simpson_strategy] * 2)
        assert tuple(strategy.best_responses.values())[0] == pytest.approx((0.0, 1.0, 0.0))
        assert strategy.value == pytest.approx(1.0)


def test_approx_best_response_rps():
    bart_simpson_strategy = ConstStrategy((1, 0, 0))
    game = RockPaperScissors()

    for p in [0, 1]:
        s = ApproxBestResponse(game, 0, [bart_simpson_strategy] * 2, iterations=200, seed=23)
        assert s.strategy((), 3) == pytest.approx((0.0, 1.0, 0.0))
        assert s.sample_value(50) == pytest.approx(1.0)


@pytest.mark.slow
def test_best_response_goofspiel():
    for n_cards, br_value in [(3, pytest.approx(4 / 3)), (4, pytest.approx(2.5))]:
        game = Goofspiel(n_cards, Goofspiel.Scoring.ZEROSUM)
        strategy = BestResponse(game, 0, [UniformStrategy()] * 2)
        for k, v in strategy.best_responses.items():
            reward = k[-1]
            played_cards = k[0::3]
            idx = len([i for i in range(n_cards) if i < reward and i not in played_cards])
            assert reward in played_cards or v[idx] == 1.0
        assert strategy.value == br_value


@pytest.mark.slow
def test_approx_best_response_goofspiel():
    for n_cards, its, br_value in [(3, 1000, 1.333), (4, 20000, 2.5)]:
        game = Goofspiel(n_cards, Goofspiel.Scoring.ZEROSUM)
        strategy = ApproxBestResponse(game, 0, [UniformStrategy()] * 2, iterations=its, seed=35)
        assert strategy.sample_value(its // 2) == pytest.approx(br_value, rel=0.2)


def test_best_response_limit():
    game = Goofspiel(3)
    BestResponse(game, 0, [UniformStrategy()] * 2)
    with pytest.raises(LimitExceeded, message="traversed more than"):
        BestResponse(game, 0, [UniformStrategy()] * 2, max_nodes=1024)
