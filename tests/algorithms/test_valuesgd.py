from gamegym.algorithms import OutcomeMCCFR, BestResponse, SSValueLearning
from gamegym.algorithms.valuesgd import LinearZeroSumValueStore
from gamegym.games.goofspiel import goofspiel_feaures_cards
from gamegym.games.matrix import matrix_zerosum_features
from gamegym.games import Goofspiel, RockPaperScissors
from gamegym.strategy import UniformStrategy
from gamegym.utils import get_rng
import numpy as np


def test_goofspiel():
    g = Goofspiel(4, scoring=Goofspiel.Scoring.ZEROSUM)
    mc = OutcomeMCCFR(g, seed=42)
    mc.compute(500)
    vs = LinearZeroSumValueStore(g, goofspiel_feaures_cards, normalize_mean=2.5)
    val = SSValueLearning(g, vs, seed=43)
    val.compute([mc, mc], 200, 0.01, 0.1)


def test_rps():
    g = RockPaperScissors()
    us = UniformStrategy()
    rng = get_rng(seed=3)
    params = rng.rand(3, 3) - 0.5
    vs = LinearZeroSumValueStore(g, matrix_zerosum_features, initializer=params,
                                 force_mean=0.0, normalize_l2=1.0)
    val = SSValueLearning(g, vs, seed=44)
    val.compute([us, us], 2000, 0.01, 0.0001)
    val.compute([us, us], 2000, 0.001, 0.00001)
    val.compute([us, us], 2000, 0.0001, 0.000001)


def non_test_goofspiel():
    g = Goofspiel(4, scoring=Goofspiel.Scoring.ZEROSUM)
    mc = OutcomeMCCFR(g, seed=42)
    for s in [10, 100, 1000]:
        mc.compute(s)
        br = BestResponse(g, 0, [None, mc])
        print("Exploit after", s, np.mean([g.play_strategies([br, mc], seed=i)[-1].values()[0] for i in range(1000)]))

    vs = LinearZeroSumValueStore(g)
    val = SSValueLearning(g, vs, seed=43)
    for alpha in [0.1, 0.01, 0.01, 0.001, 0.0001]:
        print(alpha)
        val.compute([mc, mc], 200, alpha)
