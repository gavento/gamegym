from gamegym.algorithms import OutcomeMCCFR, BestResponse, SparseStochasticValueLearning
from gamegym.algorithms.valuesgd import GoofSpielCardsValueStore
from gamegym.games import Goofspiel
import numpy as np


def test_unit():
    g = Goofspiel(4, scoring=Goofspiel.Scoring.ZEROSUM)
    mc = OutcomeMCCFR(g, seed=42)
    mc.compute(500)
    vs = GoofSpielCardsValueStore(g)
    val = SparseStochasticValueLearning(g, vs, seed=43)
    val.compute([mc, mc], 200, 0.01)


def non_test_goofspiel():
    g = Goofspiel(4, scoring=Goofspiel.Scoring.ZEROSUM)
    mc = OutcomeMCCFR(g, seed=42)
    for s in [10, 100, 1000]:
        mc.compute(s)
        br = BestResponse(g, 0, [None, mc])
        print("Exploit after", s, np.mean([g.play_strategies([br, mc], seed=i)[-1].values()[0] for i in range(1000)]))

    vs = GoofSpielCardsValueStore(g)
    val = SparseStochasticValueLearning(g, vs, seed=43)
    for alpha in [0.1, 0.01, 0.01, 0.001, 0.0001]:
        print(alpha)
        val.compute([mc, mc], 200, alpha)
