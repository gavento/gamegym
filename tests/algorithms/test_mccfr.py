import numpy as np
import pytest
import os

from gamegym.games import MatchingPennies, RockPaperScissors, MatrixZeroSumGame, Goofspiel
from gamegym.algorithms import BestResponse, OutcomeMCCFR, exploitability, RegretStrategy, approx_exploitability
from gamegym.strategy import UniformStrategy


def test_regret():
    mc = RegretStrategy()
    rs = mc.regret_matching(np.array([-1.0, 0.0, 1.0, 2.0]))
    assert rs == pytest.approx([0.0, 0.0, 1.0 / 3, 2.0 / 3])


def test_pennies():
    np.set_printoptions(precision=3)
    g = MatchingPennies()
    mc = OutcomeMCCFR(g, seed=12)
    mcs = mc.strategies
    mc.compute(500)
    s = g.start()
    assert mcs[0].strategy((), 2) == pytest.approx([0.5, 0.5], abs=0.1)
    assert mcs[1].strategy((), 2) == pytest.approx([0.5, 0.5], abs=0.1)
    s = g.play(s, index=1)
    assert mcs[0].strategy((), 2) == pytest.approx([0.5, 0.5], abs=0.1)
    assert mcs[1].strategy((), 2) == pytest.approx([0.5, 0.5], abs=0.1)


def test_mccfr_goofspiel3():
    g = Goofspiel(3, scoring=Goofspiel.Scoring.ZEROSUM)
    mc = OutcomeMCCFR(g, seed=51)
    mc.compute(600, burn=0.5)
    mcs = mc.strategies
    us = UniformStrategy()
    s1 = g.play_sequence([2])
    assert mcs[0].strategy(s1) == pytest.approx([0., 0.9, 0.], abs=0.1)
    assert g.sample_payoff(mcs, 300, seed=12)[0] == pytest.approx([0.0, 0.0], abs=0.1)
    assert g.sample_payoff((mcs[0], us), 300, seed=13)[0] == pytest.approx([1.2, -1.2], abs=0.2)
    assert exploitability(g, 0, mcs[0]) < .55
    assert exploitability(g, 1, mcs[1]) < .55


@pytest.mark.slow
def test_mccfr_goofspiel4():
    g = Goofspiel(4, scoring=Goofspiel.Scoring.ZEROSUM)
    mc = OutcomeMCCFR(g, seed=49)
    mc.compute(10000, burn=0.5)
    mcs = mc.strategies
    for p in [0, 1]:
        exp = exploitability(g, p, mcs[p])
        aexp = approx_exploitability(g, p, mcs[p], 10000, seed=31 + p)
        print(p, exp, aexp)
        assert exp == pytest.approx(0.8, abs=0.2)
        assert aexp == pytest.approx(0.7, abs=0.2)
