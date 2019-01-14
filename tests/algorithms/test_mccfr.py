import os

import numpy as np
import pytest

from gamegym.algorithms import (BestResponse, OutcomeMCCFR, RegretStrategy, approx_exploitability,
                                exploitability)
from gamegym.algorithms.stats import sample_payoff
from gamegym.games import (DicePoker, Goofspiel, MatchingPennies, MatrixZeroSumGame,
                           RockPaperScissors)
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
    s = g.play(s, "T")
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
    assert sample_payoff(g, mcs, 300, seed=12)[0] == pytest.approx([0.0, 0.0], abs=0.1)
    assert sample_payoff(g, (mcs[0], us), 300, seed=13)[0] == pytest.approx([1.2, -1.2], abs=0.2)
    assert exploitability(g, 0, mcs[0]) < 0.1
    assert exploitability(g, 1, mcs[1]) < 0.1


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
        assert exp == pytest.approx(0.7, abs=0.2)
        assert aexp == pytest.approx(0.7, abs=0.2)


@pytest.mark.slow
def test_mccfr_dicepoker():

    g = DicePoker()
    mc = OutcomeMCCFR(g, seed=52)
    mc.compute(10000, burn=0.5)

    br0 = BestResponse(g, 0, mc.strategies)
    assert br0.value < 0.3
    payoff0 = sample_payoff(g, [br0, mc.strategies[1]], 10000, seed=3)[0]
    assert br0.value == pytest.approx(payoff0[0], abs=0.05)

    br1 = BestResponse(g, 1, mc.strategies)
    assert br1.value > -0.2
    payoff1 = sample_payoff(g, [mc.strategies[0], br1], 10000, seed=4)[0]
    assert br1.value == pytest.approx(payoff1[1], abs=0.05)

    print(br0.value, br1.value, payoff0, payoff1)
    assert payoff0[0] > payoff1[0]
    assert payoff0[0] < 0.3
    assert payoff1[0] > 0.1
