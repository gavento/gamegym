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
    assert mcs[0].distribution((), 2) == pytest.approx([0.5, 0.5], abs=0.1)
    assert mcs[1].distribution((), 2) == pytest.approx([0.5, 0.5], abs=0.1)
    s = g.play(s, index=1)
    assert mcs[0].distribution((), 2) == pytest.approx([0.5, 0.5], abs=0.1)
    assert mcs[1].distribution((), 2) == pytest.approx([0.5, 0.5], abs=0.1)


def test_mccfr_goofspiel3():
    g = Goofspiel(3, scoring=Goofspiel.Scoring.ZEROSUM)
    mc = OutcomeMCCFR(g, seed=51)
    mc.compute(500)
    mcs = mc.strategies
    us = UniformStrategy()
    s1 = g.play_sequence([2])[-1]
    assert mcs[0].distribution(s1) == pytest.approx([0.2, 0.6, 0.2], abs=0.2)
    assert g.sample_payoff(mcs, 100, seed=12)[0] == pytest.approx([0.0, 0.0], abs=0.1)
    assert g.sample_payoff((mcs[0], us), 100, seed=13)[0] == pytest.approx([1.0, -1.0], abs=0.2)
    assert exploitability(g, 0, mcs[0]) < .3
    assert exploitability(g, 1, mcs[1]) < .3


@pytest.mark.slow
def test_mccfr_goofspiel4():
    g = Goofspiel(4, scoring=Goofspiel.Scoring.ZEROSUM)
    mc = OutcomeMCCFR(g, seed=49)
    mc.compute(10000)
    mcs = mc.strategies
    assert approx_exploitability(g, 0, mcs[0], 5000) == pytest.approx(0.4, abs=0.2)
    assert approx_exploitability(g, 1, mcs[1], 5000) == pytest.approx(0.4, abs=0.2)
    assert exploitability(g, 0, mcs[0]) < 1.1
    assert exploitability(g, 1, mcs[1]) < 1.1
