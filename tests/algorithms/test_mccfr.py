import numpy as np
import pytest
import os

from gamegym.games import MatchingPennies, RockPaperScissors, MatrixZeroSumGame, Goofspiel
from gamegym.algorithms import BestResponse, OutcomeMCCFR, exploitability


def test_persist(tmpdir):
    g = MatchingPennies()
    mc = OutcomeMCCFR(g, seed=42)
    fname = str(tmpdir.join("MatchingPennies"))
    assert mc.persist(fname, iterations=200) == False
    assert mc.iterations == 200
    
    mc2 = OutcomeMCCFR(g, seed=43)
    assert mc2.persist(fname, iterations=200) == True
    assert mc2.iterations == 200
   

def test_regret():
    g = MatchingPennies()
    mc = OutcomeMCCFR(g, seed=42)
    rs = mc.regret_matching(np.array([-1.0, 0.0, 1.0, 2.0]))
    assert rs == pytest.approx([0.0, 0.0, 1.0 / 3, 2.0 / 3])


def test_pennies():
    np.set_printoptions(precision=3)
    g = MatchingPennies()
    g = RockPaperScissors()
    g = MatrixZeroSumGame([[1, 0], [0, 1]])
    mc = OutcomeMCCFR(g, seed=12)
    mc.compute(500)
    s = g.initial_state()
    assert np.max(np.abs(mc.distribution(s).probabilities() - [0.5, 0.5])) < 0.1
    s = s.play(1)
    assert np.max(np.abs(mc.distribution(s).probabilities() - [0.5, 0.5])) < 0.1


def test_mccfr_goofspiel3():
    g = Goofspiel(3)
    mc = OutcomeMCCFR(g, seed=56)
    mc.compute(500)
    br = BestResponse(g, 0, {1:mc})
    assert np.mean([
        g.play_strategies([br, mc], seed=i)[-1].values()[0]
        for i in range(200)]) == pytest.approx(0.0, abs=0.2)


@pytest.mark.slow
def test_mccfr_goofspiel4_slow():
    g = Goofspiel(4, scoring=Goofspiel.Scoring.ZEROSUM)
    mc = OutcomeMCCFR(g, seed=49)
    mc.compute(10000)
    assert exploitability(g, 0, mc) < 1.2
