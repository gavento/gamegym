import numpy as np
import pytest

from gamegym.games import MatchingPennies, RockPaperScissors, MatrixZeroSumGame, Goofspiel
from gamegym.algorithms import BestResponse, OutcomeMCCFR


def test_iterations():
    pass


def test_persist(tmpdir):
    g = MatchingPennies()
    mc = OutcomeMCCFR(g, seed=42)
    fname = tmpdir.join("strat.pickle")
    assert mc.persist(fname, iterations=200) == False
    assert mc.iterations == 200
    
    mc2 = OutcomeMCCFR(g, seed=43)
    assert mc2.persist(fname, iterations=60) == True
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
    mc.compute(1000)
    s = g.initial_state()
    assert np.max(np.abs(mc.distribution(s).probabilities() - [0.5, 0.5])) < 0.1
    s = s.play("H")
    assert np.max(np.abs(mc.distribution(s).probabilities() - [0.5, 0.5])) < 0.1

def test_mccfr_goofspiel():
    g = Goofspiel(3)
    mc = OutcomeMCCFR(g, seed=56)
    mc.compute(1000)
    br = BestResponse(g, 0, {1:mc})
    assert np.mean([
        g.play_strategies([br, mc], seed=i)[-1].values()[0]
        for i in range(1000)]) == pytest.approx(0.0, abs=0.2)
