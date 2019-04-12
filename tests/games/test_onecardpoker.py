from gamegym.algorithms import OutcomeMCCFR, BestResponse
from gamegym.games import OneCardPoker
from gamegym.algorithms.stats import play_strategies
import numpy as np
import pytest


def test_onecardpoker():
    g = OneCardPoker()

    s = g.start()
    assert len(s.actions) == 6

    assert s.player == -1
    s = g.play(s, (0, 1))

    assert s.player == 0
    assert set(s.actions) == {"check", "raise"}

    s1 = g.play(s, "check")
    assert s1.player == 1
    assert set(s.actions) == {"check", "raise"}

    s2 = s1.play("check")
    assert s2.is_terminal()
    assert tuple(s2.payoff) == (-1, 1)

    s3 = g.play(s1, "raise")
    assert s3.player == 0
    assert set(s3.actions) == {"call", "fold"}

    s4 = s3.play("call")
    assert s4.is_terminal()
    assert tuple(s4.payoff) == (-2, 2)

    s4 = s3.play("fold")
    assert s4.is_terminal()
    assert tuple(s4.payoff) == (-1, 1)

    s1 = s.play("raise")
    assert s1.player == 1
    assert set(s1.actions) == {"call", "fold"}

    s2 = s1.play("call")
    assert s2.is_terminal()
    assert tuple(s2.payoff) == (-2, 2)

    s2 = s1.play("fold")
    assert s2.is_terminal()
    assert tuple(s2.payoff) == (1, -1)


def test_onecardpoker_mc():

    g = OneCardPoker()
    mc = OutcomeMCCFR(g, seed=56)
    mc.compute(1000)
    #print(mc.iss)
    br = BestResponse(g, 1, mc.strategies)
    #print(br.value)
    assert np.mean([play_strategies(g, [mc.strategies[0], br], seed=i).payoff[0]
                    for i in range(1000)]) > -0.4
