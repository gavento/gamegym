from gamegym.algorithms import OutcomeMCCFR, BestResponse
from gamegym.games import OneCardPoker
import numpy as np
import pytest


def test_onecardpoker():
    g = OneCardPoker()

    s = g.initial_state()
    assert len(s.actions()) == 9

    assert s.player() == -1
    s = s.play(1)

    assert s.player() == 0
    assert set(s.actions()) == set(["check", "raise"])

    s1 = s.play("check")
    assert s1.player() == 1
    assert set(s.actions()) == set(["check", "raise"])

    s2 = s1.play("check")
    assert s2.is_terminal()
    assert s2.values() == (-1, 1)

    s3 = s2.play("raise")
    assert s3.player() == 0
    assert set(s3.actions()) == set(["call", "fold"])

    s4 = s3.play("call")
    assert s4.is_terminal()
    assert s4.values() == (-2, 2)

    s4 = s3.play("fold")
    assert s4.is_terminal()
    assert s4.values() == (-1, 1)

    s1 = s.play("raise")
    assert s1.player() == 1
    assert set(s1.actions()) == set(["call", "fold"])

    s2 = s1.play("call")
    assert s2.is_terminal()
    assert s2.values() == (-2, 2)

    s2 = s1.play("fold")
    assert s2.is_terminal()
    assert s2.values() == (1, -1)


def test_onecardpoker_mc():

    g = OneCardPoker()
    mc = OutcomeMCCFR(g, seed=56)
    mc.compute(1000)
    #print(mc.iss)
    br = BestResponse(g, 1, {0: mc})
    #print(br.value)
    assert np.mean([g.play_strategies([mc, br], seed=i)[-1].values()[0]
                    for i in range(1000)]) > -0.4
