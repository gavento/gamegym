from gamegym.algorithms import OutcomeMCCFR, BestResponse
from gamegym.games import DicePoker
import numpy as np
import pytest


def test_dicepoker():
    g = DicePoker()

    s = g.initial_state()
    assert len(s.actions()) == 36

    assert s.player() == -1
    s = s.play((2, 5))

    assert s.player() == 0
    assert set(s.actions()) == set(["continue", "fold", "raise"])

    s1 = s.play("fold")

    assert s1.is_terminal()
    assert s1.values() == (-2, 2)

    s1 = s.play("continue")
    assert s1.player() == 1
    assert set(s.actions()) == set(["continue", "fold", "raise"])

    s2 = s1.play("fold")
    assert s2.is_terminal()
    assert s2.values() == (2, -2)

    s2 = s1.play("continue")
    assert s2.is_terminal()
    assert s2.values() == (-3, 3)

    s3 = s2.play("raise")
    assert s3.player() == 0
    assert set(s3.actions()) == set(["continue", "fold"])

    s4 = s3.play("continue")
    assert s4.is_terminal()
    assert s4.values() == (-6, 6)

    s4 = s3.play("fold")
    assert s4.is_terminal()
    assert s4.values() == (-4, 4)

    s1 = s.play("raise")
    assert s1.player() == 1
    assert set(s1.actions()) == set(["continue", "fold"])

    s2 = s1.play("continue")
    assert s2.is_terminal()
    assert s2.values() == (-6, 6)

    s2 = s1.play("fold")
    assert s2.is_terminal()
    assert s2.values() == (4, -4)


def test_dicepoker_mc():

    g = DicePoker()
    mc = OutcomeMCCFR(g, seed=56)
    mc.compute(1000)
    br = BestResponse(g, 1, {0: mc})
    assert np.mean([
        g.play_strategies([mc, br], seed=i)[-1].values()[0]
        for i in range(1000)]) <= 0.5
