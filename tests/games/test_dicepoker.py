from gamegym.algorithms import OutcomeMCCFR, BestResponse
from gamegym.games import DicePoker
from gamegym.algorithms.stats import sample_payoff
import numpy as np
import pytest


def test_dicepoker():
    g = DicePoker()

    s = g.start()
    assert len(s.actions) == 36
    assert s.is_chance()
    s = g.play(s, (2, 5))

    assert s.player == 0
    assert set(s.actions) == set(["continue", "fold", "raise"])

    s1 = g.play(s, "fold")

    assert s1.is_terminal()
    assert s1._info.payoff == (-2, 2)

    s1 = g.play(s, "continue")
    assert s1.player == 1
    assert set(s.actions) == set(["continue", "fold", "raise"])

    s2 = g.play(s1, "fold")
    assert s2.is_terminal()
    assert s2._info.payoff == (2, -2)

    s2 = g.play(s1, "continue")
    assert s2.is_terminal()
    assert s2._info.payoff == (-3, 3)

    s3 = g.play(s1, "raise")
    assert s3.player == 0
    assert set(s3.actions) == set(("continue", "fold"))

    s4 = g.play(s3, "continue")
    assert s4.is_terminal()
    assert s4._info.payoff == (-6, 6)

    s4 = g.play(s3, "fold")
    assert s4.is_terminal()
    assert s4._info.payoff == (-4, 4)

    s1 = g.play(s, "raise")
    assert s1.player == 1
    assert set(s1.actions) == set(("continue", "fold"))

    s2 = g.play(s1, "continue")
    assert s2.is_terminal()
    assert s2._info.payoff == (-6, 6)

    s2 = g.play(s1, "fold")
    assert s2.is_terminal()
    assert s2._info.payoff == (4, -4)
