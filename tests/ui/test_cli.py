from unittest.mock import Mock, patch

import numpy as np
import pytest

from gamegym.algorithms.stats import play_strategies, sample_payoff
from gamegym.games import Gomoku, Goofspiel, TicTacToe
from gamegym.situation import StateInfo
from gamegym.strategy import UniformStrategy
from gamegym.ui.cli import CliStrategy, play_in_terminal


def test_cli_on_gomoku():
    g = Gomoku(4, 4, 3)
    a = Gomoku.TextAdapter(g, colors=True)

    actions = ["1 1", "1 2", "XXX", "2 1", "2 1", "2 2", "3 1"]
    with patch('builtins.input', side_effect=actions):
        result = play_in_terminal(g, [None, None], adapter=a)

    assert result.is_terminal()
    assert tuple(result.payoff) == (1, -1)


def test_cli_on_goofspiel():
    g = Goofspiel(4, Goofspiel.Scoring.ABSOLUTE)

    actions = ["1", "2", "X", "2", "4", "3", "3", "2", "4", "\n1  "]
    #                    inv                           inv
    with patch('builtins.input', side_effect=actions):
        result = play_in_terminal(g, seed=42)

    assert result.is_terminal()
    assert tuple(result.payoff) == (1., 6.)

    a = Goofspiel.TextAdapter(g)
    assert a.get_observation(result, StateInfo.OMNISCIENT).data == "2.0:1<2 4.0:2<4 3.0:3=3 1.0:4>1"


def test_cli_on_goofspiel_symmetric_color():
    g = Goofspiel(4, Goofspiel.Scoring.ABSOLUTE)
    a = Goofspiel.TextAdapter(g, symmetrize=True, colors=True)

    actions = ["1", "2", "X", "2", "4", "3", "3", "2", "4", "\n1  "]
    #                    inv                           inv
    with patch('builtins.input', side_effect=actions):
        result = play_in_terminal(g, adapter=a, seed=42)

    assert result.is_terminal()
    assert tuple(result.payoff) == (1., 6.)
