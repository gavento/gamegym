from gamegym.games import Gomoku, TicTacToe
from gamegym.algorithms.stats import sample_payoff
from gamegym.strategy import UniformStrategy
from gamegym.ui.cli import CliStrategy
from gamegym.algorithms.stats import play_strategies


import numpy as np
import pytest
from unittest.mock import patch, Mock


def test_cli_on_gomoku():
    g = Gomoku(4, 4, 3)
    s = CliStrategy(Gomoku.TextAdapter(g, colors=True))

    actions = ["1 1", "1 2", "XXX", "2 1", "2 1", "2 2", "3 1"]
    with patch('builtins.input', side_effect=actions):
        result = play_strategies(g, [s, s])

    assert result.is_terminal()
    assert tuple(result.payoff) == (1, -1)
