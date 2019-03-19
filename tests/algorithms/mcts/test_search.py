import numpy as np
import pytest

from gamegym.algorithms.mcts import search
from gamegym.games import Gomoku


def test_search():
    g = Gomoku(3, 3, 3)
    pass
