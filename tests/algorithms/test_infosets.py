from gamegym.algorithms import InformationSetSampler
from gamegym.games import RockPaperScissors
from gamegym.strategy import UniformStrategy, Strategy
from gamegym.game import GameState
import pytest
import numpy as np


def test_infoset():
    g = RockPaperScissors()
    us = UniformStrategy()
    iss = InformationSetSampler(g, us)
    assert iss.player_dist.probabilities() == pytest.approx(np.array([0.5, 0.5]))
    assert iss.infoset_dist[0].probabilities() == pytest.approx(np.array([1.0]))
    assert iss.infoset_dist[1].probabilities() == pytest.approx(np.array([1.0]))
    assert iss.infoset_history_dist[0][(0, None)].probabilities() == pytest.approx(np.array([1.0]))
    assert iss.infoset_history_dist[1][(1, None)].probabilities() == pytest.approx(np.array([1.0, 1.0, 1.0]) / 3)
    iss.sample_player()
    iss.sample_info()
    assert iss.sample_info(0)[1] == (0, None)
    assert iss.sample_info(1)[1] == (1, None)
    assert isinstance(iss.sample_state()[2], GameState)
