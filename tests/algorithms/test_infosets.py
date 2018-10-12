from gamegym.algorithms import InformationSetSampler
from gamegym.games import RockPaperScissors
from gamegym.strategy import UniformStrategy, Strategy
from gamegym.game import GameState
from gamegym.distribution import Discrete
import pytest
import numpy as np


def test_infoset():
    g = RockPaperScissors()
    us = UniformStrategy()
    iss = InformationSetSampler(g, us)
    assert iss._player_dist.probabilities() == pytest.approx(np.array([0.5, 0.5]))
    assert iss._infoset_dist[0].probabilities() == pytest.approx(np.array([1.0]))
    assert iss._infoset_dist[1].probabilities() == pytest.approx(np.array([1.0]))
    assert iss._infoset_history_dist[0][(0, None)].probabilities() == pytest.approx(np.array([1.0]))
    assert iss._infoset_history_dist[1][(1, None)].probabilities() == pytest.approx(np.array([1.0, 1.0, 1.0]) / 3)
    iss.sample_player()
    iss.sample_info()
    assert iss.sample_info(0)[1] == (0, None)
    assert iss.sample_info(1)[1] == (1, None)
    assert isinstance(iss.sample_state()[2], GameState)
    assert isinstance(iss.player_distribution(), Discrete)
    assert isinstance(iss.info_distribution(0), Discrete)
    assert isinstance(iss.state_distribution(0, (0, None)), Discrete)
