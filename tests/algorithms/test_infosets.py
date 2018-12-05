from gamegym.algorithms import InformationSetSampler
from gamegym.games import RockPaperScissors
from gamegym.strategy import UniformStrategy, Strategy
from gamegym.game import GameState, Observation
from gamegym.utils import Distribution

import pytest
import numpy as np


def test_infoset():
    g = RockPaperScissors()
    us = UniformStrategy()
    iss = InformationSetSampler(g, [us, us])
    assert iss._player_dist.probs == pytest.approx(np.array([0.5, 0.5]))
    assert iss._infoset_dist[0].probs == pytest.approx(np.array([1.0]))
    assert iss._infoset_dist[1].probs == pytest.approx(np.array([1.0]))
    assert iss._infoset_history_dist[0][()].probs == pytest.approx(np.array([1.0]))
    assert iss._infoset_history_dist[1][()].probs == pytest.approx(np.array([1.0, 1.0, 1.0]) / 3)
    iss.sample_player()
    iss.sample_info()
    assert iss.sample_info(0)[1] == ()
    assert iss.sample_info(1)[1] == ()
    assert isinstance(iss.sample_state()[2], GameState)
    assert isinstance(iss.player_distribution(), Distribution)
    assert isinstance(iss.info_distribution(0), Distribution)
    assert isinstance(iss.state_distribution(0, ()), Distribution)
