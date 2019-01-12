from gamegym.algorithms import OutcomeMCCFR, BestResponse
from gamegym.value_learning.valuestore import LinearValueStore
from gamegym.value_learning.valuesgd import SparseSGDLinearValueLearning
from gamegym.algorithms.infosets import InformationSetSampler
from gamegym.games.goofspiel import goofspiel_feaures_cards
from gamegym.games.matrix import matrix_zerosum_features
from gamegym.games import Goofspiel, RockPaperScissors
from gamegym.strategy import UniformStrategy
from gamegym.utils import get_rng
import numpy as np
import pytest


@pytest.mark.slow
def test_goofspiel():
    g = Goofspiel(4, scoring=Goofspiel.Scoring.ZEROSUM)
    mc = OutcomeMCCFR(g, seed=42)
    mc.compute(100)
    vs = LinearValueStore(goofspiel_feaures_cards(g.initial_state()), fix_mean=2.5)
    infosampler = InformationSetSampler(g, mc)
    val = SparseSGDLinearValueLearning(g, goofspiel_feaures_cards, vs, infosampler, seed=43)
    val.compute([mc, mc], 100, 0.1, 0.01)
    print(vs.values)


def test_rps():
    g = RockPaperScissors()
    us = UniformStrategy()
    rng = get_rng(seed=3)
    params = rng.rand(3, 3) - 0.5
    vs = LinearValueStore(params, fix_mean=0.0, regularize_l1=6.0)
    infosampler = InformationSetSampler(g, us)
    val = SparseSGDLinearValueLearning(g, matrix_zerosum_features, vs, infosampler, seed=44)
    val.compute([us, us], 100, 0.1, 0.1)
    val.compute([us, us], 100, 0.01, 0.01)
    val.compute([us, us], 100, 0.001, 0.001)
