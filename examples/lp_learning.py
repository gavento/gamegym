from gamegym.algorithms import OutcomeMCCFR, BestResponse
from gamegym.value_learning.valuelp import LPZeroSumValueLearning
from gamegym.value_learning.expected_features import InfoSetExpectedFeatures
from gamegym.algorithms.infosets import InformationSetSampler
from gamegym.games.goofspiel import goofspiel_feaures_cards
from gamegym.games.matrix import matrix_zerosum_features
from gamegym.games import Goofspiel, RockPaperScissors
from gamegym.strategy import UniformStrategy
from gamegym.utils import get_rng
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.setrecursionlimit(100000)
import logging
logging.basicConfig(level=logging.INFO)

def main():
    print("#### Rock-paper-scissors value estimation")
    g = RockPaperScissors()
    us = UniformStrategy()
    infosampler = InformationSetSampler(g, us)
    val = LPZeroSumValueLearning(g, infosampler, matrix_zerosum_features, us)
    
    # Regularize: set one payoff to 1.0
    val.add_condition({(0, 1): 1.0}, 1.0)
    print("# With only non-triviality (one payoff set to 1.0)")
    print(val.compute())
    print("Flex value sum", val.flex_sum)
    # Zero diagonal
    for i in range(3):
        val.add_condition({(i, i): 1.0}, 0.0)
    print("# With zero diagonal")
    print(val.compute())
    print("Flex value sum", val.flex_sum)

    # Symmetrical payoffs
    for i in range(3):
        for j in range(i):
            val.add_condition({(i, j): -1.0, (j, i): -1.0}, 0.0)
    print("# Adding val(i,j) = -val(j,i)")
    print(val.compute())
    print("Flex value sum", val.flex_sum)

    return ### Goofspiel(3) is boring, Goofspiel(4) hits OOM
    print("#### Goofspiel(4) card value estimation")
    g = Goofspiel(4)
    mc = OutcomeMCCFR(g, seed=42)
    mc.compute(1000)
    expfeatures = InfoSetExpectedFeatures(g, goofspiel_feaures_cards, mc)
    val = LPZeroSumValueLearning(g, infosampler, goofspiel_feaures_cards, mc)
    
    # Regularize: set one payoff to 1.0
    val.add_condition({(0,): 1.0, (1,): 1.0, (2,): 1.0, (3,): 1.0}, 10.0)
    print("# Regularizing card values mean to 2.5 (mean of 1..4)")
    print(len(val.conds_eq), len(val.conds_le), len(val.flex_variables))
    print(val.compute(options=dict(tol=1e-6, disp=True, sparse=True, lstsq=True)))
    print("Flex value sum", val.flex_sum)

if __name__ == '__main__':
    main()
