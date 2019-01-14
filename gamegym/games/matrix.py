#!/usr/bin/python3

from ..game import SimultaneousGame, Situation, StateInfo
from typing import Any
import numpy as np


class MatrixGame(SimultaneousGame):
    """
    Simultaneous game in normal form specified by a payoff matrix.
    The payoffs are for player `i` are `payoffs[p0act, p1act, p2act, ..., i]`.

    Optionally, you can specify the player actions as
    `[[p1a0, p1a1, ...], [p2a0, ...], ...]` where the labels
    may be anything (numbers or strings are recommended)
    If no action labels are given, numbers are used.
    """

    def __init__(self, payoffs, player_actions=None):
        if not isinstance(payoffs, np.ndarray):
            payoffs = np.array(payoffs)
        if player_actions is None:
            player_actions = [list(range(acnt)) for acnt in payoffs.shape[:-1]]

        super().__init__(player_actions)

        self.m = payoffs
        if self.players != self.m.shape[-1]:
            raise ValueError("Last dim of the payoff matrix must be the number of players.")
        if tuple(len(i) for i in player_actions) != self.m.shape[:-1]:
            raise ValueError(
                "Mismatch of payoff matrix dims and labels provided: {} vs {}.".format(
                    self.m.shape[:-1], tuple(len(i) for i in player_actions)))
        self.player_action_m_index = [{a: i for i, a in enumerate(pa)} for pa in player_actions]

    def game_payoff(self, player_actions):
        assert len(player_actions) == self.players
        idx = tuple(self.player_action_m_index[p][a] for p, a in enumerate(player_actions))
        return self.m[idx]

    def __repr__(self):
        return "<{} {}>".format(self.__class__.__name__, 'x'.join(
            str(x) for x in self.m.shape[:-1]))


class MatrixZeroSumGame(MatrixGame):
    """
    A two-player zero-sum game specified by a payoff matrix.

    The payoffs for player 0 are `payoffs[a0, a1]`, negative for player 1.
    
    Optionally, you can specify the labels for the players as
    `(("p0a0", "p0a1", ...), ("p1a0", ...))`. If no labels are given,
    numbers are used.
    """

    def __init__(self, payoffs, player_actions=None):
        if not isinstance(payoffs, np.ndarray):
            payoffs = np.array(payoffs, dtype=np.float64)
        super().__init__(
            np.stack((payoffs, 0.0 - payoffs), axis=-1), player_actions=player_actions)


class RockPaperScissors(MatrixZeroSumGame):
    """
    Rock-paper-scissors game with -1,0,1 values.

    Actions are `R(ock)`, `P(aper)` and `S(cissors)`.
    """

    def __init__(self):
        super().__init__([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], (("R", "P", "S"), ("R", "P", "S")))


class GameOfChicken(MatrixGame):
    """
    Game of chicken with customizable values.

    Actions are `D(are)` and `C(hicken)`.
    """

    def __init__(self, win=7, lose=2, both_dare=0, both_chicken=6):
        super().__init__(
            [[[both_dare, both_dare], [win, lose]], [[lose, win], [both_chicken, both_chicken]]],
            (("D", "C"), ("D", "C")))


class PrisonersDilemma(MatrixGame):
    """
    Game of prisoners dilemma with customizable values.

    Actions are `D(efect)` and `C(ooperate)`.
    """

    def __init__(self, win=3, lose=0, both_defect=1, both_cooperate=2):
        super().__init__([[[both_defect, both_defect], [win, lose]],
                          [[lose, win], [both_cooperate, both_cooperate]]], (("D", "C"),
                                                                             ("D", "C")))


class MatchingPennies(MatrixZeroSumGame):
    """
    Game of matchig pennies, the first player is the matcher.

    Actions are `H(eads)` and `T(ails)`.
    """

    def __init__(self, mismatch=1, match_heads=1, match_tails=1):
        super().__init__([[match_heads, -mismatch], [-mismatch, match_tails]], (("H", "T"),
                                                                                ("H", "T")))


def matrix_zerosum_features(hist: Situation, sparse=False):
    assert not sparse
    assert isinstance(hist.game, MatrixGame)
    features = np.zeros(hist.game.m.shape[:-1], dtype=np.float64)
    if hist.active.is_terminal():
        features.__setitem__(hist.history_idx, 1.0)
    return features
