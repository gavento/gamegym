#!/usr/bin/python3

from ..game import Game, GameState
import numpy as np


class MatrixGame(Game):
    """
    General game specified by a payoff matrix.
    The payoffs are for player `i` are `payoffs[p0, p1, p2, ..., i]`.

    Optionally, you can specify the labels for the player actions as
    `[[p0a0, p0a1, ...], [p1a0, ...], ...]` where the labels
    may be anything (numbers or strings are recommended)
    If no action labels are given, numbers are used.
    """
    def __init__(self, payoffs, actions=None):
        self.m = payoffs
        if not isinstance(self.m, np.ndarray):
            self.m = np.array(self.m)
        if self.players() != self.m.shape[-1]:
            raise ValueError(
                "Last dim of the payoff matrix must be the number of players.")
        if actions is None:
            self.actions = [list(range(acnt)) for acnt in self.m.shape[:-1]]
        else:
            self.actions = actions
        if tuple(len(i) for i in self.actions) != self.m.shape[:-1]:
            raise ValueError(
                "Mismatch of payoff matrix dims and labels provided: {} vs {}.".format(
                    self.m.shape[:-1], tuple(len(i) for i in self.actions)))

    def players(self):
        return len(self.m.shape) - 1

    def initial_state(self):
        "Return the initial state."
        return MatrixGameState(None, None, game=self)

    def __repr__(self):
        return "<{} {}>".format(self.__class__.__name__,
                                'x'.join(str(x) for x in self.m.shape[:-1]))


class MatrixGameState(GameState):
    def player(self):
        """
        Return the number of the active player (1..N).
        0 for chance nodes and -1 for terminal states.
        """
        assert len(self.history) <= self.game.players()
        if len(self.history) == self.game.players():
            return -1
        return len(self.history) + 1

    def values(self):
        """
        Return a tuple or numpy array of values, one for every player,
        undefined if non-terminal.
        """
        if not self.is_terminal():
            raise ValueError("Value of a non-terminal node is undefined.")
        return self.game.m[self.history]

    def actions(self):
        """
        Return a list or tuple of actions valid in this state.
        Should return empty list/tuple for terminal actions.
        """
        if self.is_terminal():
            return ()
        return self.game.actions[self.player() - 1]

    def player_information(self, player):
        """
        Return the game information from the point of the given player.
        This identifies the player's information set of this state.

        Note that this must distinguish all different information sets,
        e.g. when player 3 does not see the actions of the first two turns,
        she still distinguishes whether it is the first or second round.

        On the other hand (to be consistent with the "information set" concept),
        this does not need to distinguish the player for whom this
        information set is intended, e.g. in the initial state both player 1
        and player 2 may receive `()` as the `player_information`.
        """
        return (len(self.history),
                self.history[player] if player >= len(self.history) else None)


class ZeroSumMatrixGame(MatrixGame):
    """
    A two-player zero-sum game specified by a payoff matrix.
    The payoffs for player 0 are `payoffs[a0, a1]`, negative for player 1.
    Optionally, you can specify the labels for the players as
    `["a0", "a1", ...]` where the labels may be anything
    (numbers and strings are recommended). If no labels are given,
    numbers are used.
    """
    def __init__(self, payoffs, actions1=None, actions2=None):
        if (actions1 is None) != (actions2 is None):
            raise ValueError("Provide both or no labels.")
        actions = (actions1, actions2) if actions1 is not None else None
        if not isinstance(payoffs, np.ndarray):
            payoffs = np.array(payoffs)
        super().__init__(np.stack((payoffs, -payoffs), axis=-1), actions)


class RockPaperScissors(ZeroSumMatrixGame):
    """
    Rock-paper-scissors game with -1,0,1 values.
    """
    def __init__(self):
        super().__init__(
            [[0, -1, 1], [1, 0, -1], [-1, 1, 0]],
            ["R", "P", "S"], ["R", "P", "S"])


class GameOfChicken(MatrixGame):
    """
    Game of chicken with customizable values.
    """
    def __init__(self, win=7, lose=2, both_dare=0, both_chicken=6):
        super().__init__(
            [[[both_dare, both_dare], [win, lose]],
             [[lose, win], [both_chicken, both_chicken]]],
            (("D", "C"), ("D", "C")))


class PrisonersDilemma(MatrixGame):
    """
    Game of prisoners dilemma with customizable values.
    """
    def __init__(self, win=3, lose=0, both_defect=1, both_cooperate=2):
        super().__init__(
            [[[both_defect, both_defect], [win, lose]],
             [[lose, win], [both_cooperate, both_cooperate]]],
            (("D", "C"), ("D", "C")))


def test_base():
    gs = [
        PrisonersDilemma(),
        GameOfChicken(),
        RockPaperScissors(),
        ZeroSumMatrixGame([[1, 3], [3, 2], [0, 0]], ["A", "B", "C"], [0, 1]),
        MatrixGame([[1], [2], [3]], [["A1", "A2", "A3"]]),
        MatrixGame(np.zeros([2, 4, 5, 3], dtype=np.int32)),
    ]
    for g in gs:
        s = g.initial_state()
        assert not s.is_terminal()
        assert s.player() == 1
        assert len(s.actions()) == g.m.shape[0]
        repr(s)
        repr(g)
