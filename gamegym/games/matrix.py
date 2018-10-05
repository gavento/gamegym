from ..game import Game, GameState
import numpy as np


class MatrixGame(Game):
    """
    General game specified by a payoff matrix.
    The payoffs are for player `i` are `payoffs[p0, p1, p2, ..., i]`.
    Optionally, you can specify the labels for the players as
    `[["p0a0", "p0a1", ...], ["p1a0", ...], ...]` where the labels
    may be anything (numbers and strings are recommended)
    If no labels are given, numbers are used.
    """
    def __init__(self, payoffs, labels=None):
        self.m = payoffs
        if not isinstance(self.m, np.ndarray):
            self.m = np.array(self.m)
        self.players = len(self.m.shape) - 1
        if self.players != self.m.shape[-1]:
            raise ValueError("Last dim of the payoff matrix must be the number of players.")
        if labels is None:
            self.labels = [list(range(acnt)) for acnt in self.m.shape[:-1]]
        else:
            self.labels = labels
        if tuple(len(i) for i in self.labels) != self.m.shape[:-1]:
            raise ValueError(
                "Mismatch of payoff matrix dims and labels provided: {} vs {}.".format(
                    self.m.shape[:-1], tuple(len(i) for i in self.labels)))

    def initial_state(self):
        "Return the initial state."
        return MatrixGameState(self, ())

    def __repr__(self):
        return "<{} {}>".format(self.__class__.__name__,
                                'x'.join(str(x) for x in self.m.shape[:-1]))


class MatrixGameState(GameState):

    def is_terminal(self):
        "Return whether the state is terminal."
        return len(self.history) >= self.game.players

    def values(self):
        "Return a tuple of values, one for every player. Error if non-terminal."
        if not self.is_terminal():
            raise Exception("Value of a non-terminal node is undefined.")
        assert len(self.history == self.game.players)
        return self.game.m[self.history]

    def player(self):
        "Return the number of the active player, -1 for chance nodes."
        return len(self.history)

    def information_set(self, player):
        "Return the information set (any hashable object) for this state for the given player."
        return len(self.history)

    def actions(self):
        """
        Return an iterable of `NextAction` (i.e. `label, state, probability`).
        Labels may be numbers, strings etc.
        Probability is ignored for non-chance states.
        """
        if self.is_terminal():
            return ()
        p = self.player()
        return tuple(self.next_action(i, label=self.game.labels[p][i])
                     for i in range(self.game.m.shape[p]))

    def __repr__(self):
        return "<GameState ({})>".format(
            ', '.join(str(self.game.labels[i][x])
                      for i, x in enumerate(self.history)))


class ZeroSumMatrixGame(MatrixGame):
    """
    A two-player zero-sum game specified by a payoff matrix.
    The payoffs for player 0 are `payoffs[a0, a1]`, negative for player 1.
    Optionally, you can specify the labels for the players as
    `["a0", "a1", ...]` where the labels may be anything
    (numbers and strings are recommended). If no labels are given,
    numbers are used.
    """
    def __init__(self, payoffs, labels0=None, labels1=None):
        if (labels0 is None) != (labels1 is None):
            raise ValueError("Provide both or no labels.")
        labels = (labels0, labels1) if labels0 is not None else None
        if not isinstance(payoffs, np.ndarray):
            payoffs = np.array(payoffs)
        super().__init__(np.stack((payoffs, -payoffs), axis=-1), labels)


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
        assert s.player() == 0
        assert len(s.actions()) == g.m.shape[0]
        repr(s)
        repr(g)
