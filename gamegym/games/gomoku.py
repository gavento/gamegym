from ..game import PerfectInformationGame
from ..situation import Situation, StateInfo, Action

from typing import Any, Tuple
import numpy as np


class Gomoku(PerfectInformationGame):
    """
    A game of Gomoku with variable size and winning chain length.

    The state is encoded as `(np.array board, tuple of available actions)`.
    """

    def __init__(self, w: int, h: int, chain: int = 5):
        self.w = w
        self.h = h
        self.chain = chain
        actions = tuple((r, c) for c in range(self.w) for r in range(self.h))
        super().__init__(2, actions)

    def initial_state(self) -> StateInfo:
        """
        Return the initial internal state and active player.
        """
        board = np.zeros((self.h, self.w), dtype=np.int8) - 1  # -1: empty, 0,1: player 0/1
        state = (board, self.actions)  # board, free coordinates
        return StateInfo.new_player(state, 0, state[1])

    def update_state(self, situation: Situation, action: Action) -> StateInfo:
        """
        Return the updated internal state, active player and per-player observations.
        """
        a_r, a_c = action
        board, free_fields = situation.state
        assert board[a_r, a_c] == -1
        player = (len(self.actions) - len(free_fields)) % 2
        assert player == len(situation.history) % 2
        assert player == situation.player

        new_board = board.copy()
        new_board[a_r, a_c] = player
        new_actions = tuple(a for a in free_fields if a != action)
        new_player = 1 - player
        new_state = (new_board, new_actions)  # board, free coordinates

        if ((self._extent(new_board, a_r, a_c, -1, -1) + self._extent(new_board, a_r, a_c, 1, 1) + 1 >= self.chain) or
            (self._extent(new_board, a_r, a_c, 1, -1) + self._extent(new_board, a_r, a_c, -1, 1) + 1 >= self.chain) or
            (self._extent(new_board, a_r, a_c, 0, 1) + self._extent(new_board, a_r, a_c, 0, -1) + 1 >= self.chain) or
            (self._extent(new_board, a_r, a_c, 1, 0) + self._extent(new_board, a_r, a_c, -1, 0) + 1 >= self.chain)):
            p0sc = 1.0 - (player * 2)
            return StateInfo.new_terminal(new_state, (p0sc, -p0sc))

        if len(new_actions) == 0:
            return StateInfo.new_terminal(new_state, (0.0, 0.0))

        return StateInfo.new_player(new_state, new_player, new_actions)

    def get_features(self, situation: Situation, _for_player: int = None) -> tuple:
        """
        Return the features as a tuple of numpy arrays.

        The features are: `(active player pieces, other player pieces, active player no)`.
        """
        board = situation.state[0]
        player = situation.player
        active_board = (board == player).astype(np.float32)
        other_board = (board == 1 - player).astype(np.float32)
        return (active_board, other_board, np.array([player], np.float32))

    def get_features_shape(self) -> tuple:
        """
        Return the shapes of the features as tuple of tensor shapes (tuples).
        """
        return ((self.h, self.w), (self.h, self.w), (1, ))

    def _extent(self, b: np.ndarray, r: int, c: int, dr: int, dc: int) -> int:
        """
        Return the length of a chain of the values at `b[r, c]` in the direction `(dr, dc)`.
        Does not include `b[r, c]` itself.
        """
        l = -1
        v = b[r, c]
        while r >= 0 and c >= 0 and r < self.h and c < self.w and b[r, c] == v:
            l += 1
            r += dr
            c += dc
        return l

    def __repr__(self) -> str:
        return "<{} {}x{} (chain {})>".format(
            self.__class__.__name__, self.w, self.h, self.chain)

    def show_board(self, situation) -> str:
        """
        Return a string with a pretty-printed board and one-line game information.
        """
        ps = ["player 0 (x)", "player 1 (o)"]
        cs = {-1: '.', 0: 'x', 1: 'o'}
        info = ps[situation.player] + " active"
        if situation.is_terminal():
            if situation.payoff[0] > 0.0:
                info = ps[0] + " won"
            elif situation.payoff[0] < 0.0:
                info = ps[1] + " won"
            else:
                info = "draw"
        lines = [''.join(cs[x] for x in l) for l in situation.state[0]]
        return "\n".join(lines) + "\n{} turn {}, {}".format(self, len(situation.history) + 1, info)


class TicTacToe(Gomoku):
    """
    The game of tic-tac-toe (Gomoku 3x3, chain of 3 to win).
    """
    def __init__(self):
        super().__init__(3, 3, 3)

