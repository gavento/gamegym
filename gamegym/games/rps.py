#!/usr/bin/python3

from ..game import Game, GameState


class RockPaperScissors(Game):
    """
    Manual rock-paper-scissors implementation with values -1, 0, 1
    to illustrate game implementation.
    """

    def players(self):
        return 2

    def initial_state(self):
        "Return the initial state."
        return RPSState(None, None, game=self)


class RPSState(GameState):
    def player(self):
        """
        Return the number of the active player (1..N).
        0 for chance nodes and -1 for terminal states.
        """
        if len(self.history) == 2:
            return -1
        return len(self.history) + 1

    def values(self):
        """
        Return a tuple or numpy array of values, one for every player,
        undefined if non-terminal.
        """
        assert self.is_terminal()
        if self.history[0] == self.history[1]:
            return (0, 0)
        if self.history[0] == {"R": "P", "P": "S", "S": "R"}[self.history[1]]:
            return (1, -1)
        return (-1, 1)

    def actions(self):
        """
        Return a list or tuple of actions valid in this state.
        Should return empty list/tuple for terminal actions.
        """
        if self.is_terminal():
            return ()
        return ["R", "P", "S"]

    def player_information(self, player):
        """
        Return the game information from the point of the given player.
        This identifies the player's information set of this state.
        """
        return (len(self.history),
                self.history[player] if player >= len(self.history) else None)


def test_base():
    g = RockPaperScissors()
    s = g.initial_state()
    repr(s)
    repr(g)
    assert not s.is_terminal()
    assert s.player() == 1
    assert len(s.actions()) == 3
    s1 = s.play(s.actions()[0])
    assert not s1.is_terminal()
    assert s1.player() == 2
    assert len(s1.actions()) == 3
    s2 = s1.play(s1.actions()[1])
    assert s2.is_terminal()
    assert s2.values() == (-1, 1)
