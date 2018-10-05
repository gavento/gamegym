from ..game import Game, GameState


class RockPaperScissors(Game):
    """
    Manual rock-paper-scissors implementation (with values -1, 0, 1).
    """
    def initial_state(self):
        "Return the initial state."
        return RPSState(self, ())


class RPSState(GameState):

    def is_terminal(self):
        "Return whether the state is terminal."
        return len(self.history) >= 2

    def values(self):
        "Return a tuple of values, one for every player, undef if non-terminal."
        if self.history[0] == self.history[1]:
            return (0, 0)
        if {'R': 'S', 'S': 'P', 'P': 'R'}[self.history[0]] == self.history[1]:
            return (1, -1)
        return (-1, 1)

    def player(self):
        "Return the number of the active player, -1 for chance nodes."
        return len(self.history)

    def information_set(self, player):
        "Return the information set (any hashable object) for this state for the given player."
        return len(self.history)

    def actions(self):
        """
        Return an iterable of (label, state, probability)
        Labels may be numbers, strings etc.
        Probability is ignored for non-chance states.
        """
        return tuple(self.next_action(a, label=a) for a in ("R", "P", "S"))


def test_base():
    g = RockPaperScissors()
    s = g.initial_state()
    repr(s)
    repr(g)
    assert not s.is_terminal()
    assert s.player() == 0
    assert len(s.actions()) == 3
    s1 = s.actions()[0].state
    assert not s1.is_terminal()
    assert s1.player() == 1
    assert len(s1.actions()) == 3
    s2 = s1.actions()[1].state
    assert s2.is_terminal()
    assert s2.values() == (-1, 1)
