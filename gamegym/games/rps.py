class RPSState(State):

    def is_terminal(self):
        "Return whether the state is terminal."
        return len(self.h) >= 2

    def values(self):
        "Return a tuple of values, one for every player, undef if non-terminal."
        if self.h[0] == self.h[1]:
            return (0, 0)
        if {'R': 'S', 'S': 'P', 'P': 'R'}[self.h[0]] == self.h[1]:
            return (1, -1)
        return (-1, 1)

    def player(self):
        "Return the number of the active player, -1 for chance nodes."
        if len(self.h) == 0:
            return 0
        return 1

    def information_set(self, player):
        "Return the information set (any hashable object) for this state for the given player."
        return len(self.h)

    @classmethod
    def initial_state(self):
        "Return the initial state."
        return RPSState(())

    def actions(self):
        """
        Return an iterable of (label, state, probability)
        Labels may be numbers, strings etc.
        Probability is ignored for non-chance states.
        """
        return (
            ("R", RPSState(self.seq + ("R", )), None),
            ("P", RPSState(self.seq + ("P", )), None),
            ("S", RPSState(self.seq + ("S", )), None),
        )
