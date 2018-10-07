#!/usr/bin/python3

import collections


class Game:
    """
    Base class for game instances.
    """
    def initial_state(self):
        "Return the initial state of the game."
        raise NotImplementedError

    def generate_play(self, strategies, rng=None):
        s = self.initial_state()
        while not s.is_terminal():
            p = 


class GameState:
    NextAction = collections.namedtuple(
        "NextAction", ("label", "state", "probability"))

    """
    Base class for game states.
    """
    def __init__(self, game, history):
        "Create state of `game` with given history sequence."
        self.game = game
        self.history = tuple(history)

    def is_terminal(self):
        "Return whether the state is terminal."
        raise NotImplementedError

    def values(self):
        "Return a tuple of values, one for every player, undef if non-terminal."
        raise NotImplementedError

    def player(self):
        "Return the number of the active player, -1 for chance nodes."
        raise NotImplementedError

    def information_set(self, player):
        "Return the information set (any hashable object) for this state for the given player."
        raise NotImplementedError

    def canonical_form(self):
        "Return the canonical form of the state (may merge histories)."
        return self.history

    def actions(self):
        """
        Return an iterable of `self.NextAction`.
        Probability is `None` for non-chance states.
        """
        raise NotImplementedError

    def next_action(self, action, label=None, probability=None):
        """
        Create a `NextAction` by appending the given action to self.
        Probability should be given only when self is a chance node.
        """
        assert (self.player() >= 0) == (probability is None)
        return self.NextAction(
            label, self.__class__(self.game, self.history + (action, )),
            probability)
