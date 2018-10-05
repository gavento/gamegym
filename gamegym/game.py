
class Game:
    def initial_state(self):
        "Return the initial state of the game."
        raise NotImplementedError


class GameState:
    def __init__(self, game, h):
        "Create state of `game` with given history sequence."
        self.game = game
        self.h = tuple(h)

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
        return self.h

    def actions(self):
        """
        Return an iterable of ("action_label", history, probability)
        Probability ignored for non-chance states.
        """
        raise NotImplementedError
