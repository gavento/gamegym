#!/usr/bin/python3

import collections
from .utils import debug_assert, get_rng


class Game:
    """
    Base class for game instances.
    """

    def initial_state(self):
        """
        Return the initial state (with empty history).
        Note that the initial state must be always the same. If the game start
        depends on chance, use a chance node as the first state.
        """
        raise NotImplementedError

    def players(self):
        """
        Return the number of players N.
        """
        raise NotImplementedError

    def play_strategies(
            self, strategies, *, rng=None, seed=None, state0=None, upto_fn=None):
        """
        Generate a play based on given strategies (one per player).
        Returns the list of all visited states.
        """
        rng = get_rng(rng=rng, seed=seed)
        if len(strategies) != self.players():
            raise ValueError("One strategy per player required")
        s = self.initial_state() if state0 is None else state0
        seq = [s]

        if upto_fn is None:
            upto_fn = lambda s: s.is_terminal()

        while not upto_fn(s):
            if s.is_chance():
                a = s.chance_distribution().sample(rng=rng)
            else:
                strat = strategies[s.player()]
                a = strat.distribution(s).sample(rng=rng)
            s = s.play(a)
            seq.append(s)
        return seq

    def play_history(self, history):
        """
        Play all the actions in history in sequence,
        return the list of all visited states.
        """
        s = self.initial_state()
        seq = [s]
        for a in history:
            s = s.play(a)
            seq.append(s)
        return seq

    def __repr__(self):
        return "<{}(...)>".format(self.__class__.__name__)


class GameState:
    # Returned by self.player() for chance nodes
    P_CHANCE = -1
    # Returned by self.player() for terminal nodes
    P_TERMINAL = -2

    def __init__(self, prev_state, action, game=None):
        """
        Initialize the state from `prev_state` and `action`, or as the initial
        state if `prev_state=None`, `action=None` and game is given.

        This base class keeps track of state action history in `self.history`
        and the game object in `self.game` and may be sufficient for simple games.

        If the state of the game is more complex than the history (e.g. cards in
        player hands), add this as attributes and update them in this
        constructor.
        """
        if prev_state is None and action is None:
            if game is None:
                raise ValueError("Provide (prev_state, action) or game")
            if not isinstance(game, Game):
                raise TypeError("Expected Game instance for game")
            self.game = game
            self.history = tuple()
        else:
            if game is not None:
                raise ValueError("When providing prev_state, game must be None")
            if not isinstance(prev_state, GameState):
                raise TypeError("Expected GameState instance for prev_state")
            if action is None:
                raise ValueError("None action is not valid")
            self.game = prev_state.game
            self.history = prev_state.history + (action, )

    def player(self):
        """
        Return the number of the active player (0..N-1).
        `self.P_CHANCE` for chance nodes and `self.P_TERMINAL` for terminal states.
        """
        raise NotImplementedError

    def values(self):  # one value for each player
        """
        Return a tuple or numpy array of values, one for every player.
        Undefined (and may raise an exception) in non-terminal nodes.
        """
        raise NotImplementedError

    def actions(self):
        """
        Return a list or tuple of actions valid in this state.
        Should return empty list/tuple for terminal actions.
        """
        raise NotImplementedError

    def player_information(self, player):
        """
        Return the game information from the point of the given player.
        This identifies the player's information set of this state.

        Note that this must distinguish all different information sets,
        e.g. when a player does not see any information on the first two turns,
        she still distinguishes whether it is the first or second round.

        On the other hand (to be consistent with the "information set" concept),
        this does not need to distinguish the player for whom this
        information set is intended, e.g. in the initial state both player 0
        and player 1 may receive `()` as the `player_information`.
        """
        raise NotImplementedError

    def chance_distribution(self):
        """
        In chance nodes, returns a `Discrete` distribution chance actions.
        Must not be called in non-chance nodes (and should raise an exception).
        You do not need to modify it if the game has no chance nodes.
        """
        assert self.is_chance()
        raise NotImplementedError

    def representation(self):
        """
        Create a JSON serializable representation of the game. Intended for
        use in web visualizations.

        This base class method creates a dictionary of the following form:
            history: [actions],
            player: player number (as in self.player())
            values: self.values() if terminal, None otherwise
            actions: self.actions()
        """
        return {
            "history": self.history,
            "player": self.player(),
            "values": self.values() if self.is_terminal() else None,
            "actions": self.actions(),
        }

    def is_terminal(self):
        """
        Return whether the state is terminal. Uses `self.player()` by default.
        """
        return self.player() == self.P_TERMINAL

    def is_chance(self):
        """
        Return whether the state is a chance node.
        Uses `self.player()` by default.
        """
        return self.player() == self.P_CHANCE

    def play(self, action):
        """
        Create a new state by playing `action`.
        """
        debug_assert(lambda: action in self.actions())
        return self.__class__(self, action, game=None)

    def __repr__(self):
        s = "<{} of {} {}".format(self.__class__.__name__, self.game.__class__.__name__, self.history)
        if self.is_terminal():
            return "{} terminal, vals {}>".format(s, self.values())
        if self.is_chance():
            return "{} chance, {} actions>".format(s, len(self.actions()))
        return "{} player {}, {} actions>".format(
            s, self.player(), len(self.actions()))
