# Game Gym
[![Build Status](https://travis-ci.org/gavento/gamegym.svg?branch=master)](https://travis-ci.org/gavento/gamegym)
[![Coverage Status](https://coveralls.io/repos/github/gavento/gamegym/badge.svg?branch=master)](https://coveralls.io/github/gavento/gamegym?branch=master)

A game theory framework providing a collection of games, common API and a several game-theoretic algorithms.

Algorithms:
* Exact best response and exploitability
* Outcome sampling MCCFR Nash equilibrium computation
* Naive sparse SGD value learning

Games:
* General matrix games (normal form games)
* Rock-paper-scissors, Matching pennies, Prisoner's dilemma, ...
* Goofspiel (GOPS)
* One-card poker

*Under development, looking for users and contributors!*

## Game interface

*For an exploration of API in Rust, see [GTCogs](https://github.com/gavento/gtcogs).*

To implement game you define one class derived from `gamegym.Game` and one from
`gamegym.GameState`.

The first class holds any static game configuration but no game state.
All the state and most of the functionality is in the state instances.
State instances are assumed to hold the full game history (starting from an
empty sequence) but you can override it and add more state information.

Note that players are numbered 0..N-1, player `CHANCE`=-1 represents chance.

The game interface is the following:

```python
class Game:
    def initial_state(self) -> GameState:
        """
        Return the initial state (with empty history).
        Note that the initial state must be always the same. If the game start
        depends on chance, use a chance node as the first state.
        """

    def players(self) -> int:
        """
        Return the number of players N. Chance player is not counted here.
        """

class GameState:

    ### The following methods must be implemented

    def player(self) -> int:
        """
        Return the number of the active player (0..N-1).
        `self.P_CHANCE=-1` for chance nodes and `self.P_TERMINAL=-2` for terminal states.
        """

    def values(self) -> (p0val, p1val, ...):  # one value for each player
        """
        Return a tuple or numpy array of values, one for every player,
        undefined if non-terminal.
        """

    def actions(self) -> [any_hashable]:
        """
        Return a list or tuple of actions valid in this state.
        Should return empty list/tuple for terminal actions.
        """

    def player_information(self, player) -> any_hashable:
        """
        Return the game information from the point of the given player.
        This identifies the player's information set of this state.

        Note that this must distinguish all different information sets,
        e.g. when a player does not see any information on the first two turns,
        she still distinguishes whether it is the first or second round.

        On the other hand (to be consistent with the "information set" concept),
        this does not need to distinguish the player for whom this
        information set is intended, e.g. in the initial state both player 1
        and player 2 may receive `()` as the `player_information`.
        """

    ### The following methods have a sensible default

    def __init__(self, prev_state: GameState, action: any_hashable, game=None):
        """
        Initialize the state from `prev_state` and `action`, or as the initial
        state if `prev_state=None`, `action=None` and game is given.

        This base class keeps track of state action history in `self.history`
        and the game object in `self.game` and may be sufficient for simple games.

        If the state of the game is more complex than the history (e.g. cards in
        player hands), add this as attributes and update them in this
        constructor.
        """

    def chance_probability(self, action) -> Discrete:
        """
        In chance nodes, returns the actions distribution.
        Must not be called in non-chance nodes (and should raise an exception).
        You do not need to modify it if the game has no chance nodes.
        """

    def representation(self) -> json_serializable:
        """
        Create a JSON serializable representation of the game. Intended for
        use in web visualizations.

        This base class method creates a dictionary of the following form:
            history: [actions],
            player: player number (as in self.player())
            values: self.values() if terminal, None otherwise
            actions: self.actions()
        """

    ### The following methods are provided

    def is_terminal(self) -> bool:
        """
        Return whether the state is terminal.
        Uses `self.player()` by default.
        """

    def is_chance(self) -> bool:
        """
        Return whether the state is a chance node.
        Uses `self.player()` by default.
        """

    def play(self, action) -> GameState:
        """
        Create a new state by playing `action`.
        """
```

## Integration with Gambit

### Installing gambit in Python 3

As of 2018-12, Gambit did not work under Python 3 (see [#203](https://github.com/gambitproject/gambit/issues/203)). A fix is pending in [#242](https://github.com/gambitproject/gambit/pull/242). A temporary workaround until this is resolved is to use branch from the author of the fix:

```
git clone https://github.com/rhalbersma/gambit gambit-future
cd gambit-future
git checkout future
aclocal && libtoolize && automake --add-missing && autoconf && ./configure
cd src/python
python3 setup.py build
python3 setup.py install
```
