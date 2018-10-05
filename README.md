# Game Gym

A game theory framework providing a collection of games, common API and a several game-theoretic algorithms.

*Under heavy development*

## Game interface

To implement game you define one class derived from `gamegym.Game` and one from `gamegym.GameState`.

The first class holds any static game configuration but no game state.
All the state and most of the functionality is in the state instances.
State instances are assumed to hold the full game history (starting from an empty sequence) but you
can override it.

Note that if you want to merge some histories (treat them as the same, simplified state), the right place for this is
in `GameState.information_set()` and `GameState.canonical_form()`.

The basic game interface is the following:

```python
class Game:
    def initial_state(self) -> GameState:
        "Return the initial state (with empty history)"

class GameState:
    def __init__(self, game, history):
        """
        Initialize the state, keeping the reference to the game in `self.game`
        and the history in `self.h`.
        """

    def is_terminal(self) -> bool:
        "Return whether the state is terminal."

    def values(self) -> (p0, p2, ...):  # one value for each player
        "Return a tuple of values, one for every player, undefined if non-terminal."

    def player(self) -> int:
        "Return the number of the active player, -1 for chance nodes."

    def information_set(self, player) -> anything_hashable:
        "Return the information set for this state for the active player."

    def canonical_form(self) -> anything_hashable:
        "Return a canonical representation of the state."

    def actions(self) -> [(action_label, new_state, probability)]:
        """
        Return an iterable of (action_label, new_state, probability)
        where probability is `None` for non-chance states.
        """
```
