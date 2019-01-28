# Game Gym
[![MIT Licence](https://img.shields.io/github/license/gavento/gamegym.svg)](https://github.com/gavento/gamegym/blob/master/LICENCE)
[![Build Status](https://travis-ci.org/gavento/gamegym.svg?branch=master)](https://travis-ci.org/gavento/gamegym)
[![PyPI version](https://badge.fury.io/py/gamegym.svg)](https://pypi.org/project/gamegym/)
[![codecov](https://codecov.io/gh/gavento/gamegym/branch/master/graph/badge.svg)](https://codecov.io/gh/gavento/gamegym)
[![Coverage Status](https://coveralls.io/repos/github/gavento/gamegym/badge.svg?branch=master)](https://coveralls.io/github/gavento/gamegym?branch=master)

A game theory framework providing a collection of games, common API and a several game-theoretic algorithms.

**The goal of the project** is to provide tools for buildng complex games (e.g. board games, with partial information or simultaneous moves), computation of approximate strateges and creating artificial intelligence for such games, and to be a base for robust value learning experiments.

*Under active development, looking for ideas and contributors!*

## Overview

Algorithms:

* Outcome sampling MCCFR Nash equilibrium computation
* Exact best response and exploitability
* Approximate best response and exploitablity
* Sparse SGD value learning (with values linear in known features)
* Plotting strategy development (see plots for [Matching Pennies](https://gavento.ucw.cz/view/plot_mccfr_trace_pennies_PCA_all.html), [Rock-Paper-Scissors](https://gavento.ucw.cz/view/plot_mccfr_trace_rps_PCA_all.html), [Goofspiel(4)](https://gavento.ucw.cz/view/plot_mccfr_trace_goof4_PCA_all.html))

Games:

* General matrix games (normal form games), Rock-paper-scissors, Matching pennies, Prisoner's dilemma, ...
* Goofspiel (GOPS)
* One-card poker, Dice-poker

## Game interface

*For an exploration of API in Rust, see [GTCogs](https://github.com/gavento/gtcogs).*

To implement game you define one class derived from `gamegym.Game` with the following interface:

```python
class MyRockPaperScissor(PartialInformationGame):
    ACTIONS = ("rock", "paper", "scissors")
    def __init__(self):
        # Set thenumber of players and all game actions
        super().__init__(2, self.ACTIONS)

    def initial_state(self) -> StateInfo:
        # Return node information, here player 0 is active and has all actions.
        # Note that in this simple game we do not use any game state.
        return StateInfo.new_player(state=None, player=0, actions=self.ACTIONS)

    def update_state(self, situation: Situation, action) -> StateInfo:
        # Return the node information after playing `action` in `situation`.
        if len(situation.history) == 0:
            return StateInfo.new_player(state=None, player=1, actions=self.ACTIONS)
        p1payoff = 1.0 # TODO: compute from `situation`, e.g. from `situation.history`
        return StateInfo.new_terminal(state=None, payoff=(x, -x))

# Create game instance
game = MyRockPaperScissor()
# Initial situation
s1 = game.start()
# Play some actions
s2 = game.play(s1, "rock")
s3 = s2.play("paper") # alias for game.play(s2, "paper")
# See game result
assert s3.is_terminal()
assert s3.payoff == (-1.0, 1.0)
```

The available base classes are `PerfectInformationGame` and `PartialInformationGame`
(with specialised subclasses `ObservationSequenceGame`, `SimultaneousGame` and
`MatrixGame` - which would be a better fit for Rock-Paper-Scissors).

The main auxiliary structs common to all games are `StateInfo` that contains the information
about the game node itself, and `Situation` which additionally contains game history,
accumulated payoffs, the game itself etc.

*Game state* is any structure the game uses to keep track of the actual game state, e.g. cards in all hands, game board state, map, ... This is not generally visible to players in partial information game, any *observations* are passed with `observations=(p0_obs, p1_obs, public_obs)` to `StateInfo`.
