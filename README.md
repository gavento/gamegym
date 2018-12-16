# Game Gym
[![MIT Licence](https://img.shields.io/github/license/gavento/gamegym.svg)](https://github.com/gavento/gamegym/blob/master/LICENCE)
[![Build Status](https://travis-ci.org/gavento/gamegym.svg?branch=master)](https://travis-ci.org/gavento/gamegym)
[![Coverage Status](https://coveralls.io/repos/github/gavento/gamegym/badge.svg?branch=master)](https://coveralls.io/github/gavento/gamegym?branch=master)
[![PyPI version](https://badge.fury.io/py/gamegym.svg)](https://pypi.org/project/gamegym/)

A game theory framework providing a collection of games, common API and a several game-theoretic algorithms.

**The goal of the project** is to provide tools for buildng complex games (e.g. board games, with partial information or simultaneous moves), computation of approximate strateges and creating artificial intelligence for such games, and to be a base for robust value learning experiments.

*Under active development, looking for users, ideas and contributors!*

## Overview

Algorithms:

* Outcome sampling MCCFR Nash equilibrium computation
* Exact best response and exploitability
* Approximate best response and exploitablity
* Sparse SGD value learning (with values linear in known features)

Games:

* General matrix games (normal form games)
* Rock-paper-scissors, Matching pennies, Prisoner's dilemma, ...
* Goofspiel (GOPS)
* One-card poker

## Game interface

*For an exploration of API in Rust, see [GTCogs](https://github.com/gavento/gtcogs).*

To implement game you define one class derived from `gamegym.Game` with the following interface:

```python
class Game:
    """
    Base class for game instances.

    Every descendant must have an attribute `self.players`.
    Players are numbered `0 .. players - 1`.
    """

    def __init__(self):
        self.players = 2

    def initial_state(self) -> Tuple[Any, ActivePlayer]:
        """
        Return the initial internal state and active player.

        Note that the initial game state must be always the same. If the game start
        depends on chance, use a chance node as the first state.
        """
        raise NotImplementedError

    def update_state(self, sit: Situation, action: Any) -> Tuple[Any, ActivePlayer, tuple]:
        """
        Return the updated internal state, active player and per-player observations.

        The observations must have length 0 (no obs for anyone)
        or (players + 1) (last observation is the public one).
        """
        raise NotImplementedError

    # ... more methods provided
    # Use sit1 = game.start() to obtain a starting situation, and
    # sit2 = game.play(sit1, action) to obtain an updated situation.
```

The main auxiliary structs common to all games are `Situation` and `ActivePlayer`.

```python
@attr.s
class Situation:
    """
    One gae history and associated structures: observations, active player and actions, state.
    """
    history = attr.ib(type=tuple)
    history_idx = attr.ib(type=tuple)
    active = attr.ib(type=ActivePlayer)
    observations = attr.ib(type=tuple)
    state = attr.ib(type=Any)
    game = attr.ib(type='Game')

@attr.s
class ActivePlayer:
    """
    Game mode description: active player, actions, payoffs (in terminals), chance distribution.
    """
    CHANCE = -1
    TERMINAL = -2

    player = attr.ib(type=int)
    actions = attr.ib(type=tuple)
    payoff = attr.ib(type=Union[tuple, np.ndarray])
    chance = attr.ib(type=Union[tuple, np.ndarray])
```

## Integration with Gambit

In `gamegym.contrib.gambit` there is basic integration with [Gambit project](https://github.com/gambitproject/gambit) game library (export `Game` to `.efg`, import computed
strategy). However, gambit is only suitable for very small games (e.g. <100 states) and is not
actively developed anymore.

### Installing gambit in Python 3

As of 2018-12, Gambit did not work under Python 3 (see [#203](https://github.com/gambitproject/gambit/issues/203)) and there were some problems building it with recent GCC (see [#220](https://github.com/gambitproject/gambit/issues/220)). A fix is pending in [#242](https://github.com/gambitproject/gambit/pull/242). A temporary workaround until this is resolved is to use a git branch from the author of the fix:

```shell
git clone https://github.com/rhalbersma/gambit gambit-future
cd gambit-future
git checkout future
aclocal && libtoolize && automake --add-missing && autoconf && ./configure
cd src/python
python3 setup.py build
python3 setup.py install
```
