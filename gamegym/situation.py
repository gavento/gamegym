import collections
from typing import (Any, Callable, Hashable, Iterable, List, Optional, Tuple, Union, NewType)

import attr
import numpy as np

from .utils import debug_assert, get_rng, uniform

Action = NewType('Action', Any)
"""
Action is a NewType(Any) to allow for simple typechecking.
"""


@attr.s(slots=True, cmp=False, frozen=True)
class StateInfo:
    """
    Game state information: active player, actions, payoffs (in terminals), chance distribution.

    Attrbiutes:
        state (any): 
            Internal state of the game.
        player (int): 
            Active player number (or `self.CHANCE`, `self.TERMINAL`).
        actions (tuple or array):
            Available actions.
        payoff (tuple or array):
            Tuple of rewards for all the players. Allowed even outside terminal nodes. `None` for all zeros.
        chance (tuple or array):
            In chance nodes, distribution across `self.actions`, otherwise `None`.
        observations (tuple or array):
            `players+1` observations (for every player and the public) or `None` for no observation.
            For `Generi`SequentialGame`
    """
    CHANCE = -1
    TERMINAL = -2

    # Internal game state
    state = attr.ib(type=Any)
    # Active player number
    player = attr.ib(type=int)
    # Tuple of action numbers
    actions = attr.ib(type=Iterable[Action])
    # Player rewards in this node
    payoff = attr.ib(type=Union[None, Iterable[float]])
    # In chance nodes
    chance = attr.ib(type=Union[None, Iterable[float]])
    # Overall or immediate observations for the players and the public
    observations = attr.ib(type=tuple)

    @classmethod
    def new_player(cls,
                   state: Any,
                   player: int,
                   actions: Iterable[Action],
                   payoff=None,
                   observations=None):
        assert player >= 0
        assert len(actions) >= 0
        return cls(state, player, actions, payoff, None, observations)

    @classmethod
    def new_chance(cls,
                   state: Any,
                   actions: Iterable[Action],
                   chance,
                   payoff=None,
                   observations=None):
        assert len(actions) >= 0
        if chance is None:
            chance = uniform(len(actions))
        assert len(actions) == len(chance)
        return cls(state, cls.CHANCE, actions, payoff, chance, observations)

    @classmethod
    def new_terminal(cls, state: Any, payoff, observations=None):
        return cls(state, cls.TERMINAL, (), payoff, None, observations)

    def is_chance(self):
        return self.player == self.CHANCE

    def is_terminal(self):
        return self.player == self.TERMINAL


@attr.s(slots=True, cmp=False, frozen=True)
class Situation:
    """
    One game history and associated structures: observations, active player and actions, state.

    Attributes:

        game
        state
        history
        observations
        player
        actions
        chance
        payoff
    """
    # Link to the underlying game
    # NOTE: may be replaced by a weak_ref and attribute
    game = attr.ib(type='Game')
    # Sequence of actions indices
    history = attr.ib(type=Iterable[Action])
    # Tuple of (players+1) observations (last is the public information)
    observations = attr.ib(type=Iterable[Any])
    # Accumulated player payoffs.
    payoff = attr.ib(type=np.ndarray)
    # Node and active player information
    _info = attr.ib(type=StateInfo)

    @property
    def state(self) -> Any:
        return self._info.state

    @property
    def player(self) -> int:
        return self._info.player

    @property
    def actions(self) -> Iterable[Action]:
        return self._info.actions

    @property
    def chance(self) -> Optional[Iterable]:
        return self._info.chance

    def is_terminal(self) -> bool:
        return self._info.is_terminal()

    def is_chance(self) -> bool:
        return self._info.is_chance()

    def history_indexes(self) -> Iterable[int]:
        """
        Return the history as a tuple of action numbers.
        """
        return tuple(self.game.actions_index[a] for a in self.history)

    @classmethod
    def new(cls, game, state_info: StateInfo):
        """
        Create a new Situation for game and state info.
        """
        from .game import Game

        assert isinstance(game, Game)
        assert state_info.player < game.players

        obs = state_info.observations
        if obs is None:
            obs = ((), ) * (game.players + 1)
        assert len(obs) == game.players + 1

        payoff = state_info.payoff
        if payoff is None:
            payoff = np.zeros(game.players, dtype=np.float64)
        assert len(payoff) == game.players

        return cls(game, (), obs, payoff, state_info)

    def updated(self, action: Action, new_state_info: StateInfo, observations=None) -> 'Situation':
        """
        Create an updated situation from self.

        Computes cumulative payoff.
        Observations are taken from `new_state_info` and 
        may be overriden with the parameter `observations`.
        """
        obs = new_state_info.observations
        if observations is not None:
            obs = observations
        if obs is None:
            obs = ((), ) * (self.game.players + 1)

        payoff = self.payoff
        if new_state_info.payoff is not None:
            payoff = np.add(payoff, new_state_info.payoff)
        return self.__class__(self.game, self.history + (action, ), obs, payoff, new_state_info)

    def play(self, action: Action) -> 'Situation':
        """
        Shortcut for `self.game.play(self, action)`
        """
        return self.game.play(self, action)
