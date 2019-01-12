import collections
import attr
import numpy as np
from typing import List, Tuple, Optional, Hashable, Callable, Any, Union, Iterable

from .utils import debug_assert, get_rng, uniform
from .game import Game


@attr.s(slots=True, cmp=False, frozen=True)
class StateInfo:
    """
    Game state information: active player, actions, payoffs (in terminals), chance distribution.

    Attrbiutes:
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

    # Active player number
    player = attr.ib(type=int)
    # Tuple of actions (not indexes)
    actions = attr.ib(type=tuple)
    # Tuple of rewards
    payoff = attr.ib(type=Union[tuple, np.ndarray])
    # In chance nodes
    chance = attr.ib(type=Union[tuple, np.ndarray])
    observations = attr.ib(type=tuple)

    @classmethod
    def new_player(cls, p, actions):
        assert p >= 0
        return cls(p, actions, None, None)

    @classmethod
    def new_chance(cls, chance, actions):
        if chance is None:
            chance = uniform(len(actions))
        assert len(actions) == len(chance)
        assert len(actions) >= 0
        return cls(cls.CHANCE, actions, None, chance)

    @classmethod
    def new_terminal(cls, payoffs):
        return cls(cls.TERMINAL, (), payoffs, None)

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
    game = attr.ib(type=Game)
    # Game-specific state object (full information state)
    state = attr.ib(type=Any)
    # Sequence of actions indices
    history = attr.ib(type=tuple)
    # Tuple of (players+1) observations (last is the public information) 
    observations = attr.ib(type=Optional(tuple))
    # Accumulated player payoffs.
    payoff = attr.ib(type=np.ndarray)
    # Node and active player information
    _info = attr.ib()  #type=StateInfo)

    @property
    def player(self) -> int:
        return self._info.player

    @property
    def actions(self) -> Iterable:
        return self._info.actions

    @property
    def chance(self) -> Optional(Iterable):
        return self._info.chance

    def is_terminal(self) -> bool:
        return self._info.is_terminal()

    def is_chance(self) -> bool:
        return self._info.is_chance()

    @classmethod
    def new(cls, game, state, info):
        assert isinstance(game, Game)
        assert info.player < game.players
        obs = info.observations
        if obs is None:
            obs = ((), ) * (game.players + 1)
        payoff = info.payoff
        if payoff is None:
            payoff = np.zeros(game.players, dtype=np.float64)
        return cls(game, state, (), obs, payoff, info)

    def updated(self, action_idx, new_state, new_info):
        """
        Create an updated situation from self.

        Computes cumulative payoff.
        """

        if self._info.payoff is None:
            return np.zeros(self.game.players)
        return cls(self.game, state, hist + action, obs, payoff, info)

    def play(self, action=None, index=None) -> 'Situation':
        return self.game.play(self, action=action, index=index)
