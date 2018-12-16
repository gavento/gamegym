#!/usr/bin/python3

import collections
import attr
import numpy as np
from typing import List, Tuple, Optional, Hashable, Callable, Any, Union

from .utils import debug_assert, get_rng, uniform


@attr.s(slots=True, cmp=True, frozen=True, repr=False)
class Observation:
    """
    Single piece of new observation.
    
    Represents either the action of the active player or other observation.
    """
    OBSERVATION = 1
    OWN_ACTION = 2

    kind = attr.ib(type=int)
    obs = attr.ib(type=Hashable)

    def __repr__(self):
        if self.kind == self.OBSERVATION:
            return "Obs({})".format(self.obs)
        return "Own({})".format(self.obs)


@attr.s(slots=True, cmp=False, frozen=True)
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
    One gae history and associated structures: observations, active player and actions, state.
    """
    history = attr.ib(type=tuple)
    history_idx = attr.ib(type=tuple)
    active = attr.ib()  #type=ActivePlayer)
    observations = attr.ib(type=tuple)
    state = attr.ib(type=Any)
    game = attr.ib(type='Game')

    def __len__(self):
        return len(self.history)

    @property
    def player(self):
        return self.active.player

    @property
    def actions(self):
        return self.active.actions

    @property
    def chance(self):
        return self.active.chance

    @property
    def payoff(self):
        return self.active.payoff

    def is_terminal(self):
        return self.active.is_terminal()

    def is_chance(self):
        return self.active.is_chance()


@attr.s(slots=True, cmp=False, frozen=True)
class SituationUpdate:
    active = attr.ib()
    state = attr.ib(default=None)
    observations = attr.ib(default=None)


class Game:
    """
    Base class for game instances.

    Every descendant must have an attribute `self.players`.
    Players are numbered `0 .. players - 1`.
    """

    def __init__(self):
        self.players = None

    def initial_state(self) -> Tuple[Any, ActivePlayer]:
        """
        Return the initial internal state and active player.

        Note that the initial game state must be always the same. If the game start
        depends on chance, use a chance node as the first state.
        """
        raise NotImplementedError

    def update_state(self, state: Situation, action: Any) -> Tuple[Any, ActivePlayer, tuple]:
        """
        Return the updated internal state, active player and per-player observations.

        The observations must have length 0 (no obs for anyone) 
        or (players + 1) (last observation is the public one).
        """
        raise NotImplementedError

    def start(self) -> Situation:
        """
        Create a new initial game state.
        """
        state, active = self.initial_state()
        assert active.player < self.players
        return Situation((), (), active, ((), ) * (self.players + 1), state, self)

    def play(self, hist, action=None, index=None, reuse=False) -> Situation:
        """
        Create and return a new game state by playing given action.

        Action can be given either by value or by index in the available actons (or both).
        """
        if hist.game != self:
            raise ValueError("Playing in wrong game {} (state has {})".format(self, hist.game))
        if (action is None) and (index is None):
            raise ValueError("Pass at least one of `action` and `index`.")
        assert not hist.active.is_terminal()
        if action is None:
            action = hist.active.actions[index]
        if index is None:
            index = hist.active.actions.index(action)
        if hist.active.is_terminal():
            raise ValueError("Playing in terminal state {}".format(hist))
        state, active, obs = self.update_state(hist, action)
        assert active.player < self.players
        assert len(obs) in (0, self.players + 1)
        new_obs = hist.observations
        new_obs = []
        for i in range(self.players + 1):
            o_p, o_a = (), ()
            if i == hist.active.player:
                o_a = (Observation(Observation.OWN_ACTION, action), )  # type: ignore
            if len(obs) > 0 and obs[i] is not None:
                o_p = (Observation(Observation.OBSERVATION, obs[i]), )  # type: ignore
            new_obs.append(hist.observations[i] + o_a + o_p)
        new_obs = tuple(new_obs)
        return Situation(hist.history + (action, ), hist.history_idx + (index, ), active, new_obs,
                         state, self)

    def play_sequence(self, actions=None, *, indexes=None, start: Situation = None,
                      reuse=False) -> List[Situation]:
        """
        Play a sequence of actions, return the last visited state.

        Starts from a given state or `self.start()`. The actions may be given by values or their
        indices in the available action lists.
        """
        if start is None:
            start = self.start()
            reuse = True
        if (actions is None) == (indexes is None):
            raise ValueError("Pass exactly one of `actions` and `indexes`.")
        hist = start
        if actions is not None:
            for a in actions:
                hist = self.play(hist, action=a, reuse=reuse)
                reuse = True
        else:
            for i in indexes:
                hist = self.play(hist, index=i, reuse=reuse)
                reuse = True
        return hist

    def play_strategies(self,
                        strategies,
                        *,
                        rng=None,
                        seed=None,
                        start: Situation = None,
                        reuse=False,
                        stop_when: Callable = None,
                        max_moves: int = None):
        """
        Generate a play based on given strategies (one per player), return the last state.

        Starts from a given state or `self.start()`. Plays until a terminal state is hit, `stop_when(hist)` is True or
        for at most `max_moves` actions (whenever given).
        """
        moves = 0
        rng = get_rng(rng=rng, seed=seed)
        if len(strategies) != self.players:
            raise ValueError("One strategy per player required")
        if start is None:
            start = self.start()
            reuse = True
        hist = start
        while not hist.active.is_terminal():
            if stop_when is not None and stop_when(hist):
                break
            if max_moves is not None and moves >= max_moves:
                break
            if hist.active.is_chance():
                dist = hist.active.chance
            else:
                p = hist.active.player
                dist = strategies[p].strategy(hist)
            assert len(dist) == len(hist.active.actions)
            idx = rng.choice(len(hist.active.actions), p=dist)
            hist = self.play(hist, index=idx, reuse=reuse)
            moves += 1
            reuse = True
        return hist

    def sample_payoff(self, strategies, iterations=100, *, seed=None, rng=None):
        """
        Play the game `iterations` times using `strategies`.
        
        Returns `(mean payoff, payoff variances)` as two numpy arrays.
        """
        rng = get_rng(rng=rng, seed=seed)
        payoffs = [
            self.play_strategies(strategies, rng=rng).active.payoff for i in range(iterations)
        ]
        return (np.mean(payoffs, axis=0), np.var(payoffs, axis=0))

    def __repr__(self):
        return "<{}(...)>".format(self.__class__.__name__)

    def __str__(self):
        "By default, strips the outer '<..>' from `repr(self)`."
        s = repr(self)
        if s.startswith('<') and s.endswith('>'):
            s = s[1:-1]
        return s
