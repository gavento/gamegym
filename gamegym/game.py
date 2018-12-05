#!/usr/bin/python3

import collections
from attr import attrs, attrib
import numpy as np
from typing import List, Tuple, Optional, Hashable, Callable, Any, Union

from .utils import debug_assert, get_rng, uniform


@attrs(slots=True, cmp=False, frozen=True)
class Observation:
    OBSERVATION = 1
    OWN_ACTION = 2

    kind = attrib(type=int)
    obs = attrib(type=Hashable)


@attrs(slots=True, cmp=False, frozen=True)
class Active:
    CHANCE = -1
    TERMINAL = -2

    player = attrib(type=int)
    actions = attrib(type=tuple)
    payoff = attrib(type=Union[tuple, np.ndarray])
    chance = attrib(type=Union[tuple, np.ndarray])

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


@attrs(slots=True, cmp=False, frozen=True)
class GameState:
    history = attrib(type=tuple)
    history_idx = attrib(type=tuple)
    active = attrib(type=Active)
    observations = attrib(type=tuple)
    state = attrib(type=Any)
    game = attrib(type='Game')

    def __len__(self):
        return len(self.history)


class Game:
    """
    Base class for game instances.

    Any descendant must have attribute `self.players`.
    Players are numbered `0 .. players - 1`.
    """

    def initial_state(self) -> Tuple[Any, Active]:
        """
        Return the initial internal state and active player.

        Note that the initial game state must be always the same. If the game start
        depends on chance, use a chance node as the first state.
        """
        raise NotImplementedError

    def update_state(self, state: GameState, action: Any) -> Tuple[Any, Active, tuple]:
        """
        Return the updated internal state, active player and per-player observations.

        The observations must have length 0 (no obs for anyone) 
        or (players + 1) (last observation is the public one).
        """
        raise NotImplementedError

    def start(self) -> GameState:
        """
        Create a new initial game state.
        """
        state, active = self.initial_state()
        assert active.player < self.players
        return GameState((), (), active, ((), ) * (self.players + 1), state, self)

    def play(self, hist, action=None, index=None) -> GameState:
        """
        Create and return a new game state by playing given action.

        Action can be given either by value or by index in the available actons (or both).
        """
        if hist.game != self:
            raise ValueError("Playing in wrong game {} (state has {})".format(self, hist.game))
        if (action is None) and (index is None):
            raise ValueError("Pass at least one of `action` and `index`.")
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
        if len(obs) > 0:
            new_obs = []
            for i in range(self.players + 1):
                o_p, o_a = (), ()
                if i == hist.active.player:
                    o_a = (Observation(Observation.OWN_ACTION, action), )  # type: ignore
                if obs[i] is not None:
                    o_p = (Observation(Observation.OBSERVATION, obs[i]), )  # type: ignore
                new_obs.append(hist.observations[i] + o_a + o_p)
            new_obs = tuple(new_obs)
        return GameState(hist.history + (action, ), hist.history_idx + (index, ), active, new_obs,
                         state, self)

    def play_sequence(self, actions=None, *, indexes=None,
                      start: GameState = None) -> List[GameState]:
        """
        Play a sequence of actions, return a list of the visited states (including the start).

        Starts from a given state or `self.start()`. The actions may be given by values or their
        indices in the available action lists.
        """
        if start is None:
            start = self.start()
        if (actions is None) == (indexes is None):
            raise ValueError("Pass exactly one of `actions` and `indexes`.")
        hist = start
        res = [hist]
        if actions is not None:
            for a in actions:
                hist = self.play(hist, action=a)
                res.append(hist)
        else:
            for i in indexes:
                hist = self.play(hist, index=i)
                res.append(hist)
        return res

    def play_strategies(self,
                        strategies,
                        *,
                        rng=None,
                        seed=None,
                        start: GameState = None,
                        stop_when: Callable = None,
                        max_moves: int = None):
        """
        Generate a play based on given strategies (one per player), return list of visited states (including the start).

        Starts from a given state or `self.start()`. Plays until a terminal state is hit, `stop_when(hist)` is True or
        for at most `max_moves` actions (whenever given).
        """
        rng = get_rng(rng=rng, seed=seed)
        if len(strategies) != self.players:
            raise ValueError("One strategy per player required")
        if start is None:
            start = self.start()
        hist = start
        res = [hist]
        while not hist.active.is_terminal():
            if stop_when is not None and stop_when(hist):
                break
            if max_moves is not None and len(res) > max_moves:
                break
            if hist.active.is_chance():
                dist = hist.active.chance
            else:
                p = hist.active.player
                dist = strategies[p].distribution(hist.observations[p], hist.active)
            assert len(dist) == len(hist.active.actions)
            idx = rng.choice(len(hist.active.actions), p=dist)
            hist = self.play(hist, index=idx)
            res.append(hist)
        return res

    def sample_payoff(self, strategies, iterations, seed=None, rng=None):
        """
        Play the game `iterations` times using `strategies`.
        
        Returns `(mean payoff, payoff variances)` as two numpy arrays.
        """
        rng = get_rng(rng=rng, seed=seed)
        payoffs = [self.play_strategies(strategies, rng=rng)[-1].active.payoff for i in range(200)]
        return (np.mean(payoffs, axis=0), np.var(payoffs, axis=0))

    def __repr__(self):
        return "<{}(...)>".format(self.__class__.__name__)

    def __str__(self):
        "By default, strips the outer '<..>' from `repr(self)`."
        s = repr(self)
        if s.startswith('<') and s.endswith('>'):
            s = s[1:-1]
        return s
