import collections
import itertools
from typing import (Any, Callable, Hashable, Iterable, List, Optional, Tuple, Union)

import attr
import numpy as np

from .situation import Situation, StateInfo, Action
from .utils import debug_assert, get_rng, uniform, first_occurences


class Game:
    """
    Base class for game instances.

    To define your own game. inherit your game from the appropriate subclass of `Game` and:

    * Call `super().__init__(number_of_players, all_actions)`
    * Define `initial_state(self)`
    * Define `update_state(self, action)`

    Attributes:
        players
        actions
        actions_index
   
    """

    def __init__(self, players: int, actions: Iterable[Any]):
        assert players > 0
        self.players = players
        actions = tuple(actions)
        assert len(actions) == len(set(actions))
        self.actions = tuple(actions)
        self.actions_index = {a: ai for ai, a in enumerate(self.actions)}

    def initial_state(self) -> StateInfo:
        """
        Return the initial game state and state information.

        Note that the initial game state must be always the same. If the game start
        depends on chance, use a chance node as the first state.
        """
        raise NotImplementedError("Define update_state and initial_state for your game.")

    def update_state(self, situation: Situation, action: Action) -> StateInfo:
        """
        Return the updated internal state and state information.
        """
        raise NotImplementedError("Define update_state and initial_state for your game.")

    def start(self) -> Situation:
        """
        Create a new initial game state.
        """
        state_info = self.initial_state()
        return Situation.new(self, state_info)

    def __repr__(self):
        return "<{}(...)>".format(self.__class__.__name__)

    def __str__(self):
        "By default, returns `repr(self)` without the outer `<...>`."
        s = repr(self)
        if s.startswith('<') and s.endswith('>'):
            s = s[1:-1]
        return s

    def play(self, situation: Situation, action: Action) -> Situation:
        """
        Return the situation after playing the given action.

        The original situation is unchanged.
        """
        raise NotImplementedError(
            "Inherit your game from one of the subclasses of `Game`, not `Game` directly.")

    def play_sequence(self, actions: Iterable[Action], *, start: Situation = None) -> Situation:
        """
        Play a sequence of actions, return the last one.

        Starts from a given state or `self.start()`.
        """
        if start is None:
            start = self.start()
        sit = start
        for a in actions:
            sit = self.play(sit, action=a)
        return sit

    def _common_play(self, situation: Situation, action: Action) -> StateInfo:
        """
        A common part of `play()` methods for subclasses. Internal.

        Verifies various invariants before and after `update_state()` call.
        Returns new `StateInfo`.
        """
        if situation.game != self:
            raise ValueError("Playing in wrong game {} (situation has {})".format(
                self, situation.game))
        if situation.is_terminal():
            raise ValueError("Playing in terminal state {}".format(situation))
        debug_assert(lambda: action in situation.actions)

        state_info = self.update_state(situation, action)

        assert state_info.player < self.players
        if state_info.observations is not None:
            assert len(state_info.observations) == self.players + 1
        if state_info.payoff is not None:
            assert len(state_info.payoff) == self.players

        return state_info


class PerfectInformationGame(Game):
    """
    Base for games where the game state is public knowledge.

    All observations are the entire current state (but not the history).

    Note that these games may contain randomness but the game structure is fixed
    (i.e. may be assumed to be public knowledge).
    In this sense these are also complete information games except for knowledge of
    other players' strategies (that is sometimes assumed).

    This base class also serves as marker class for relevant algorithms.
    """

    def play(self, situation: Situation, action: Action) -> Situation:
        """
        Return the situation after playing the given action.

        The original situation is unchanged.
        """
        state_info = self._common_play(situation, action)
        obs = (state_info.state, ) * (self.players + 1)
        return situation.updated(action, state_info, observations=obs)


class ImperfectInformationGame(Game):
    """
    Base for general sequential games with randomness and imperfect player information.

    Player observations are taken directly from `update_state()` and you need to make
    sure that the game is perfect recall.

    This base class also serves as marker class for relevant algorithms.
    """

    def play(self, situation: Situation, action: Action) -> Situation:
        """
        Return the situation after playing the given action.

        The original situation is unchanged.
        """
        state_info = self._common_play(situation, action)
        return situation.updated(action, state_info)


class ObservationSequenceGame(ImperfectInformationGame):
    """
    Base for general sequential games where the observations are accumulated over time.

    Observations from `update_state()` are considere to be the *new information*.
    Player observations are sequences of new observation (from `update_state()`) and
    player actions (immediatelly after played).
    In self-observations, the *value* (rather than number) of the action is observed.
    These games are always perfect recall.

    This base class also serves as marker class for relevant algorithms.
    """

    def play(self, situation: Situation, action: Action) -> Situation:
        """
        Return the situation after playing the given action.

        The original situation is unchanged.
        """
        state_info = self._common_play(situation, action)
        new_obs = state_info.observations
        seq_obs = list(situation.observations)
        for p in range(self.players + 1):
            # Own-action observation
            append_obs = ()
            if p == situation.player:
                append_obs += (action, )
            if new_obs is not None:
                append_obs += (new_obs[p], )
            seq_obs[p] += append_obs
        return situation.updated(action, state_info, observations=tuple(seq_obs))


class SimultaneousGame(ImperfectInformationGame):
    """
    Base for normal-form simultaneous games.

    Player observations are `()` before their turn, their action value after their turn,
    and the tule of all player actions in terminal state.
    
    Super-class of `MatrixGame`. These games are always perfect recall.
    This base class also serves as marker class for relevant algorithms.
    """

    def __init__(self, player_actions: Iterable):
        assert len(player_actions) > 0
        actions = first_occurences(itertools.chain(*player_actions))
        super().__init__(len(player_actions), actions)
        self.player_actions = player_actions

    def initial_state(self) -> StateInfo:
        obs = ((), ) * (self.players + 1)
        return StateInfo.new_player(0, 0, self.player_actions[0], observations=obs)

    def update_state(self, situation: Situation, action: Action) -> StateInfo:
        # next player
        p = situation.state + 1
        assert p == len(situation.history) + 1
        new_history = situation.history + (action, )
        # Terminal?
        if p >= self.players:
            payoff = self.game_payoff(new_history)
            obs = (new_history, ) * (self.players + 1)
            return StateInfo.new_terminal(p, payoff, observations=obs)
        # Next player
        obs = new_history + ((), ) * (self.players + 1 - p)
        return StateInfo.new_player(p, p, self.player_actions[p], observations=obs)

    def game_payoff(self, player_actions) -> Iterable[float]:
        raise NotImplementedError("A simultaneous game needs to implement `_game_payoff()`")
