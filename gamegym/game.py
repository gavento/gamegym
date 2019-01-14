import collections
import itertools
from typing import (Any, Callable, Hashable, Iterable, List, Optional, Tuple, Union)

import attr
import numpy as np

from .situation import Situation, StateInfo
from .utils import debug_assert, get_rng, uniform, first_occurences


class Game:
    """
    Base class for game instances.

    Every instance *must* have attributes `self.players` and `self.actions`.
    Actions are any (hashable) objects
    Players are numbered `0 .. players - 1`.

    Attributes:
        players
        actions
        actions_index
    
    To define your own game. you inherit from `Game` or a similar class, and then

    * call `super().__init__(number_of_players, all_actions)`
    * define `initial_state(self)`
    * define `update_state(self)`
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

    def update_state(self, situation: Situation, action: Any) -> StateInfo:
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

    def play(self, situation: Situation, action_no: int = None, *,
             action: Any = None) -> Situation:
        """
        Return the situation after playing the given action.

        The original situation is unchanged. Action may be given either by value or number.
        """
        raise NotImplementedError(
            "Inherit your game from one of the subclasses of `Game`, not `Game` directly.")

    def play_sequence(self,
                      actions_no: Iterable[int] = None,
                      *,
                      actions: Iterable[Any] = None,
                      start: Situation = None) -> Situation:
        """
        Play a sequence of actions, return the last one.

        Starts from a given state or `self.start()`. The actions may be given by values or their
        indices in the available action lists.
        """
        if start is None:
            start = self.start()
        if (actions is None) == (actions_no is None):
            raise ValueError("Pass exactly one of `actions` and `actions_no`.")
        sit = start
        if actions_no is not None:
            for an in actions_no:
                sit = self.play(sit, action_no=an)
        else:
            for a in actions:
                sit = self.play(sit, action=a)
        return sit

    def _common_play(self, situation: Situation, action_no: int = None,
                     action: Any = None) -> (Any, int, StateInfo):
        """
        A common part of `play()` methods for subclasses. Internal.

        Verifies various invariants before and after `update_state()` call.
        Action can be given either by value or by index in the available actons (or both).
        Returns `(action_no, action, StateInfo)`.
        """
        if situation.game != self:
            raise ValueError("Playing in wrong game {} (situation has {})".format(
                self, situation.game))
        if (action_no is None) and (action is None):
            raise ValueError("Pass at least one of `action` and `index`.")
        if situation.is_terminal():
            raise ValueError("Playing in terminal state {}".format(situation))
        if action is None:
            action = self.actions[action_no]
        elif action_no is None:
            action_no = self.actions_index[action]
        else:
            assert self.actions[action_no] == action

        state_info = self.update_state(situation, action)

        assert state_info.player < self.players
        if state_info.observations is not None:
            assert len(state_info.observations) == self.players + 1
        if state_info.payoff is not None:
            assert len(state_info.payoff) == self.players

        return (action_no, action, state_info)


class PerfectInformationGame(Game):
    """
    Games where the game state is public knowledge.

    All observations are the entire current state (but not the history).

    Note that these games may contain randomness but the game structure is fixed
    (i.e. may be assumed to be public knowledge).
    In this sense these are also complete information games except for knowledge of
    other players' strategies (that is sometimes assumed).

    This base class also serves as marker class for relevant algorithms.
    """

    def play(self, situation: Situation, action_no: int = None, *,
             action: Any = None) -> Situation:
        """
        Return the situation after playing the given action.

        The original situation is unchanged. Action may be given either by value or number.
        """
        action_no, action, state_info = self._common_play(situation, action_no, action)
        obs = (state_info.state, ) * (self.players + 1)
        return situation.updated(action_no, state_info, observations=obs)


class ImperfectInformationGame(Game):
    """
    General sequential games with randomness and imperfect player information.

    Player observations are taken directly from `update_state()` and you need to make
    sure that the game is perfect recall.

    This base class also serves as marker class for relevant algorithms.
    """

    def play(self, situation: Situation, action_no: int = None, *,
             action: Any = None) -> Situation:
        """
        Return the situation after playing the given action.

        The original situation is unchanged. Action may be given either by value or number.
        """
        action_no, action, state_info = self._common_play(situation, action_no, action)
        return situation.updated(action_no, state_info)


class ObservationSequenceGame(ImperfectInformationGame):
    """
    General sequential games where the observations are accumulated over time.

    Observations from `update_state()` are considere to be the *new information*.
    Player observations are sequences of new observation (from `update_state()`) and
    player actions (immediatelly after played).
    In self-observations, the *value* (rather than number) of the action is observed.
    These games are always perfect recall.

    This base class also serves as marker class for relevant algorithms.
    """

    def play(self, situation: Situation, action_no: int = None, *,
             action: Any = None) -> Situation:
        """
        Return the situation after playing the given action.

        The original situation is unchanged. Action may be given either by value or number.
        """
        action_no, action, state_info = self._common_play(situation, action_no, action)
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
        return situation.updated(action_no, state_info, observations=tuple(seq_obs))


class SimultaneousGame(ImperfectInformationGame):
    """
    Normal-form simultaneous games.

    Player observations are `()` before their turn, their action value after their turn,
    and the tule of all player actions in terminal state.
    
    Super-class of `MatrixGame`. These games are always perfect recall.
    This base class also serves as marker class for relevant algorithms.
    """

    def __init__(self, player_actions: Iterable):
        assert len(player_actions) > 0
        actions = first_occurences(itertools.chain(*player_actions))
        super().__init__(len(player_actions), actions)
        self.player_actions_no = tuple(
            tuple(self.actions_index[a] for a in pa) for pa in player_actions)

    def initial_state(self) -> StateInfo:
        obs = ((), ) * (self.players + 1)
        return StateInfo.new_player(0, 0, actions_no=self.player_actions_no[0], observations=obs)

    def update_state(self, situation: Situation, action: Any) -> StateInfo:
        # next player
        p = situation.state + 1
        assert p == len(situation.history) + 1
        player_actions = situation.history_actions() + (action, )
        # Terminal?
        if p >= self.players:
            payoff = self._game_payoff(player_actions)
            obs = (player_actions, ) * (self.players + 1)
            return StateInfo.new_terminal(p, payoff, observations=obs)
        # Next player
        obs = player_actions + ((), ) * (self.players + 1 - p)
        return StateInfo.new_player(p, p, actions_no=self.player_actions_no[p], observations=obs)

    def _game_payoff(self, player_actions) -> Iterable[float]:
        raise NotImplementedError("A simultaneous game needs to implement `_game_payoff()`")
