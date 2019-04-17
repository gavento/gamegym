from typing import Any, NewType, Tuple
from .game import Game, Action
from .situation import StateInfo

import attr

ObservationData = NewType("ObservationData", Any)


@attr.s(slots=True)
class Observation:
    """
    Observation from the point of a single player (or the public).

    Is a subset of information available in the Situation.
    Semantics of `data` depends on the adapter and strategy. The usual are
    textual description of observations, tuple of ndarrays,
    JSON or general hashable objects.
    """
    # Game this observation is for
    game = attr.ib(type=Game)
    # Tuple of available actions
    actions = attr.ib(type=Tuple[Action])
    # Active player number
    player = attr.ib(type=int)
    # Observed data, depends on strategy and adapter
    data = attr.ib(type=ObservationData)

    def is_terminal(self) -> bool:
        return self.player == StateInfo.TERMINAL

    def is_chance(self) -> bool:
        return self.player == StateInfo.CHANCE
