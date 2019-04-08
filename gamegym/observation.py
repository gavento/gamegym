
from typing import Any, NewType

ObservationData = NewType("ObservationData", Any)
ActionData = NewType("ActionData", Any)

class InvalidActionData(Exception):
    pass

@attr.s(slots=True)
class Observation:

    game = attr.ib(type=Game)
    data = attr.ib(type=Any)
    actions = attr.ib()
    # Active player number
    player = attr.ib(type=int)


    @classmethod
    def new_observation(cls,
                        sitation: Situation):
        player = sitation.player
        assert player >= 0
        return cls(situation.game,
                   self.observation_data(situation),
                   situation.actions,
                   player)

    @classmethod
    def observation_data(cls,
                         situation: Situation):
        raise NotImplementedError()


    @classmethod
    def decode_actions(cls, action_data: ActionData) -> Distribution:
        """
        may throw an InvalidActionData exception
        """
        raise NotImplementedError()


    @classmethod
    def encode_actions(cls, distribution: Distribution) -> ActionData:
        raise NotImplementedError()