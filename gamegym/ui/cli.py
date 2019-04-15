from ..errors import DecodeObservationInvalidData
from ..observation import Observation
from ..strategy import Strategy
from ..utils import Distribution


class CliStrategy(Strategy):
    def make_policy(self, observation: Observation) -> Distribution:
        print(self.adapter.colored("~~~~ player: {} ~~~~".format(observation.player), 'yellow'))
        print(observation.data)

        while True:
            try:
                line = input(">> ")
                policy = self.adapter.decode_actions(observation, line)
                return policy
            except DecodeObservationInvalidData:
                print("Invalid action. Available actions:\n{}".format(
                    self.adapter.actions_to_text(observation.actions)))
