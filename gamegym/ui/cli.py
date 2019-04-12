
from ..strategy import Strategy
from ..observation import Observation
from ..utils import Distribution

class CliStrategy(Strategy):

        def _get_policy(self, observation):
            line = input(">> ")
            return self.adapter.decode_actions(observation, line)


        def make_policy(self, observation: Observation) -> Distribution:
            print("~~~~ player: {} ~~~~".format(observation.player))
            print(observation.data)

            policy = self._get_policy(observation)
            while policy is None:
                print("Invalid action")
                policy = self._get_policy(observation)
            return policy

