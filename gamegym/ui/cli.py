from ..errors import DecodeObservationInvalidData, ObservationNotAvailable
from ..situation import StateInfo
from ..observation import Observation
from ..strategy import Strategy
from ..utils import Distribution
from ..algorithms.stats import play_strategies


class CliStrategy(Strategy):

    DEFAULT_ADAPTER = "TextAdapter"

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


def play_in_terminal(game,
                     strategies=None,
                     *,
                     adapter=None,
                     rng=None,
                     seed=None):
    """
    Plays the game in terminal. 
    
    Strategies is either one strategy per player, with `None` replaced by the CLI player,
    or `None` for all CLI players. Adapter is used to create `CliStrategy`,
    `seed` or `rng` for chance nodes randomness.
    """
    cli_strat = CliStrategy(game, adapter=adapter)
    if strategies is None:
        strategies = [None] * game.players
    strategies = [cli_strat if s is None else s for s in strategies]
    res = play_strategies(cli_strat.adapter.game, strategies, rng=rng, seed=seed)

    print(cli_strat.adapter.colored("~~~~ terminal with payoffs {} ~~~~".format(res.payoff), 'yellow'))
    try:
        obs = cli_strat.adapter.observe_data(res, StateInfo.OMNISCIENT)
        print(obs)
    except ObservationNotAvailable:
        print(adapter.colored("[not available]", 'white', None, ['dark']))

    return res