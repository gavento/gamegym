import numpy as np

from .game import Game
from .observation import Observation, ObservationData
from .situation import Situation
from .utils import Distribution, flatten_array_list


class Adapter():
    """
    Adapter extracts specific type of observation from a game situation.

    Adapter both filters the visible observation (leaving only what is visible
    to the requested played) and feormats the observation in a format suitable for
    the strategy. The following types of adapters and strategies exist
    (but you can create your own):

    * Text representation (for console play, or textual neural nets)
    * Hashable data representation (for tabular strategies)
    * Numeric ndarrays (for neural networks)
    * JSON/SVG/... (for displying in a web gui)

    Some games have symmetric representation for both players (e.g. gomoku),
    for those games the default adapter behavious is that to report two
    symmetric situations as distinct. When you create such adapters with 
    `symmetrize=True`, they will produce all observations as if the active player
    was player 0. Note that this should be done also for public observation.

    Adapter may or may not define observation from a non-current player's
    point of view, but is not required to do so.
    Note that full information games give the same information from any player's
    point of view, regardless of symmetrization.
    """
    SYMMETRIZABLE = False

    def __init__(self, game: Game, symmetrize=False):
        self.game = game
        assert self.SYMMETRIZABLE or not symmetrize
        self.symmetrize = symmetrize

    def get_observation(self, sit: Situation, player: int = None) -> Observtion:
        """
        Create an `Observation` object for given situation.

        Internally uses `observe_data`. By default, provides an
        observation from the point of active player.
        Use `player=-1` to request public state.

        Some adapters may not provide observations for e.g. inactive players or,
        in rare cases, even for all situations of the active player.
        """
        if player is None:
            player = sit.player
        d = self.observe_data(sit, player)
        return Observation(sit.game, sit.actions, player, d)

    def observe_data(self, situation: Situation, player: int) -> ObservationData:
        """
        Provide the observation data from the point of view of the
        specified player.

        NOTE: symm and public
        
        Raise `ObservationNotAvailable` where the observation is not
        specified.
        """
        raise NotImplementedError


class TensorAdapter(Adapter):
    """
    Used to encode 

    Also provides methods to decode action distribution from neural net output,
    and encode target policy into neural net output for training.

    By default the decoding assumes the neural net output is a 1D probability vector
    indexed by actions. Other shapes and action ordering in the output can be
    obtained by overriding `_generate_shaped_actions`, or by reimplementing both
    `decode_actions` and `encode_actions`.    
    """
    def __init__(self, game, symmetrize=False):
        super().__init__(game, symmetrize=symmetrize)
        shaped = self._generate_shaped_actions()
        if isinstance(shaped, np.ndarray):
            shaped = (shaped, )
        self.shaped_actions = shaped
        self.actions_index = {a: i for i, a in enumerate(flatten_array_list(shaped))}

    def _generate_shaped_actions(self) -> Tuple[np.ndarray]:
        """
        Return a tuple of shaped ndarrays of actions, or a single ndarray.

        The default implementation returns `game.actions`.
        """
        return (np.array(self.game.actions, dtype=object), )

    def decode_actions(self, observation: Observation, action_arrays: Tuple[np.ndarray]) -> Distribution:
        """
        Decode a given tuple of likelihood ndarrays to a (normalized) distribution on valid actions.
        """
        # check shapes
        assert len(self.shaped_actions) == len(action_arrays)
        for i in range(len(action_arrays)):
            assert self.shaped_actions[i].shape == action_arrays[i].shape
        policy = flatten_array_list(action_arrays)
        ps = np.zeros(len(self.shaped_actions))
        for i in range(len(observation.actions)):
            ps[i] = policy[self.actions_index[observation.actions[i]]]
        if np.sum(ps) < 1e-30:
            ps = None  # Uniform dstribution
        return Distribution(observation.actions, ps)

    def encode_actions(self, dist: Distribution) -> Tuple[np.ndarray]:
        raise NotImplementedError  # TODO


#def test_adapter():
#    g = Gomoku(3,3,3)
#    ad = Gomoku.TextAdapter(g)
#    s1 = ConsolePlayer(ad, prompt="Player 1")
#    s2 = ConsolePlayer(ad, prompt="Player 2")
#    g.play_strategies([s1, s2])
