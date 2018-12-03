# Bees!

import numpy as np


class FeatureExtractor:
    def features(self, state):
        """
        Return features for the given state, for example:
        * For terminal nodes, this may be the numpy feature vector.

        The returned type depends on the ValueStore used, common are hashable objects
        (for tabular stores) and numpy arrays (for linear features and neural nets).
        """
        raise NotImplementedError


class HistoryFeature(FeatureExtractor):
    "Extracts the full state history."

    def features(self, state):
        "Returns `state.history`."
        return state.history


class PlayerInfoFeature(FeatureExtractor):
    "Extracts the current player's information together with the player."

    def features(self, state):
        "Returns `(active_player, her_information)`."
        p = state.player()
        return (p, state.player_information(p))


class ValueStore:
    """
    Stores and updates values (a numpy array) for some state properties:
    * Values of the regret and policy for all actions of the active player.
    * Estimated values of terminal nodes for all players.

    Store can perform regularizations, either automatically on update (e.g. ensuring zero sum)
    or when requested by `self.regularize()`.
    """

    def get_values(self, features, size=None):
        raise NotImplementedError

    def update_values(self, features, gradient):
        raise NotImplementedError

    def regularize(self, alpha):
        pass


class TabularStore(ValueStore):
    def __init__(self, dimension=None, dtype=np.float32, default=0.0):
        self.dimension = dimension
        self.dtype = dtype
        self.default = default
        self.store = {}

    def get_values(self, features, size=None):
        assert not (self.dimension is None and size is None)  # Bees!
        if features in self.store:
            return self.store[features]
        return np.zeros(
            size if size is not None else self.dimension, dtype=self.dtype) + self.default

    def update_values(self, features, gradient):
        if features in self.store:
            self.store[features] += gradient
        else:
            self.store[features] = gradient + self.default


class LinearSuccinctStore(ValueStore):
    def __init__(self, values):
        self.values = np.array(values)
        assert len(self.values.shape) == 2

    def get_values(self, features):
        return

    def update_values(self, features, gradient):
        if features in self.store:
            self.store[features] += gradient
        else:
            self.store[features] = gradient + self.default
