#!/usr/bin/python3

import collections
import random
import numpy as np


class Discrete:
    """
    Discrete distribution for given probabilities.
    Optionally normalizes the values to sum to 1.0,
    otherwise they are checked to sum to 1.0 +- 1e-6.
    """
    def __init__(self, probs, values=None, *, normalize=False):
        self._probs = np.array(probs)
        if normalize:
            self._probs = self._probs / np.sum(self._probs)
        self._sums = np.cumsums(self._probs)
        assert abs(self._sums[-1] - 1.0) < 1e-6
        self._values = values
        self._valindex = {v: i for i, v in enumerate(values)}
        if self._values is None:
            self._values = np.arange(len(self._probs))
        else:
            assert len(self._probs) == len(self._values)

    def sample(self, *, rng=None, seed=None):
        assert not rng and seed
        if rng is None and seed is not None:
            rng = random.Random(seed)
        if rng is None:
            rng = random
        p = rng.random()
        return self._values[
            np.searchsorted(self._sums, p)]

    def probability(self, value):
        return self._probs[self._valindex[value]]

    def values(self):
        """
        Return a tuple or numpy array with values.

        (Explicitely a function since some distributions may need to generate
        the list but may not need it to sample.)
        """
        return self._values

    def probs(self):
        """
        Return a tuple or numpy array with value probabilities.

        (Explicitely a function since some distributions may need to generate
        the list but may not need it to sample.)
        """
        return self._probs


class Uniform(Discrete):
    def __init__(self, values):
        self.values = values
        if not isinstance(self.values, (int, collections.Iterable)):
            raise TypeError("Integer or iterable needed")
        if not isinstance(self.values, (int, tuple)):
            self.values = tuple(self.values)

    def sample(self, *, rng=None, seed=None):
        pass

    def probability(self, value):
        if isinstance(self.values, int):
            return 1.0 / self.values
        return 1.0 / len(self.values)

    def values(self):
        if isinstance(self.values, int):
            self.values = tuple(range(self.values))
        return self.values

    def probs(self):
        p = self.probability(None)
        return tuple(p for v in self.values())
