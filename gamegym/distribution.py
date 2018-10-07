#!/usr/bin/python3

import collections
import random
import numpy as np


class Discrete:
    """
    Base class for discrete distributions, mostly used for actions.
    Includes various proxy distributions, can support large action sets without
    explicitly enumerating them.
    """
    def sample(self, *, rng=None, seed=None):
        """
        Generate a single sample from the distribution.
        If neither `rgn` nor `seed` is provided, uses `random`.
        """
        raise NotImplemented

    def values(self):
        """
        Return a tuple or numpy array with values.

        (Explicitely a function since some distributions may need to generate
        the list but may not need it to sample.)
        """
        raise NotImplemented

    def probability(self, value):
        """
        Return the probability of a single value.
        """
        raise NotImplemented

    def probabilities(self):
        """
        Return a tuple or numpy array with value probabilities.
        The default implementation computes a tuple using `self.probability()`.

        (Explicitely a function since some distributions may need to generate
        the list but may not need it to sample.)
        """
        return tuple(self.probability(v) for v in self.values())


def _get_rng(rng=None, seed=None):
    "Hepler returning given `rng`, new one based on `seed` or `random`."
    if rng is not None and seed is not None:
        raise ValueError("provide ony one of `seed` and `rng`.")
    if seed is not None:
        rng = random.Random(seed)
    if rng is None:
        rng = random
    return rng


class Explicit(Discrete):
    """
    Discrete distribution determined by given probabilities.

    Optionally normalizes the values to sum to 1.0,
    otherwise they are checked to sum to 1.0 +- 1e-6.
    """
    def __init__(self, probs, values=None, *, normalize=False):
        self._probs = np.array(probs)
        if normalize:
            self._probs = self._probs / np.sum(self._probs)
        # Cumulative sum for fast sampling and to check the overall sum
        self._sums = np.cumsum(self._probs)
        if abs(self._sums[-1] - 1.0) > 1e-6:
            raise ValueError("given probabilities do not sum to 1.0 +- 1e-6")
        self._values = values
        if self._values is None:
            self._values = np.arange(len(self._probs), dtype=int)
        else:
            assert len(self._probs) == len(self._values)
        self._valindex = {v: i for i, v in enumerate(self._values)}

    def sample(self, *, rng=None, seed=None):
        p = _get_rng(rng, seed).random()
        return self._values[np.searchsorted(self._sums, p)]

    def probability(self, value):
        return self._probs[self._valindex[value]]

    def values(self):
        return self._values

    def probabilities(self):
        return self._probs


class Uniform(Discrete):
    """
    An uniform distributon over the given values (iterable or number).
    If `values` is number, range `0..values - 1` is used and the full list is
    not actually generated for sampling.
    """
    def __init__(self, values):
        self._values = values
        if not isinstance(self._values, (int, collections.Iterable)):
            raise TypeError("Integer or iterable needed")
        if not isinstance(self._values, (int, tuple)):
            self._values = tuple(self._values)

    def sample(self, *, rng=None, seed=None):
        rng = _get_rng(rng, seed)
        if isinstance(self._values, int):
            return rng.randint(0, self._values - 1)
        else:
            return rng.choice(self._values)

    def probability(self, value):
        if isinstance(self._values, int):
            return 1.0 / self._values
        return 1.0 / len(self._values)

    def values(self):
        if isinstance(self._values, int):
            self._values = tuple(range(self._values))
        return self._values

    def probabilities(self):
        p = self.probability(None)
        return tuple(p for v in self.values())


def test_dists():
    a = Explicit([0.3, 0.2, 0.5])
    a.sample()
    b = Uniform(10)
    b.sample()
    c = Uniform(["a", "b"])
    c.sample()
