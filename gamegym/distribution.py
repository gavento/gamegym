#!/usr/bin/python3

import collections
import random
import numpy as np

from .utils import get_rng


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
        return np.array((self.probability(v) for v in self.values()))


class Explicit(Discrete):
    """
    Discrete distribution determined by given probabilities.

    Optionally normalizes the values to sum to 1.0,
    otherwise they are checked to sum to 1.0 +- 1e-6.
    """
    def __init__(self, probs, values=None, *, normalize=False):
        if isinstance(probs, dict):
            assert values is None
            self._values = tuple(probs.keys())
            self._probs = tuple(probs.values())
        elif isinstance(probs, collections.Iterable):
            self._probs = np.array(probs)
            self._values = values
        else:
            raise TypeError("probs must be dict or iterable")
        if normalize:
            self._probs = self._probs / np.sum(self._probs)
        # Cumulative sum for fast sampling and to check the overall sum
        self._sums = np.cumsum(self._probs)
        if abs(self._sums[-1] - 1.0) > 1e-6:
            raise ValueError("given probabilities do not sum to 1.0 +- 1e-6")
        if self._values is None:
            self._values = np.arange(len(self._probs), dtype=int)
        else:
            assert len(self._probs) == len(self._values)
        self._valindex = {v: i for i, v in enumerate(self._values)}

    def sample(self, *, rng=None, seed=None):
        p = get_rng(rng, seed).rand()
        return self._values[np.searchsorted(self._sums, p)]

    def probability(self, value):
        return self._probs[self._valindex[value]]

    def values(self):
        return self._values

    def probabilities(self):
        return self._probs

    def __repr__(self):
        return "<Explicit {} {}>".format(self._probs, self._values)


class EpsilonUniformProxy(Discrete):
    """
    Wrap given distribution by sampling a uniformly random value
    with probability epsilon, and sampling the actual distribution otherwise.

    Does not generate the probability list unless requested.
    However, needs the value list from `dist.values()`.
    """
    def __init__(self, dist, epsilon):
        self.dist = dist
        self.epsilon = epsilon

    def sample(self, *, rng=None, seed=None):
        rng = get_rng(rng, seed)
        if rng.rand() < self.epsilon:
            return rng.choice(self.dist.values())
        return self.dist.sample(rng=rng)

    def values(self):
        return self.dist.values()

    def probability(self, value):
        return (self.epsilon * 1 / len(self.values()) +
                (1.0 - self.epsilon) * self.dist.probability(value))

    def probabilities(self):
        ps = np.array(self.dist.probabilities())
        return (self.epsilon * 1 / len(self.values()) +
                (1.0 - self.epsilon) * ps)


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
        rng = get_rng(rng, seed)
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
        p = self.probability(None)  # Hacky?
        return np.zeros(len(self.values()), dtype=float) + p


def test_unit():
    for d in [
        Explicit([0.3, 0.2, 0.5]),
        Explicit([0.3, 4.2, 1.5], normalize=True),
        Explicit([0.3, 0.2, 0.5], ["A", "B", "C"]),
        Uniform(10),
        Uniform([3, 6, 9]),
        EpsilonUniformProxy(Explicit([0.3, 0.2, 0.5]), 0.5),
    ]:
        d.sample()
        d.sample(seed=42)
        d.sample(rng=np.random)
        d.values()
        d.probability(d.values()[0])
        d.probabilities()
