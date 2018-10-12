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
        raise NotImplementedError

    def sample_with_p(self, *, rng=None, seed=None):
        """
        Same as `sample()` but returns `(value, p_sampled)`.
        """
        raise NotImplementedError

    def values(self):
        """
        Return a tuple or numpy array with values.

        NOTE: This is explicitely a function since some distributions may need to generate
        the list but may not need it to just sample.
        """
        raise NotImplementedError

    def probability(self, value):
        """
        Return the probability of a single value.

        NOTE: May not be implemented if values are not hashable, or just very slow.
        """
        raise NotImplementedError

    def probabilities(self):
        """
        Return a tuple or numpy array with value probabilities.
        The default implementation computes a tuple using `self.probability()`.

        NOTE: This is explicitely a function since some distributions may need to generate
        the list but may not need it to just sample.
        """
        return np.array((self.probability(v) for v in self.values()))


class Explicit(Discrete):
    """
    Discrete distribution determined by given probabilities.

    Optionally normalizes the values to sum to 1.0,
    otherwise they are checked to sum to 1.0 +- 1e-6.

    With `index=True` (default) allows probability lookups for values and
    requires hashable values. Otherwise the values may be arbitrary.
    """
    def __init__(self, probs, values=None, *, normalize=False, index=True):
        if isinstance(probs, dict):
            assert values is None
            self._values = tuple(probs.keys())
            self._probs = np.fromiter(probs.values(), dtype=np.float32)
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

        if index == True:
            self._valindex = {v: i for i, v in enumerate(self._values)}
        else:
            self._valindex = None

    def sample(self, *, rng=None, seed=None):
        p = get_rng(rng, seed).rand()
        return self._values[np.searchsorted(self._sums, p)]

    def sample_with_p(self, *, rng=None, seed=None):
        p = get_rng(rng, seed).rand()
        idx = np.searchsorted(self._sums, p)
        return self._values[idx], self._probs[idx]

    def probability(self, value):
        if self._valindex is None:
            raise Exception("Value index not present (i.e. index=False was given to __init__)")
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

    def sample_with_p(self, *, rng=None, seed=None):
        v = self.sample(rng = rng, seed=seed)
        return (v, self.probability(v))

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

    def sample_with_p(self, *, rng=None, seed=None):
        rng = get_rng(rng, seed)
        if isinstance(self._values, int):
            return (rng.randint(0, self._values - 1), 1.0 / self._values)
        else:
            return (rng.choice(self._values), 1.0 / len(self._values))

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

