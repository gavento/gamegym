#!/usr/bin/python3

import numpy as np
import contextlib
import logging
import time
import pytest
import pickle
from typing import Iterable, Union, Any, Optional, Iterator
from attr import attrs, attrib


def uniform(n: int) -> tuple:
    """
    Return a Uniform(n) distribution as a tuple.
    """
    return (1.0 / n, ) * n


def np_uniform(n: int) -> np.ndarray:
    """
    Return a Uniform(n) distribution as numpy array.
    """
    return np.full(n, 1.0 / n)


def sample_with_p(vals: Union[int, Iterable], probs: Optional[Iterable[float]],
                  rng=None) -> (Any, float):
    """
    Sample a single value according to probabilities.

    Returns `(val, prob_of_val)`.
    `vals` may be indexable or an `int` (then samples from `range(vals)`).
    `probs` may be indexable or `None` (then samples uniformly).
    Warning: repeated values are permitted but return their inidividual probabilities (not their sum).
    """
    return Distribution(vals, probs).sample_with_p(rng)


@attrs(cmp=True, slots=True, init=False)
class Distribution:
    """
    A minimal distribution wrapper.

    `vals` may be indexable or an `int` (then samples from `range(vals)`).
    `probs` may be indexable or `None` (then samples uniformly).
    Optionally perform distribution normalization.

    Warning: repeated values are permitted but `sample_with_p` will return their
    inidividual probabilities (not their sum).
    """

    vals = attrib(type=Union[None, int, Iterable])
    probs = attrib(type=Union[None, Iterable[float]])

    def __init__(self, vals, probs, norm=False):
        assert vals is not None or probs is not None
        self.vals = len(probs) if vals is None else vals
        self.probs = probs
        if norm and self.probs is not None:
            s = np.sum(self.probs)
            assert s > 0.0
            self.probs = self.probs / s

    def sample_with_p(self, rng=None) -> (Any, float):
        """
        Sample a single value according to probabilities, return `(val, prob_of_val)`.
        """
        rng = rng or np.random.RandomState()
        assert isinstance(rng, np.random.RandomState)
        n = self.vals if isinstance(self.vals, int) else len(self.vals)
        i = rng.choice(n, p=self.probs)
        return (i if isinstance(self.vals, int) else self.vals[i],
                self.probs[i] if self.probs is not None else 1.0 / n)

    def sample(self, rng=None) -> Any:
        """
        Sample a single value according to probabilities, return `(val, prob_of_val)`.
        """
        return self.sample_with_p(rng)[0]

    def items(self) -> Iterator:
        """
        Iterator over `(val, prob)` pairs.
        """
        vs = range(self.vals) if isinstance(self.vals, int) else self.vals
        for i in range(len(vs)):
            if self.probs is None:
                yield (vs[i], 1.0 / len(vs))
            else:
                yield (vs[i], self.probs[i])


def debug_assert(cond):
    if hasattr(pytest, "_called_from_pytest"):
        assert cond()


def _open_by_ext(fname, mode, *args, **kwargs):
    if fname.endswith('.bz2'):
        import bz2
        return bz2.BZ2File(fname, mode, *args, **kwargs)
    if fname.endswith('.gz'):
        import gzip
        return gzip.GzipFile(fname, mode, *args, **kwargs)
    if fname.endswith('.xz'):
        import lzma
        return lzma.LZMAFile(fname, mode, *args, **kwargs)
    return open(fname, mode, *args, **kwargs)


def store(obj, fname):
    """
    Pickle object into file, compressing by extension (bz2, gz, xz).
    """
    with _open_by_ext(fname, "w") as f:
        pickle.dump(obj, f, protocol=4)


def load(fname):
    """
    Unpickle object from a file, decompressing by extension (bz2, gz, xz).
    """
    with _open_by_ext(fname, "w") as f:
        return pickle.load(f)


def get_rng(rng=None, seed=None):
    """
    Hepler returning nupy RandomState, either `rng`, new one based on `seed` or random one.
    """
    if rng is not None and seed is not None:
        raise ValueError("provide only one of `seed` and `rng`.")
    if seed is not None:
        rng = np.random.RandomState(seed)
    if rng is None:
        rng = np.random.RandomState()
    if not isinstance(rng, np.random.RandomState):
        raise TypeError("Provided `rng` must be instance of `np.random.RandomState`.")
    return rng
