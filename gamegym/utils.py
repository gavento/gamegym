#!/usr/bin/python3

import numpy as np
import contextlib
import logging
import time
import pytest
import pickle


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
