from typing import Any, Dict, List, Tuple, Union

import numpy as np


NestedArray = Union[np.ndarray, List['NestedArrays'], Tuple['NestedArrays'], Dict[str, 'NestedArrays']]


def stack(insts: List[NestedArray]) -> NestedArray:
    """
    Given a list of NestedArrays, return a NestedArray where the leaves are stacked.
    """
    assert isinstance(insts, list)
    t = insts[0]
    if isinstance(t, (np.ndarray, float, int, np.floating)):
        return np.stack(insts)
    if isinstance(t, (list, tuple)):
        return tuple(stack([i[k] for i in insts]) for k in range(len(t)))
    if isinstance(t, dict):
        return {k: stack([i[k] for i in insts]) for k in t.keys()}
    raise TypeError("Unknown nested type {}".format(type(t)))


def flatten(nested, dtype=None) -> np.ndarray:
    """
    Flatten the nested array structure into a 1D ndarray.

    If `dtype is object`, allows any data, i.e. allows non-list/dict/tuple/ndarray
    objects. Dicts are flattened ordered by `sorted(keys)`.
    """
    r = [np.array([], dtype=dtype)]

    def _rec(n):
        if isinstance(n, np.ndarray):
            r.append(n.flatten())
        elif isinstance(n, (list, tuple)):
            for i in n:
                _rec(i)
        elif isinstance(n, dict):
            ks = sorted(n.keys())
            for k in ks:
                _rec(n[k])
        elif isinstance(n, (float, int, np.floating)):
            r.append(np.array([n], dtype=dtype))
        else:
            if dtype is object:
                r.append(np.array([n], dtype=dtype))
            else:
                raise TypeError("Unknown nested type {}".format(type(n)))

    _rec(nested)
    return np.concatenate(r)
