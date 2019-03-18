import numpy as np
import pytest

from gamegym.algorithms.mcts import buffer


def test_nested():
    d = [
        {'a': 1.0, 'b': (np.array([1,2,3]), np.float128(42), np.array([]), [0, 1, (2, )])},
        {'a': 2.0, 'b': (np.array([4,5,6]), np.float128(43), np.array([]), [1, 2, (3, )])},
        ]
    e = buffer.nested_stack(d)
    assert all(e['a'] == np.array([1.0, 2.0]))
    (b1, b2, b3, b4) = e['b']
    assert (b1 == np.array([[1,2,3], [4,5,6]])).all()
    assert all(b2 == np.array([42, 43]))
    assert b2.dtype == np.float128
    assert b3.shape == (2, 0)
    assert all(b4[0] == np.array([0, 1]))
    assert all(b4[1] == np.array([1, 2]))
    assert all(b4[2][0] == np.array([2, 3]))


def test_buffer():
    R = buffer.ReplayRecord
    rb = buffer.ReplayBuffer(capacity=4)
    rb.add_records([R(i, i, i) for i in range(10)])
    assert [r.values for r in rb.buffer] == [6, 7, 8, 9]
    b = rb.get_batch(2)
    nums = b.action_logits
    assert all(b.values == nums)
    assert all(b.inputs == nums)
    assert b.situations == [None, None]