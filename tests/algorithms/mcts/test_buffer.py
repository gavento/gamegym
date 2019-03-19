import numpy as np
import pytest

from gamegym.algorithms.mcts import buffer


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