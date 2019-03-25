import numpy as np
import pytest

from gamegym.algorithms.mcts import buffer


def test_buffer():
    R = buffer.ReplayRecord
    rb = buffer.ReplayBuffer(capacity=3)

    v = []
    for i in range(10):
        v.append(R(i, i, i))
        rb.add_record(R(i, i, i))
        assert sorted(rb.records) == sorted(v[-3:])

    b = rb.get_batch(2)
    assert len(b.inputs) == 2
    assert len(b.target_policy_logits) == 2

    b = rb.get_batch(3)
    assert sorted(b.inputs) == [7, 8, 9]
    assert sorted(b.target_policy_logits) == [7, 8, 9]