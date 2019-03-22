#!/usr/bin/python3

import attr
import random
import numpy as np

from ...nested import stack

@attr.s
class ReplayRecord:
    input = attr.ib()
    values = attr.ib()
    action_logits = attr.ib()
    situation = attr.ib(default=None)


@attr.s
class ReplayRecordBatch:
    """
    Batch of ReplayRecords with all data stacked.
    """
    inputs = attr.ib()
    values = attr.ib()
    action_logits = attr.ib()
    situations = attr.ib(default=None)

    @classmethod
    def from_records(cls, records):
        return cls(
            stack([i.input for i in records]),
            stack([i.values for i in records]),
            stack([i.action_logits for i in records]),
            [i.situation for i in records],
        )


@attr.s
class ReplayBuffer:
    capacity = attr.ib(default=1000)
    added = attr.ib(default=0)
    sampled = attr.ib(default=0)
    buffer = attr.ib(factory=list)

    def add_records(self, records: ReplayRecord):
        """
        Adds records (or a single record) to the buffer, dropping old records over capacity.
        """
        if isinstance(records, ReplayRecord):
            records = (records,)
        self.buffer.extend(records)
        self.added += len(records)
        if len(self.buffer) > self.capacity:
            self.buffer = self.buffer[-self.capacity:]

    def get_batch(self, batch_size=32):
        """
        Return ReplayRecordBatch sampled from the buffer.
        Assumes the buffer contains enough samples.
        """
        assert len(self.buffer) >= batch_size
        s = random.sample(self.buffer, batch_size)
        self.sampled += batch_size
        return ReplayRecordBatch.from_records(s)

    def stats(self):
        return "samples: {} in, {} sampled (on average {:.2f} times each)".format(
            self.added, self.sampled, self.sampled / self.added)


