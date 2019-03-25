#!/usr/bin/python3

import attr
import random
import numpy as np

from ...nested import stack

@attr.s(slots=True)
class ReplayRecord:
    input = attr.ib()
    target_values = attr.ib()
    target_policy_logits = attr.ib()


@attr.s
class ReplayRecordBatch:
    """
    Batch of ReplayRecords with all data stacked.
    """
    inputs = attr.ib()
    target_values = attr.ib()
    target_policy_logits = attr.ib()

    @classmethod
    def from_records(cls, records):
        return cls(
            stack([i.input for i in records]),
            stack([i.target_values for i in records]),
            stack([i.target_policy_logits for i in records]),
        )


@attr.s
class ReplayBuffer:

    capacity = attr.ib(default=1000)
    records = attr.ib(default=attr.Factory(list), init=False)
    added = attr.ib(default=0, init=False)
    sampled = attr.ib(default=0, init=False)

    def add_record(self, record):
        """
        Adds records to the buffer, dropping old records over capacity.
        """
        records = self.records
        capacity = self.capacity
        if len(records) < capacity:
            records.append(record)
        else:
            records[self.added % capacity] = record
        self.added += 1

    @property
    def records_count(self):
        return len(self.records)

    def get_batch(self, batch_size):
        """
        Return ReplayRecordBatch sampled from the buffer.
        Assumes the buffer contains enough samples.
        """
        assert len(self.records) >= batch_size
        s = random.sample(self.records, batch_size)
        self.sampled += batch_size
        return ReplayRecordBatch.from_records(s)

    def stats(self):
        return "samples: {} in, {} sampled (on average {:.2f} times each)".format(
            self.added, self.sampled, self.sampled / self.added)
