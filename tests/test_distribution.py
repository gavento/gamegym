from gamegym.distribution import Explicit, Uniform, EpsilonUniformProxy
import numpy as np
import pytest


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
        d.sample(rng=np.random.RandomState())
        d.values()
        d.probability(d.values()[0])
        d.probabilities()
        v, p = d.sample_with_p()
        assert d.probability(v) == p


def test_uniform():
    for u in [Uniform(3), Uniform(['a', 'b', 'c'])]:
        assert u.values() == (0, 1, 2) or u.values() == ('a', 'b', 'c')
        s = [u.sample() for i in range(100)]
        for i in u.values():
            assert s.count(i) > 20
        sp = [u.sample_with_p() for i in range(100)]
        for i in u.values():
            assert sp.count((i, pytest.approx(1 / 3.0))) > 20
