from gamegym.distribution import Explicit, Uniform, EpsilonUniformProxy
import numpy as np


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
