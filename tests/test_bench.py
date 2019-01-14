"""
Various benchmarks.
"""

import pytest
from gamegym import games, algorithms, contrib


def goof_seq(n):
    return [1 + i // 3 for i in range(n * 3)]


@pytest.mark.parametrize("gname,g,seq", [
    ("DicePoker", games.DicePoker(), [(1, 1), "continue", "raise", "fold"]),
    ("RPS", games.RockPaperScissors(), ["R", "P"]),
    ("Goofspiel(4)", games.Goofspiel(4), goof_seq(4)),
    ("Goofspiel(6)", games.Goofspiel(6), goof_seq(6)),
])
def test_playthrough(benchmark, gname, g, seq):
    def bench():
        s = g.start()
        for a in seq:
            s = g.play(s, a)

    benchmark(bench)


@pytest.mark.parametrize("gname,g", [
    ("DicePoker", games.DicePoker()),
    ("RPS", games.RockPaperScissors()),
    ("Goofspiel(4)", games.Goofspiel(4)),
    ("Goofspiel(6)", games.Goofspiel(6)),
])
def test_outer_mccfr_sample(benchmark, gname, g):
    mc = algorithms.OutcomeMCCFR(g, seed=52)
    # small warmup
    mc.compute(64)

    def bench():
        mc.sampling(0, 0.6, 1.0)
        mc.sampling(1, 0.6, 1.0)

    benchmark(bench)
