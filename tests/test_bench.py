"""
Various benchmarks.
"""

import pytest
from gamegym import games, algorithms, contrib


@pytest.mark.parametrize("gname,g,seq,reuse", [
    ("DicePoker", games.DicePoker(), [0, 1, 0], False),
    ("DicePoker", games.DicePoker(), [0, 1, 0], True),
    ("RPS", games.RockPaperScissors(), [0, 0], False),
    ("RPS", games.RockPaperScissors(), [0, 0], True),
    ("Goofspiel(4)", games.Goofspiel(4), [0] * 12, False),
    ("Goofspiel(4)", games.Goofspiel(4), [0] * 12, True),
    ("Goofspiel(6)", games.Goofspiel(6), [0] * 18, False),
    ("Goofspiel(6)", games.Goofspiel(6), [0] * 18, True),
])
def test_playthrough(benchmark, gname, g, seq, reuse):
    def bench():
        s = g.start()
        for i in seq:
            s = g.play(s, index=i, reuse=reuse)

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
