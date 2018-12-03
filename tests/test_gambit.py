import pytest
import io
from gamegym.games import Goofspiel
from gamegym.gambit import write_efg


def test_dump_gambit():
    g = Goofspiel(3, scoring=Goofspiel.Scoring.ZEROSUM)

    s = io.StringIO()
    write_efg(g, s, names=False)
    assert(len(s.getvalue()) > 1024)

    s = io.StringIO()
    write_efg(g, s, names=True)
    assert(len(s.getvalue()) > 1024)

    with open('gs3.efg', 'wt') as f:
        write_efg(g, f, names=False)
