import pytest
import io
from gamegym.games import Goofspiel
from gamegym.gambit import write_efg


def test_dump_gambit():
    g = Goofspiel(3, scoring=Goofspiel.Scoring.ZEROSUM)

    s = io.StringIO()
    write_efg(g, s, names=False)
    assert (len(s.getvalue()) > 1024)

    s = io.StringIO()
    write_efg(g, s, names=True)
    assert (len(s.getvalue()) > 1024)

    g2 = Goofspiel(2, scoring=Goofspiel.Scoring.WINLOSS)
    s = io.StringIO()
    write_efg(g2, s, names=True)
    assert len(s.getvalue().splitlines()) == 40
