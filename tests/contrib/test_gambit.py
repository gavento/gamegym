import pytest
import io
from gamegym.games import Goofspiel
from gamegym.contrib.gambit import write_efg, parse_strategy
from gamegym.algorithms import exploitability


def test_dump_gambit_game():
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


def test_parse_gambit_strategy_g2():
    g = Goofspiel(2, scoring=Goofspiel.Scoring.ZEROSUM)
    txt = "NE,1,0,1,1,1,1,0,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1"
    strats = parse_strategy(g, txt)
    assert exploitability(g, 0, strats[0]) < 1e-6
    assert exploitability(g, 0, strats[1]) < 1e-6


@pytest.mark.xfail(reason="Waiting for analsis of verbose output from `gambit-lcp -D g3.efg`")
def test_parse_gambit_strategy_g3():
    g = Goofspiel(3, scoring=Goofspiel.Scoring.ZEROSUM)
    txt = "NE,1,0,0,1,0,1,1,1,1,0,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,1,0,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1,1,0,1,1,1,0,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1,0,0,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,0,0,1,1,1,0,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1,1,0,0,1,0,1,1,1,1,0,1,1,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1,0,0,1,1,1,1,0,0,1,1,1,1,1,0,1,1,1,0,1,1,1,1,0,1,1,1,1,1,0,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1,0,0,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,0,0,1,1,1,0,1,1,1,1,0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,0,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1"
    strats = parse_strategy(g, txt)
    assert exploitability(g, 0, strats[0]) < 1e-6
    assert exploitability(g, 0, strats[1]) < 1e-6
