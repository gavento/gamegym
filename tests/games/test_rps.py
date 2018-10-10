from gamegym.games.rps import RockPaperScissors

def test_base():
    g = RockPaperScissors()
    s = g.initial_state()
    repr(s)
    repr(g)
    assert not s.is_terminal()
    assert s.player() == 0
    assert len(s.actions()) == 3
    s1 = s.play(s.actions()[0])
    assert not s1.is_terminal()
    assert s1.player() == 1
    assert len(s1.actions()) == 3
    s2 = s1.play(s1.actions()[1])
    assert s2.is_terminal()
    assert s2.values() == (-1, 1)
