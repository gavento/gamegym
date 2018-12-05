def write_efg(game, f, names=False):
    """
    Write out the game tree in Gambit EFG text format.

    Optionally also write names of all the nodes, actions and info-sets.
    """

    def esc(s):
        return str(s).replace('"', '\\"').replace("\n", " ")

    def escn(s):
        return esc(s) if names else ""

    # infoset: (player, obs) -> int
    infosets = {}
    # counters for unique outcomes and chance infosets
    outcomes = 0
    chance_infosets = 0

    pls = range(game.players)
    f.write('EFG 2 R "{}" {{ {} }}\n'.format(
        esc(game), ' '.join('"Player {}"'.format(p + 1) for p in pls)))

    def traverse(state):
        nonlocal outcomes, chance_infosets
        if state.active.is_terminal():
            v = state.active.payoff
            outcomes += 1
            f.write('t "{}" {} "" {{ {} }}\n'.format(
                escn(state.history), outcomes, ' '.join("{:.6f}".format(v[p]) for p in pls)))
        elif state.active.is_chance():
            chance_infosets += 1
            d = state.active.chance
            f.write('c "{}" {} "" {{ {} }} 0\n'.format(
                escn(state.history), chance_infosets, ' '.join(
                    '"{}" {:.6f}'.format(esc(a), p) for a, p in zip(state.active.actions, d))))
            for a in state.active.actions:
                traverse(game.play(state, a))
        else:
            obs = state.observations[state.active.player]
            iset = infosets.setdefault(obs, len(infosets) + 1)
            f.write('p "{}" {} {} "OBS{}" {{ {} }} 0\n'.format(
                escn(state.history), state.active.player + 1, iset, escn(obs),
                ' '.join('"{}"'.format(esc(a)) for a in state.active.actions)))
            for a in state.active.actions:
                traverse(game.play(state, a))

    traverse(game.start())
