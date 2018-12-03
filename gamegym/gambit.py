


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

    pls = range(game.players())
    f.write('EFG 2 R "{}" {{ {} }}\n'.format(
        esc(game),
        ' '.join('"Player {}"'.format(p + 1) for p in pls)))
        
    def traverse(state):
        nonlocal outcomes, chance_infosets
        if state.is_terminal():
            v = state.values()
            outcomes += 1
            f.write('t "{}" {} "" {{ {} }}\n'.format(
                escn(state.history),
                outcomes,
                ' '.join("{:.6f}".format(v[p]) for p in pls)))
        elif state.is_chance():
            chance_infosets += 1
            d = state.chance_distribution()
            f.write('c "{}" {} "" {{ {} }} 0\n'.format(
                escn(state.history),
                chance_infosets,
                ' '.join('"{}" {:.6f}'.format(esc(a), p) for a, p in d.items())))
            for a in state.actions():
                traverse(state.play(a))
        else:
            obs = state.player_information(state.player())
            iset = infosets.setdefault(obs, len(infosets) + 1)
            f.write('p "{}" {} {} "OBS{}" {{ {} }} 0\n'.format(
                escn(state.history),
                state.player() + 1, iset,
                escn(obs),
                ' '.join('"{}"'.format(esc(a)) for a in state.actions())))
            for a in state.actions():
                traverse(state.play(a))

    traverse(game.initial_state())

