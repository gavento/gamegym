from ..strategy import DictStrategy


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
        if state.is_terminal():
            v = state.payoff
            outcomes += 1
            f.write('t "{}" {} "" {{ {} }}\n'.format(
                escn(state.history), outcomes, ' '.join("{:.6f}".format(v[p]) for p in pls)))
        elif state.is_chance():
            chance_infosets += 1
            d = state.chance
            f.write('c "{}" {} "" {{ {} }} 0\n'.format(
                escn(state.history), chance_infosets,
                ' '.join('"{}" {:.6f}'.format(esc(a), p) for a, p in zip(state.actions, d))))
            for a in state.actions:
                traverse(game.play(state, a))
        else:
            obs = state.observations[state.player]
            iset = infosets.setdefault(obs, len(infosets) + 1)
            f.write('p "{}" {} {} "OBS{}" {{ {} }} 0\n'.format(
                escn(state.history), state.player + 1, iset, escn(obs),
                ' '.join('"{}"'.format(esc(a)) for a in state.actions)))
            for a in state.actions:
                traverse(game.play(state, a))

    traverse(game.start())


def parse_strategy(game, s):
    d = s.strip().split(",")
    if d[0].strip() not in ("end", "NE"):
        raise Exception("Unknown strategy tag {r}".format(d[0]))
    probs = [float(x) for x in d[1:]]

    # infoset: player -> (obs -> int)
    infosets = [set() for _ in range(game.players)]
    # obss: player -> infoset_no -> (obs, actions)
    obss = [[] for _ in range(game.players)]

    def traverse(state):
        if state.is_terminal():
            pass
        elif state.is_chance():
            for a in state.actions:
                traverse(game.play(state, a))
        else:
            p = state.player
            obs = state.observations[p]
            if obs not in infosets[p]:
                infosets[p].add(obs)
                obss[p].append((obs, len(state.actions)))
            for a in state.actions:
                traverse(game.play(state, a))

    traverse(game.start())

    dicts = [{} for _ in range(game.players)]
    for p in range(game.players):
        for obs, actions in obss[p]:
            ps = tuple(probs[:actions])
            if len(ps) < actions:
                raise Exception("Not enough values, got {} for observation {} ({} actions)".format(
                    ps, obs, actions))
            if abs(sum(ps) - 1.0) >= 0.01:
                raise Exception("Player {} info set {} probabilities {} do not sum to 1.0".format(
                    p, obs, ps))
            probs = probs[actions:]
            dicts[p][obs] = ps
    return [DictStrategy(d) for d in dicts]
