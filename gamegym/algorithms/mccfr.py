#!/usr/bin/python3

import random
import collections
from ..strategy import Strategy


class MCCFR(Strategy):
    Regret = collections.namedtuple("Regret", ("sum", "cnt"))

    def __init__(self, game, seed=None, rng=None):
        super().__init__(game, seed=seed, rng=rng)
        self.regret = {}  # (infoset, action_label) -> Regret

    def get_regret(self, infoset, action):
        return self.regret.setdefault((infoset, action), self.Regret(0.0, 0))

    def add_regret(self, infoset, action, r):
        r0 = self.get_regret(infoset, action)
        self.regret[(infoset, action)] = self.Regret(r0.sum + r)

    def distribution(self, state, epsilon=0.0):
        iset = state.information_set()
        res = []
        if state.is_terminal():
            return []
        for label, st, prob in state.actions():
            if st.player() >= 0: # update for non-chance nodes
                r = self.get_r(iset)
                prob = max(r.sum / max(r.cnt, 1), 0.0)
            res.append(self.game.NextAction(label, st, prob))
        return self.make_normalized_epsilon_greedy(res, epsilon=epsilon)



    def generate_play(self, state0, epsilon=0.0):
        state = state0
        play = [state0]
        while not state.is_terminal():
            self.get_strategy(state, epsilon=epsilon)

            play.append(state)
        return play
