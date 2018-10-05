import random
import collections


class MCCFR:
    Regret = collections.namedtuple("Regret", ("sum", "cnt"))
    #Action = collections.namedtuple("Action", ("label", "state", "prob"))

    def __init__(self, game):
        assert isinstance(game, GameState)
        self.game = game
        self.r = {}  # (infoset, action_label) -> Regret

    def get_r(self, infoset, action):
        return self.r.setdefault((infoset, action), self.Regret(0.0, 0))

    def add_r(self, infoset, action, val):
        r0 = self.get_r(infoset, action)
        self.r[(infoset, action)] = self.Regret(r0.sum + val)

    def _norm_epsilon_greedy(self, actions, epsilon):
        assert isinstance(actions, (list, tuple))
        s = sum(a[1] for a in actions)
        if s <= 1e-20:
            s = 1.0
            epsilon = 1.0
        return [(label, st, (1.0 - epsilon) * prob / s + epsilon / len(actions))
                for label, st, prob in actions]

    def get_strategy(self, state, epsilon=0.0):
        iset = state.information_set()
        res = []
        for label, st, prob in state.actions():
            if st.player() >= 0: # update for non-chance nodes
                r = self.get_r(iset)
                prob = max(r.sum / max(r.cnt, 1), 0.0)
            res.append((label, st, prob))
        return self._norm_epsilon_greedy(res, epsilon=epsilon)

    def generate_play(self, state0, epsilon=0.0):
        state = state0
        play = [state0]
        while not state.is_terminal():
            self.get_strategy(state, epsilon=epsilon)

            play.append(state)
        return play
