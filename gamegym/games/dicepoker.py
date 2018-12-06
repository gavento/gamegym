from ..game import Game, GameState, Active
from ..utils import uniform


class DicePoker(Game):
    r"""
        chance
         |
         p1
        /| \
       F C  R
         |   \
         p2   p2
        /|\   /\
       F C R  F C
           |
           p1
           /\
           F C

    F - fold
    C - continue
    R - raise (double value)
    """
    ACTIONS1 = ("continue", "raise", "fold")
    ACTIONS2 = ("continue", "fold")

    def __init__(self, dice_size=6, fold_cost=2):
        self.dice_size = dice_size
        self.fold_cost = fold_cost
        dice_combinations = []
        for i in range(dice_size):
            for j in range(dice_size):
                dice_combinations.append((i, j))
        self.dice_combinations = dice_combinations
        self.dice_distribution = uniform(len(self.dice_combinations))
        self.players = 2

    def initial_state(self):
        return ((), Active.new_chance(self.dice_distribution, self.dice_combinations))

    def _player(self, state, action):
        h = state.history
        s = len(h)
        if action == "fold":
            return Active.TERMINAL
        if s == 0:
            return 0
        if s == 1:
            return 1
        if action == "raise" and h[-1] == "continue":
            return 0
        return Active.TERMINAL

    def update_state(self, state, action):
        h = state.history
        s = len(h)
        pair = h[0] if h else action

        if action == "raise":
            actions = self.ACTIONS2
        else:
            actions = self.ACTIONS1
        player = self._player(state, action)

        if player >= 0:
            if s == 1:
                obs = (action[0], action[1], None)
            else:
                obs = (action, ) * 3
            return ((), Active.new_player(player, actions), obs)

        if action == "fold":
            v = self.fold_cost
            if len(h) != 2:
                v = -v
        else:
            v = pair[0] - pair[1]

        if h[-1] == "raise":
            v *= 2
        return ((), Active.new_terminal((v, -v)), (action, ) * 3)
